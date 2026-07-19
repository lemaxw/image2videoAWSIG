import tempfile
import unittest
from pathlib import Path

from PIL import Image

from services.decision.semantic_planner import build_memory_query, compile_semantic_plan, sanitize_analysis_for_decision
from services.orchestrator.comfy_client import ComfyClient, tiled_vae_decode_node
from services.orchestrator.mux import _normalized_video_filter
from services.orchestrator.review import update_feedback
from services.orchestrator.run_batch import _video_variants_for_decision
from services.orchestrator.validate import _compose_audio_prompt, validate_and_clamp_decision


def plan(backend="wan22", operation="static"):
    return {
        "classification": {
            "scene_classes": ["landscape"],
            "environment": "test",
            "important_subjects": [],
            "incidental_subjects": [],
            "sensitive_content": [],
            "preservation_risk": "medium",
        },
        "motion_plan": {
            "primary_target": "clouds",
            "primary_action": "drift",
            "secondary_target": "",
            "secondary_action": "",
            "keep_stable": ["geography"],
        },
        "generation": {
            "mode": "generative" if backend != "none" else "deterministic",
            "backend": backend,
            "prompt": "Clouds drift while geography remains stable. Preserve the original composition and viewpoint.",
            "negative_prompt": "flicker, jitter, unstable geometry, low quality",
            "candidate_count": 2,
            "reason": "test",
        },
        "presentation": {
            "aspect": "square_1_1",
            "operation": operation,
            "crop_anchor": "center_center",
            "pan_start": 0.42,
            "pan_end": 0.50,
            "must_keep_visible": ["geography"],
        },
        "audio": {"prompt": "quiet natural ambience, no music", "duration_s": 5},
    }


class SemanticPipelineTests(unittest.TestCase):
    def test_wan_profile_for_environmental_scene(self):
        analysis = {
            "summary": "mountain valley and stream",
            "people": [],
            "subjects": [{"label": "valley", "spatial": {"relative_size": "dominant"}}],
            "dynamic_potential": {"natural_motion_elements": ["clouds", "stream"]},
            "confidence": {"overall": 0.9},
        }
        decision = compile_semantic_plan(plan(), analysis, "a" * 64)
        self.assertEqual(decision["video"]["preset"], "WAN22_NATURAL")
        self.assertEqual((decision["video"]["frames"], decision["video"]["fps"]), (97, 20))
        self.assertEqual((decision["video"]["params"]["steps"], decision["video"]["params"]["shift"]), (20, 8.0))
        self.assertTrue(decision["video"]["params"]["preserve_source_aspect"])
        self.assertFalse(decision["video"]["params"]["tiled_vae"])

    def test_important_vehicle_forces_hunyuan(self):
        analysis = {
            "people": [{"label": "two riders", "spatial": {"relative_size": "medium"}}],
            "subjects": [{"label": "red scooter", "spatial": {"relative_size": "medium"}}],
            "composition": {"focal_points": ["red scooter"]},
            "confidence": {"overall": 0.9},
        }
        decision = compile_semantic_plan(plan(), analysis, "b" * 64)
        self.assertEqual(decision["video"]["preset"], "HUNYUAN15_I2V_720P")
        self.assertEqual((decision["video"]["frames"], decision["video"]["fps"]), (61, 12))
        self.assertEqual(decision["video"]["params"]["steps"], 50)
        self.assertTrue(decision["video"]["params"]["tiled_vae"])

    def test_hunyuan_focus_follows_people_instead_of_rigid_breeze_target(self):
        street_plan = plan(operation="push_in")
        street_plan["presentation"].update(
            {"aspect": "instagram_reel_9_16", "focus_target": "food cart", "must_keep_visible": ["food cart", "people"]}
        )
        street_plan["motion_plan"] = {
            "primary_target": "food cart",
            "primary_action": "the cart wheels and grill sway in a subtle breeze",
            "secondary_target": "people",
            "secondary_action": "the seated people gently adjust their posture",
            "keep_stable": ["stone building", "shop entrance"],
        }
        analysis = {
            "image_metadata": {"aspect_ratio": 1.5},
            "subjects": [
                {"label": "food cart", "spatial": {"relative_size": "large"}},
                {"label": "people", "spatial": {"relative_size": "medium"}},
            ],
            "composition": {
                "attention_regions": [
                    {"label": "food cart", "region": "0.0 0.17 0.46 0.99"},
                    {"label": "people", "region": "0.59 0.34 0.67 0.55"},
                ]
            },
            "people": [],
        }
        decision = compile_semantic_plan(street_plan, analysis, "h" * 64)
        params = decision["video"]["params"]
        self.assertEqual(decision["video"]["preset"], "HUNYUAN15_I2V_720P")
        self.assertEqual(decision["semantic_plan"]["motion_plan"]["primary_target"], "people")
        self.assertEqual(params["focus_region"]["label"], "people")
        self.assertAlmostEqual(params["zoom_focus_x"], 0.63)
        self.assertIn("food cart", decision["semantic_plan"]["motion_plan"]["keep_stable"])
        self.assertNotIn("cart wheels", params["prompt"].lower())

    def test_incidental_small_vehicle_does_not_force_hunyuan(self):
        analysis = {
            "people": [],
            "objects": [{"label": "vehicles", "spatial": {"relative_size": "small"}}],
            "composition": {"focal_points": ["winding road"]},
            "confidence": {"overall": 0.8},
        }
        decision = compile_semantic_plan(plan(), analysis, "c" * 64)
        self.assertEqual(decision["video"]["preset"], "WAN22_NATURAL")

    def test_low_minimal_motion_still_uses_wan(self):
        analysis = {
            "people": [],
            "objects": [{"label": "vehicles", "spatial": {"relative_size": "small"}}],
            "dynamic_potential": {
                "level": "low",
                "natural_motion_elements": ["clouds", "vehicles", "depth layers"],
                "notes": "Clouds and vehicles provide minimal motion cues",
            },
            "confidence": {"overall": 0.9},
        }
        decision = compile_semantic_plan(plan(), analysis, "f" * 64)
        self.assertEqual(decision["video"]["preset"], "WAN22_NATURAL")

    def test_mannequins_are_not_routed_as_people(self):
        analysis = {
            "people": [{"label": "display mannequins", "spatial": {"relative_size": "large"}}],
            "subjects": [{"label": "mannequins wearing garments", "spatial": {"relative_size": "dominant"}}],
            "composition": {"focal_points": ["mannequins wearing garments"]},
            "dynamic_potential": {
                "level": "low",
                "natural_motion_elements": ["depth layers"],
                "notes": "No visible motion elements or camera motion cues.",
            },
            "confidence": {"overall": 0.9},
        }
        requested_hunyuan = plan("hunyuan15")
        decision = compile_semantic_plan(requested_hunyuan, analysis, "e" * 64)
        self.assertEqual(decision["video"]["preset"], "WAN22_NATURAL")

    def test_presentation_enum_underscores_survive_compilation(self):
        decision = compile_semantic_plan(
            plan(operation="pan_right_to_left"),
            {"people": [], "confidence": {"overall": 0.8}},
            "p" * 64,
        )
        self.assertEqual(decision["framing"], {"target_aspect": "square_1_1", "crop_anchor": "center_center"})
        self.assertEqual(decision["video"]["params"]["final_crop_motion"], "pan_right_to_left")

    def test_city_static_plan_gets_subtle_deterministic_push(self):
        city_plan = plan(operation="static")
        city_plan["classification"]["scene_classes"] = ["city", "architecture"]
        decision = compile_semantic_plan(
            city_plan,
            {"people": [], "dynamic_potential": {"camera_motion_affordances": ["gentle push toward skyline"]}},
            "u" * 64,
        )
        self.assertEqual(decision["video"]["params"]["final_crop_motion"], "push_in")

    def test_high_risk_full_width_panorama_uses_square_traversal(self):
        city_plan = plan(operation="push_in")
        city_plan["classification"]["scene_classes"] = ["urban landscape", "cityscape", "panoramic view"]
        city_plan["presentation"].update({"aspect": "instagram_reel_9_16", "pan_start": 0.5, "pan_end": 0.5})
        analysis = {
            "people": [],
            "composition": {"layout": "panoramic", "focal_points": ["cityscape", "road"]},
            "reframe_constraints": {
                "wide_composition": True,
                "full_width_important_content": True,
                "vertical_crop_risk": "high",
            },
            "confidence": {"overall": 0.9},
        }
        decision = compile_semantic_plan(city_plan, analysis, "j" * 64)
        params = decision["video"]["params"]
        self.assertEqual(decision["framing"]["target_aspect"], "square_1_1")
        self.assertEqual(params["output_aspect"], "square_1_1")
        self.assertEqual(params["final_crop_motion"], "pan_left_to_right")
        self.assertEqual((params["pan_start"], params["pan_end"], params["pan_max_span"]), (0.1, 0.8, 0.7))
        self.assertEqual(params["visibility_validation"]["status"], "traversal")
        validated = validate_and_clamp_decision(decision)
        self.assertEqual(validated["video"]["params"]["pan_max_span"], 0.7)

    def test_framed_landscape_push_enters_opening(self):
        framed_plan = plan(operation="static")
        framed_plan["classification"]["scene_classes"] = ["city", "architecture"]
        analysis = {
            "summary": "Urban skyline viewed through a modern architectural structure",
            "composition": {
                "layout": "framed",
                "focal_points": ["urban skyline"],
                "foreground": ["foreground structure"],
                "attention_regions": [{"label": "urban skyline", "description": "visible through the architectural frame"}],
            },
            "spatial_map": {
                "primary_regions": [
                    {"label": "urban skyline", "center": {"x": 0.535, "y": 0.625}},
                    {"label": "foreground structure", "center": {"x": 0.495, "y": 0.275}},
                ]
            },
            "people": [],
        }
        decision = compile_semantic_plan(framed_plan, analysis, "z" * 64)
        params = decision["video"]["params"]
        self.assertEqual(params["final_crop_motion"], "push_in")
        self.assertEqual(params["zoom_mode"], "enter_frame")
        self.assertEqual(params["zoom_end"], 2.2)
        self.assertAlmostEqual(params["zoom_focus_x"], 0.535)
        self.assertAlmostEqual(params["zoom_focus_y"], 0.625)
        validated = validate_and_clamp_decision(decision)
        self.assertEqual(validated["video"]["params"]["zoom_mode"], "enter_frame")
        self.assertEqual(validated["video"]["params"]["zoom_end"], 2.2)
        framed_plan["presentation"]["aspect"] = "instagram_reel_9_16"
        reel_decision = compile_semantic_plan(framed_plan, analysis, "z" * 64)
        self.assertEqual(reel_decision["video"]["params"]["zoom_end"], 2.0)

    def test_flower_push_uses_observed_upper_right_region_and_local_motion(self):
        flower_plan = plan(operation="push_in")
        flower_plan["motion_plan"] = {
            "primary_target": "red poppies",
            "primary_action": "gentle push toward the red poppies",
            "secondary_target": "cacti",
            "secondary_action": "subtle swaying of cacti",
            "keep_stable": ["trees", "hillside terrain", "photographer watermark"],
        }
        flower_plan["presentation"].update(
            {
                "aspect": "instagram_reel_9_16",
                "must_keep_visible": ["red poppies", "cacti", "trees", "photographer watermark"],
            }
        )
        analysis = {
            "summary": "A sunlit hillside with red poppies, cacti, rocks, and trees.",
            "image_metadata": {"aspect_ratio": 1.777},
            "composition": {
                "attention_regions": [
                    {"label": "red poppies", "region": "0.65 0.25 0.75 0.45", "importance": "high"},
                    {"label": "cacti", "region": "0.75 0.15 0.95 0.35", "importance": "medium"},
                    {"label": "trees", "region": "0.0 0.0 0.35 0.75", "importance": "high"},
                ]
            },
            "spatial_map": {
                "primary_regions": [
                    {"label": "poppies cluster", "box_normalized": {"x": 0.35, "y": 0.35, "w": 0.2, "h": 0.2}}
                ]
            },
            "people": [],
        }
        decision = compile_semantic_plan(flower_plan, analysis, "v" * 64)
        params = decision["video"]["params"]
        prompt = params["prompt"].lower()
        self.assertNotIn("push", prompt)
        self.assertIn("petals flutter visibly", prompt)
        self.assertIn("cacti", prompt)
        self.assertEqual(decision["framing"]["crop_anchor"], "right_top")
        self.assertEqual(params["zoom_end"], 1.25)
        self.assertAlmostEqual(params["zoom_focus_x"], 0.70)
        self.assertAlmostEqual(params["zoom_focus_y"], 0.35)
        self.assertEqual(params["must_keep_visible"], ["red poppies"])
        self.assertEqual(params["visibility_validation"]["status"], "adjusted")

    def test_winter_prompt_separates_branch_motion_from_static_geometry(self):
        winter_plan = plan(operation="push_in")
        winter_plan["motion_plan"] = {
            "primary_target": "sunlight",
            "primary_action": "gentle push toward the sunlight source",
            "secondary_target": "snow-covered trees",
            "secondary_action": "subtle movement of snow on the trees",
            "keep_stable": ["trees", "snow", "sunlight"],
        }
        winter_plan["presentation"]["must_keep_visible"] = ["sunlight", "snow-covered trees"]
        analysis = {
            "summary": "A sunlit winter forest with snow-covered evergreen trees.",
            "image_metadata": {"aspect_ratio": 1.3777},
            "dynamic_potential": {"camera_motion_affordances": ["gentle push toward sunlight"]},
            "composition": {
                "attention_regions": [
                    {"label": "sunlight", "region": "0.65, 0.15, 0.75, 0.25", "importance": "high"},
                    {"label": "snow-covered trees", "region": "0.1, 0.2, 0.3, 0.8", "importance": "high"},
                ]
            },
            "people": [],
        }
        decision = compile_semantic_plan(winter_plan, analysis, "w" * 64)
        params = decision["video"]["params"]
        prompt = params["prompt"].lower()
        self.assertNotIn("push", prompt)
        self.assertIn("outer boughs near the sunlit focal area flex gently", prompt)
        self.assertNotIn("powder", prompt)
        self.assertIn("tree trunks", prompt)
        self.assertNotIn("snow-covered trees fixed", prompt)
        self.assertEqual(params["zoom_end"], 1.15)
        self.assertAlmostEqual(params["zoom_focus_x"], 0.70)
        self.assertAlmostEqual(params["zoom_focus_y"], 0.20)

    def test_deterministic_fallback_is_not_a_regular_candidate(self):
        decision = compile_semantic_plan(plan(), {"people": []}, "r" * 64)
        validated = validate_and_clamp_decision(decision)
        selected = _video_variants_for_decision(validated, "all")
        self.assertEqual([item["preset"] for item in selected], ["WAN22_NATURAL", "WAN22_NATURAL"])
        recovery = [item for item in validated["fallbacks"] if item.get("recovery_only")]
        self.assertEqual([item["preset"] for item in recovery], ["DETERMINISTIC_ORIGINAL"])

    def test_retired_presets_are_rejected(self):
        for preset in ("SVD_SUBTLE", "ANIMATEDIFF_GRASS_WIND"):
            with self.subTest(preset=preset), self.assertRaisesRegex(ValueError, "Unsupported video preset"):
                validate_and_clamp_decision({"video": {"preset": preset}, "fallbacks": []})

    def test_selected_pair_uses_selected_model_seeds(self):
        decision = validate_and_clamp_decision(compile_semantic_plan(plan(), {"people": []}, "s" * 64))
        selected = _video_variants_for_decision(decision, "selected_pair")
        self.assertEqual([item["preset"] for item in selected], ["WAN22_NATURAL", "WAN22_NATURAL"])
        self.assertNotEqual(selected[0]["seed"], selected[1]["seed"])

    def test_memory_query_uses_analysis_properties(self):
        query = build_memory_query(
            {
                "summary": "Night waterfront",
                "scene": {"environment": "urban lake"},
                "subjects": [{"label": "city skyline"}],
                "dynamic_potential": {"natural_motion_elements": ["water reflections"]},
            }
        )
        self.assertIn("night waterfront", query)
        self.assertIn("water reflections", query)

    def test_static_properties_are_removed_from_natural_motion_evidence(self):
        cleaned = sanitize_analysis_for_decision(
            {
                "dynamic_potential": {
                    "natural_motion_elements": ["sunlight", "snow", "depth layers", "foliage", "falling snow"]
                }
            }
        )
        dynamic = cleaned["dynamic_potential"]
        self.assertEqual(dynamic["natural_motion_elements"], ["foliage", "falling snow"])
        self.assertEqual(dynamic["excluded_static_motion_elements"], ["sunlight", "snow", "depth layers"])

    def test_validator_keeps_new_profiles_and_square_output(self):
        raw = compile_semantic_plan(plan(), {"people": [], "confidence": {"overall": 0.8}}, "d" * 64)
        validated = validate_and_clamp_decision(raw)
        self.assertEqual(validated["video"]["preset"], "WAN22_NATURAL")
        self.assertEqual(validated["framing"]["target_aspect"], "square_1_1")
        self.assertEqual(validated["video"]["frames"], 97)
        self.assertEqual((validated["video"]["params"]["steps"], validated["video"]["params"]["shift"]), (20, 8.0))
        self.assertFalse(validated["video"]["params"]["tiled_vae"])
        self.assertEqual(validated["audio"]["mix_db"], -6.0)

    def test_nature_audio_does_not_inject_wind_or_insects(self):
        prompt = _compose_audio_prompt(
            "WAN22_NATURAL",
            {"tags": ["nature", "mountains", "water"]},
            {},
            "quiet stream ambience, no music",
        )
        self.assertNotIn("insect", prompt.lower())
        self.assertNotIn("wind", prompt.lower())
        self.assertIn("distant occasional birds", prompt.lower())

    def test_hunyuan_fast_does_not_under_sample_full_checkpoint(self):
        validated = validate_and_clamp_decision(
            {
                "scene": {"tags": ["street"]},
                "video": {"preset": "HUNYUAN15_I2V_FAST"},
                "fallbacks": [],
            }
        )
        self.assertEqual((validated["video"]["frames"], validated["video"]["fps"]), (30, 6))
        self.assertEqual(validated["video"]["params"]["steps"], 50)

    def test_source_aspect_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.jpg"
            Image.new("RGB", (4080, 3060)).save(path)
            dimensions = ComfyClient._resolve_dimensions(
                {
                    "preset": "HUNYUAN15_I2V_720P",
                    "resolution_width": 704,
                    "params": {"preserve_source_aspect": True},
                },
                str(path),
            )
        self.assertEqual(dimensions, (704, 528))

    def test_postprocess_filters_include_static_position_and_zoom(self):
        static_filter = _normalized_video_filter("static_crop", output_aspect="square_1_1", zoom_focus_x=0.7, zoom_focus_y=0.2)
        zoom_filter = _normalized_video_filter("push_in", output_aspect="square_1_1", zoom_focus_x=0.7, zoom_focus_y=0.2)
        self.assertIn("iw*0.7000-ow/2", static_filter)
        self.assertIn("ih*0.2000-oh/2", static_filter)
        self.assertIn("zoompan", zoom_filter)
        self.assertIn("iw*0.7000-1080/2", zoom_filter)
        self.assertLess(zoom_filter.index("fps=30"), zoom_filter.index("zoompan"))
        enter_filter = _normalized_video_filter(
            "push_in", output_aspect="square_1_1", zoom_end=2.2, zoom_focus_x=0.535, zoom_focus_y=0.625
        )
        self.assertIn("1+1.2000*on/149", enter_filter)
        self.assertIn("0.5350", enter_filter)

    def test_tiled_vae_uses_safe_temporal_window(self):
        node = tiled_vae_decode_node("9", "3")
        self.assertEqual(node["class_type"], "VAEDecodeTiled")
        self.assertEqual(node["inputs"]["tile_size"], 512)
        self.assertEqual(node["inputs"]["overlap"], 64)
        self.assertEqual(node["inputs"]["temporal_size"], 64)
        self.assertEqual(node["inputs"]["temporal_overlap"], 8)

    def test_review_feedback_is_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.result.json"
            path.write_text('{"state":"HUMAN_REVIEW","human_feedback":{}}', encoding="utf-8")
            updated = update_feedback(
                path,
                status="rejected",
                rating=2,
                issues=["motion_too_strong"],
                notes="earthquake-like global shake",
            )
            persisted = __import__("json").loads(path.read_text(encoding="utf-8"))
        self.assertEqual(updated["state"], "REJECTED")
        self.assertEqual(persisted["human_feedback"]["issue_codes"], ["motion_too_strong"])


if __name__ == "__main__":
    unittest.main()
