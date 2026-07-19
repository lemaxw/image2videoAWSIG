# Video model routing and experience-memory decision — 2026-07-13

## Purpose and status

This document consolidates the tested image-to-video behavior, explains why the current decision flow produced weak or inconsistent videos, and records the agreed replacement direction. It is an architecture decision and experiment summary, not yet a completed production implementation.

The main decision is:

- Use HunyuanVideo 1.5 when an important visible person, fauna/animal, or vehicle must be preserved, especially when that subject should move.
- Use Wan2.2 TI2V 5B for ordinary photographic scenes without important sensitive moving subjects: nature, plants/flora, clouds, water, landscapes, cities, architecture, and mixed environmental scenes.
- Treat small distant incidental people, animals, or vehicles as a risk signal rather than an automatic Hunyuan requirement.
- Do not use VACE or masked generation in the normal production route.
- Generate in the original aspect ratio, validate the full-frame result, and apply deterministic crop/pan/zoom afterward.
- Produce multiple candidates when useful, collect structured human feedback, store one comparable JSON record per produced video, and index accepted and rejected experience in MemPalace for later retrieval.

## Current scheme before the redesign

The decision service uses a local two-step flow:

1. `qwen3-vl:8b` image2json analysis describes scene, subjects, objects, spatial layout, dynamic potential, framing risks, generation risks, and soundscape.
2. `qwen3:14b` receives that JSON, `TEXT_MODEL_SYSTEM_PROMPT`, and `DECISION_SCHEMA`, then returns `scene`, `framing`, `video`, `audio`, and exactly two `fallbacks`.

The existing schema knows Hunyuan, AnimateDiff, and SVD presets, but not Wan. It forces top-level framing to Instagram 9:16, requires the text model to emit low-level values such as FPS, frames, resolution, seed, and prompt parameters, and requires exactly two fallback video objects. The orchestrator normally renders the selected primary plus the first fallback from another backend family. Validation then clamps values to backend defaults. Additional image2json motion guidance later merges backend phrases into the text-model prompts and can change camera/crop behavior.

`use_original_input_for_video`, model-generated camera motion, and final deterministic crop motion evolved separately and have previously been confused. The current code now separates original-input preservation from final crop motion, but the decision prompt still carries many historical rules and competing priorities.

## Why the current scheme worked poorly

### Backend knowledge was outdated

The strongest tested nature backend, Wan2.2, is absent from the schema. The prompt therefore selects among older choices and frequently puts Hunyuan first regardless of scene risk or motion type.

### The objective was aesthetic rather than risk-aware

The prompt asks for the most visually appealing treatment but does not first classify important moving subjects, preservation sensitivity, or rejection requirements. A visually dynamic result can therefore win even when it changes a person, vehicle, building, reflection, or geography.

### Too many responsibilities were assigned to the language model

The text model chooses semantic treatment, backend, technical frames/FPS/resolution/seed, generation prompt, negative prompt, input preservation, camera motion, and final crop motion at once. Some emitted technical values are then overridden by validator defaults. This makes the decision hard to reason about and prevents clear learning from past results.

### Prompt transformations were layered

The text model creates a prompt, then `_apply_image2json_motion_guidance` adds family-specific phrases. The result may be longer, contradictory, or mention concepts that should not be generated. Wan tests showed that naming an unwanted concrete concept, even as a constraint or negative, can cause a related hallucination. Short positive, action-focused prompts were more reliable.

### Generation and presentation were mixed

Pre-cropping to 9:16 throws away scene context before generation. Model-generated camera movement can be too fast or invent content. Deterministic pan/crop is predictable but can move protected subjects out of frame. The safer design is original-aspect generation followed by independently validated deterministic framing.

### Fixed fallback rendering did not express useful experiments

Primary plus a cross-family fallback is not the same as two meaningful candidates. For stochastic video generation, two seeds of the correct backend are often more useful than an unrelated stylized backend. The old flow also had no structured human comparison, lineage, acceptance, or rejection record.

### There was no reusable result-level memory

`debug.json` captures diagnostics but is not a normalized comparable result record. Successful prompts, failed artifacts, seeds, crop choices, model versions, and human opinions were not automatically available to future decisions.

## Model evidence and decisions

### Wan2.2 TI2V 5B

#### Successful nature behavior

The valley landscape produced the strongest overall result with full-frame Wan:

- 768x576, preserving the original 4:3 aspect.
- 97 native frames at 20 FPS: exactly 4.85 seconds.
- 20 steps, seed 42, NORMAL_VRAM.
- About 129 seconds on the RTX 3090.
- Action-first prompt made clearly visible cloud motion primary, natural stream flow secondary, and protected mountains/geography.

The result looked natural, coherent, and substantially better than hard-masked alternatives. Artifact: `.local/outputs/wan-benchmark/valley_wan22_visible_clouds_97f_20fps.mp4`.

#### Waterfront behavior and prompt sensitivity

The first waterfront prompt explicitly mentioned a locked tripod/camera concept. Wan later invented related foreground equipment and arms. Removing those concepts from the prompt eliminated that hallucination with the same seed. The neutral-prompt result kept the skyline stable but exaggerated a blue building reflection. Explicit negative wording about water patches replaced that problem with long luminous streaks. This demonstrates that concrete unwanted concepts should not be named in positive or negative prompts; use abstract quality negatives and automated rejection instead.

Artifacts:

- Better neutral prompt: `.local/outputs/wan-benchmark/waterfront_wan22_neutral_prompt_49f.mp4`.
- Explicit reflection negatives with new streak artifact: `.local/outputs/wan-benchmark/waterfront_wan22_reflection_negative_49f.mp4`.

#### Scooter behavior

At 704x528, 61 frames, 12 FPS, 20 steps, seed 1, Wan completed in about 82 seconds. It preserved the red scooter and two riders surprisingly well and was roughly three times faster than Hunyuan. It nevertheless invented a large blue band along the bottom of the road and changed background color/vegetation more. Thus Wan was competitive for subject appearance but unsafe for the complete structured frame.

Artifact: `.local/outputs/wan-benchmark/scooter_wan22_original4x3_61f_12fps_seed1.mp4`.

#### Wan decision

Wan is the default for scenes without important sensitive moving subjects. Use full-frame original-aspect generation. Start with one candidate for low-risk scenes and two candidates when time permits or the first has weak motion. Use abstract negative quality terms only. Require a quality gate for new large regions, geometry drift, reflection artifacts, and background changes.

Recommended default from the successful valley case:

```text
resolution: approximately 768px long edge, original aspect
frames: 97
fps: 20
steps: 20
duration: 4.85 seconds
candidate seeds: 1-2
```

### HunyuanVideo 1.5 720p I2V

#### Static city/water behavior

Hunyuan preserved structured waterfront geometry better than the failed Wan waterfront prompt, but 49 and 61 frames did not make water motion clearly readable. Extra frames mainly increased lighting and reflection drift. Model-generated camera motion moved the whole scene but did not improve water. A static square crop preserved more context than 9:16.

#### People/vehicle behavior

For the scooter image, a descriptive prompt replaced the scooter/riders with a car late in the clip. A short action-focused prompt retained a scooter and two riders but still changed clothing, helmets, and identity details. Original-aspect generation improved the overall composition compared with the earlier portrait pre-crop.

The practical winning configuration on the RTX 3090 with 16 GB host RAM was:

```text
resolution: 704x528, original 4:3 aspect
frames: 61
fps: 12
duration: 5.083 seconds
steps: 14
cfg: 5.8
shift: 7.0
sampler/scheduler: Euler/simple
seed: compare at least two; seed 1 won this image
VAE: tiled decode, tile_size 512, overlap 64,
     temporal_size 64, temporal_overlap 8
```

The successful seed-1 run took about 243 seconds. Full-batch VAE decoding restarted Comfy after sampling, while tiled temporal decoding completed reliably. At 97 frames/20 FPS the model completed but temporarily collapsed the scooter into a grid-like structure. At 960x720, 97 frames sampled at about 75 seconds per step and was operationally unusable; 960x720/61 also proved unstable during handoff. More frames were not better for identity-sensitive motion.

Correction from the 2026-07-14 raw-frame audit: the short `16/4` temporal decode
window itself caused periodic grid frames in both Wan and Hunyuan. Same-seed
`64/8` tiled-decode controls removed the grid while remaining stable. Production
therefore uses spatial tiles of 512/64 and temporal tiles of 64/8. The 61-frame
Hunyuan recommendation remains, but the old 16/4 decoder recipe is rejected.

Artifacts:

- Accepted 61-frame candidate: `.local/outputs/hunyuan-benchmark/scooter_original4x3_61f_12fps_14steps_seed1_704x528_tiled.mp4`.
- Rejected 97-frame identity failure: `.local/outputs/hunyuan-benchmark/scooter_original4x3_97f_20fps_14steps_seed1_704x528_tiled.mp4`.
- Wan/Hunyuan comparison: `.local/outputs/wan-benchmark/scooter_hunyuan_vs_wan22_61f_side_by_side.mp4`.

#### Hunyuan decision

Use Hunyuan when an important person, fauna/animal, or vehicle is visible, especially when it should move. Also consider Hunyuan or deterministic treatment for exact products, readable text, faces, and highly sensitive structure. Generate two seeds for important moving subjects, keep prompts short and action-focused, preserve original aspect, and make tiled VAE decoding mandatory for long clips. Use deterministic framing afterward.

### VACE 1.3B

Unmasked VACE radically redrew the valley and was unsuitable as normal image-to-video. A coarse stream mask improved global preservation but invented vegetation near the boundary. A SAM stream mask localized changes better but still reconstructed water and bank texture and showed an unstable opening transition. The combined clouds/stream hard-composite test was spatially constrained but visually less coherent and took about 189 seconds versus about 69 seconds for the comparable Wan test.

#### VACE decision

Do not include VACE in the normal production route. Keep it only as an optional research backend for exact regional editing if a future larger/better model proves useful. Current masked output was never clearly better than full-frame Wan.

### SAM2.1 Base+

SAM ran in about 1.6 seconds at 2048x1536. A box captured much of the stream but also included twigs and banks. Positive/negative point refinement would be required for production segmentation. Since masking did not improve final video quality, SAM should not be part of normal generation. It can remain useful for automatic evaluation of protected subjects or motion regions.

### Depth Anything V2 Small

Depth inference ran in about 0.48 seconds and produced a plausible relative depth map. It remains useful for deterministic 2.5D pan/push/parallax when generative camera movement is unsafe. It is a postprocessing or fallback tool, not a competing video generator.

### AnimateDiff

Earlier static-position results did not show readable cloud/water movement despite the intended motion. It also redraws the image into a stylized/anime representation, which is not desired for normal photographic output. AnimateDiff should no longer be a standard cross-family fallback. Keep it only as explicit opt-in stylization.

### SVD

SVD remains an older conservative/emergency option but was not competitive with the successful Wan nature result or the Hunyuan moving-subject route. It should not be selected merely because a scene has generic geometry risk. Prefer deterministic original-image motion when both Wan and Hunyuan are unsafe.

## Agreed decision and prompt policy

Image2json must identify:

- scene class and environment;
- people, fauna, vehicles, plants, structures, text, and products;
- subject importance: primary, secondary, or incidental;
- intended visible motion for each important subject;
- primary and secondary environmental motion targets;
- preservation risk and framing risks;
- original spatial relationship and preferred delivery framing.

Routing:

```text
important primary/secondary person, fauna, or vehicle -> Hunyuan
everything else -> Wan
exact text/logo/product/identity -> conservative Hunyuan or deterministic
incidental distant sensitive subject -> risk flag; Wan may still be used
failed generative candidates -> deterministic original-image treatment
```

Prompt structure:

```text
[existing primary subject and one requested action].
[secondary environmental motion].
[existing important content remains stable].
Preserve the original composition and viewpoint.
```

Do not mention concrete unwanted objects or artifacts. Negative prompts should contain abstract quality terms such as `flicker`, `jitter`, `unstable geometry`, `inconsistent appearance`, `scene transition`, and `low quality`.

The decision model should return semantic classification, backend, short model prompt, preservation requirements, candidate count, postprocess operation, and audio intent. Deterministic code should assign tested resolution, frames, FPS, steps, CFG, sampler, seed policy, and tiled-decode settings.

## Framing decision

Generation and presentation are separate:

1. Generate from the original aspect ratio.
2. Validate and select the full-frame candidate.
3. Apply deterministic static crop, pan, or zoom.
4. Export a matching still directly from the original image with the same crop.

The preferred valley presentation was a centered square crop. For 768x576 input, crop 576x576 at x=96, y=0, then scale to 1080x1080. General centered-square original-image crop is `crop=ih:ih:x=(iw-ih)/2:y=0`.

For a deterministic right-to-left horizontal pan of crop width `crop_w` over duration `D`, use normalized movement equivalent to `x=(iw-crop_w)*(1-t/D)`. Use partial slow pans when possible; a complete sweep can be too fast and can remove important subjects. Prefer square or 4:5 when 9:16 discards important landscape or city context.

## Experience records and human-feedback protocol

Every produced video must have a comparable companion JSON. The JSON archive is the reproducible source of truth; MemPalace is the semantic search layer.

Each record must contain:

- schema version, experiment ID, candidate ID, parent candidate, round, and status;
- source path/hash/dimensions/aspect;
- normalized image classification and important/incidental subjects;
- intended motion and preservation targets;
- image2json and decision model/version;
- chosen backend and reason;
- exact generation and negative prompts;
- checkpoint name/hash, seed, dimensions, frames, FPS, steps, CFG, sampler, scheduler, VAE settings, and runtime;
- deterministic crop/pan/zoom using normalized coordinates;
- artifact paths;
- automated metrics when available;
- human rating, accepted/rejected/pending status, issue codes, notes, and timestamp;
- lineage and changed fields;
- generated natural-language `memory.search_text` for semantic indexing.

Normalized human issue codes include `subject_missing`, `subject_count_changed`, `subject_identity_changed`, `vehicle_shape_changed`, `new_object`, `background_changed`, `motion_too_weak`, `motion_too_strong`, `crop_too_tight`, `crop_wrong_anchor`, `pan_too_fast`, `zoom_too_fast`, `flicker`, `temporary_transformation`, `water_artifact`, `audio_wrong`, and `accepted`.

Protocol state:

```text
ANALYZED -> PLANNED -> CANDIDATES_RENDERED -> HUMAN_REVIEW
         -> REVISE (new child candidates) or ACCEPT
         -> RESULT_RECORDED -> MEMORY_INDEXED
```

Change only one or two variables per revision round so the reason for improvement remains learnable. Preserve rejected results and reasons; they are as valuable as accepted recipes.

For an exact same image hash and request, reuse the accepted artifact or exact recipe. For a similar image, retrieve prior model family, prompt structure, technical profile, crop strategy, and failure warnings, but rebuild descriptions from the current image2json analysis. A winning seed is reproducible for the same workflow but is only a weak prior for a different image.

Before the text decision step, search MemPalace with a compact signature containing scene class, environment, important subject types, intended motion, aspect, composition, lighting, and preservation risk. Inject at most a few compact accepted and rejected examples. The prompt must state that prior cases are evidence only, absent objects must never be copied, and seeds do not transfer reliably between images.

## Planned implementation order

1. Add a versioned per-video result JSON schema and write records for existing backends.
2. Add production Wan routing and original-aspect workflow support.
3. Make tiled Hunyuan VAE decoding the long-video default.
4. Replace the current decision schema/prompt with semantic routing and postprocess directions.
5. Replace cross-family selected-pair rendering with same-backend candidate/seed experiments.
6. Add human review, issue codes, candidate lineage, and revision rounds.
7. Mine result records and experiment summaries into MemPalace.
8. Retrieve similar accepted/rejected experience between image2json and the text decision model.
9. Add automatic rejection metrics after the human-feedback loop is usable.

## Classification of the proposed change

This is a clean architectural redesign. It is broader than a prompt-only fix because the observed failures came from routing, technical defaults, prompt mutation, framing, lack of quality gates, and lack of feedback memory together. Safe validation requires replaying the known valley, waterfront, and scooter cases, confirming exact record generation, testing accepted/rejected human feedback updates, verifying MemPalace retrieval, and ensuring retrieved prior experience cannot introduce objects absent from the current image.
