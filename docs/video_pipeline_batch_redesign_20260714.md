# Video pipeline batch redesign — 2026-07-14

## Goal

Run every image under the local incoming directory (`video_input/`, mounted as the
orchestrator local input directory), produce candidates with the model and framing
policy established by the Wan/Hunyuan benchmarks, and leave every image in an
explicit, reviewable state.

Batch completion does not mean only that MP4 files exist. A completed batch must
prove that:

- every discovered image has a durable manifest entry;
- image analysis, routing, candidate settings, artifacts, and failures are recorded;
- each successful candidate has one comparable `*.result.json` companion;
- no image disappears from the batch because analysis or rendering timed out;
- candidates wait for human acceptance instead of being treated as correct merely
  because generation succeeded;
- accepted and rejected experience can later be retrieved without copying objects
  or seeds blindly into a different image.

## Current gaps blocking this goal

The current code cannot yet execute the agreed flow:

1. `DECISION_SCHEMA` and validation do not contain Wan.
2. Production Comfy workflows contain Hunyuan, SVD, and AnimateDiff only.
3. The decision model selects technical values that validation later overwrites.
4. The normal `selected_pair` mode renders different backend families instead of
   meaningful same-backend seed candidates.
5. All images are pre-cropped to 9:16 before rendering unless a historical special
   case chooses the original input.
6. Even the “original input” path resolves render dimensions as 16:9; it does not
   preserve the source aspect ratio.
7. Hunyuan production defaults are still 30 frames at 6 FPS and do not use the
   validated 61-frame tiled-VAE profile.
8. `debug.json` is diagnostic output, not a normalized candidate result record.
9. The batch has no resume ledger, human-review state, revision lineage, or final
   acceptance summary.
10. An image2json timeout currently fails the whole image immediately. This already
    happened for `1_20260627_130019.jpg` at both 300 and 600 seconds and for
    `4_DSC08334.jpg` at 300 seconds.

This is a clean architectural fix. It replaces mixed decision/render/presentation
responsibilities with explicit stages. It is intentionally broader than a prompt
workaround because the current failures cross all of those boundaries.

## Target flow

```text
DISCOVERED
  -> ANALYZED
  -> EXPERIENCE_RETRIEVED
  -> SEMANTIC_PLAN_CREATED
  -> TECHNICAL_PLAN_COMPILED
  -> CANDIDATES_RENDERED
  -> PRESENTATIONS_EXPORTED
  -> HUMAN_REVIEW
       -> ACCEPTED -> MEMORY_INDEXED
       -> REVISE -> child candidates -> HUMAN_REVIEW
       -> REJECTED -> MEMORY_INDEXED
```

Each state transition is written atomically to a per-image batch manifest. A
process restart resumes from the last durable state and does not regenerate an
already completed candidate unless `--force` is supplied.

## Separation of responsibilities

### 1. Image analysis

Image2json remains responsible for observed facts:

- scene/environment and image dimensions;
- subjects and objects with normalized positions;
- person, fauna, vehicle, flora, structure, text, and product presence;
- importance (`primary`, `secondary`, `incidental`);
- plausible natural motion and motion risk;
- content that must remain visible;
- framing constraints and soundscape.

It must not choose the video backend or low-level generation settings.

Analysis is cached by source SHA-256 plus vision model/schema version. Before
analysis, a bounded proxy image is created for the vision model while original
dimensions and hash remain authoritative. A timeout retries once with the proxy;
if it still fails, the image becomes `ANALYSIS_FAILED`, remains visible in the
batch report, and may be retried independently.

### 2. Experience retrieval

After analysis and before the text decision, search accepted and rejected result
records using a compact signature:

```text
scene class + environment + important subject types + intended motion +
source aspect + composition + lighting + preservation risks
```

Inject at most three compact cases: preferably two accepted and one rejected.
Retrieved evidence may influence backend, prompt shape, technical profile, crop,
and warnings. It must not add an object absent from the current analysis, and a
previous seed is only a weak candidate prior for a different source image.

MemPalace is the semantic index. The JSON result archive remains the reproducible
source of truth. MemPalace lookup/indexing failure must be recorded but must not
make local rendering impossible.

### 3. Semantic decision

The text model returns intent, not sampler configuration. Proposed version-2
contract:

```json
{
  "schema_version": "2.0",
  "classification": {
    "scene_classes": ["landscape"],
    "environment": "mountain valley",
    "important_subjects": [],
    "incidental_subjects": [],
    "sensitive_content": [],
    "preservation_risk": "medium"
  },
  "motion_plan": {
    "primary": {"target": "cloud layers", "action": "clearly drift across the sky"},
    "secondary": {"target": "stream", "action": "continue flowing naturally"},
    "keep_stable": ["mountain contours", "vegetation layout", "stream banks"]
  },
  "generation": {
    "mode": "generative",
    "backend": "wan22",
    "prompt": "Cloud layers clearly drift across the valley while the stream continues flowing naturally. Mountain contours, vegetation layout, and stream banks remain stable. Preserve the original composition and viewpoint.",
    "negative_prompt": "flicker, jitter, unstable geometry, inconsistent appearance, scene transition, low quality",
    "candidate_count": 2,
    "reason": "Environmental motion with no important sensitive moving subject."
  },
  "presentation": {
    "aspect": "square_1_1",
    "operation": "static_crop",
    "anchor": {"x": 0.5, "y": 0.5},
    "pan": null,
    "zoom": null,
    "must_keep_visible": ["stream", "central valley"]
  },
  "audio": {
    "intent": "nearby flowing stream, distant birds and insects, minimal wind",
    "duration_s": 5
  }
}
```

Allowed generation modes are `generative` and `deterministic`. Allowed backends
for the first production version are `wan22`, `hunyuan15`, and `none`.

Routing rules are stated directly in the prompt and are also enforced in code:

```text
important primary/secondary person, fauna, or vehicle -> Hunyuan
everything else with useful natural motion -> Wan
exact identity/text/logo/product or no safe useful motion -> deterministic candidate
small distant incidental sensitive subject -> risk flag, not automatic Hunyuan
failed generative candidates -> deterministic original-image candidate
```

The deterministic route is important for static product/fashion/architecture
scenes where inventing motion is less useful than a controlled crop, pan, or small
push. It is also the guaranteed fallback that lets a batch complete without
silently accepting a damaged generative video.

### 4. Technical plan compiler

Code maps the semantic plan to versioned profiles. The language model cannot set
frames, FPS, steps, CFG, sampler, scheduler, checkpoint paths, or VAE tiling.

Initial profiles based on the completed benchmarks:

#### `wan22_natural_v1`

```json
{
  "backend": "wan22",
  "checkpoint": "wan2.2_ti2v_5B_fp16.safetensors",
  "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
  "vae": "wan2.2_vae.safetensors",
  "long_edge": 768,
  "frames": 97,
  "fps": 20,
  "steps": 20,
  "cfg": 5.0,
  "shift": 8.0,
  "sampler": "uni_pc",
  "scheduler": "simple",
  "duration_s": 4.85
}
```

Width and height are derived from the source aspect, rounded to model-safe
multiples while keeping the long edge near 768. The source is fitted without a
center crop. Low-risk scenes start with one seed; normal review batches use two.

#### `hunyuan15_sensitive_motion_v1`

```json
{
  "backend": "hunyuan15",
  "checkpoint": "hunyuanvideo1.5_720p_i2v_fp16.safetensors",
  "long_edge": 704,
  "frames": 61,
  "fps": 12,
  "steps": 14,
  "cfg": 5.8,
  "shift": 7.0,
  "sampler": "euler",
  "scheduler": "simple",
  "duration_s": 5.083,
  "vae_decode": {
    "tiled": true,
    "tile_size": 512,
    "overlap": 64,
    "temporal_size": 64,
    "temporal_overlap": 8
  }
}
```

Width and height preserve source aspect. Important moving-subject scenes render
two seeds by default. The known 97-frame Hunyuan mode is excluded because it
temporarily transformed the scooter and cost substantially more time.

#### `deterministic_original_v1`

Uses the original still only. It produces a 5-second H.264 presentation using a
static crop, bounded slow pan, or small push. It never synthesizes new scene
content and therefore serves as both a valid treatment and the final safety net.

Candidate seed generation is deterministic from `(source_sha256, profile_version,
candidate_index, revision_round)` unless an exact accepted recipe supplies a seed.
This makes retries reproducible without treating seed numbers as semantic presets.

### 5. Original-aspect generation

The render planner computes native dimensions from the source image, not from the
delivery aspect. For source width `W`, height `H`, and profile long edge `L`:

```text
scale = L / max(W, H)
render_width  = safe_round(W * scale)
render_height = safe_round(H * scale)
```

For the tested Wan/Hunyuan nodes, `safe_round` rounds each dimension to the
nearest multiple of 16, then adjusts the shorter dimension by one 16-pixel step
if that reduces aspect error. It must reject a result whose aspect error exceeds
1.5% instead of silently falling back to 16:9. The chosen dimensions and aspect
error are stored in the candidate record.

There is no `prepare_instagram_input_image` before generative rendering in the new
path. Delivery crop/pan/zoom is applied only after the full-frame candidate exists.

### 6. Presentation compiler

Presentation is independently reproducible from normalized parameters. Proposed
normalized format:

```json
{
  "aspect": "square_1_1",
  "operation": "pan",
  "crop": {"width": 0.75, "height": 1.0},
  "start_center": {"x": 0.62, "y": 0.5},
  "end_center": {"x": 0.50, "y": 0.5},
  "easing": "linear",
  "duration_s": 5.0,
  "output": {"width": 1080, "height": 1080, "fps": 30}
}
```

The compiler verifies at both pan endpoints that all `must_keep_visible` boxes are
inside the crop. A requested pan that cannot satisfy this becomes a static crop.
Pan span and zoom delta are bounded by profile, not free-form prompt text.

The matching JPG is cropped from the original source using the same normalized
presentation window. It is not extracted from a generated video frame.

### 7. Candidate result record

Every produced MP4 has a sibling `<candidate-id>.result.json`. Required top-level
sections:

```json
{
  "schema_version": "1.0",
  "record_id": "...",
  "experiment_id": "...",
  "candidate_id": "...",
  "parent_candidate_id": null,
  "revision_round": 0,
  "state": "HUMAN_REVIEW",
  "source": {},
  "analysis": {},
  "retrieved_experience": [],
  "semantic_plan": {},
  "technical_plan": {},
  "generation": {},
  "presentation": {},
  "artifacts": {},
  "metrics": {},
  "human_feedback": {
    "status": "pending",
    "rating": null,
    "issue_codes": [],
    "notes": "",
    "reviewed_at": null
  },
  "lineage": {"changed_fields": []},
  "memory": {"indexed": false, "search_text": "..."}
}
```

Checkpoint file SHA-256, workflow-template SHA-256, prompt, negative prompt, seed,
native dimensions, frames, FPS, sampler, scheduler, VAE settings, runtime, Comfy
prompt ID, final crop, and artifact hashes are mandatory for a successful
generative record.

Issue codes use the agreed normalized vocabulary: `subject_missing`,
`subject_count_changed`, `subject_identity_changed`, `vehicle_shape_changed`,
`new_object`, `background_changed`, `motion_too_weak`, `motion_too_strong`,
`crop_too_tight`, `crop_wrong_anchor`, `pan_too_fast`, `zoom_too_fast`, `flicker`,
`temporary_transformation`, `water_artifact`, `audio_wrong`, and `accepted`.

### 8. Human review and revision

The first implementation can use a local CLI rather than a web UI:

```text
python -m services.orchestrator.review list --job-id <id>
python -m services.orchestrator.review show --candidate <id>
python -m services.orchestrator.review accept --candidate <id> --rating 5
python -m services.orchestrator.review reject --candidate <id> \
  --issues water_artifact,motion_too_strong --notes "reflection changed shape"
python -m services.orchestrator.review revise --candidate <id> \
  --change presentation.operation=static_crop
```

A revision changes no more than two fields. It creates a child record and never
overwrites its parent. Acceptance of one candidate marks sibling candidates
`not_selected`, not technically failed.

## Batch manifest and resume behavior

Each job writes `batch_manifest.json` containing the input snapshot and per-image
state. The snapshot includes relative path, SHA-256, dimensions, discovered time,
and expected candidate count. Files added after job start belong to a later job
unless `--refresh-inputs` is explicit.

The manifest distinguishes:

- `analysis_failed`;
- `planning_failed`;
- `render_failed`;
- `presentation_failed`;
- `awaiting_review`;
- `accepted`;
- `rejected`;
- `not_selected`.

The process exit code is based on execution health, while acceptance is reported
separately. A batch with seven rendered images and zero human reviews is
operationally successful but has `0/7 accepted`; it is not described as having
seven expected final results.

Audio is generated once per image/semantic plan and reused by sibling seed
candidates unless the revision changes audio intent. This avoids reloading the
audio model for each seed and reduces batch time.

## Expected routing for the current incoming set

This table is an initial acceptance fixture, not a substitute for image2json. It
lets the first redesigned batch detect obviously wrong routing and presentation.

| Image | Expected route | Intended motion | Initial presentation | Key rejection conditions |
|---|---|---|---|---|
| `0__MG_0006.jpg` | Wan | Clearly readable cloud drift; temple and rocks static | Square or 4:5 static crop; preserve the temple span | Temple/scaffolding deformation, new objects, watermark loss caused by crop, cloud motion too weak |
| `1_20260627_130019.jpg` | Wan | Faster cloud/mist motion; stream flow secondary | Centered square static crop, matching the accepted valley treatment | Mountain/stream-bank change, flicker, weak cloud motion, unnatural water |
| `2_DSC05444.jpg` | Deterministic first; Wan only as an explicit challenger | No mannequin body motion; at most extremely subtle fabric/light change | Slow bounded push or static square crop preserving the row | Mannequin movement, garment redesign, count change, crop losing row depth |
| `3_DSC04483.jpg` | Wan, two seeds | Subtle water/reflection motion; skyline stable | Square static crop or short right-to-left pan | Blue patch/streak, reflection geometry change, new foreground equipment/people, skyline drift |
| `4_DSC08334.jpg` | Wan with deterministic sibling | Slow cloud movement; framed structure and skyline static | Static square/4:5 crop using the architectural frame | Beam/building deformation, crane change, aggressive camera motion, crop breaking the frame-within-frame composition |
| `5_DSC00233.jpg` | Wan; vehicles are incidental risk signals | Subtle cloud drift and heat/air atmosphere; no invented vehicle action | Static square/4:5 crop or very short pan preserving the winding road | Road geometry change, vehicle multiplication, campsite redraw, pan too fast |
| `6_20250516_125456.jpg` | Hunyuan, two seeds | Scooter continues naturally with exactly two riders; background stable | Original-aspect generation, then static square/4:5 crop centered on scooter/building relationship | Rider/scooter count or identity change, vehicle-shape change, temporary transformation, graffiti/building redesign |

Expected native dimensions from the initial profiles are:

| Image | Source size | Native generation size |
|---|---:|---:|
| `0__MG_0006.jpg` | 7500x5000 | Wan 768x512 |
| `1_20260627_130019.jpg` | 2048x1536 | Wan 768x576 |
| `2_DSC05444.jpg` | 3815x3155 | deterministic; no generative resize |
| `3_DSC04483.jpg` | 6000x4000 | Wan 768x512 |
| `4_DSC08334.jpg` | 5095x4000 | Wan 768x608 |
| `5_DSC00233.jpg` | 6000x4000 | Wan 768x512 |
| `6_20250516_125456.jpg` | 4080x3060 | Hunyuan 704x528 |

The two timed-out images are visually known for this fixture: image 1 is the green
mountain valley/stream case used in the successful Wan benchmark; image 4 is a
city skyline framed by close architectural beams. The production run must still
obtain and store fresh image2json analysis rather than relying on this prose.

## Acceptance gates for the first full batch

### Structural gates (automatic)

- exactly seven source entries for the current `video_input/` snapshot;
- every source has SHA-256, dimensions, analysis state, and terminal execution state;
- route matches the fixture unless the plan records an explicit risk-based reason;
- Wan candidates use original-aspect `wan22_natural_v1` settings;
- Hunyuan scooter candidates use 61 frames, 12 FPS, 14 steps, and tiled VAE;
- every candidate has MP4, matching original-source JPG, and `result.json` hashes;
- MP4 dimensions/aspect match the compiled presentation and duration is within
  0.15 seconds of the intended duration;
- rerunning the same job with `--resume` does not regenerate successful candidates;
- no SVD, AnimateDiff, VACE, or masked generation appears unless explicitly
  requested as a research override.

### Visual gates (human in version 1)

- requested motion is visible without fast whole-frame movement;
- important subjects and their count remain stable;
- no large new region, reflection patch, streak, or scene transition appears;
- deterministic framing keeps required regions visible throughout;
- presentation speed feels natural;
- ambient audio matches the visible environment and is not dominated by wind.

At least one candidate per source must be accepted before the job is declared
content-complete. Rejected records remain in the archive and are indexed with
their issue codes.

## Proposed code boundaries

```text
services/decision/
  decision_service.py          image2json/Ollama orchestration only
  semantic_schema.py           version-2 schema and validation
  routing.py                   hard routing invariants
  experience.py                MemPalace query and compact evidence

services/orchestrator/
  run_batch.py                 job/state orchestration
  profiles.py                  versioned technical profiles
  planner.py                   semantic -> candidate technical plans
  records.py                   atomic manifest/result JSON writes and hashes
  presentation.py              normalized crop/pan/zoom compiler
  review.py                    local feedback/revision CLI
  quality.py                   structural media checks; visual metrics later

services/comfy/workflow_templates/
  wan22_i2v_workflow.json
  hunyuan15_i2v_workflow.json  updated with tiled decode
```

Existing SVD/AnimateDiff support can remain behind explicit legacy/research
selection during migration, but it is removed from automatic routing.

## Implementation slices

### Slice 1 — records, manifest, and tests

Add result/manifest schemas, atomic writes, hashing, resume behavior, and unit tests.
Wrap the current backends first so every subsequent benchmark produces reusable
records. No routing behavior changes in this slice.

### Slice 2 — production Wan and true source-aspect dimensions

Promote the tested Wan workflow into `ComfyClient`, add `WAN22_NATURAL`, compute
dimensions from the source, and render one manually selected image through the
normal orchestrator. Validate against the accepted valley benchmark.

### Slice 3 — Hunyuan long profile and tiled VAE

Promote the 61-frame/12-FPS scooter settings and tiled VAE nodes to production.
Validate seed 1 against the benchmark and one additional deterministic seed.

### Slice 4 — semantic decision version 2

Replace preset/fallback output with classification, motion, generation,
presentation, and audio intent. Add hard routing tests for important versus
incidental person/fauna/vehicle cases. Remove layered prompt mutation.

### Slice 5 — candidate planning and deterministic fallback

Replace `selected_pair` with profile-driven same-backend candidate seeds. Generate
audio once per image. Add deterministic original-still presentation as a normal
mode and render-failure fallback.

### Slice 6 — review, lineage, and MemPalace

Add feedback/revision CLI, searchable `memory.search_text`, result indexing, and
retrieval before semantic planning. Test that retrieved cases cannot introduce
objects absent from the current analysis.

### Slice 7 — seven-image acceptance batch

Snapshot all current inputs, run with resume enabled, review every candidate,
perform bounded revisions, and publish a batch summary containing execution and
human acceptance counts separately.

## Test strategy

There is currently no repository test suite, so implementation starts by adding
focused tests before the expensive GPU acceptance batch:

- semantic schema and routing table tests;
- original-aspect dimension calculation tests for all seven source dimensions;
- profile compiler and deterministic seed tests;
- presentation endpoint visibility and FFmpeg filter tests;
- result-schema, atomic-write, lineage, and resume tests;
- fake Comfy/audio integration tests for batch state transitions;
- fixture tests using sanitized existing debug/image2json records;
- one real Wan valley render and one real Hunyuan scooter render;
- finally, the complete seven-image batch.

The full batch should not begin until the two reference renders and all non-GPU
tests pass. That prevents spending hours on a batch whose routing, dimensions, or
records are already known to be wrong.
