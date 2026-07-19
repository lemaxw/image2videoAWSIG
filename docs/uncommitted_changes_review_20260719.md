# Uncommitted changes review — 2026-07-19

## Scope

This document describes the current uncommitted working tree, emphasizing the
pipeline work completed in the current conversation/session. Git does not record
when an uncommitted line was changed, so it cannot prove session boundaries. Use
`git diff` before committing to verify that every listed file belongs in the same
commit.

At the time of writing, the tracked diff contains roughly 892 additions and
2,782 deletions across 20 tracked files, plus three deleted legacy files and new
untracked pipeline, memory, documentation, and test files.

Generated videos and their feedback JSON files live under `video_output/`, which
is ignored by Git. They contain important evaluation evidence but will not be
included in a normal source commit.

## New pipeline at a glance

```text
source image
    |
    v
image2json / qwen3-vl:8b
    |
    +--> sanitize false motion evidence
    |
    +--> semantic MemPalace search
    |
    v
qwen3:14b semantic plan
    |
    v
deterministic compiler and validator
    |                         |
    |                         +--> model, fixed technical profile, seeds
    |                         +--> focus region and visibility checks
    |                         +--> deterministic crop/pan/push plan
    v
full-frame WAN or Hunyuan render in ComfyUI
    |
    v
FFmpeg presentation + normalized ambient audio
    |
    v
MP4 + JPG + debug JSON + comparable result JSON
    |
    v
human accept/reject feedback for later MemPalace reuse
```

The language model proposes semantics. It does not directly choose arbitrary
samplers, frame counts, or renderer-specific implementation details. Those are
compiled into tested profiles.

## 1. Decision service was reduced to one mandatory local flow

The old multi-backend/legacy decision logic was removed. The remaining service
has one public decision path:

1. Validate that image2json is enabled and that the vision/text models have the
   correct roles.
2. Run `qwen3-vl:8b` through image2json.
3. Sanitize the analysis and unload the vision model.
4. Search MemPalace using properties from the analysis.
5. Give image2json plus retrieved cases to `qwen3:14b`.
6. Compile and validate the semantic response.
7. Unload the text model and return full diagnostic metadata.

Important review lines:

- Mandatory local entry point and failure policy:
  [`decision_service.py`](../services/decision/decision_service.py#L121)
- Vision/text model role validation:
  [`decision_service.py`](../services/decision/decision_service.py#L134)
- image2json execution and sanitation:
  [`decision_service.py`](../services/decision/decision_service.py#L144)
- MemPalace lookup:
  [`decision_service.py`](../services/decision/decision_service.py#L187)
- Text-model input includes both analysis and similar cases:
  [`decision_service.py`](../services/decision/decision_service.py#L217)
- Compiler and validator handoff:
  [`decision_service.py`](../services/decision/decision_service.py#L229)
- Returned decision/debug metadata:
  [`decision_service.py`](../services/decision/decision_service.py#L264)

Review consequence: if image2json or the text decision step fails, there is no
OpenAI or old heuristic decision backend. The job fails explicitly.

## 2. Semantic planning is now a distinct compiler layer

The new [`semantic_planner.py`](../services/decision/semantic_planner.py) is the
largest architectural change.

### 2.1 Schema and prompt contract

The text model returns classification, motion, generation, presentation, and
audio fields under a strict schema. The prompt makes image2json authoritative
for visible content and treats MemPalace only as experience about technique.

- Output schema: [`semantic_planner.py`](../services/decision/semantic_planner.py#L15)
- System prompt and routing rules:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L85)
- Generation/presentation separation:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L104)
- Region-aware framing and motion-discipline rules:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L113)
- Quiet audio rules: [`semantic_planner.py`](../services/decision/semantic_planner.py#L153)

The prompt prohibits generated camera instructions. `push`, `pan`, and `zoom`
are deterministic post-production operations.

### 2.2 Analysis-derived memory search

The MemPalace query contains scene, subject, motion, risk, style, complexity,
and vertical-crop properties rather than filenames alone.

- Query construction: [`semantic_planner.py`](../services/decision/semantic_planner.py#L177)
- Result deduplication/compaction:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L227)
- HTTP retrieval and optional/required failure behavior:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L258)

By default, memory is useful but not required. Set `MEMPALACE_REQUIRED=true` if
a failed memory search must abort the decision.

### 2.3 False motion evidence is removed

Sunlight, shadows, accumulated snow, and depth layers are static properties,
not motion targets. They are removed before the text decision and recorded in
`excluded_static_motion_elements` for diagnostics.

- Sanitizer: [`semantic_planner.py`](../services/decision/semantic_planner.py#L201)

### 2.4 Model routing is compiled from important moving subjects

Important medium/large people, fauna, or vehicles select Hunyuan. Everything
else selects WAN. Tiny incidental subjects do not force Hunyuan. Mannequins,
statues, posters, and similar representations are excluded.

- Sensitive-subject detection:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L295)
- Whole-token matching avoids interpreting `food cart` as `car`:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L307)
- Final enforced backend selection:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L698)

### 2.5 Motion plans are normalized

The compiler rejects static targets and physically implausible rigid-object
motion, removes target/stable conflicts, gives the Hunyuan-sensitive subject
priority, and reconstructs a short action prompt.

- Motion vocabulary and localized actions:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L403)
- Rigid breeze-motion rejection and sensitive-subject priority:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L444)
- Stable geometry and reconstructed prompt:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L480)

Current deliberate heuristics include:

- flowers: visible localized stem sway and petal flutter;
- snowy forests: flexible outer boughs move while trunks and ground snow stay
  fixed;
- clouds/water/grass: short physically grounded actions;
- carts/wheels/buildings do not sway because of wind.

These heuristics deserve careful review because they encode product behavior,
not merely data validation.

### 2.6 Normalized regions control presentation

The compiler parses attention-region and spatial-map boxes, finds a focal
region, positions the target-aspect crop around it, calculates initial/final
viewports, and retains only required regions that fit at both endpoints.

- Box parsing: [`semantic_planner.py`](../services/decision/semantic_planner.py#L352)
- Region priority: [`semantic_planner.py`](../services/decision/semantic_planner.py#L370)
- Region-aware crop/zoom and visibility validation:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L537)
- Framed-opening push logic:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L640)

This fixes the upper-right poppy regression: importance is no longer descriptive
only; its observed region controls the crop and 1.25x push.

### 2.7 Wide panoramas receive a traversal

If all four facts are present—panoramic layout, wide composition, important
full-width content, and high vertical-crop risk—the compiler overrides a fixed
portrait crop with square delivery and a deterministic left-to-right traversal.

- Trigger: [`semantic_planner.py`](../services/decision/semantic_planner.py#L687)
- Applied aspect/operation and 0.10-to-0.80 traversal:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L719)
- Saved traversal validation reason:
  [`semantic_planner.py`](../services/decision/semantic_planner.py#L752)

This rule came from rejected `6_DSC06580` portrait pushes. Ordinary scenes retain
the conservative 0.18 pan span.

### 2.8 Fixed renderer profiles and seeds

- WAN: 97 frames, 20 FPS, 20 steps, CFG 5, shift 8, `uni_pc/simple`, 768-pixel
  long edge, full VAE decode.
- Hunyuan: 61 frames, 12 FPS, 50 steps, CFG 6, shift 7, 704-pixel long edge,
  tiled VAE decode for memory safety.
- Seeds derive deterministically from source hash plus backend. Additional
  candidates use consecutive seeds.
- Deterministic-original is recovery-only and is not a routine primary result.

Review these values at [`semantic_planner.py`](../services/decision/semantic_planner.py#L773)
and candidate/fallback construction at
[`semantic_planner.py`](../services/decision/semantic_planner.py#L788).

## 3. ComfyUI supports only the retained production families

The client now builds Hunyuan 1.5, WAN 2.2, and deterministic-recovery workflows.
AnimateDiff, SVD, VACE, SAM2, and Depth Anything production paths were removed.

- Robust request/prompt monitoring:
  [`comfy_client.py`](../services/orchestrator/comfy_client.py#L39)
- Timeout, queue disappearance, and diagnostics:
  [`comfy_client.py`](../services/orchestrator/comfy_client.py#L122)
- Model unload and orphaned-queue cleanup:
  [`comfy_client.py`](../services/orchestrator/comfy_client.py#L179)
- Original-aspect dimension resolution:
  [`comfy_client.py`](../services/orchestrator/comfy_client.py#L218)
- Hunyuan workflow construction:
  [`comfy_client.py`](../services/orchestrator/comfy_client.py#L248)
- WAN workflow construction:
  [`comfy_client.py`](../services/orchestrator/comfy_client.py#L299)
- Deterministic recovery workflow:
  [`comfy_client.py`](../services/orchestrator/comfy_client.py#L331)

The tiled decoder uses 512/64 spatial tiles and a 64/8 temporal window. The old
16/4 window caused periodic grid/checkerboard frames.

- Tiled VAE constants: [`comfy_client.py`](../services/orchestrator/comfy_client.py#L12)

Review hotspot: model resolution currently returns the first available model if
the requested filename is absent. See
[`comfy_client.py`](../services/orchestrator/comfy_client.py#L67). Decide whether
production should fail instead of silently substituting another checkpoint.

## 4. Orchestration now owns candidates, model handoff, and comparable records

The batch runner:

- renders one to three seeds from the selected family;
- treats deterministic output only as recovery after renderer failure;
- uses the original source for generation when source-aspect preservation is on;
- starts/stops ComfyUI and audio to hand host/GPU memory between stages;
- logs structured start/done/failed events;
- captures Comfy queue/history/system diagnostics on failure;
- normalizes video and audio into the final five-second presentation;
- always writes debug JSON;
- writes a comparable `.result.json` beside each successful candidate.

Important review lines:

- Variant selection: [`run_batch.py`](../services/orchestrator/run_batch.py#L109)
- Original vs cropped render input:
  [`run_batch.py`](../services/orchestrator/run_batch.py#L133)
- Presentation operation/aspect/pan extraction:
  [`run_batch.py`](../services/orchestrator/run_batch.py#L162)
- Docker model-service control:
  [`run_batch.py`](../services/orchestrator/run_batch.py#L371)
- Main per-image pipeline: [`run_batch.py`](../services/orchestrator/run_batch.py#L762)
- Render loop and model/audio handoff:
  [`run_batch.py`](../services/orchestrator/run_batch.py#L870)
- Post-processing invocation:
  [`run_batch.py`](../services/orchestrator/run_batch.py#L975)
- Failure diagnostics and recovery enqueue:
  [`run_batch.py`](../services/orchestrator/run_batch.py#L1046)
- Comparable result record, hashes, and pending feedback:
  [`run_batch.py`](../services/orchestrator/run_batch.py#L1109)

One successful candidate is enough for an image-level success. Therefore a
second seed may time out while the image and batch still finish successfully;
this happened for one Hunyuan candidate.

## 5. Post-production crop, pan, and push are deterministic

FFmpeg now positions a target-aspect crop around the normalized source focus
before a push begins. Previously, a centered 9:16 crop could discard an
off-center focal subject before zooming.

- Static region-positioned crop: [`mux.py`](../services/orchestrator/mux.py#L66)
- Five-second normalized push/pull:
  [`mux.py`](../services/orchestrator/mux.py#L76)
- Region-positioned crop before zoompan:
  [`mux.py`](../services/orchestrator/mux.py#L87)
- Deterministic panoramic pan:
  [`mux.py`](../services/orchestrator/mux.py#L105)
- Audio normalization and final encoding:
  [`mux.py`](../services/orchestrator/mux.py#L130)

Final outputs are 1080x1920 or 1080x1080, 30 FPS, five seconds, H.264/AAC.

## 6. Validation rejects retired presets and preserves presentation metadata

Validation now recognizes only the retained WAN/Hunyuan profiles plus
deterministic recovery, supplies fixed defaults, clamps backend-specific values,
and passes focus/required-region/visibility metadata to runtime.

- Presentation metadata validation:
  [`validate.py`](../services/orchestrator/validate.py#L118)
- Audio prompt construction:
  [`validate.py`](../services/orchestrator/validate.py#L193)
- Fixed profile defaults: [`validate.py`](../services/orchestrator/validate.py#L257)
- Retired-preset rejection: [`validate.py`](../services/orchestrator/validate.py#L345)
- Full decision validation: [`validate.py`](../services/orchestrator/validate.py#L456)

The maximum allowed pan span was expanded to 0.80 so the compiler's explicitly
flagged panorama traversal can pass validation. Ordinary compiler output still
uses 0.18.

## 7. MemPalace became a local pipeline service

The new memory container exposes health and semantic search over the existing
local palace. The orchestrator sends an image-derived query and receives
verbatim cases.

- Search request and FastAPI service:
  [`memory_service.py`](../services/memory/memory_service.py#L21)
- Health endpoint: [`memory_service.py`](../services/memory/memory_service.py#L33)
- Search endpoint: [`memory_service.py`](../services/memory/memory_service.py#L38)
- Compose service and local palace mount:
  [`docker-compose.yml`](../services/comfy/docker-compose.yml#L2)
- Orchestrator memory configuration:
  [`docker-compose.yml`](../services/comfy/docker-compose.yml#L97)

Human feedback is persisted atomically in each result JSON:

- Feedback updater: [`review.py`](../services/orchestrator/review.py#L12)
- CLI arguments: [`review.py`](../services/orchestrator/review.py#L40)

Accepted records become `ACCEPTED`; rejected records become `REJECTED`; pending
records remain `HUMAN_REVIEW`.

## 8. Audio was made quieter and less windy/insect-heavy

Nature prompts no longer add insects by default. Water/coast prompts remove
automatic wind, and default post-processing moved from -14 LUFS to -18 LUFS.

- Scene sound hints: [`audio_service.py`](../services/audio/audio_service.py#L64)
- Post-processing loudness: [`audio_service.py`](../services/audio/audio_service.py#L261)
- Compose default: [`docker-compose.yml`](../services/comfy/docker-compose.yml#L58)

Changing audio backend/device settings still requires recreating the audio
container, not merely restarting it.

## 9. Container/build changes

ComfyUI installs the exact CUDA 13 PyTorch packages first, removes only
`torch`, `torchvision`, and `torchaudio` entries from ComfyUI's requirements,
then installs the remaining requirements. This prevents a second installation
from replacing the pinned CUDA build and avoids duplicate large downloads.

- Pinned CUDA packages and filtered requirements:
  [`Dockerfile`](../services/comfy/Dockerfile#L25)
- VideoHelperSuite is the only extra Comfy custom node installed:
  [`Dockerfile`](../services/comfy/Dockerfile#L37)
- GPU Comfy runs with `--cache-none --disable-pinned-memory`, not `--lowvram`:
  [`docker-compose.gpu.yml`](../services/comfy/docker-compose.gpu.yml#L3)
- Local memory, model, input/output, image2json, and Docker-socket mounts:
  [`docker-compose.yml`](../services/comfy/docker-compose.yml#L82)

Review hotspots:

- `COMFYUI_REF` defaults to `master`, so rebuilding later is not fully
  reproducible. Consider pinning a tested commit.
- The requirement-filter regex is intentionally anchored, so it removes
  `torch==...` but does not remove unrelated packages such as `torchsde`.
- `.dockerignore` is new and should be reviewed to ensure required build inputs
  are not excluded.

## 10. Retired model and benchmark cleanup

Deleted production artifacts:

- `services/comfy/workflow_templates/animatediff_workflow.json`
- `services/comfy/workflow_templates/svd_workflow.json`
- `services/comfy/scripts/benchmark_vision_models.py`

VACE/masked-composite paths were removed from the WAN benchmark. The WAN script
now benchmarks only the accepted Wan 2.2 family and exposes CFG, shift, and
sampler controls. Hunyuan benchmark defaults now match the full installed
checkpoint rather than the old 14-step experiment.

- WAN benchmark: [`benchmark_wan_models.py`](../scripts/benchmark_wan_models.py#L1)
- Hunyuan benchmark parameters:
  [`benchmark_hunyuan.py`](../scripts/benchmark_hunyuan.py#L24)
- New WAN workflow:
  [`wan22_i2v_workflow.json`](../services/comfy/workflow_templates/wan22_i2v_workflow.json)
- New deterministic recovery workflow:
  [`deterministic_workflow.json`](../services/comfy/workflow_templates/deterministic_workflow.json)

Historical benchmark documents were corrected to explain that the 16/4 tiled
temporal decoder caused grid artifacts; the accepted production window is 64/8.

## 11. Tests and validation evidence

The new suite contains 24 tests covering routing, fixed profiles, seed pairs,
retired presets, region-aware flowers, winter motion discipline, people-focused
Hunyuan framing, panorama traversal, audio hints, mux filters, VAE settings,
feedback persistence, and source-aspect sizing.

- Test suite: [`test_semantic_pipeline.py`](../tests/test_semantic_pipeline.py#L1)
- People/cart regression:
  [`test_semantic_pipeline.py`](../tests/test_semantic_pipeline.py#L81)
- Panorama regression:
  [`test_semantic_pipeline.py`](../tests/test_semantic_pipeline.py#L175)
- Flower regression:
  [`test_semantic_pipeline.py`](../tests/test_semantic_pipeline.py#L232)
- Winter regression:
  [`test_semantic_pipeline.py`](../tests/test_semantic_pipeline.py#L277)
- Retired presets: [`test_semantic_pipeline.py`](../tests/test_semantic_pipeline.py#L319)
- Feedback persistence:
  [`test_semantic_pipeline.py`](../tests/test_semantic_pipeline.py#L422)

Last observed validation: 24 tests passed, `git diff --check` passed, and the
orchestrator image was rebuilt/recreated.

## Suggested review order

1. Review the product policy in `SEMANTIC_SYSTEM_PROMPT`.
2. Review routing and motion normalization.
3. Review region/crop mathematics and panorama override.
4. Review the fixed WAN/Hunyuan technical profiles.
5. Review runtime fallback semantics and service handoff.
6. Review the comparable result/feedback schema.
7. Review container reproducibility and the model-resolution fallback.
8. Review deletions and historical documentation last.

Useful commands:

```bash
git diff -- services/decision/decision_service.py services/decision/semantic_planner.py
git diff -- services/orchestrator/comfy_client.py services/orchestrator/validate.py
git diff -- services/orchestrator/run_batch.py services/orchestrator/mux.py
git diff -- services/comfy/Dockerfile services/comfy/docker-compose.yml services/comfy/docker-compose.gpu.yml
git diff -- services/audio/audio_service.py services/orchestrator/review.py services/memory
git diff -- tests/test_semantic_pipeline.py
git diff --summary
git diff --check
docker exec pipeline-orchestrator python -m unittest discover -s /app/tests -v
```

---

# Self-examination questionnaire

Try to answer without opening the answer key. Questions 1–12 check the basic
architecture; questions 13–20 test whether you can apply it to failures.

## Architecture and policy

1. Put these stages in order: semantic compiler, image2json, Comfy generation,
   MemPalace search, Ollama text decision, deterministic presentation.

2. Which input is authoritative for objects visible in the current image:
   image2json or a similar MemPalace case? What is MemPalace allowed to influence?

3. What conditions select Hunyuan? Why does a tiny distant car normally remain
   on WAN?

4. Why is deterministic-original present if deterministic video is not a normal
   semantic backend?

5. Name at least four values removed from `natural_motion_elements` and explain
   why.

6. Where are frame count, FPS, sampler, CFG, shift, and VAE policy decided: by
   the text model or by the compiler/profile?

7. Why are generated-camera words removed from the generation prompt?

8. What is the difference between `focus_target`, `focus_region`,
   `must_keep_visible`, and `visibility_validation`?

9. What exact evidence triggers the special square panorama traversal?

10. Why does the Comfy Dockerfile filter three packages out of
    `requirements.txt`? Why does it not accidentally remove `torchsde`?

11. Does a MemPalace outage abort the pipeline by default? Which setting changes
    that behavior?

12. Which files are produced for each successful candidate, and which fields
    make candidates comparable/searchable later?

## Failure scenarios

13. image2json describes sunlight, accumulated snow, trees, and a snow-covered
    bough. The text model asks sunlight and snow to move and also places snow in
    `keep_stable`. What should the compiler produce instead?

14. A high-importance flower is at x=0.70 in a 16:9 source. A centered 9:16 crop
    would remove it. What information now prevents that, and what is validated?

15. A street image contains a large food cart and medium seated people. The text
    plan asks the cart wheels to sway in the breeze and people to adjust posture.
    Which backend, primary motion target, stable target, and crop focus should
    result?

16. Why was matching `car` with a substring unsafe for `food cart`, and what form
    of matching replaced it?

17. A city panorama has pale clouds and very distant foliage. Two WAN seeds look
    equally static after centered portrait pushes. Should the next action be a
    third seed, a new model, or a presentation change? Describe the compiled
    presentation.

18. A Hunyuan pair has one successful candidate and one prompt that remains in
    Comfy's running queue until timeout, without OOM. Is the image-level result
    necessarily a failure? What diagnostics should be saved?

19. Grid/checkerboard frames appear periodically in both WAN and Hunyuan while
    using tiled decoding. Which decoder parameters should you inspect before
    blaming prompt or seed?

20. After reviewing candidates, how is a user's acceptance/rejection represented
    in the result JSON, and why must this feedback be recorded rather than only
    mentioned in conversation?

---

# Answer key

1. image2json -> MemPalace search -> Ollama text decision -> semantic compiler
   and validator -> Comfy generation -> deterministic presentation/mux.

2. image2json is authoritative for visible content. MemPalace may influence
   model/prompt technique, known risks, seed/candidate policy, and presentation,
   but must not introduce absent objects or actions.

3. An important medium/large/focal person, fauna, or vehicle selects Hunyuan.
   A tiny distant incidental subject is mainly a preservation risk and does not
   justify Hunyuan's identity-sensitive motion path.

4. It is an execution recovery when generative candidates fail. It prevents an
   empty job result without allowing the semantic planner to routinely replace
   requested motion with a repeated still.

5. Sunlight, light, shadows, accumulated/ground snow, and depth layers are
   examples. They describe static appearance or geometry, not independently
   moving physical elements.

6. The deterministic compiler/profile chooses them. The text model chooses
   semantics: target/action, backend intent, candidate count, and presentation.

7. Model-generated camera motion caused speed, composition, and object-retention
   problems. A deterministic FFmpeg camera operation is predictable and
   separable from physical scene motion.

8. `focus_target` is the semantic label. `focus_region` is its observed normalized
   source box. `must_keep_visible` is the subset of essential labels compatible
   with the crop. `visibility_validation` records whether those regions fit in
   initial/final viewports or why a panorama uses traversal.

9. `composition.layout=panoramic`, `wide_composition=true`,
   `full_width_important_content=true`, and `vertical_crop_risk=high`.

10. PyTorch is installed first from the pinned CUDA index. Filtering prevents
    Comfy requirements from downloading/replacing it. The anchored regex requires
    the package name to end or be followed by a version operator/space, so
    `torchsde` does not match.

11. No. Memory is optional by default. `MEMPALACE_REQUIRED=true` makes search
    failure fatal.

12. Final MP4, presentation JPG, debug JSON, and per-candidate result JSON. The
    result includes source hash, analysis, retrieved experience, semantic plan,
    technical plan, seed/generation attempt, presentation values, artifact
    hashes, and human-feedback state.

13. Move only the flexible snow-laden outer boughs. Keep tree trunks, accumulated
    ground snow, and forest geometry stable. Sunlight is a focus/appearance cue,
    not a motion target. Remove target/stable contradictions.

14. The normalized attention/spatial region supplies the focus center. The
    target-aspect crop is positioned around it before zoom. Both initial and
    final viewports are checked, and the flower uses localized stem/petal motion
    with approximately a 1.25x deterministic push.

15. Hunyuan; people as primary moving target; the rigid food cart/building as
    stable; the people region as crop focus. Rigid wheels do not sway because of
    a breeze.

16. `car` is a substring of `cart`, causing a false vehicle classification.
    Whole normalized label-token intersection replaced substring matching for
    sensitive-subject routing.

17. Change presentation first. The source is already clean and another seed will
    not make tiny details readable. Compile square 1:1 output and a smooth
    deterministic left-to-right traversal, currently 0.10 to 0.80 over five
    seconds.

18. No. One valid candidate makes the image successful. Save timeout/error type,
    prompt ID, queue, prompt history, system/VRAM statistics, elapsed time, and
    model-free/unload results.

19. Inspect spatial tile/overlap and especially temporal tile/overlap. The old
    16/4 temporal window caused periodic grids; production uses 64/8 with 512/64
    spatial tiling where Hunyuan needs tiled decode.

20. `human_feedback.status` becomes `accepted` or `rejected`, with optional
    rating, issue codes, notes, and review timestamp; top-level state becomes
    `ACCEPTED` or `REJECTED`. Recording it makes the outcome machine-searchable
    for later MemPalace retrieval and prevents repeating rejected plans.
