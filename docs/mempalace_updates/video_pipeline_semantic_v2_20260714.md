# Semantic video pipeline v2 — implementation and validation record

Date: 2026-07-14

## Implemented flow

The production decision path is now:

1. `qwen3-vl:8b` image2json analysis records observable scene facts.
2. A read-only local MemPalace service searches wing `image2videoAWSIG` with a
   compact signature derived from that analysis.
3. `qwen3:14b` receives the image analysis, up to three retrieved experiences,
   the semantic schema, and routing rules. It returns generation intent, prompt,
   negative prompt, and deterministic presentation instructions.
4. A deterministic compiler enforces backend routing and maps intent to a
   versioned technical profile. The language model does not select sampler-level
   parameters.
5. Generation preserves source aspect. Square or vertical delivery crop, pan, or
   zoom happens only after full-frame rendering.
6. Every video gets a comparable `.result.json` containing image classification,
   analysis, retrieved experiences, semantic and technical plans, artifact paths
   and hashes, and pending human-feedback fields.

## Enforced backend policy

- Important visible person, fauna, or vehicle motion uses Hunyuan.
- Environmental motion without an important sensitive moving subject uses Wan.
- Scenes without an important visible person, fauna, or vehicle use Wan, even
  when environmental motion should be restrained.
- Mannequins, statues, sculptures, dolls, posters, and photographs do not count as
  real people for Hunyuan routing.
- Small distant incidental people, fauna, or vehicles remain preservation risks;
  they do not automatically force Hunyuan.
- A failed generative candidate may use deterministic presentation only as a
  recovery fallback; deterministic output is never a routine primary candidate.
- Masked VACE is not part of the default path because testing did not improve the
  result over full-frame generation.

## Validated profiles

`WAN22_NATURAL` uses the Wan 2.2 TI2V 5B checkpoint, original-aspect dimensions
with a 768-pixel long edge, 97 generated frames, 20 FPS, 20 steps, CFG 5.0,
shift 8.0, `uni_pc`/`simple`, and full VAE decode. This exactly matches the
accepted first valley configuration. Later claims that 50 steps/shift 5 fixed
the grid were disproved by native-resolution comparison with that accepted run.

`HUNYUAN15_SENSITIVE` uses HunyuanVideo 1.5 I2V, original-aspect dimensions with
a 704-pixel long edge, 61 generated frames, 12 FPS, 50 steps, CFG 6.0, shift 7.0,
`euler`/`simple`, and tiled VAE decode for host-memory safety. The earlier 14-step profile remains
unsupported because the installed checkpoint is not step-distilled, but step
count alone was not proven to be the grid cause. The 97-frame
Hunyuan experiment is not a
production default because it transformed the scooter and was substantially more
expensive.

`DETERMINISTIC_ORIGINAL` is a recovery-only five-second treatment used after
requested generative candidates fail. It is excluded from normal candidate output.

ComfyUI runs on the tested RTX 3090 without `--lowvram`, using `--cache-none` and
`--disable-pinned-memory`. Wan uses full VAE decode. Hunyuan retains tiled decode
because full 61-frame decode previously restarted Comfy during VAE handoff; the
user-rejected Hunyuan grid remains a separate unresolved validation item.
Compose commands must include `--env-file .env` so model and output mounts resolve
to the intended host directories.

## Clean acceptance batch

The earlier batch `semantic-v2-final-selected` had zero technical failures but
was not visually clean: exhaustive review later found grid/blinking in several
Wan outputs and the Hunyuan scooter. Technical media validation and sparse frame
sampling did not constitute visual acceptance.

- Parthenon: Wan, static square presentation.
- Mountain valley: Wan, static square presentation.
- Store mannequins: initially Wan; visual review rejected it because the image
  blurred and drifted late in the clip.
- Waterfront: Wan, static square presentation.
- City architecture: Wan, slow push-in square presentation.
- Desert/campsite with distant incidental vehicles: an intermediate revision
  routed it to deterministic; the 2026-07-15 review rejected that policy and
  restored Wan routing.
- Street scooter with visible riders: corrected Hunyuan 50-step quality profile,
  static square presentation.

The later full-VAE batch showed that low-motion promotion to deterministic was
contrary to the agreed routing and produced unwanted repeated stills. That rule
has been removed; mannequin and other non-sensitive scenes route to Wan.

Superseded artifacts are under:

`video_output/semantic_v2_final_selected/semantic-v2-final-selected/`

Corrected selected artifacts are under:

`video_output/semantic_v2_qualityfix_selected/`

The corrected mannequin artifact is under:

`video_output/semantic_v2_final_selected/semantic-v2-corrected-mannequin/`

## Reliability fixes and classification

- The image-level semantic planner and technical compiler separation is a clean
  architectural fix. It prevents prompts from directly choosing unsafe backend
  settings.
- A process-level batch lock and Comfy queue reset are clean architectural fixes.
  They prevent concurrent batch shells and orphan prompts from corrupting later
  results or restarting Comfy unexpectedly.
- Removing the explicit no-motion deterministic override is a clean routing fix:
  deterministic output is recovery-only and cannot replace a requested Wan result.
- Restoring full VAE decode for Wan is a targeted regression fix grounded in the
  clean valley baseline. It is not yet applied to Hunyuan because of the known
  host-memory restart. A faint model/latent texture can still exist, so
  this does not replace native-resolution human acceptance.
- Wan 20 steps/shift 8 is the empirically accepted local profile. Hunyuan keeps
  50 steps because the installed checkpoint is not step-distilled. The earlier
  inference that Wan also required 50 steps was not supported by the controls.
- `--cache-none --disable-pinned-memory` is a hardware-specific operational
  configuration validated on the current RTX 3090; other GPUs should benchmark it
  rather than assume it is universally optimal.
- `.dockerignore` is a build reliability fix: it excludes local models, media,
  Git history, and caches so image rebuild context remains small.

## Verification

- Sixteen semantic routing, presentation, feedback, dimension, profile, audio,
  and mux unit tests pass.
- Python compilation, Compose configuration validation, `git diff --check`, and
  Docker image rebuilds pass.
- The earlier `semantic_v2_qualityfix_selected` Wan valley and Hunyuan scooter
  outputs are rejected: user review still found grid. A full-decoded 97-frame
  Wan control was generated for renewed human comparison. The later full-VAE
  batch accepted the valley and scooter video settings but rejected its audio;
  deterministic desert/mannequin/city results were rejected.
- Human visual review remains authoritative. A technically successful render is
  stored with `human_feedback.status = pending` until accepted or rejected.
- Accepted and rejected feedback should be mined later as comparable experiences;
  seeds are reproducibility values, not universal visual-style controls.
