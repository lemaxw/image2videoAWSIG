# Hunyuan benchmark — 2026-07-11

Source cases:

- `3_DSC04483.jpg`: wide nighttime waterfront with water/reflection motion opportunity.
- `6_20250516_125456.jpg`: red scooter and two riders; identity-persistence stress case.

All generated variants used HunyuanVideo 1.5 720p I2V, 14 steps, CFG 5.8,
shift 7.0, and FP8 weights. Production code/defaults were not changed.

## Results

- The live Comfy `HunyuanVideo15ImageToVideo` node accepted production length 30
  and native `4n+1` lengths 49 and 61.
- Waterfront 49/10 fps and 61/12 fps produced denser five-second clips, but did
  not make water motion clearly readable. More frames mainly added lighting and
  reflection drift. Average frame YDIF: baseline 30=0.4124, 49=0.5857, 61=1.0521.
- Static short-prompt seeds 0/1/2 all remained stable and nearly static. YDIF was
  0.3904, 0.3165, and 0.3237 respectively. Multiple seeds did not discover a
  clearly successful water-motion result for this case.
- Generated camera prompt produced visible global movement (YDIF 1.6817) without
  the deterministic crop's rigid slide, but it did not improve water animation.
- Deterministic crop pan 0.36->0.50 was visually obvious and moved the left
  building through/out of the narrow crop. It should only be used after checking
  protected-region visibility across the full crop path.
- A static square cover preserved substantially more waterfront context than
  static 9:16 cover or 9:16 pan.
- Direct 9:16 pre-cropped Hunyuan input was stable, but necessarily discarded the
  full-width building/skyline relationship before generation.
- Scooter descriptive prompt replaced the scooter/riders with a car late in the
  clip (YDIF 10.7051). The short action prompt preserved a scooter and two riders
  throughout and reduced overall change (YDIF 6.9207), but helmets, clothing, and
  rider appearance still changed. Prompting reduces failure severity but does not
  guarantee identity.

## Recommendation

- Keep 30 frames as the cost-effective default; 49/61 are not justified by this
  test for static environmental scenes.
- Prefer short motion/action prompts over long image descriptions.
- Generate 2–3 seeds only for cases with a meaningful motion target, then score
  them; do not expect seed variation alone to solve unavailable local motion.
- Separate generated camera motion from deterministic final crop motion.
- Prefer static square output for high-risk full-width compositions.
- Treat people/vehicles/distinct objects as persistence-risk scenes and reject
  candidates where protected subjects change or disappear.

## Original-aspect long scooter benchmark — 2026-07-13

The source `6_20250516_125456.jpg` is 4080x3060 (4:3). Literal native generation at that resolution is not practical for HunyuanVideo 1.5 720p or the RTX 3090/16 GB host-RAM configuration. The correct flow is to preserve the original 4:3 aspect during model generation and upscale the accepted render afterward.

- 960x720, 97 frames, 20 FPS, 14 steps was accepted by the node but ran at about 75 seconds per sampling step and was interrupted after 13 minutes. This is operationally unusable.
- 960x720, 61 frames caused a Comfy restart during model/VAE handoff. It is not stable on the current host-RAM budget.
- 704x528, 61 frames, 12 FPS, 14 steps, seed 0 succeeded in 236.3 seconds with the original 4:3 composition.
- The same 61-frame configuration with seed 1 initially finished sampling but restarted during full-batch VAE decode. Replacing `VAEDecode` with `VAEDecodeTiled` (`tile_size=512`, `overlap=64`, `temporal_size=16`, `temporal_overlap=4`) made it repeatably successful in 243.0 seconds.
- Seed 1 was visually better than seed 0: the red scooter and both riders remain present and recognizable while moving, and the complete street context is retained.
- 704x528, 97 frames, 20 FPS, seed 1 with tiled VAE completed in 423.7 seconds, but the scooter temporarily collapsed into a grid-like structure and identity degraded. More generated frames reduced quality for this moving-subject case.

### 2026-07-14 tiled-decode correction

Later frame-by-frame inspection showed that the periodic checkerboard/grid and
brightness pulses were produced by the unusually short `temporal_size=16`,
`temporal_overlap=4` VAE decode window, not solely by Hunyuan sampling. The same
artifact appeared in Wan clips using that decoder configuration. Controlled
same-seed rerenders changed only the decoder to ComfyUI's `64/8` temporal window:

- Wan valley, 97 frames: the grid disappeared and Comfy remained stable; total
  prompt execution was 126.55 seconds.
- Hunyuan scooter, 61 frames: the grid disappeared and Comfy remained stable;
  total client-observed execution was 222.17 seconds.

Production keeps spatial tiling (`tile_size=512`, `overlap=64`) for memory safety
but uses `temporal_size=64`, `temporal_overlap=8`. The earlier `16/4` artifacts
must not be interpreted as evidence that a seed or prompt caused checkerboarding.

The recommended Hunyuan parameters for this hardware and subject type are therefore:

```text
model: HunyuanVideo 1.5 720p I2V, FP8 runtime weights
generation size: 704x528 (original 4:3 aspect)
frames: 61
fps: 12
duration: 5.083 seconds
steps: 14
cfg: 5.8
shift: 7.0
sampler/scheduler: Euler / simple
seed strategy: render 2 candidates; seed 1 won this case
VAE decode: tiled, 512px tiles, 64px overlap, 64-frame temporal tiles, 8-frame overlap
presentation: upscale accepted output after generation; do not pre-crop the source
```

The accepted seed-1 output was upscaled with Lanczos to the source delivery resolution of 4080x3060 while retaining all 61 native frames. Upscaling restores delivery dimensions, not model detail.

The benchmark manifest and all MP4/contact-sheet artifacts are under
`.local/outputs/hunyuan-benchmark/`.
