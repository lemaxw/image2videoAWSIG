# Wan/VACE landscape benchmark — 2026-07-12

## Outcome

The downloaded checkpoints load and render successfully on the local RTX 3090. SAM2.1 and Depth Anything V2 also run successfully in the rebuilt ComfyUI image. No tested generative video model is reliable enough to replace the current renderer unconditionally.

The recommended direction is a layered landscape pipeline: preserve the source image as the immutable base, identify motion regions with refined SAM masks, create conservative depth/procedural motion, and use Wan/VACE only as optional candidates whose output is composited locally and rejected when protected content changes.

## Runtime

- ComfyUI 0.27.0
- PyTorch 2.10.0+cu130
- NVIDIA GeForce RTX 3090, 24 GB VRAM
- ComfyUI runs in `NORMAL_VRAM` mode (the explicit `--lowvram` flag was removed after validation)
- Container health after rebuild: healthy

## Results

| Test | Configuration | Render time | Result |
|---|---:|---:|---|
| VACE 1.3B smoke | 512x320, 17 requested frames, 8 steps | 51.1 s | Runs, but redraws the landscape heavily; unsuitable as ordinary image-to-video. |
| VACE 1.3B, coarse stream mask | 640x480, 49 frames, 20 steps | 144.2 s | Better scene retention, but coarse boundaries invent vegetation and alter nearby banks. |
| VACE 1.3B, SAM stream mask | 640x480, 49 frames, 20 steps | 141.1 s | More localized than the coarse mask, but still reconstructs water/bank texture and has a poor opening transition. |
| Wan2.2 TI2V 5B, valley | 768x576, 49 frames, 20 steps | 90.5 s | Best generative result: stable landscape and visible subtle stream motion. Still not guaranteed to preserve geometry. |
| Wan2.2 TI2V 5B, valley, NORMAL_VRAM | 768x576, 49 frames, 20 steps | 66.1 s | Successful with the same seed and workload; approximately 27% faster than the earlier low-VRAM run. |
| Wan2.2 TI2V 5B, waterfront | 768x512, 49 frames, 20 steps | 90.1 s | Hard failure: invents a camera/tripod and arms after about two seconds despite a locked-camera prompt. |
| Wan2.2 TI2V 5B, waterfront, neutral prompt | 768x512, 49 frames, 20 steps | 60.1 s | Removing the foreground equipment/body-part concepts eliminates that hallucination with the same seed. Skyline remains stable, but the building reflection becomes an oversized rectangular blue patch. |
| Wan2.2 TI2V 5B, waterfront, explicit reflection negatives | 768x512, 49 frames, 20 steps | 60.1 s | The rectangular patch is reduced, but new long luminous streaks appear across the water. Explicitly naming unwanted water artifacts in the negative prompt is not reliable. |
| SAM2.1 Base+ | 2048x1536 source | 1.606 s | Runs. A box finds much of the stream but includes twigs/banks; positive and negative point refinement is required. |
| Depth Anything V2 Small | 2048x1536 source | 0.478 s | Runs and produces a plausible relative depth map suitable for restrained 2.5D parallax. |
| Wan2.2, SAM clouds+stream, hard composite | 768x576, 49 frames, 20 steps | 69.1 s | Successful. Only SAM-approved regions receive generated pixels; outside-mask temporal difference is 0.107/255 after video encoding. |
| VACE 1.3B, SAM clouds+stream, hard composite | 768x576, 49 frames, 20 steps | 189.2 s | Successful and spatially constrained, but almost three times slower than Wan; outside-mask temporal difference is 0.176/255 after encoding. |

VACE returns 21 encoded frames for the 17-frame smoke request because of the native temporal packing/trim behavior. The 49-frame runs produce the intended longer clip configuration.

## Interpretation

Wan2.2 5B is fast enough for candidate generation and can work well on a natural valley, but the waterfront hallucination shows that prompt constraints alone cannot protect scene contents. VACE 1.3B is useful only as a region-control experiment: its output should never replace the complete source frame. Even with a mask, it changes texture around the target and needs exact compositing plus an automated quality gate.

SAM2.1 and Depth Anything V2 are practical on this hardware. Segmentation needs refinement beyond one box. Depth is fast enough to use routinely for deterministic camera motion that does not hallucinate objects.

## Recommended production flow

1. Classify the desired motion from image2json into water, clouds/mist, foliage, subject action, or camera motion. Mark buildings, faces, signs, horizon, and other important structures as protected regions.
2. Keep the original image as the immutable base. Generate motion masks using SAM2.1 with positive/negative points and conservative erosion/feathering.
3. Prefer deterministic motion first: depth-based 2.5D pan/push, water displacement/reflection animation, cloud/mist advection, and small foliage deformation.
4. For organic motion that procedural effects cannot provide, generate candidates:
   - Wan2.2 5B for broader low-risk natural scenes.
   - VACE 1.3B only for tightly masked regional motion.
5. Composite generated pixels only inside the approved motion mask. Preserve all pixels outside it from the deterministic base, and discard unstable opening frames when needed.
6. Score every candidate for protected-region drift, new-object appearance, optical-flow speed, temporal flicker, and requested-motion strength. Reject or fall back to deterministic animation when thresholds fail.
7. Apply final 9:16 framing as a controlled crop/pan of the accepted full scene, then add audio.

## Decision

Proceed with the layered pipeline, but do not make Wan2.2 or VACE the unconditional renderer. The first implementation milestone should be SAM mask refinement, depth/procedural animation, masked compositing, and rejection metrics. Add Wan2.2/VACE candidates only after that deterministic path is working.

## Reusable successful Wan and framing parameters

The strongest valley result used Wan2.2 TI2V 5B with the original 4:3 composition, 768x576 generation, 97 native frames, 20 FPS playback, 20 sampling steps, and seed 42. Wan temporal lengths should use `4n+1`; 97 frames therefore gives an exact encoded duration of 4.85 seconds at 20 FPS. The action-first prompt made cloud motion primary, retained natural stream flow as secondary motion, required a stable viewpoint and protected the geography. Generation in NORMAL_VRAM mode took 129.1 seconds on the RTX 3090.

The preferred Instagram presentation is a deterministic centered 1:1 crop, not a generative camera move. For a 768x576 video, crop a 576x576 window at `x=(768-576)/2=96`, `y=0`, then scale to 1080x1080 with Lanczos and convert to 30 FPS:

```text
crop=576:576:96:0,scale=1080:1080:flags=lanczos,fps=30
```

For a horizontal pan, animate only the crop window after generation. A right-to-left crop of width `crop_w` over duration `D` uses `x=(iw-crop_w)*(1-t/D)`, clamped by the clip duration. This preserves generated scene geometry and separates presentation motion from model motion. The matching original-image still uses the same centered geometry: `crop=ih:ih:x=(iw-ih)/2:y=0`, followed by scaling to 1080x1080.

## Artifacts

All videos, contact sheets, masks, depth maps, and machine-readable timings are in `.local/outputs/wan-benchmark/`. The isolated benchmark entry point is `scripts/benchmark_wan_models.py`; direct vision-model checks are in `services/comfy/scripts/benchmark_vision_models.py`.
