# Case: periodic grid and blinking in Wan and Hunyuan outputs

Date: 2026-07-14

## Symptom

Final videos for `0__MG_0006`, `1_20260627_130019`, `4_DSC08334`,
`5_DSC00233`, and Hunyuan scooter `6_20250516_125456` blinked and showed a
fine checkerboard/grid over detailed regions. Sparse frame sampling initially
missed the remaining defect after a decoder change.

## Diagnostic evidence

The raw Comfy MP4s contained the corruption, excluding final crop/pan, upscale,
audio mux, H.264 export, and FPS conversion as causes. Exhaustive every-frame
contact sheets and full-resolution spike frames were required; first/middle/last
sampling was insufficient.

Changing `VAEDecodeTiled` from a 16/4 to 64/8 temporal window removed a coarse
periodic temporal-chunk collapse, but did not remove the fine lattice. The fine
lattice remained unchanged with:

- spatial VAE tiles increased from 512/64 to 1024/128;
- short 49-frame Wan and 30-frame Hunyuan sequences;
- true non-tiled `VAEDecode`;
- earlier seeds previously described as accepted.

Those controls show that a faint periodic latent texture can exist even without
tiling. They did not justify accepting the tiled production renders: direct
comparison with the first valley video shows a materially stronger, blinking
grid in the current output.

## Root cause

The semantic-v2 pipeline enabled `VAEDecodeTiled` for Wan, while the accepted
first valley used plain `VAEDecode`. That is the concrete valley regression.
Tiling amplified periodic texture and made it temporally objectionable in
detailed regions. Hunyuan also has a user-rejected grid, but its decoder cannot
simply be changed: full 61-frame decode previously restarted Comfy during VAE
handoff on this host, so its root cause remains separately unresolved.

The earlier diagnosis that Wan 20 steps/shift 8 was under-denoised was wrong.
The accepted first valley used exactly 20/8, and a 50/5 production render still
showed the user-visible grid. Seed and prompt controls also did not establish a
single universal clean seed; selection remains image-specific.

## Controlled validation

The trustworthy reference is
`.local/outputs/wan-benchmark/valley_wan22_visible_clouds_97f_20fps.mp4`:
768x576, 97 frames, 20 FPS, 20 steps, CFG 5, shift 8, seed 42,
`uni_pc`/`simple`, full `VAEDecode`.

A matched semantic-v2 control was generated with the current prompt and seed,
97 frames, 20/8, and full `VAEDecode`:
`.local/outputs/comfy/valley-fullvae-regression-6f0ab2_00001.mp4`.
Its square presentation is
`.local/outputs/diagnostics/valley-fullvae-fixed/valley_fullvae_fixed_1x1.mp4`.

Hunyuan FP16 runtime loading restarted Comfy before sampling because the host has
only 16 GB system RAM. FP8 runtime loading is therefore retained on this machine;
the RTX 3090's 24 GB VRAM was not the limiting resource.

Validation artifacts:

- `.local/outputs/diagnostics/valley-clean-baseline-compare/decoder_compare_f48.png`
- `.local/outputs/wan-benchmark/valley_wan22_visible_clouds_97f_20fps.mp4`
- `.local/outputs/comfy/valley-fullvae-regression-6f0ab2_00001.mp4`
- `video_output/semantic_v2_qualityfix_selected/` (rejected)

## Fix classification

Disabling tiled decode by default for Wan is a targeted regression fix grounded
in the accepted baseline. It is not a claim that every full-decoded generative result
will be clean; native-resolution human review is still required.

Wan returns to the empirically accepted 20-step/shift-8 schedule. Hunyuan retains
50 steps because the installed full checkpoint is not step-distilled, but the
50-step Hunyuan artifact is rejected until a memory-safe clean decode strategy
is benchmarked and reviewed.

`HUNYUAN15_I2V_FAST` also uses 50 steps with the installed full checkpoint. It
is faster only by generating 30 frames at 6 FPS instead of 61 at 12 FPS. An
8/12-step profile must not be restored unless the separate step-distilled
checkpoint is installed and selected explicitly.

Low-motion scenes explicitly described by image2json as having minimal or no
useful motion now use deterministic presentation. This is a safe general routing
fix; it prevented the desert/campsite image from paying generation cost and
artifact risk for imperceptible motion.

Acceptance must inspect every native frame. Technical success, sparse still
sampling, and temporal-difference statistics alone do not prove visual quality.
