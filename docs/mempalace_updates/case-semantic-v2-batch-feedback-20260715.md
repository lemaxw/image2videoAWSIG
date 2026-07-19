# Case: semantic-v2 full-VAE batch human review

Date: 2026-07-15

Batch: `video_output/semantic_v2_fullvae_batch/semantic-v2-fullvae-batch-20260714/`

## Human findings

- Valley `1_20260627_130019` WAN candidates were visually accepted.
- Scooter `6_20250516_125456` Hunyuan candidates were visually accepted.
- Parthenon `0__MG_0006` WAN candidate 2 was preferable but rejected in its
  current form because of earthquake-like global camera shake.
- Deterministic repeated-still outputs were rejected as normal candidates,
  including cases `2_DSC05444`, `3_DSC04483`, `4_DSC08334`, and `5_DSC00233`.
- No delivered video visibly used the requested deterministic pan or push.
- Audio was too loud and overemphasized wind and insects.

The sibling `.result.json` records contain this accepted/rejected feedback.

## Root causes

The compiler used `_text()` to parse enum values. `_text()` replaces underscores
with spaces, so `push_in`, `pan_right_to_left`, `square_1_1`, and
`center_center` failed enum validation and silently fell back to static/center
defaults.

A low-motion compiler heuristic overrode otherwise valid WAN decisions with
`DETERMINISTIC_ORIGINAL`. In multi-candidate runs, the deterministic recovery
fallback was also rendered as an ordinary final candidate.

Audio was normalized aggressively at two stages: generated audio targeted
-14 LUFS, final mux targeted -12 LUFS, and mux added 3 dB. Nature prompt helpers
also injected `insects buzzing` and wind-related phrases automatically.

## Fixes and classification

- Clean architectural fix: schema and compiler now enforce the agreed routing:
  important person/fauna/vehicle uses Hunyuan; everything else uses WAN.
  Deterministic treatment is recovery-only after requested generative candidates
  fail.
- Safe general fix: enum parsing preserves underscores. City/architectural
  landscapes prefer a subtle deterministic push or bounded pan unless it would
  crop required content.
- Safe general audio fix: generated and mux loudness targets are -18 LUFS, mux
  gain defaults to 0 dB, candidate mix defaults to -6 dB, and nature helpers no
  longer inject insects or wind.
- Targeted experimental workaround: a stabilized Parthenon candidate is at
  `video_output/semantic_v2_feedback_review/0__MG_0006/final_20260714_192955_wan_c2_stabilized.mp4`.
  Stabilization is not a universal default until human review confirms that its
  crop/warping tradeoff is preferable.

## Validation

- Replaying the saved semantic plans routes cases 2–5 to WAN.
- Saved operations survive compilation: waterfront/city cases 3 and 4 become
  `push_in`; the accepted scooter plan also retains `push_in`.
- Routine candidate selection excludes recovery-only deterministic fallbacks.
- Sixteen unit tests, Python compilation, Compose validation, and whitespace
  validation pass.
- Live case `4_DSC08334` completed with WAN, `push_in`, vertical presentation,
  and no deterministic output. Artifact:
  `video_output/semantic_v2_feedbackfix/semantic-v2-feedbackfix-20260715/4_DSC08334/final_20260715_050113_wan_c1.mp4`.
- Its final audio measured -26.1 dB mean / -14.4 dB peak, compared with
  -13.7 dB mean / -1.3 dB peak in the rejected batch output: approximately
  12–13 dB quieter.

## Framed-opening push follow-up (2026-07-16)

Human review of `4_DSC08334/final_20260715_050113_wan_c1.mp4` found that its
push was too weak: the near wooden/metal architectural frame was still visible
at the end. This was not a WAN generation problem.

Two presentation defects caused the result:

- The generic deterministic push had a fixed endpoint of only 1.06x.
- FFmpeg applied `zoompan` to the native 97 WAN frames before extending the
  clip to 150 delivery frames. The zoom therefore reached only about 1.039x.
  The delivered file also ended at 97 frames / 3.23 seconds instead of the
  requested 150 frames / 5 seconds.

Safe general fix: when image2json reports `composition.layout=framed`, textual
evidence that the focal scene is viewed through a near foreground frame/opening,
and a usable distant focal region, compile `zoom_mode=enter_frame`. Use the focal
region center and a 2.0x portrait or 2.2x square endpoint, currently capped at
2.2. Normalize duration
and FPS before `zoompan`, so the full move runs across 150 frames. Ordinary
pushes remain subtle at 1.06x.

The exact case used image2json's skyline center `(0.535, 0.625)`. A 2.0x 9:16
push removes both the upper beam and lower ledge from the last frame while
retaining more skyline than the 2.2x endpoint. A square delivery needs 2.2x to
clear both edges. Validated review artifact:
`video_output/semantic_v2_feedbackfix/semantic-v2-feedbackfix-20260715/4_DSC08334/final_20260715_050113_wan_c1_enter_frame.mp4`
(`1080x1920`, 30 FPS, 150 frames, 5.0 seconds).
