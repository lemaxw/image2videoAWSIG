# Case: dry-008 human review — accepted references, horse framing, and Wan jump

Date: 2026-07-24

Batch: `video_output/out/dry-008/`

## Exact accepted candidates

The user explicitly accepted these exact c1 files; sibling c2 candidates were
not implicitly accepted:

- `1.IMG_2278/final_20260724_100221_wan_c1.mp4`
  - WAN, square 1:1, deterministic pan left-to-right `0.10 -> 0.80`.
- `4.20260519_1400.33/final_20260724_103806_wan_c1.mp4`
  - WAN, square 1:1, push-in `zoom_end=1.15`, focus about `(0.475, 0.595)`.
- `5.DSC01952/final_20260724_104505_hunyuan_c1.mp4`
  - Hunyuan, square 1:1, push-in `zoom_end=1.15`, focus about `(0.60, 0.61)`.

Their sibling `.result.json` records are marked `ACCEPTED`, rating 5.

## Additional accepted MCP candidate

The user explicitly accepted this manually constrained retry as a good case:

- `video_output/out/manual-dsc09278-ig-pan-rigid-v2-20260724/2.DSC09278/final_20260724_174552_wan_c1.mp4`
  - WAN, IG portrait 9:16, deterministic pan right-to-left `0.78 -> 0.18`.
  - Seed `1326754209`; source-preserving raw input.
  - The wheel is held rigid and stationary through 3 seconds, then prompted
    for only 15 degrees of rotation during the final 2 seconds.
  - MCP quality check passed with mean step `0.005586` and maximum step
    `0.055527`.

Its `.result.json` record is marked `ACCEPTED`. The preceding seed
`1326754208` remains explicitly `REJECTED` for wheel deformation, pose change,
and excessive motion.

## Case: DSC08496 horse/rider framing

Symptom:

- Both original Hunyuan candidates used 9:16 plus `push_in`.
- The horse was not fully included.
- The desired treatment is 1:1 with a left-to-right camera crop that follows
  the horse and rider.

Exact debug:

`video_output/out/dry-008/3.DSC08496/debug_20260724_101610.json`

Key evidence:

- Source is `6000x4000` landscape.
- image2json described a motion-blur/panning photograph.
- The high-importance `rider and horse` region touched the left edge.
- The old visibility result explicitly excluded `man and horse`, `man`,
  `horse`, `saddle`, and `bridle`.
- Raw Hunyuan c1 is `704x464`; original final c1 is `1080x1920`.

Root cause:

The text model selected portrait push-in despite conflicting observed evidence:
a prominent sensitive subject touched the edge in a photographed tracking-pan
composition.

Decision direction:

- The image2json evidence and human recommendation should be presented to the
  text model, which owns the aspect and pan decision.
- Do not add a deterministic semantic compiler rule for this composition.
- Codex can produce a targeted 1:1 left-to-right tracking clip through explicit
  overrides in the `image2video-pipeline` MCP server.

Validation preview, remuxed from the existing raw Hunyuan output and audio:

`video_output/out/dry-008/3.DSC08496/final_20260724_101610_hunyuan_c1_square_tracking.mp4`

It is `1080x1080`, 30 FPS, 150 frames, and 5.0 seconds. This is a manual
reference artifact, not evidence of a compiler rule. Contact-sheet inspection
shows the horse and rider staying substantially complete while the crop follows
their left-to-right motion.

## Case: DSC07662 Wan jump under a good push-in

Symptom:

The square 1.15x push-in was visually good, but c1 jumped.

Exact paths:

- Debug:
  `video_output/out/dry-008/6.DSC07662/debug_20260724_111546.json`
- Rejected final:
  `video_output/out/dry-008/6.DSC07662/final_20260724_111546_wan_c1.mp4`
- Raw c1:
  `.local/outputs/comfy/dry-008-6.DSC07662-0-wan_c1-162d41_00001.mp4`

Key evidence:

- Raw Wan c1 is `768x512`, 20 FPS, 97 frames, 4.85 seconds.
- Final c1 is `1080x1080`, 30 FPS, 150 frames, 5.0 seconds.
- The largest discontinuities already occur in raw c1 near 0.6 s and 1.55 s.
- Frame-to-source SSIM continuity measured
  `mean_similarity_step=0.029154` and `max_similarity_step=0.123675`.
- Therefore the push/mux did not create the jump; it made the raw discontinuity
  easier to see.

Fix classification: targeted quality guard for a general fragile scene class.

- Before audio/mux, low-motion Wan push/pull scenes with dense details,
  fine geometry, or repeating patterns get a raw frame-to-source SSIM
  continuity check.
- Reject only when both mean step exceeds `0.025` and maximum step exceeds
  `0.10`.
- An unavailable quality measurement is non-fatal and remains recorded.
- Rejected c1 now causes the next planned seed to be tried while preserving the
  requested push-in.

Saved c1 correctly fails the new gate. The already-rendered c2 passes with
`mean_similarity_step=0.010300`, so it was recovered with the original audio:

`video_output/out/dry-008/6.DSC07662/final_20260724_111546_wan_c2_qualitypass.mp4`

The recovery preview is `1080x1080`, 30 FPS, 150 frames, and 5.0 seconds.

## Validation

- Host syntax compilation passed with bytecode writes disabled.
- Container unit suite passed: 27 tests.
- The DSC08496 hardcoded semantic compiler override was removed; its square
  tracking treatment remains a human recommendation and manual reference.
- Saved DSC07662 c1 fails and c2 passes the new temporal gate.
- Original weak `.result.json` files are marked rejected with normalized issue
  codes; accepted examples remain exact-candidate records.

## Read-only temporal gate audit

The current `dry-008` artifacts were audited after the gate change without
altering any video:

- Gate-eligible `4.20260519_1400.33` c1 passes:
  mean step `0.021535`, max step `0.091366`.
- Gate-eligible `6.DSC07662` c1 is the known failure:
  mean step `0.029154`, max step `0.123675`.
- Gate-eligible `6.DSC07662` c2 passes:
  mean step `0.010300`, max step `0.127686`. Its isolated maximum exceeds the
  max limit, but the mean does not; the gate intentionally requires both.
- Shadow-audited medium-motion Wan push clips also pass:
  `2.DSC09278` c1 is `0.012419 / 0.068558`, and `7.DSC08260` c1 is
  `0.014317 / 0.085051`.
- c2 raw renders for cases 2, 4, and 7 had already been cleaned up, so they
  could not be remeasured. They were reported as unavailable, not as passes.

No additional saved raw render failed the production thresholds.

## Codex MCP architecture

Fix classification: clean architectural fix for targeted human-guided clips.

The local STDIO server in `services/pipeline_mcp/` exposes only:

- `analyze_case`
- `render_with_overrides`
- `remux_existing_raw`
- `quality_check`
- `record_review`

It deliberately exposes no arbitrary shell command. Paths are restricted to
the project, render inputs to `video_input`, and remux outputs to
`video_output`. This lets Codex turn a human recommendation such as “square,
pan left-to-right following the horse and rider” into an explicit one-off clip
without adding semantic compiler rules. The server is registered in Codex as
`image2video-pipeline`.

## Follow-up batch review: 2026-08-08

Batch: `video_output/out/dry-008/`

The user's exact candidate feedback was persisted in the sibling
`.result.json` records. Accepted references are:

- `2_DSC00574/final_20260807_165716_wan_c2.mp4`
- `4_DSC04057/final_20260807_170854_hunyuan_c1.mp4`, with a model-selection
  concern described below
- `6_DSC09289/final_20260807_173845_wan_c2.mp4`

Presentation-only review artifacts were added without changing generation:

- `1_DSC06876/final_20260808_review_zoom_wan_c1.mp4` is square 1:1 with a
  1.25x push-in focused near `(0.55, 0.55)`.
- `7_DSC07185-Enhanced-NR/final_20260808_review_pan_rtl_wan_c1.mp4` is square
  1:1 with a right-to-left traversal from `0.80 -> 0.10`.

Both are `1080x1080`, 30 FPS, 150 frames, and 5.0 seconds. Contact-sheet
inspection confirms that the first zooms into the city and the second begins
on the source's right side before traversing left.

### Case: tiny non-primary person incorrectly triggers Hunyuan

Symptom:

`4_DSC04057/final_20260807_170854_hunyuan_c1.mp4` is visually good, but the
user correctly questioned why an environmental sunset/beach scene used
Hunyuan when it has no important person.

Exact debug and output paths:

- `video_output/out/dry-008/4_DSC04057/debug_20260807_170854.json`
- `video_output/out/dry-008/4_DSC04057/final_20260807_170854_hunyuan_c1.mp4`
- Replacement:
  `video_output/out/dry-008/4_DSC04057/final_20260808_072615_wan_c1.mp4`
- Replacement debug:
  `video_output/out/dry-008/4_DSC04057/debug_20260808_072615.json`

Key fields and dimensions:

- The decision saved `scene.has_people=true`, selected
  `HUNYUAN15_I2V_720P`, and prompted `person move gently and locally`.
- Image2json observed one tiny distant swimmer silhouette in the water. It is
  context, not a primary subject.
- The Wan replacement raw clip is `768x512`, 20 FPS, 97 frames, 4.85 seconds.
- Its final is `1080x1920`, 30 FPS, 150 frames, 5.0 seconds.

Root cause:

The semantic selection treated the presence of any detected person as enough
to route Hunyuan. This violates the intended rule that Hunyuan is for an
important visible person/fauna/vehicle.

Fix direction and classification:

- Applied output fix: targeted workaround. An explicit `WAN22_NATURAL`
  override retained the successful 9:16, 1.15x sunset push while moving only
  coherent ocean waves and holding the sky, horizon, buildings, and distant
  silhouette stable.
- Clean architectural fix: model routing should use subject prominence and
  importance, not merely non-empty people detection. This broader compiler
  change was not applied in this review turn.

Validation:

The replacement's raw continuity check passed with
`mean_similarity_step=0.006096` and `max_similarity_step=0.093510` against
thresholds `0.025 / 0.10`. Contact-sheet inspection shows stable sunset and
horizon with continuous wave motion.

### Case: localized jumping sky escapes the full-frame continuity gate

Symptom:

Both original `5_IMG_5822` Wan candidates were rejected by the user, mainly
because the clouded sky jumps.

Exact debug and output paths:

- `video_output/out/dry-008/5_IMG_5822/debug_20260807_173202.json`
- `video_output/out/dry-008/5_IMG_5822/final_20260807_173202_wan_c1.mp4`
- `video_output/out/dry-008/5_IMG_5822/final_20260807_173202_wan_c2.mp4`
- Old raw c1:
  `.local/outputs/comfy/dry-008-5_IMG_5822-0-wan_c1-41db1a_00001.mp4`
- Replacement:
  `video_output/out/dry-008/5_IMG_5822/final_20260808_073036_wan_c1.mp4`
- Replacement raw:
  `.local/outputs/comfy/dry-008-5_IMG_5822-0-wan_c1-a900bd_00001.mp4`
- Replacement debug:
  `video_output/out/dry-008/5_IMG_5822/debug_20260808_073036.json`

Key fields and dimensions:

- The original prompt explicitly requested clouds to move visibly and change
  position and edge shape by the end.
- Both old finals are square 1:1 push-ins. The surviving old raw c1 is
  `768x512`, 20 FPS, 97 frames, 4.85 seconds.
- The replacement uses the same raw/final dimensions; the final is
  `1080x1080`, 30 FPS, 150 frames, 5.0 seconds.
- The old raw c1 still passed the full-frame gate at `0.006963 / 0.047629`, so
  the current global SSIM metric does not capture this localized semantic sky
  failure.

Root cause:

The problem is generative cloud evolution, not deterministic square framing or
the push-in. The prompt invited aggressive cloud morphology, while the global
continuity gate diluted a sky-localized defect across the detailed city frame.

Fix direction and classification:

- Applied output fix: targeted workaround. A new Wan seed requests only local
  city-light shimmer and slight outer-branch movement, explicitly holding the
  entire cloud layer and skyline stable. The push endpoint was reduced from
  1.15x to 1.12x.
- Clean architectural follow-up: use region-aware temporal checks for large
  high-risk sky/cloud regions, or compile static-sky motion when prior human
  feedback reports cloud jumping. No broad gate change was applied here.

Validation:

The replacement raw passes with `mean_similarity_step=0.002776` and
`max_similarity_step=0.015287`, substantially below the old raw's values.
A 10-frame contact sheet shows a coherent cloud layer without the prior jump.
