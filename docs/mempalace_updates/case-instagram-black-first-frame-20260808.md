# Case: Instagram grid cover renders black first frame

Date: 2026-08-08

## Symptom

Instagram displayed black grid thumbnails for uploaded `dry-008` final MP4s even
though local playback began on the source image.

## Evidence

- The decoded frame at video time 0 was not black in any selected result. First-frame
  `signalstats` YAVG values ranged from 63.75 to 138.9.
- The original H.264 streams reported `has_b_frames=2`.
- Their first video packet had PTS `0.000000`, DTS `-0.066667`, and was a keyframe.
- Therefore the visible frame was valid, but the MP4 contained negative-DTS decode
  preroll from B-frame reordering. Instagram's thumbnail decoder handling this preroll
  as black is the compatibility root-cause inference; it cannot be directly proven
  from the local pipeline.
- Re-uploading the first `_instagram_safe.mp4` workaround still produced a black
  Instagram Web edit canvas and black thumbnails across the full trim timeline,
  although playback worked and a separately uploaded cover image was visible.
- The first workaround still contained `edts/elst` edit lists on both tracks. Its
  AAC stream was 96 kHz, its first physical AAC packet was negative, and the video
  stream described 150 frames as 4.9667 seconds (`avg_frame_rate=4500/149`, about
  30.20 fps). These remaining inconsistencies are the stronger explanation for the
  web editor's thumbnail failure.

## Affected selected outputs and dimensions

- `video_output/out/dry-008/1_DSC06876/final_20260808_103401_hunyuan_c1.mp4`: 1080x1080
- `video_output/out/dry-008/2_DSC00574/final_20260808_085704_hunyuan_clouds_c1_4x5_both_flowers.mp4`: 1080x1350
- `video_output/out/dry-008/4_DSC04057/final_20260807_170854_hunyuan_c1.mp4`: 1080x1920
- `video_output/out/dry-008/5_IMG_5822/final_20260808_080726_hunyuan_c1.mp4`: 1080x1080
- `video_output/out/dry-008/6_DSC09289/final_20260807_173845_wan_c2.mp4`: 1080x1080
- `video_output/out/dry-008/7_DSC07185-Enhanced-NR/final_20260808_review_pan_rtl_wan_c1.mp4`: 1080x1080

## Fix

The initial no-B-frame `_instagram_safe.mp4` change was a targeted workaround and
proved insufficient in Instagram Web.

V2 classification: safe general compatibility fix.

`services/orchestrator/mux.py` now exports H.264 Main 4.1 with no B-frames, exact
constant frame rate, a one-second keyframe interval, a fixed integer video track
timescale, AAC-LC stereo at 44.1 kHz, `mp42` branding, and no MP4 edit lists. The AAC
filter compensates for its 1024-sample encoder delay before muxing so both physical
packet timelines begin at zero without relying on `elst`. Original accepted renders
were preserved.

Existing selected results received sibling `_instagram_safe.mp4` re-encodes and
matching `_instagram_cover.jpg` images extracted from exact frame 0. After that
workaround failed, all six received `_instagram_web_v2.mp4` siblings.

## Validation

- All six safe MP4s report `has_b_frames=0` and `start_time=0.000000`.
- Every safe MP4 first packet is a keyframe with PTS/DTS `0.000000/0.000000`.
- Frame-0 YAVG values: 138.888, 126.301, 132.996, 63.7771, 111.836, 77.2559.
- All six V2 MP4s have 150 video frames, `r_frame_rate=avg_frame_rate=30/1`, exact
  video duration `5.000000`, H.264 Main, `has_b_frames=0`, and first video packet
  PTS/DTS/duration `0.000000/0.000000/0.033333` marked as a keyframe.
- Every V2 audio stream is AAC-LC, 44.1 kHz stereo, with `start_time=0.000000`.
- MP4 trace finds zero `edts` or `elst` boxes in every V2 file.
- V2 frame-0 YAVG values: 138.933, 126.33, 132.999, 63.7845, 111.892, 77.2751.
- `tests/test_mux.py`: 1 passing test.
- `tests/test_semantic_pipeline.py`: 29 passing tests in `pipeline-orchestrator`.

Final confirmation requires re-uploading an `_instagram_web_v2.mp4`, because an
already published Instagram post cannot be repaired by changing the local file.
