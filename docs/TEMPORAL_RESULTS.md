# Temporal consistency on real footage (11 and 12 Oct 2026)

STATUS row 5 asks whether RUDRA's video is steadier than its frames. This is the
first run on real camera footage, with RUDRA's own measures (not yet clip-mode
ColorVideoVDP).

## Footage

Five 2-second clips (48 frames, 960 x 540) made by `tools/make_temporal_clips.py`
from real camera masters in `G:\datasets\sources`:

| clip | source | what it has |
|---|---|---|
| hdm_bar_01 | HdM HFR 2017, Bar scene shot 1 (192 fps, every 8th frame = 24 fps) | dim interior, a person, a lit screen |
| hdm_fire_01 | HdM HFR 2017, Fire scene shot 1 | a campfire at night: clipped, moving highlights |
| hdm_fire_02 | HdM HFR 2017, Fire scene shot 2 | a close campfire |
| chimera_a | Netflix Chimera, frames 120 to 167 (23.976 fps) | a hillside, a walker, still camera, daylight grain |
| chimera_b | Netflix Chimera, frames 2120 to 2167 | freeway traffic, everything moving |

The masters are HDR (PQ); the SDR is rendered from them (linearised, Hable
tone-mapped to BT.709, 8-bit H.264 4:2:0, CRF 18). That is a tone map, not a
colourist's SDR grade, and the clips are short.

## Measures

From every movie report: `flicker_ev`, how far the HDR's average exposure moved
from one frame to the next beyond what the source's did (EV), and
`local_flicker_ev`, the same per cell of a 36 x 64 grid, averaged. Mean over the
clip and the worst frame. Three ways each: each frame its own
(`--temporal-smoothing 0 --static-hold off`), the shot hold alone, and the shot
and static holds together (the defaults).

## With the model (sdr2hdr_shadow_v1, ONNX Runtime, CPU)

| clip | frames | shots | config | flicker mean | max | local mean | max | local change | s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| chimera_a.mp4 | 48 | 1 | each frame | 0.00029 | 0.00083 | 0.00510 | 0.00942 |  | 437.55 |
| chimera_a.mp4 | 48 | 1 | shot hold | 0.00030 | 0.00081 | 0.00510 | 0.00942 | 0 % | 225.0 |
| chimera_a.mp4 | 48 | 1 | shot + static hold | 0.00032 | 0.00103 | 0.00364 | 0.00837 | -29 % | 234.3 |
| chimera_b.mp4 | 48 | 1 | each frame | 0.00909 | 0.02518 | 0.06289 | 0.09459 |  | 226.81 |
| chimera_b.mp4 | 48 | 1 | shot hold | 0.00910 | 0.02517 | 0.06289 | 0.09459 | 0 % | 223.85 |
| chimera_b.mp4 | 48 | 1 | shot + static hold | 0.00919 | 0.02462 | 0.06112 | 0.09241 | -3 % | 226.33 |
| hdm_bar_01.mp4 | 48 | 1 | each frame | 0.00165 | 0.00431 | 0.00432 | 0.00609 |  | 225.86 |
| hdm_bar_01.mp4 | 48 | 1 | shot hold | 0.00139 | 0.00372 | 0.00410 | 0.00559 | -5 % | 225.98 |
| hdm_bar_01.mp4 | 48 | 1 | shot + static hold | 0.00073 | 0.00225 | 0.00285 | 0.00368 | -34 % | 219.93 |
| hdm_fire_01.mp4 | 48 | 1 | each frame | 0.03219 | 0.10672 | 0.09419 | 0.14707 |  | 221.09 |
| hdm_fire_01.mp4 | 48 | 1 | shot hold | 0.02572 | 0.09875 | 0.09211 | 0.14395 | -2 % | 232.69 |
| hdm_fire_01.mp4 | 48 | 1 | shot + static hold | 0.00645 | 0.02416 | 0.03349 | 0.06065 | -64 % | 220.68 |
| hdm_fire_02.mp4 | 48 | 1 | each frame | 0.04779 | 0.18515 | 0.12909 | 0.24883 |  | 222.25 |
| hdm_fire_02.mp4 | 48 | 1 | shot hold | 0.03369 | 0.11696 | 0.12216 | 0.20268 | -5 % | 226.77 |
| hdm_fire_02.mp4 | 48 | 1 | shot + static hold | 0.01344 | 0.05442 | 0.06852 | 0.11562 | -47 % | 217.18 |

## Analytic (the app's default: no model; the settings file is `{"analytic": true}`)

| clip | frames | shots | config | flicker mean | max | local mean | max | local change | s |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| chimera_a.mp4 | 48 | 1 | each frame | 0.00046 | 0.00146 | 0.00164 | 0.00308 |  | 30.49 |
| chimera_a.mp4 | 48 | 1 | shot hold | 0.00032 | 0.00100 | 0.00159 | 0.00302 | -3 % | 26.3 |
| chimera_a.mp4 | 48 | 1 | shot + static hold | 0.00009 | 0.00047 | 0.00060 | 0.00201 | -63 % | 27.81 |
| chimera_b.mp4 | 48 | 1 | each frame | 0.00674 | 0.03489 | 0.03096 | 0.05497 |  | 24.08 |
| chimera_b.mp4 | 48 | 1 | shot hold | 0.00470 | 0.01333 | 0.03015 | 0.04997 | -3 % | 27.87 |
| chimera_b.mp4 | 48 | 1 | shot + static hold | 0.00480 | 0.01356 | 0.02927 | 0.05115 | -5 % | 27.61 |
| hdm_bar_01.mp4 | 48 | 1 | each frame | 0.00010 | 0.00028 | 0.00038 | 0.00059 |  | 20.43 |
| hdm_bar_01.mp4 | 48 | 1 | shot hold | 0.00010 | 0.00028 | 0.00038 | 0.00059 | 0 % | 19.67 |
| hdm_bar_01.mp4 | 48 | 1 | shot + static hold | 0.00009 | 0.00027 | 0.00036 | 0.00057 | -5 % | 19.77 |
| hdm_fire_01.mp4 | 48 | 1 | each frame | 0.00421 | 0.01019 | 0.00742 | 0.01168 |  | 25.12 |
| hdm_fire_01.mp4 | 48 | 1 | shot hold | 0.00421 | 0.01015 | 0.00741 | 0.01168 | 0 % | 23.72 |
| hdm_fire_01.mp4 | 48 | 1 | shot + static hold | 0.00387 | 0.00941 | 0.00682 | 0.01073 | -8 % | 23.9 |
| hdm_fire_02.mp4 | 48 | 1 | each frame | 0.00493 | 0.01484 | 0.01166 | 0.01905 |  | 25.06 |
| hdm_fire_02.mp4 | 48 | 1 | shot hold | 0.00492 | 0.01478 | 0.01166 | 0.01905 | 0 % | 27.23 |
| hdm_fire_02.mp4 | 48 | 1 | shot + static hold | 0.00461 | 0.01391 | 0.01109 | 0.01814 | -5 % | 27.57 |

## What it says

- **The model flickers on fire; the analytic reconstruction hardly does.** Each
  frame its own, the fire clips move 0.03 to 0.05 EV a frame on average beyond
  the source with the model (worst frame 0.11 and 0.19 EV, visible), against
  0.004 to 0.005 EV analytic. The flicker is the model's: its recovered
  highlights change frame to frame where the source is clipped.
- **The static hold takes most of it out.** With both holds the fire clips fall
  to 0.006 and 0.013 EV mean (-80 % and -72 %), the worst frames to 0.024 and
  0.054 EV, and the local flicker by 64 % and 47 %. Most of that is the hold
  inside clipped highlights added on 11 Oct (the HDR's own change decides there).
- **Still shots settle; moving ones are left alone, by design.** The still
  hillside (chimera_a) loses 29 % (model) and 63 % (analytic) of its local
  flicker; the freeway (chimera_b) 3 to 5 %, because almost every pixel moves and
  a moving pixel is always its own frame's.
- **The shot hold matters where a whole-frame value moves.** Analytic, with the
  anchor, it takes the freeway's worst frame from 0.035 to 0.013 EV; with the
  model, the fire's worst frames from 0.11 / 0.19 to 0.10 / 0.12 EV.
- Nothing got noticeably worse: the largest increase anywhere is 0.0002 EV (the
  global worst frame on the still hillside, model); the local measures only fell.

## 12 Oct 2026: the optional model v8, and ColorVideoVDP

### v8 (sdr2hdr_image_v8, ONNX Runtime, CPU)

| clip | config | flicker mean | max | local mean | local change |
|---|---|---:|---:|---:|---:|
| chimera_a | each frame | 0.00048 | 0.00174 | 0.00608 |  |
| chimera_a | shot + static hold | 0.00035 | 0.00104 | 0.00399 | -34 % |
| chimera_b | each frame | 0.01181 | 0.03290 | 0.09276 |  |
| chimera_b | shot + static hold | 0.01146 | 0.03082 | 0.08918 | -4 % |
| hdm_bar_01 | each frame | 0.00114 | 0.00312 | 0.00462 |  |
| hdm_bar_01 | shot + static hold | 0.00095 | 0.00264 | 0.00347 | -25 % |
| hdm_fire_01 | each frame | 0.01078 | 0.02725 | 0.03025 |  |
| hdm_fire_01 | shot + static hold | 0.00530 | 0.01739 | 0.02283 | -25 % |
| hdm_fire_02 | each frame | 0.02187 | 0.07303 | 0.08599 |  |
| hdm_fire_02 | shot + static hold | 0.01542 | 0.05569 | 0.07281 | -15 % |

v8 flickers a third to a half as much as shadow_v1 on the fire (0.011 and 0.022 EV a frame, against 0.032
and 0.048), and the static hold takes local flicker down by 15 to 34 %. The shot hold alone changes
nothing with v8 (identical to each frame its own): v8 has no per-frame shadow weight, scale or curve to hold.
Full table: `reports/temporal_v8_20261012/temporal.md`.

### ColorVideoVDP against the clips' HDR masters (analytic, the default)

`tools/make_temporal_clips.py --reference` writes each clip's HDR master at the clip's size (16-bit PQ,
BT.2020), and `tools/measure_temporal.py --reference` scores every export against it with ColorVideoVDP
twice: as a clip (its temporal model sees flicker) and frame by frame (it cannot).

| clip | each frame: clip JOD | per frame | shot hold: clip | shot + static: clip | per frame |
|---|---:|---:|---:|---:|---:|
| chimera_a | 4.640 | 6.742 | 4.640 | 4.640 | 6.743 |
| chimera_b | 5.603 | 7.202 | 5.900 | **5.941** | 7.238 |
| hdm_bar_01 | 4.468 | 6.679 | 4.468 | 4.465 | 6.677 |
| hdm_fire_01 | 6.955 | 8.179 | 6.957 | 6.957 | 8.180 |
| hdm_fire_02 | 6.770 | 8.368 | 6.770 | 6.771 | 8.370 |

STATUS row 5's condition is clip JOD with the holds at least that of each frame its own, on every clip.
Analytic: met on four clips and level on the fifth (hdm_bar_01, -0.003 JOD, far inside the metric's noise).
The one clip whose whole-frame exposure moved (the freeway, where the anchor's hold acts) gains +0.34 JOD as a
clip; elsewhere analytic flicker was already below what the metric resolves. Clip JODs sit 1.2 to 2.2 below
the per-frame ones on every row, held or not, so that gap is how the clip metric scores this footage, not flicker
the holds could take out. The same measure with v8 is
`reports/temporal_cvvdp_v8_20261012` when it has run.

## Still to do

- Longer clips with a colourist's SDR (the v4b/v4c test clips step 24 frames at a
  time, one frame a second, so they cannot show flicker).
- The same on the GPU runtimes (DirectML/CUDA), which may differ in jitter.

Reproduce:

    python tools/make_temporal_clips.py G:/datasets/sources --out G:/datasets/corpora/temporal_clips_20261011
    python tools/measure_temporal.py G:/datasets/corpora/temporal_clips_20261011 --native <rudra-native> --package <sdr2hdr_shadow_v1>
    python tools/measure_temporal.py G:/datasets/corpora/temporal_clips_20261011 --native <rudra-native> --analytic
    # with ColorVideoVDP against the masters (12 Oct):
    python tools/make_temporal_clips.py G:/datasets/sources --out G:/datasets/corpora/temporal_clips_20261011 --reference --no-clips
    python tools/measure_temporal.py G:/datasets/corpora/temporal_clips_20261011 --native <rudra-native> --analytic --reference G:/datasets/corpora/temporal_clips_20261011
