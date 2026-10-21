<p align="center">
  <img src="ui/assets/rudra-mark.png" width="150" alt="RUDRA">
</p>

<h1 align="center">RUDRA</h1>

<p align="center">
  <b>R</b>adiance <b>U</b>niversal <b>D</b>ynamic <b>R</b>ange <b>A</b>dapter<br>
  Turns 8-bit SDR footage into scene-linear HDR, and tells you where it did it.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img src="https://img.shields.io/badge/torch-2.x-ee4c2c">
  <img src="https://img.shields.io/badge/weights-noncommercial-orange">
  <img src="https://github.com/FXTD-Studios/RUDRA/actions/workflows/tests.yml/badge.svg">
  <a href="https://github.com/FXTD-Studios/RUDRA/releases"><img alt="Desktop beta" src="https://img.shields.io/github/v/release/FXTD-Studios/RUDRA?include_prereleases&label=desktop%20beta"></a>
  <a href="https://github.com/FXTD-Studios/RUDRA/actions/workflows/native.yml?query=branch%3Anative"><img alt="Windows build" src="https://img.shields.io/github/check-runs/FXTD-Studios/RUDRA/native?nameFilter=core%2C%20windows-2022&label=Windows&logo=windows"></a>
  <a href="https://github.com/FXTD-Studios/RUDRA/actions/workflows/native.yml?query=branch%3Anative"><img alt="macOS build" src="https://img.shields.io/github/check-runs/FXTD-Studios/RUDRA/native?nameFilter=core%2C%20macos-14&label=macOS&logo=apple"></a>
  <a href="https://github.com/FXTD-Studios/RUDRA/actions/workflows/native.yml?query=branch%3Anative"><img alt="Linux build" src="https://img.shields.io/github/check-runs/FXTD-Studios/RUDRA/native?nameFilter=core%2C%20ubuntu-24.04&label=Linux&logo=linux"></a>
</p>

---

**Contents:**
[What it does](#what-it-does) ·
[What's new](#whats-new) ·
[See it](#see-it) ·
[Results](#results) ·
[Install](#install) ·
[Quick start](#quick-start) ·
[Desktop app](#desktop-app) ·
[The browser Studio](#the-browser-studio) ·
[Command line](#command-line) ·
[Validation diagnostics](#validation-diagnostics) ·
[Known limitations](#known-limitations) ·
[Documentation](#documentation) ·
[Licence](#licence)

---

## What it does

An 8-bit frame throws away everything above the clip. RUDRA puts it back:
it reads the SDR through the inverse of the curve that made it, fills the
highlights the SDR clipped, and leaves the artist in control of every step,
live on an HDR display.

**RUDRA 1.0 ships on the analytic inverse with the artist in the loop**
(re-scoped 7 Oct 2026). No learned model has beaten the inverse on real SDR,
so the model is optional and off by default. What the artist gets:

- **Source**: a picker for the curve that made the SDR (BT.1886, the default
  for new sessions, ACES, filmic/Hable, AgX, camera log, Rec.709 clip),
  three-click calibration, and a match to a graded HDR reference.
- **Grade**: exposure in stops, peak, knee, Region EV bands with painted
  masks, anchor to the SDR's mid-tones, every control under one frame at 4K.
- **Clipped highlights** filled from the SDR alone, with a peak per region
  the artist can set, and an **Invented** layer that marks every pixel RUDRA
  made up.
- **Delivery**: ACES EXR masters, HDR10, HLG, ProRes 422 HQ / 4444 (PQ,
  ACEScct or LogC4) from a movie or a folder of frames, with audio, QC before
  publishing, HDR10+ and Dolby Vision L1/L6 metadata per shot, and holds that
  keep a shot steady over time.
- **Provenance**: every master records the app version, build commit, the
  source file's SHA-256 and, with the model on, the checkpoint's.

It ships as a native desktop app (Windows, macOS, Linux; no Python), the
browser Studio and two command lines (`rudra`, `rudra-native`).

**Scope.** RUDRA is one thing: SDR to HDR for any image or video, whatever made
it, whether a camera, a phone, an archive or any generative model. It works on
pixels, not on a model's latent space. The per-backbone VAE decoders (Flux,
Wan, LTX, SDXL, Qwen, Klein) and the latent-conditioning research pipeline are
**paused** as of 23 Sep 2026. Their code stays in `rudra/` and `training/` and
their results in [`STATUS.md`](STATUS.md), but they are not part of the product
and are not maintained.

---

## What's new

On `native` since `v1.0.0-rc.2`, for the next release candidate. Per-beta
detail: [`docs/BETA.md`](docs/BETA.md).

- **Shot keyframes (roadmap 4.3).** A programme's grade can be kept per shot
  in a `<name>.shots.json`: keys inside a shot blend the continuous settings
  (strength, exposure, knees, Region EV bands, calibration nits) and step the
  rest; nothing reaches across a cut, and the holds reset at every shot.
  `rudra-native video ... --shots FILE` (or a queue's `shots`) exports a whole
  programme in one pass; the report lists each shot with its own flicker.
- **Conform by shot (5.4).** `rudra-native conform <movie>` makes the shots
  file from a CMX3600 EDL (drop frame, dissolves, clip names), an FCPXML's
  primary storyline, or the movie's own cuts (`--detect`).
- **Opt-in correction logging (5.2).** `rudra-native correction-log open | add |
  withdraw | export`: nothing is kept without a recorded consent (who, when,
  terms, rights holder); an accepted grade is kept as a 16-bit SDR / PQ HDR
  pair with its source hash, and `export` writes a training manifest the
  commercial preflight reads. A withdrawal stops logging and every export.
- **Dolby Vision from RUDRA itself (5.5).** `rudra-native dolby-vision` writes
  the per-shot metadata as a profile 8.1, CM v4.0 RPU (byte for byte what
  dovi_tool 2.3.1 generates, on 165 configs; injects into the HDR10 master) and
  as a Dolby Vision metadata XML (CM v4.0) that dovi_tool reads back.
- **OFX plug-in (5.1).** "RUDRA SDR to HDR" (`-DRUDRA_BUILD_OFX=ON`,
  `RUDRA.ofx.bundle`): the analytic reconstruction and the master chain on the
  CPU, Source, exposure, strength, recovery, clip fill and anchor as
  parameters, scene-linear Rec.2020 / ACEScg or PQ out. Loaded and rendered in
  a test host on the OpenFX SDK; not yet run in a commercial host.

- **A folder of frames exports to a movie.** HDR10, HLG, ProRes 422 HQ and
  4444 are offered for frame folders too: the frames are mastered as a linear
  Rec.2020 EXR sequence beside the movie, then encoded and checked as
  `rudra-native deliver` encodes them (measured MaxCLL/MaxFALL, tags verified,
  a frame rate picked on the export sheet). Master EXR stops it.
- **Stills keep their colour and orientation.** ICC profiles (PNG, JPEG, TIFF,
  WebP) and PNG cICP / sRGB / cHRM set the master's source primaries: sRGB /
  Rec.709, Display P3, Rec.2020 and Adobe RGB (1998) are recognised, an
  unknown profile is reported and read as Rec.709, untagged files are
  unchanged. EXIF orientation is applied. A tagged 16-bit PQ reference image
  converts from its own primaries.
- **Master provenance.** App version, build commit, source SHA-256, the
  checkpoint SHA-256 when the model ran, and whether the network really tiled
  (`rudra:tiled` was hard-coded).
- **Source: BT.1886 shadows (opt-in).** BT.1886 from SDR code 40 up and, below
  it, black levels measured on real graded HDR. BT.1886 stays the default.
- **Getting started.** A five-step guide on the first start, any time from
  Help > Getting started.
- **Notes.** [`NOTICE`](NOTICE) credits the Netflix Open Content titles; the
  super-white design and its 1.0 limit
  ([`docs/SUPER_WHITES_DESIGN.md`](docs/SUPER_WHITES_DESIGN.md)); open licensing
  questions in [`docs/LEGAL_QUESTIONS.md`](docs/LEGAL_QUESTIONS.md).

---

## See it

![Degraded input, four methods](docs/compare/hard_deployment.png)

Three held-out frames. Unknown tone curve, 4:2:0 chroma, banding, JPEG: the
condition real footage arrives in. The analytic inverse has a hard ceiling and
clips into it, so the sky goes flat. RUDRA keeps the roll-off the reference has.
HDR columns are stopped down by whole stops so a browser can show them, and the
exposure comes from the reference alone, so nothing is flattered by its own
output.

---

## Results

### Against the market, on real SDR (R3, 12 to 13 Oct 2026)

Real SDR video with its own graded HDR: Netflix's *Meridian* (287 held-out
frames, 1080p) and the HDRTV1K test set (117 frames). Paired against the old
default (the ACES inverse), 95 % bootstrap intervals; PU21-PSNR and CVVDP.

| Method | Meridian, vs the ACES inverse | HDRTV1K, vs the ACES inverse |
|---|---:|---:|
| **RUDRA, Source = BT.1886 (the default)** | **+13.96 dB / +2.28 JOD** | **+4.16 dB / +2.11 JOD** |
| BT.2408 direct mapping | +14.24 dB / +2.29 JOD | +4.31 dB / +2.04 JOD |
| RUDRA, Source = Rec.709 | +1.70 dB / +1.33 JOD | +0.89 dB / +0.59 JOD |
| RUDRA + model v8 | level (not significant) | not run |
| HDRTVDM (CVPR 2023) | −1.15 dB (n.s.) / −1.38 JOD | not run |
| RUDRA, ACES inverse (the old default) | 0 (21.65 dB, 6.443 JOD) | 0 |

BT.1886 is level with the best open mapping on both sets and is the default
for new sessions; CG and games made with a known tone curve still want ACES,
filmic or AgX in the picker. The commercial tools (Resolve, Topaz, RTX Video
HDR) are run by hand before 1.0 and published whatever they say.

### The model's bench (rendered SDR)

429 held-out frames, native 1280×720, scene-linear with diffuse white at 1.0.
Whole scenes are held out, so no scene appears on both sides of a split.

| Condition | Method | PU21-PSNR | CVVDP (JOD) |
|---|---|---:|---:|
| **hard** | analytic baseline | 25.92 | 7.362 |
| **hard** | **RUDRA + shadow gate** | **27.15** | **7.751** |
| clean | analytic baseline | 45.99 | 9.448 |
| clean | **RUDRA + shadow gate** | **46.06** | **9.561** |

`hard` is degraded input (4:2:0, banding, JPEG), `clean` well-graded input.
These frames were rendered with ACES, so they favour the ACES inverse by
construction; on real SDR (above) no model beats the inverse, which is why
the model is optional.

> **Read both metrics together.** The ungated model gains +1.43 dB on degraded
> input and *loses* 3.0 dB on clean input, where CVVDP puts the same gap at
> −0.046 JOD, far below a just-noticeable difference. Quoting the gain
> without the loss, or the clean PSNR without the JOD beside it, misrepresents
> the result in opposite directions.

Full tables, the failure analysis, and how to recompute every number:
[`docs/RESULTS.md`](docs/RESULTS.md); flicker on real clips:
[`docs/TEMPORAL_RESULTS.md`](docs/TEMPORAL_RESULTS.md).

---

## Install

**Desktop app (no Python).** From
[Releases](https://github.com/FXTD-Studios/RUDRA/releases), the latest
pre-release (`v1.0.0-rc.2` now, then 1.0): for Windows x64 `RUDRA-<version>-windows-x64-setup.exe`, which
installs for your user (no administrator prompt) with the model and the Visual
C++ runtime included, or the portable ZIP beside it; for Apple silicon
`RUDRA-<version>-macos-arm64.dmg`. Details: [`docs/BETA.md`](docs/BETA.md).

**Python package.** Requires Python 3.10 to 3.13. CUDA is optional: everything runs on CPU, slower.
FFmpeg is needed for video, not for stills.

```bash
git clone https://github.com/FXTD-Studios/RUDRA.git
cd RUDRA
pip install -e .
(cd checkpoints && sha256sum -c SHA256SUMS)
```

---

## Quick start

**Desktop app.** Install it and start RUDRA. The first start opens **Welcome
to RUDRA**: it checks your display (HDR or not, and its real peak) and,
optionally, the model on your GPU or CPU; the model is not needed. Then
**Getting started** shows the five steps once (Help > Getting started any
time):

1. **Source**: open a still, a folder of frames or a movie; Reconstruct >
   Source says how the SDR was made (BT.1886, the default, for camera and
   broadcast video; ACES, filmic or AgX for CG and games).
2. **Calibrate** (optional): patches of known brightness, or a graded HDR
   reference.
3. **Reconstruct**: exposure in stops, Fill clipped highlights (on), the
   model if you want it (off by default).
4. **Check**: B for the baseline, W to wipe, the view peak, False colour,
   the scopes, the Invented layer.
5. **Deliver**: ACES EXR masters, or HDR10, HLG and ProRes movies with their
   HDR10+ and Dolby Vision metadata.

The whole guide: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) (F1 in the app).

**Browser Studio (Python).** It loads the shipped checkpoint and opens a browser tab:

```bash
python ui/server.py
```

Convert a complete SDR clip to HDR10, keeping its audio:

```bash
rudra video input.mp4 --output delivery/master_hdr10.mp4 \
    --checkpoint checkpoints/sdr2hdr_shadow_v1.pt --device cuda
```

Reconstruct a frame or a folder of frames:

```bash
python training/infer_sdr2hdr.py input/ --output-dir out/ \
    --image-checkpoint checkpoints/sdr2hdr_shadow_v1.pt --recovery-mode all
```

---

## Desktop app

A native RUDRA for Windows, Linux and macOS, with no Python at runtime: Qt 6
for the interface, a QRhi viewer on each OS's native GPU API (Direct3D 12,
Metal, Vulkan, OpenGL fallback) with HDR output, a C++20 core, and inference
through LibTorch (CUDA, MPS) and ONNX Runtime (DirectML, Core ML, ROCm,
OpenVINO). It lives in [`native/`](native/README.md) on the `native` branch.
The browser Studio and the Python CLI stay as they are and remain the
reference every native module is tested against.

In it today (`v1.0.0-rc.2` and the changes since, [`docs/BETA.md`](docs/BETA.md)):
the Studio's layout, actions, undo and projects; the HDR viewer with scopes,
wipe, Compare, the Changes tint and the Invented layer; Source, calibration,
reference match, exposure, peak, knee, Region EV and painted masks, live on
the GPU and held to the C++ core by `rudra-gpu-parity` on four backends;
stills with their colour profile and orientation, sequences, EXR masters,
movie exports from movies and frame folders with QC, HDR10+ and Dolby
Vision metadata, a resumable queue; the shot and static holds in playback and
export; the optional model checked against its reference frames on every
device; installers for Windows (setup.exe, portable ZIP) and macOS (DMG);
CI green on all three OSes.

Plan: [`docs/DESKTOP_APP_PLAN.md`](docs/DESKTOP_APP_PLAN.md) · design:
[`docs/NATIVE_ARCHITECTURE.md`](docs/NATIVE_ARCHITECTURE.md) · build:
[`native/README.md`](native/README.md).

---

## The browser Studio

![RUDRA Studio](docs/rudra_studio.png)

Drop a frame or a shot. The network runs once per frame on the GPU and hands
the page raw fields. Everything after that composes on your own GPU, so the
controls move at frame rate.

| Layer | What it shows |
|---|---|
| **Compare** | RUDRA, the analytic baseline, or a draggable wipe between them |
| **False colour** | luminance zones in nits, against a diffuse white of 203 |
| **Difference** | how far RUDRA moved from the baseline; black means it changed nothing there |
| **Probe** | one pixel: baseline, RUDRA, the delta in stops, and whether the SDR clipped there at all |
| **Scopes** | waveform, RGB histogram and vectorscope, all in nits on a log axis, computed from the frame's own pixels |
| **Frame** | what the frame contains: MaxCLL, MaxFALL, the share above 1 000 nits, and the share the SDR actually clipped |

The bar along the bottom is the colour pipeline, and it is always on: what the
input is being read as, the working space, what the viewer is doing to the
picture, and what the master will be written as. It warns when the frame
carries pixels above the peak the viewer is showing, because an SDR monitor
clipping a highlight looks exactly like a highlight that was never there.

The surround is a neutral grey on purpose, because a tinted one biases the
judgement of the picture inside it.

Playback runs at the footage's frame rate, with a read-ahead in front of the
playhead, and the transport reads timecode.

### Mastering from the browser Studio

**Master EXR** writes OpenEXR in ACES 2065-1 or linear Rec.2020 directly to an
absolute **Render folder** on the computer running the Studio. Choose
**Current image** or **All loaded frames (sequence)**, then set the render name
and starting frame. A sequence named `shot` starting at 1001 writes
`shot.001001.exr`, `shot.001002.exr` and matching JSON sidecars, using the
loaded frame order and one frozen copy of the current grade.

- Master keeps the source resolution and does not trigger a browser download.
- Flat highlights are settled: above the anchor's knee the expansion curve is
  steep enough to turn the source's one-code grain into tens of nits, so where
  the source is flat the master's luminance is smoothed (hue, edges and glints
  untouched). On a sunset plate, flat sky at source luma 0.95 to 0.98 went from
  17.0 to 1.8 nits of grain; `tools/measure_highlight_grain.py` measures any
  plate. `settle_grain: false` in the master parameters turns it off.
- Existing outputs stop the render before processing; files are never overwritten.
- Keep the Studio and the browser open until the render completes.
- If a sequence stops, completed frames remain on disk. Choose the remaining
  inputs and the matching start number to continue, or use a fresh render folder.

---

## Command line

Two command lines: `rudra` (Python: video, batch queues, stills) and
`rudra-native` (the desktop app's engine, no Python: `diff`, `video`,
`deliver`, `batch`, `ffmpeg-check`).

### Video delivery

`rudra video` converts a complete SDR clip. Run `python -m rudra.video` with the
same arguments if the CLI is not installed. It requires FFmpeg/ffprobe with
`libx265` and `zscale`; ProRes also needs `prores_ks`.

#### Presets

Select a preset with `--format`:

| Preset | Output | Signal | Alpha |
|---|---|---|---|
| `hdr10` (default) | MP4/MOV/MKV, HEVC 10-bit | BT.2020 PQ, static HDR10 metadata | No |
| `hlg` | MP4/MOV/MKV, HEVC 10-bit | BT.2020 HLG | No |
| `prores422` | MOV, ProRes 422 | BT.2020 PQ | No |
| `prores422hq` | MOV, ProRes 422 HQ | BT.2020 PQ | No |
| `prores4444` | MOV, ProRes 4444 | BT.2020 PQ | Straight alpha |

```bash
rudra video input.mp4 --output delivery/broadcast_hlg.mp4 --format hlg \
    --checkpoint checkpoints/sdr2hdr_shadow_v1.pt --device cuda
rudra video input.mp4 --output delivery/editorial.mov --format prores422hq \
    --checkpoint checkpoints/sdr2hdr_shadow_v1.pt --device cuda
rudra video transparent.mov --output delivery/composite.mov --format prores4444 \
    --alpha-mode straight --checkpoint checkpoints/sdr2hdr_shadow_v1.pt --device cuda
```

#### Input

Supported: progressive, square-pixel, constant-frame-rate SDR video with even
dimensions. The rational frame rate and frame count are preserved, video start
time is normalized to zero, and audio stays aligned to the video. Audio outside
the video interval is trimmed; all input audio streams are copied by default.
Use `--audio aac` when the input audio codec cannot be copied into MP4.

Colour tags determine the input transfer, primaries, YUV matrix and range.
Missing or unsupported tags require explicit overrides, for example
`--input-transfer rec709 --input-primaries rec709 --input-matrix bt709 --input-range limited`.
Only use those overrides when they describe the source.

Rejected rather than silently changed: HDR, alpha-bearing, interlaced, rotated,
anamorphic and variable-frame-rate inputs. The exception is straight alpha with
the ProRes 4444 preset. No preset carries subtitles.

#### Output and QC

HDR10 is 10-bit HEVC, BT.2020/PQ, with measured MaxCLL/MaxFALL and mastering
display metadata; `--peak-nits 1000` selects the mastering peak. Every preset is
published only after checking dimensions, every frame timestamp, frame count,
colour tags, HDR metadata, audio alignment and a complete decode. A matching
`.json` sidecar records the checkpoint hash, settings, per-frame statistics and
QC results. Existing outputs are never overwritten.

#### HLG

HLG uses BT.2100's inverse OOTF followed by its OETF, with zero reference black,
the selected peak and the corresponding system gamma (1.2 at 1000 nits). It
does not merely relabel PQ pixels. Saturated values outside legal HLG scene RGB
receive a common RGB gain reduction. HLG has no HDR10 static SEI.

#### ProRes and alpha

ProRes stores PQ colour tags, while the mastering and content-light analysis
remains in the sidecar. ProRes MOV's `nclc` atom may omit a separate range flag;
conversion uses limited video range.

`prores_ks` accepts 10-bit colour and alpha input planes, so a 4444 stream
decoding to 12-bit colour or configured for 16-bit alpha storage does not
restore precision lost at its input. Alpha bypasses reconstruction and grading.
Every decoded output alpha pixel is checked against the input with a tolerance
of 128/65535 (two 10-bit steps), so arbitrary 16-bit alpha is **not lossless**.
Premultiplied sources must first be unpremultiplied; `--alpha-mode straight`
declares the supplied interpretation.

#### Performance and options

- CPU is the default device; select `--device cuda` explicitly when available.
- Full source dimensions are kept, using 512-pixel tiles by default;
  `--tile-size 0` runs untiled when memory permits.
- Temporary 16-bit PNGs are spooled to disk rather than keeping a whole clip in
  RAM. Use `--work-dir` to select a disk with space.
- A single `rudra video` job is not resumable; use a [batch queue](#batch-queues)
  for per-clip resume.
- `--shadow-smoothing 0.8` smooths the scalar shadow gate and resets it at
  detected hard cuts. It does not blend image pixels and is not a validated
  temporal model. Off by default.

### Batch queues

Save a queue as `queue.json`. Paths resolve relative to that file. Defaults and
per-job `options` accept the same option names as `rudra video` (underscores or
hyphens), without the leading `--`.

```json
{
  "version": 1,
  "defaults": {
    "checkpoint": "checkpoints/sdr2hdr_shadow_v1.pt",
    "device": "cpu",
    "format": "hdr10"
  },
  "jobs": [
    {"input": "clips/shot01.mp4", "output": "masters/shot01.mp4"},
    {"input": "clips/shot02.mp4", "output": "masters/shot02.mov",
     "options": {"format": "prores422hq"}}
  ]
}
```

```console
rudra batch run queue.json
rudra batch status queue.json
rudra batch run queue.json --retry-failed
```

- A job whose options carry `grade` (the export settings file the app's queues
  name) runs through `rudra-native video`, the native app's command line, found
  on PATH or through `RUDRA_NATIVE`; the rest of the job is as above.
- Jobs run sequentially. Progress and errors are saved atomically in
  `queue.json.state.json`; a process lock prevents two runners using the same queue.
- Repeating `run` verifies the SHA-256 hashes of completed video/report pairs,
  sources and weights before skipping them.
- Failed jobs remain visible and require `--retry-failed`; other jobs continue.
  The command returns nonzero if any job is incomplete.
- Keep the queue unchanged after starting it; use a new filename for a revised
  queue, and separate output paths across different queues.
- Resume is **per clip**: interrupted clips restart from frame one. An existing
  output or report is never overwritten.
- If a crash occurs during final publication or before completion is saved,
  review and relocate that job's output/report pair before retrying. Temporary
  folders may remain after a hard process termination.

### Stills and sequences

```bash
python training/infer_sdr2hdr.py input/ --output-dir out/ \
    --image-checkpoint checkpoints/sdr2hdr_shadow_v1.pt --recovery-mode all
```

Inference writes float EXR delivery masters at 203 nits per stored unit. TIFF
outputs keep the network's separate 10,000-nit convention. For a nested input
sequence, pass the corresponding output shot directory to delivery. No GPU is
required.

---

## Validation diagnostics

### Quality benchmark

Run a fixed, scene-balanced sample without consuming the final test set:

```console
python training/quality_benchmark.py --manifest outputs/finetune_views_20260920/data/manifest.jsonl --checkpoint checkpoints/sdr2hdr_shadow_v1.pt --out outputs/quality_diagnostic_new --scenes 12
```

The output directory must be new. The tool freezes source and checkpoint hashes,
scores the shipped model against the analytic inverse-ACES baseline at native
resolution on CPU, and writes `REPORT.md`, `summary.json` and `scores.jsonl`.
It selects one frame per validation scene by a stable hash, then tests clean and
seeded degraded inputs. It reports PU21-PSNR, real ColorVideoVDP image JOD, and
shadow/highlight region errors, with paired bootstrap intervals. Missing metrics
remain unavailable; proxy values never enter the comparison.

This is an image-quality diagnostic, not a motion benchmark or an external comparison.
Candidate training assessment and final held-out testing remain separate. Do not
read a small validation sample as proof of general superiority.

### Recovery ablation

Isolate the residual recovery paths on the exact same frozen sample:

```console
python training/recovery_ablation.py --benchmark outputs/quality_benchmark_20260920 --out outputs/recovery_ablation_new
```

This verifies the original source, checkpoint and implementation hashes, reuses
the original baseline and all-recovery scores, and measures highlights-only,
shadows-only and recovery-off with real ColorVideoVDP. The new output directory
holds paired comparisons and an ablation report. No inference defaults or
training settings are changed; confirm findings on broader validation before
promoting a different recovery policy.

---

## Known limitations

- **Super-whites are clipped.** In limited-range video, detail recorded above
  white (code 235, or 940 at 10 bits) is clipped when the movie is read, and
  the clip fill then rebuilds those highlights instead of using them. A mode
  that keeps them for BT.1886 movies is built and in testing, not yet in the
  app. Until then, convert such footage to full range with the super-whites
  kept (in Resolve, data levels set to full) before opening it in RUDRA.
- **The clip fill at +0 EV and on fireworks.** It overshoots highlights that
  sit just above the clip, and loses 4.9 to 7.7 dB on the four fireworks clips;
  turn it down (Strength) or off (Reconstruct > Fill clipped highlights).
- **Unknown ICC profiles** are reported and read as Rec.709; RUDRA matches the
  four profiles above, it is not a general colour-management system.
- **The model is optional.** None beats the analytic inverse on real SDR yet.
- **Non-commercial.** Commercial clearance of sources, weights and formats is
  open ([`docs/LEGAL_QUESTIONS.md`](docs/LEGAL_QUESTIONS.md)).

---

## Documentation

| Document | What is in it |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | every benchmark table, and how to recompute it |
| [`docs/RELEASE_V1.md`](docs/RELEASE_V1.md) | what is left before 1.0, and the steps to ship it |
| [`docs/BETA.md`](docs/BETA.md) | the desktop beta: what is in it, the changes per beta, install |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | using the desktop app, control by control |
| [`docs/TEMPORAL_RESULTS.md`](docs/TEMPORAL_RESULTS.md) | flicker and clip-mode CVVDP on real footage |
| [`docs/TRAINING.md`](docs/TRAINING.md) | training on your own footage, end to end |
| [`docs/TRAINING_STEPS.md`](docs/TRAINING_STEPS.md) | the next training run, step by step, with the gate each step has to pass |
| [`docs/RETRAIN_RUNBOOK.md`](docs/RETRAIN_RUNBOOK.md) | rebuilding the corpus: sources, licences, the ingest, the gates |
| [`docs/INTERNALS.md`](docs/INTERNALS.md) | the composite, the units, the gate |
| [`docs/CORPUS.md`](docs/CORPUS.md) | what a training set has to contain |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | repo layout, how it is checked, and the decisions behind it |
| [`docs/DESKTOP_APP_PLAN.md`](docs/DESKTOP_APP_PLAN.md) | the native desktop Studio: architecture, phases, acceptance |
| [`docs/NATIVE_ARCHITECTURE.md`](docs/NATIVE_ARCHITECTURE.md) | the native app design: layers, types, threading, patterns, numerics |
| [`docs/composite.spec.md`](docs/composite.spec.md) | the composite, master chain and measurements as a spec: every stage, precision and tolerance |
| [`native/README.md`](native/README.md) | building the native app, the model package, what is done and what is not |
| [`STATUS.md`](STATUS.md) | what is finished, what is open, and the next steps in order |

---

## Licence

**RUDRA is non-commercial.** The code is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE): research, teaching,
evaluation, personal projects and noncommercial organizations may use,
change and share it. **The weights are non-commercial too**, under their own
terms, because of their training corpus; that is a term FXTD Studios cannot
waive for you. See [`checkpoints/LICENSE`](checkpoints/LICENSE) and
[`NOTICE`](NOTICE).

Commercial use of the code or the weights needs a licence from
[FXTD Studios](https://fxtdstudios.com). Copies of the code obtained before
24 September 2026 were released under Apache 2.0 and keep that licence.

---

<p align="center"><a href="https://fxtdstudios.com">FXTD Studios</a> · Cairo</p>
