# RUDRA 0.9.0 beta 4

The desktop RUDRA: SDR footage in, scene-linear HDR out, with the places the
model reconstructed shown to you. This is the fourth public beta of the native
app (C++20, Qt 6.8, no Python at runtime). It is for evaluation and
non-commercial use (PolyForm Noncommercial 1.0.0, see LICENSE; the model
weights: LICENSE-weights).

## What is in it

- The Studio in a native window: Reconstruct, Grade and Deliver; residual
  strength, recovery modes, preserve, Region EV, undo; the scopes (waveform,
  histogram, vectorscope), the probe, false colour, wipe, guides.
- An HDR viewer: on an HDR display the image is shown above SDR white
  (Windows: Direct3D 12 scRGB and HDR10; macOS: Metal EDR on the XDR panel).
- Compare that tells you what changed: under the Compare bar, the share of
  the frame RUDRA changed against the analytic baseline, by how many stops,
  and how much of it the view peak shows, with a button that raises the peak
  far enough to see it. **Changes** tints what RUDRA changed (amber brighter,
  blue darker) on either side of a wipe.
- The invented-pixel map: the **Invented** layer marks in magenta the values
  RUDRA made up where the SDR clipped or crushed, and in cyan where it read SDR
  detail differently from the baseline. Every master's sidecar carries the
  same shares (`invented_pixels`).
- Stills and sequences: PNG, JPEG, TIFF, WebP, BMP, a frame or a folder.
- Movies: open, scrub, and export HDR10 (HEVC 10-bit with its metadata), HLG,
  ProRes 422 HQ or ProRes 4444 (the finishing master), with the audio,
  checked before it is published, through a queue you can stop and resume
  (ProRes 4444 with alpha from the command line:
  `rudra-native video --format prores4444`). The ProRes files come graded in
  PQ, or scene-referred in **ACEScct** (AP1) or **ARRI LogC4** (AWG4): the
  master before peak and knee, to grade like camera footage. A log file is
  tagged unknown primaries and transfer (there are no codes for these
  curves); set the clip's input to the curve in Resolve. Its sidecar names
  the curve and gamut.
- Masters as ACES 2065-1 EXR sequences.
- The model (optional; masters default to the analytic inverse):
  `sdr2hdr_image_v8`, level with the inverse on real SDR, run by ONNX
  Runtime (DirectML on Windows GPUs, Core ML on Apple silicon, CPU
  everywhere). Every package is checked against its reference frames the
  first time it runs on a device, and refused if it drifts.
- `rudra-native`, the command line: `diff`, `video`, `deliver`, `batch`,
  `ffmpeg-check`.

## Changes since beta 3

- **Source panel.** Name the curve that made the SDR (unknown, ACES,
  filmic/Hable, AgX, camera log, plain Rec.709 clip) and the baseline becomes
  its inverse, drawn against the ACES inverse with readouts in nits.
- **Three-click calibration.** Click black, 18% grey or a highlight and type
  the nits it should be; one to three anchors are fitted over the picked curve.
- **Reference match.** Load the graded HDR of the frame (EXR, or 16-bit PQ
  PNG/TIFF) and the curve is fitted to it at every code, with the residual in
  stops.
- **Painted masks** for the first four Region EV bands (left drag paints, Alt
  erases, the wheel sizes the brush); they travel with the master as
  `<master>.masks.png`.
- **Anchor live on the viewer**, with an Anchor knee slider: the master's
  anchor stage is what you see while Anchor is on.
- **ProRes 4444** beside 422 HQ, and on both an **Encoding** row: graded PQ,
  or scene-referred **ACEScct** (AP1) or **ARRI LogC4** (AWG4).
- **The model is `sdr2hdr_image_v8`** (optional; masters default to the
  analytic inverse). `shadow_v1` lost to the inverse on every real Meridian
  frame (−4.11 dB) and is retired. A bare start now follows the default when
  it changes, so an install upgraded from an earlier beta opens v8 instead of
  the package it used last; a model you picked yourself stays picked.
- **Model parity on real 1080p frames:** the ONNX graphs reduce GroupNorm in
  stages, so ONNX Runtime matches PyTorch to 9e-6 on all 287 Meridian test
  frames (it was 5.6e-3).
- Half precision: packages carry an fp16 tile graph, held to a 1% tolerance;
  `--precision fp16` on `rudra-native diff` and `bench`, and a TensorRT device
  (`--device tensorrt`) where ONNX Runtime has it.
- **Starts the same on every machine.** Qt settings left in the environment by
  other software (a conda env with PyQt, a Qt SDK) no longer stop RUDRA with
  "no Qt platform plugin could be initialized": the package uses its own
  plugins and the log says what it ignored.
- **A log file**, rotated at 5 MB: on Windows
  `%LOCALAPPDATA%\FXTD Studios\RUDRA\logs\rudra.log`, on macOS
  `~/Library/Application Support/FXTD Studios/RUDRA/logs/rudra.log`. Attach it
  to a report.
- **Failed writes say why**: "The EXR file could not be written (the disk is
  full: 1.2 GB free on D:\, 3.4 GB needed)", access denied (with a pointer to
  Windows Security's Controlled folder access), or a missing folder. A master
  cut short by a full disk is removed rather than left truncated.
- Every live control (peak, region EV, strength, mode, preserve, source curve,
  calibration) is one composite and display pass: 1.1 to 1.3 ms at 4K on an
  RTX 4080, under a tenth of a 60 Hz frame. Inference time is separate:
  `rudra-native bench` reports and budgets it per machine.

## Install

**macOS (Apple silicon, macOS 13 or later).** Open the DMG and drag RUDRA to
Applications. The DMG is signed with RUDRA's Developer ID and notarized, so it
opens with no warning. A DMG built without the certificate (the build log says
"ad hoc") makes macOS warn on first open: System Settings > Privacy &
Security > Open Anyway, or `xattr -dr com.apple.quarantine /Applications/RUDRA.app`.

**Windows (x64, Windows 10 or 11).** Run `RUDRA-0.9.0-beta.4-windows-x64-setup.exe`:
it installs for your user by default (no administrator prompt), adds RUDRA to
the Start menu and can be removed from Settings > Apps. Or take the portable
ZIP: unzip anywhere and run `RUDRA.exe`. A GPU with DirectX 12 is used when
there is one; the CPU otherwise.

**Movies, both systems.** RUDRA runs `ffmpeg` and `ffprobe` from the PATH and
needs a build with `libx265`, `prores_ks` and `zscale`:
macOS `brew install ffmpeg@6`, with `$(brew --prefix ffmpeg@6)/bin` on PATH; Windows the "full" build from gyan.dev with
its `bin` folder on the PATH. `rudra-native ffmpeg-check` says whether yours
has everything. Stills need nothing extra.

## Known in this beta

- macOS: HDR output on Metal EDR, and inference on Core ML, are being
  measured on hardware; until then, check the first-run card (Help > Check the
  display and the model), which reports the display's real peak.
- The model does not yet beat the analytic baseline on every condition of
  the bench (see STATUS.md in the repository); the reconstruction is shown
  where it applied, so you can judge it frame by frame.
- Windows: not signed with a code-signing certificate yet, so SmartScreen
  may ask once.
- Linux builds are made from source (`scripts/native_app.sh`); there is no
  Linux package in this beta.

## Reporting

Issues and results: https://github.com/fxtdstudios/RUDRA/issues. Include the
version (RUDRA.app > Get Info, or RUDRA.exe > Properties > Details), what
Help > About shows, and for a crash the report macOS or Windows offers.
