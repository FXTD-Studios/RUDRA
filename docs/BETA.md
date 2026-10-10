# RUDRA 0.9.0 beta 8

The desktop RUDRA: SDR footage in, scene-linear HDR out, with the places the
model reconstructed shown to you. This is the eighth public beta of the native
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
- The model (optional, off by default: Reconstruct > Use the model):
  `sdr2hdr_image_v8`, level with the inverse on real SDR, run by ONNX
  Runtime (DirectML on Windows GPUs, Core ML on Apple silicon, CPU
  everywhere). Every package is checked against its reference frames the
  first time it runs on a device, and refused if it drifts.
- `rudra-native`, the command line: `diff`, `video`, `deliver`, `batch`,
  `ffmpeg-check`.

## Changes since beta 7

- **The analytic reconstruction is the default, and says so.** Until now the
  app ran the model on every frame and every master although these notes
  called it optional. Now the view and Master EXR use the analytic inverse
  (your source curve, calibration, reference and grade) and run no model at
  all; RUDRA opens, views and masters with no model package installed.
  Reconstruct > Use the model (AI-assisted) turns the model on; the status
  bar says which is in use, and a master's sidecar records it
  (`"reconstruction": "analytic (no model)"`, or the model's name under
  `checkpoint`). The choice is kept between runs and in a project.
  Movie exports (HDR10, HLG, ProRes) still run the model package, whatever the
  switch says, and do not yet carry the grade: see Known in this beta.

## Changes in beta 7 (since beta 6)

Fixes that bring the native app in line with the 9 Oct review of the
Studio's master path:

- **Region EV lands in the master with Anchor on.** The anchor (on by
  default) used to divide a Region EV push below its knee straight back out,
  in the viewer and in the master, while the sidecar said the grade was
  applied. The master now grades after the anchor, chroma and grain stages,
  and the viewer shows the same.
- **The linear EXR container is really Rec.2020.** It carried the plate's
  primaries (Rec.709 for most footage) under a "scene-linear Rec.2020" label
  with no chromaticities; it is now converted and tagged.
- **`rudra-native deliver`, HLG:** the BT.2100 system gamma (log10; it was
  right at 1,000 nits only) and saturated colours scaled together instead of
  clipping per channel (no hue shift). Footage in Rec.709 or P3 is converted
  to Rec.2020 before the roll-off, so MaxCLL/MaxFALL describe what is encoded.
- **No half-written movies:** `deliver` encodes under a hidden partial name
  and renames it into place, so a failed encode leaves nothing that looks like
  a delivery.

## Changes in beta 6 (since beta 5)

- **Projects.** File > Save project (Ctrl+S) keeps the shot, the frame, the
  model and the whole grade (mode, strength, Region EV, source curve,
  calibration, reference, painted masks, view peak, Deliver settings) in a
  `.rudra` file; File > Open project (Ctrl+O), a drop, or on Windows a
  double-click opens it. Footage moved together with the project is found
  again; anything missing is named and the rest still opens.
- **Autosave.** The session is saved a few seconds after each change and on
  close; File > Reopen last session brings back the one the last run left.
- **ffmpeg in the Windows package** (the gyan.dev "full" build, GPL v3, in
  `ffmpeg\` with its licence and source note): movies work without
  installing anything. `RUDRA_FFMPEG_DIR` picks another build.
- **Update check.** Once a day RUDRA asks GitHub whether a newer release is
  out and says so in the status bar; Help > Check for updates asks now.
  Nothing is downloaded. Off with `RUDRA_NO_UPDATE_CHECK=1`.
- **Help for a first session:** Help > Getting started (the five steps,
  shown once after the first-run check), Help > User guide (F1, also in the package
  as "User guide.md"), Help > Open log folder, and tooltips on every control
  of the Reconstruct, Grade and Deliver tabs.
- **Signing hooks:** the Windows build signs RUDRA.exe, rudra-native.exe and
  the installer when a code-signing certificate is supplied (the release
  workflow's `WINDOWS_CERT_PFX` secret).
- Open recent opens a movie as a shot again (it was added as a still).

## Changes in beta 5 (since beta 4)

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

## New in beta 4 (since beta 3)

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

**Windows (x64, Windows 10 or 11).** Run `RUDRA-0.9.0-beta.8-windows-x64-setup.exe`:
it installs for your user by default (no administrator prompt), adds RUDRA to
the Start menu and can be removed from Settings > Apps. Or take the portable
ZIP: unzip anywhere and run `RUDRA.exe`. A GPU with DirectX 12 is used when
there is one; the CPU otherwise.

**Movies.** RUDRA needs an ffmpeg with `libx265`, `prores_ks` and `zscale`.
Windows: the package carries one, nothing to install. macOS: `brew install
ffmpeg@6`, with `$(brew --prefix ffmpeg@6)/bin` on PATH (the DMG does not carry
ffmpeg yet). `RUDRA_FFMPEG_DIR` points either at another build;
`rudra-native ffmpeg-check` says whether one has everything. Stills need
nothing extra.

## Known in this beta

- macOS: HDR output on Metal EDR, and inference on Core ML, are being
  measured on hardware; until then, check the first-run card (Help > Check the
  display and the model), which reports the display's real peak.
- The model does not yet beat the analytic baseline on every condition of
  the bench (see STATUS.md in the repository); the reconstruction is shown
  where it applied, so you can judge it frame by frame.
- Windows: signed only when the release was built with a code-signing
  certificate (RUDRA.exe > Properties > Digital Signatures shows it); without
  one SmartScreen may ask once.
- macOS: ffmpeg is not in the DMG yet (see Install).
- Movie exports run the model package (`rudra video`) with recovery "all"
  and without the Reconstruct/Grade settings; the analytic switch, the grade
  and the masks apply to the view and to Master EXR. To deliver a graded
  movie today, master EXR frames and encode those (`rudra-native deliver`).
- Real-time playback with the model is not offered: the model takes about
  150 ms per 1080p frame on an RTX 4080; the controls themselves redraw in
  about 1 ms at 4K.
- Linux builds are made from source (`scripts/native_app.sh`); there is no
  Linux package in this beta.

## Reporting

Issues and results: https://github.com/FXTD-Studios/RUDRA/issues. Include the
version (RUDRA.app > Get Info, or RUDRA.exe > Properties > Details), what
Help > About shows, and for a crash the report macOS or Windows offers.
