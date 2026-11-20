# RUDRA 1.2.7

RUDRA 1.2.7 fixes what a review of 1.2.6 by a colourist, a compositor, an editor, a DoP and an
HDR mastering engineer found. AI fill now shows what it did as soon as a preview finishes.
Rebuilt highlights keep the colour of the source. A single hot speck can no longer set a
master's MaxCLL. A render from a script now gets the same settings as one from the app.

## Opening it the first time

- **Windows**: if Windows asks, choose **More info** › **Run anyway**. Signing through SignPath
  Foundation follows once SignPath has approved RUDRA ([code signing policy](CODE_SIGNING_POLICY.md)).
- **macOS**: right-click RUDRA.app › **Open** › **Open** (or System Settings › Privacy & Security ›
  **Open Anyway**).
- **Linux**: unpack `RUDRA-1.2.7-linux-x86_64.tar.gz` and run `RUDRA` inside it.

Download it only from this GitHub page; the checksums are listed with each release.

## AI fill: see what it did

- **A finished preview opens where you can see it.** What the fill draws is brighter than SDR
  white, so at 0 EV it looked the same as the master. Now, when a preview finishes, RUDRA turns
  off the SDR | RUDRA wipe and sets the view exposure to −2 EV (only if it was at 0). A line over
  the viewer says what it changed. **Alt+E** takes the view back to 0 EV, and **W** turns the
  wipe back on. Nothing in the master changes.
- **"Where it fills" shows what v4.3 really redraws.** v4.3 leaves a saturated highlight, such as a
  sunset or a flame, to the master. 1.2.6 still painted it orange (11 % of the 07V sunset, where
  v4.3 redraws none of it). The overlay and the frame line now show the near-white region in
  solid orange and what stays the master tinted. The line also says how much stays the master
  ("v4.3 redraws 11.0 % · 0.1 % stays the master").
- **Keep small and thin bright parts** (new, v4.3, off by default). Bright parts that are thin or
  broken up (a floor grating, a railing), and areas smaller than 0.05 % of the frame (a practical
  lamp, a glint), keep the master instead of being redrawn. "Where it fills" follows it. Off is
  v4.3 exactly as its checks ran.
- **When the v4.3 model is missing, the card says where to get it.** 1.2.6 said it was "not
  published" and to "turn Re-render off". v4.3 is a download in **File › Model packages…**, and
  v3 / v4.3 is the switch in the card.

## Highlights keep the colour of the source

- **A door or window seen from a dark room no longer takes the room's colour.** The highlight fill
  borrowed the colour of the clean pixels around a partly clipped area. When those pixels belong
  to a darker room lit differently, the sky came out lavender. Pixels more than 3 stops below the
  clip now lend less of their colour, and none from 6 stops. On 09V the sky in the cargo door goes
  from B/R 1.39 to 1.13 (the source is 1.09), and the purple blotches at −2 EV are gone.
- **A speck can no longer outshine its own band.** A few partly clipped pixels beside a dark
  silhouette could jump stops above everything around them. In 07V, 24 pixels under an engine
  reached 21× white, so the frame's MaxCLL was 4,386 nits while 99.99 % of it stayed under 182.
  A pixel's lift is now held to 1 stop over the mean of the partly clipped pixels around it. 07V's
  MaxCLL is now 213 nits, and the fire's goes from 2,988 to 1,048.

Masters made from the same shot differ from 1.2.6 only where these two rules act. For one of
those, `clip_fill_options_1_2_6()` in the engine gives 1.2.6's fill exactly.

## QC report

- **A MaxCLL set by a few pixels wants a look.** Every master's sidecar now records
  `light_p99_98_nits`, the light its brightest 0.02 % leave out. A frame over 1,000 nits whose
  MaxCLL is more than 4× that gets a **Review** with the frame's name. A TV tone-maps the whole
  programme against MaxCLL.

## Renders from scripts match the app

`rudra-native` (and anything that drives it) now takes the app's defaults when its settings don't
say otherwise:

- **Source curve BT.1886**, the app's default. The engine's own default (`unknown`) made the same
  09V frame four times brighter on average (MaxFALL 142 nits against 39).
- **The lossy-source fill** for a JPEG / WebP / AVIF / HEIC still, and for an 8-bit H.264, HEVC,
  VP9 or AV1 movie, as the app has done since 1.2.6.
- **A P3-D65 mastering display** in an HDR10 movie's SEI, as the app writes it. Use
  `--mastering-display rec2020` for the old BT.2020 one.

Each default is printed as a `note:` line. Give `"source_curve"` or `"lossy_source"` in the
parameters to override it.

## Fixed

- An HDR10 movie whose HDR10+ could not be embedded (ffmpeg without HDR10+) now lists it under
  `warnings` in the render report, not only under `hdr_metadata`.
- `rudra/delivery/exr.py` reads the ZIP and ZIPS masters RUDRA has written since 1.2.6, so scripts
  built on the Python package open 1.2.6 and 1.2.7 ACEScg masters again.

## For developers

- `core/clip_fill`: `ClipFillOptions::chroma_from_stops` / `chroma_to_stops` (3 / 6) and
  `partial_outlier_stops` / `partial_outlier_sigma_px` (1 / 8); `clip_fill_options_1_2_6()`. Tests
  `ClipFill.ADarkRoomDoesNotTintTheDoorItLooksOutOf`, `ClipFill.ASpeckCannotOutshineItsBand`,
  `ClipFill.TheOldBehaviourIsOneCallAway`.
- `core/fill_region`: `ai_fill_v4_weight` and `ai_fill_region_v4`, the C++ twin of
  `composite.neutral_weight` and the new `composite.detail_keep_weight`. Tests `AiFillRegionV4.*`.
- `rudra.genfill.run --rerender-keep-detail`; the app sends it as `rerender_keep_detail` to the warm
  server. `GenFillOptions::rerender_keep_detail`, setting `deliver/aiKeepDetail`.
- `MainWindow::frame_genfill_preview` (setting `genfill/previewFrames` = false turns it off); label
  `viewerAiHint` over the viewer.
- Master sidecar: `light_p99_98_nits`. QC: `kQcSpeckRatio` (4), `kQcSpeckMinNits` (1,000);
  `light_p99_98_nits` in each frame of `<name>.qc.json`.
- CLI: `with_app_defaults` (cli/master.hpp), `--mastering-display p3d65|rec2020` on any command.
  The engine's own default stays BT.2020 / `unknown`, the Python reference's, so the parity tests
  are unchanged. `tools/validate_rudra_1_1.py` passes `source_curve: unknown` to keep its
  registered measurements.
- `rudra/video.py` and `video/src/convert.cpp`: the same HDR10+ warning (parity).
- Tests: `tests/test_release_1_2_7.py`; app tests `GenFill.KeepDetailIsOptInAndReachesTheFill`,
  `GenFill.TheMissingV4LineSaysWhereToGetIt`, `QcReport.AMaxCllSetByAFewPixelsWantsALook`.
