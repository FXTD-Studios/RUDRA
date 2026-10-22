# Headroom preview in the application — 17–18 October 2026

This covers ledger items E1 and E2, and parts of E3 and E4 (see
[REMAINING_WORK.md](REMAINING_WORK.md)). The preview stays **experimental and
closed**. No public menu, setting or default project reader enables it. The
only entry points are `MainWindow(viewer, /*headroom_capable=*/true)`, used by
tests, and the developer check `RUDRA --headroom-check`.

**Windows acceptance is still needed.** The headroom decoder only runs on
Windows, because its source read leases are Windows-only. The QRhi viewer needs
Qt 6.6 or newer. All the code here was built and tested in a Linux workspace
with Qt 6.4 and no viewer. That covers the logic, using a fake renderer, and
the refusal paths of the real one. The real decode, viewer, export and timing
checks still have to be run on the Windows machine with
`scripts/validate_headroom_app_2026-10-18.ps1`, and their evidence recorded in
PRODUCTION_PROGRESS before the ledger is changed.

## What was added

### E1: application presentation integration

**The driver.** `engine/headroom_preview.{hpp,cpp}` adds
`HeadroomPreviewDriver`. It has no Qt dependency and sits over a
`HeadroomPreviewPort` seam. The production port wraps `HeadroomPresentation`.
- One worker thread calls only `render_next`.
- Rendered frames are converted on the worker to the viewer's precomposited
  convention: linear Rec.709, 1.0 = 10 000 nits, with no clamp. Frames are
  posted to the UI thread as **opaque tokens, never naked pixels**.
- On the UI thread, `present()` revalidates each token under the bridge lock
  before the display callback sees any pixels.
- At most one token is in flight (back-pressure).
- Each request has an epoch. Stale tokens are rejected by the port's own
  revision check. The driver does not drop them itself, because doing so lost
  the only frame of a paused seek in testing.
- Failures are reported and never replaced by older pixels. Destruction
  cancels the port and joins the worker; tasks still queued become no-ops.

**MainWindow routing.** Routing is on when the window is capable, the grade's
policy is `bt1886-headroom-v1`, and the shot is a movie. Then:
- `step_to`, play and stop, grade changes, undo and redo, project restore,
  model and temporal-hold changes, shot replacement and close all route
  through the driver.
- Legacy engine frames are ignored, and legacy measurements and probe state
  are cleared. Turning the policy off, for example by undo, returns the frame
  to the legacy engine. That is an explicit change of route, not a fallback.

**Refused controls.** The v1 preview refuses the anchor, chroma carry, clip
fill and temporal hold. The window reports why and shows nothing; it never
switches controls off for the artist.

**Projects.** A capable window reads headroom projects (`ProjectReadSupport::HeadroomV1`).
A project saved with the policy on now records `settle_grain = false`, which
is what the preview renders. Before this change the window could not save a
headroom project at all: it had no grain control, and the stored default
(`true`) failed validation.

**Viewer.** The viewer path uses the existing `ViewerWindow::set_composited`
upload. No shader changes were needed. The upload happens inside the
synchronous present callback.

**Playback.** Playback runs as fast as frames render, with no clock and no
skipping. It is an inspection mode, not real time.

### E2: headroom workflow

**Scopes and readouts.** `measure_headroom(model, baseline)` in
`engine/measure` produces the frame's own metrics, waveform, histogram,
vectorscope, compare line and probe.
- There is no SDR or network mask on this path. Masks, clip coverage and the
  invented-pixel shares are empty, never borrowed from the legacy path.
- The clip bar and the strip's clip lanes are hidden on this route instead of
  showing a false 0 %.

**Export.** `engine/headroom_export.{hpp,cpp}` adds `export_headroom_movie`, a
transaction in this order:
1. Refuse an existing movie or sidecar.
2. Validate the grade.
3. Hash the source (SHA-256) and prepare it with protection.
4. Decode once with a single `HeadroomStream` owner.
5. Grade each frame with `render_headroom_frame` and write it as a float32
   linear Rec.2020 EXR (1.0 = 203 nits) into a private staging folder beside
   the output.
6. Require the **decoder completion receipt and source identity**.
7. Encode with `deliver_linear_frames`: Rec.2020, roll-off into the peak,
   measured MaxCLL/MaxFALL, colour tags verified.
8. Probe the staged file (frame count, size) and re-hash the source.
9. Write the provenance sidecar: policy, source hash and cache id, decoder,
   target, peak, MaxCLL/MaxFALL, grade, app version, build commit, QC results
   and limits.
10. Publish by renaming without replacing anything.

Any failure or cancellation leaves nothing at the output and removes the
staging folder.

**Export in the app.** With the policy on, Deliver's movie export
(HDR10, HLG, ProRes 422 HQ or 4444) runs this transaction off the UI thread.
A second click cancels it.
- Log masters (ACEScct, LogC4) and Master EXR stills still refuse the policy,
  with a reason.
- There is no audio, temporal smoothing or dynamic metadata on this path.

**Public selection.** It stays closed: there is no menu item, and default
windows and readers still refuse the policy.

### E3: performance and durability (partial)

- **Memory accounting.** The driver reports rendered CPU memory for tokens in
  flight and for the frame on screen (`HeadroomPreviewStats`). This is
  separate from the decoder's source-pixel allowance.
- **View conversion.** This is now one fused float pass split across threads.
  On a 2-core Linux workspace with a UHD synthetic frame:

  | Version | Time per image |
  |---|---:|
  | Before (median of 2) | ~103 ms |
  | After | ~72 ms |

  The two versions differ by at most 2e-5 relative. This is a view image only;
  masters never use it.
- **Not measured yet.** The headroom check times paused frames, seeks, grade
  changes, playback and export. These timings exist only once it runs on
  Windows. **No real-time or process-wide memory claim is made.** The ~0.56 fps
  UHD CPU preview found on 17 Oct is not resolved by this change.

### E4: source controls (partial)

`super_white_stats` classifies each frame's decoded code values into:
- **detail**: above reference white but below the code ceiling, so super-white
  information was recorded;
- **ceiling**: at the format's highest legal code (8-bit 254, 10-bit 1019,
  within half a code), which is true clipping;
- **at white**: within half a code of reference white. This is reported but
  not judged.

The app's status line shows the three shares for every headroom frame.
**Validated input overrides are not done.** v1 accepts only one input contract,
so any override would just be refused.

## Validation in this workspace

All runs were on Linux: Ubuntu 24.04, GCC 13, Qt 6.4.2, FFmpeg 6.1, OpenCV on,
no viewer and no ONNX Runtime.

**Native tests: 378 total.**
- 330 pass, including 14 new driver tests (one of them the super-white case)
  and 3 new export refusal tests.
- 48 skip. These are the existing Windows-only headroom tests.
- 2 more new export tests (publish with provenance, and cancellation) are
  compiled only on Windows, so they are not in this count.
- Five shuffled runs of the driver tests under ThreadSanitizer were clean.
- 60 shuffled repeats of the driver tests passed after one race was fixed.
  That race was found by repetition: a frame of a paused seek was being
  discarded.

**App tests: 73 total.**
- 71 pass, including 8 new `AppHeadroom` tests: routing, undo and redo back to
  legacy, refusal without fallback, playback and stop, failure, export refusal
  and start, project restore, and shot replacement.
- 10 shuffled repeats of the headroom set passed.
- **2 failures** are environmental and also fail on the untouched baseline
  build: `FTriggersFitButStillTypesInTextFields` and
  `AnalyticWorkflowExportsWithoutLoadingAnyModel`. On this Linux build the
  baseline also failed `TheScriptedWorkflowPassesAndReportsEveryStep`.

**Real-port runs on Linux** stop where they should:
- `RUDRA --headroom-check` refuses at the paused step with "Headroom source
  read leases require Windows in this build". The route stays on, and no
  legacy picture is shown.
- The app's HDR10 export fails at preparation, and nothing is staged or
  published.

## Still open in these items

- **E1/E3 on Windows.** Run `scripts/validate_headroom_app_2026-10-18.ps1` and
  record its reports. It covers the real decode, viewer upload, present and
  readback, timings, and the export.
- **E2.** Masters of individual frames (EXR stills) and log exports still
  refuse the policy. There is no headroom-aware probe of the source SDR codes,
  and no HDR10+/DoVi metadata.
- **E3.** Faster initial preparation, long clips, rapid seeks, multiple
  consumers, slow storage, FFmpeg and GPU memory, and the CPU preview rate.
- **E4.** Validated input overrides.
- **Public enablement.** This waits until all of the above pass and the owner
  decides.
