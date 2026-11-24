# RUDRA 1.3 plan

Written 23 Nov 2026, while the AI fill v5 LoRA trains (registration
`configs/experiments/research_genfill_v5_20261122.json`). 1.2.9 is released (PR #101).
Status markers: **done**, **in progress**, **next**, **waits on v5**, **needs others**.

## The order

| # | Item | Why | Size | Status |
|---|---|---|---|---|
| 1 | Tone-mapped view when the AI fill is shown | The default RUDRA view (nits) does no tone mapping: a fill above SDR white clips to plain white on an SDR screen, so the user sees nothing (09V, 21 Nov) | Small | done 23 Nov: colour_view.cpp tone_map_for_fill; test AppFillView.TheFillIsShownThroughAToneMappedViewAndTheViewComesBack |
| 2 | "Show what it invented" on the viewer bar | The check every fill needs sits at the bottom of Reconstruct | Small | done 23 Nov: vtAiInvented ("What AI drew"); test AppFillView.WhatAiDrewSitsBesideTheAiFillButton |
| 3 | Pre-flight before an AI fill: disk space, time estimate, a busy GPU | A fill writes about 45 MB a frame (09V ran out of space at frame 22) and takes about 13 min on 09V, far longer with the GPU shared (47 min seen) | Small to medium | done 23 Nov: genfill_preflight, genfill_gpu_load, genfill/secPerFrame learned from each fill; test GenFill.ThePreflightRefusesAShortDiskAndWarnsOfABusyGpu |
| 4 | The three tests that fail only on the studio PC | AppCheck.FlorenceIsOfferedWithTheModelsAndUsedOnceDownloaded, GenFill.FlorenceIsDownloadableWithoutThePublishedList, AppVideo.TheExportTilesQueueAMovieAndTheQueueWindowFollowsIt: known failures hide real ones | Small | done 23 Nov: the two Florence tests found the PC's real Florence-2 beside its Wan base; they now use a stand-in base (stand_in_genfill_base). The export-queue test passed in the 22 Nov run |
| 5 | Keep super-whites on read | Video with detail in codes 236-254 is clipped when decoded; RUDRA then guesses it back | Medium | part done 23 Nov: the Source panel now measures it (media probe_source_whites, 8 frames, the movie's own codes, no range conversion) and says whether anything is lost, pointing to Keep super-whites. The keep mode itself (bt1886-headroom-v1) already exists, experimental and opt-in; still open before it can drop that label: its Windows validation (archive/docs/HEADROOM_APP_INTEGRATION_2026-10-17.md, E1/E3), still and log exports (E2), and the controls it turns off (the model, clip fill, steady, anchor, chroma carry; BT.1886 only) |
| 6 | Per-shot QC report | One page a studio can send with the delivery: the gates, flicker, window joins, where the AI drew, MaxCLL / MaxFALL | Medium | done 23 Nov: the report beside every render (qc_report.cpp, HTML + JSON) already had the checks, MaxCLL / MaxFALL, frames and settings; it now has an AI fill section (share of generated pixels, worst frames, frames flagged for flicker or fallen back), a rewritten <title>_aifill report after each fill, and a verdict that says "Pass · contains AI fill" rather than a plain Pass. Window joins are not in it yet (measured by tools/window_joins.py, Python only); test QcReport.TheAiFillIsReportedAndItsFlaggedFramesWantALook |
| 7 | AI fill v5 replaces v4.3, still experimental | Lamps, thin edges, doors, glare (the v5 training material) | Model | done 23 Nov: v5 with the thin hold (research_genfill_v5d_thin_20261123) PASS on every registered check: Porch T 0.080 (v5 alone 0.229), 09V 0.036, testwindow 0.027, water 0.079 (v4.3 0.112), 09V lamp range 0.19 stops (v4.2 0.54), worst join x1.38 (x7.69), H 21/22 and 7/7 (PU21 -30.3); the 09V antenna as v5 draws it. Chosen offline on replays of the saved fills (predicted 0.0804 / 0.0358, measured the same). Tried and dropped: v5b (hold everywhere: antenna ghosts), v5c (7-frame median: Porch 0.219). App: v5 preferred, --rerender-hold-clipped thin; v4.3 unchanged. Package docs/models/genfill_v5. Still owed: a fresh thin-practical shot |
| 8 | Water flicker | If v5 fails or narrowly passes water T (limit 0.115 EV): the continuation option or calm, on the GPU | Medium | waits on v5 |
| 9 | Radiance (ComfyUI) port: 1.2.8 / 1.2.9 fixes and v5 | The two products should draw the same fill | Medium | waits on v5 |
| 10 | Output size in Deliver (2x, UHD) | A clean resize in HDR, no ringing round highlights; no invented detail | Small | done 23 Nov: Deliver > Format > Output size (Source, 2x, UHD 3840 wide, HD 1920 wide; shape kept, sides even). core/resample.hpp: Lanczos 3 in linear light held between the two nearest source pixels (no halo, no dark ring, no negatives), a tent for downscales; about 0.2 s a frame 1080p to UHD. Both routes: the learned master (EXR, DPX, movie; aces_v2 serve and cmd, EXR attribute rudra:outputSize) and the analytic master (its alpha, mattes and brackets too; sidecar output_size). The model and the measurements run at the source size. Not with the AI fill (it works at the source size; the choice rests while the fill is on) or the analytic movie of a movie (the export queue). Tests: OutputSize.*, Resample.*, MasterProvenance.TheOutputSizeWritesTheMasterItsAlphaAndMattesAtIt, AcesDeliver.TheOutputSizeResizesTheMasterItsMaskAndAlpha, Deliver.TheOutputSizeGoesToTheEngineOnlyWhenItIsNotTheSources, AppDeliver.TheOutputSizeWritesTheMastersAtItAndTheAiFillKeepsTheSources |
| 11 | Signing: Microsoft Store (MSIX) and Apple Developer ID | Removes the install warning for everyone (Store: free, Microsoft signs) | Medium (setup) | needs others: Store and Apple accounts |
| 12 | The Batch tab (moved into 1.3 on 24 Nov) | The two queues (learned batch dialog, analytic Queue window) in one minimal tab; the mockup approved by Ahmed | Medium | built 24 Nov, to build and test on the PC: batch_page.cpp; tests AppBatch.*; docs/PLAN_1_4_BATCH.md ("In 1.3.0": what is in, what waits) |
| 13 | The 24 Nov audit's four fixes, and EXR compression in Deliver | docs/AUDIT_1_3_2026-11-24.md items 2-5 and 17; Ahmed asked for ZIP / none and the like | Small | written 24 Nov, syntax-checked (g++ 11 on every changed file), to build and test on the PC: children in a kill-on-close job (platform/process.cpp, app/child_process.hpp); ffprobe and nvidia-smi by full path; Batch thumbnails on a worker thread; no output side past 8192; the Batch queue keeps size and compression; Deliver > EXR compression ZIP / ZIPS / None (ACES 2065-1 stays uncompressed), `--exr-compression`. Tests Process.AChildEndsWithThisProgram, Exr.ZipsMastersAreLosslessOneLineABlock, OutputSize.NoSideIsWrittenPast8192, AppDeliver.TheExrCompressionGoesToTheEnginesAndAcesStaysUncompressed, Batch.TheQueueKeepsItsJobsAndBuildsTheirRequests |
| 14 | AI fill v5 on the learned ACES 2.0 route | Is v5 safe on the learned masters too? | Check only | done 24 Nov 01:42, FAIL (`research_genfill_v5_learned_20261123.json`, tmp/09V_test/v4_gates/learned_v5_verdict.json): 5 of 10 shots pass; 01V T 0.112, 02V T 0.124, 03V T 0.318, water T 0.136 (limit 0.115), Porch L 4.074; held-out PU21 +43.1 on 21 scenes (closer in 1), +71.2 on 6 (closer in 0). In the release notes, MODELS.md and report_v5.json (`learned_route`); finalize_genfill_v5.py run. Whether the app should keep offering v5 on the learned route is Ahmed's call |
| 15 | v5's flicker on 01V and 03V (both routes), and the learned route's shots | Ahmed: fix it before 1.3.0 (24 Nov); the learned held-out check later | Medium | registered 24 Nov about 10:40 (`research_genfill_v5e_20261124.json`), GPU run 24 Nov 12:16-15:30 (tmp/RUN_V5E.ps1): FAIL by its rule, on 02V of the learned route only (T 0.118 against 0.10); learned 9 of 10 pass (v5d 5), analytic 7 of 7 (v5d 5), held-out -30.5 and -31.3 (v5d -30.3, -32.3). Ahmed chose to ship v5e anyway (about 15:45, recorded in the registration's `outcome`). The finish's mix settles while its code stays within 3 of where its hold began (`--rerender-blend-still 0.95`), a held thin shape keeps 0.95 for a jump (`--rerender-hold-calm 0.95`); on the learned route also `--rerender-cap-master --mask-first-frame`. Replayed on the registered raw fills (analytic): 01V T 0.164 -> 0.079, 03V 0.173 -> 0.094, the other five pass as before; 09V's antenna has no copies (holding on the frame-to-frame change instead left them: tmp/v5_review/09V_*_antenna3x.jpg). Wired into the app for v5 only; RELEASE_1_3_0.ps1 tags only with a PASS or that recorded decision |
| 16 | zlib on Windows | The ZIP tests skipped on the PC's build ("this build has no zlib"): ZIP/ZIPS wrote uncompressed EXRs there, and PACKAGE_WINDOWS.ps1 has no vcpkg either | Small | 24 Nov: native/cmake/RudraDeps.cmake fetches zlib 1.3.1 (SHA-256 pinned) when none is found and links it static; Deliver greys the menu in a build without it. To build and test |

## Fixed on launch day (24 Nov, Ahmed: "fix it now")

- BT.1886 on the app's own codes. The app holds every shot's own codes (a still; a movie's frame through ffmpeg to
  PNG), and BT.1886 read them through sRGB and then the BT.709 OETF: 0.36 stop dark at code 128, 1.3 at 51, 2.7 at
  26 (09V: 1.95 stops at the median, 3.6 in the darkest 5 %; measured on the CLI's masters of 09V's frames). Now
  `own_codes` (CompositeParams, MasterRequest, Session; set by the app and by the CLI's still masters, forced off
  on the video path, which canonicalizes) reads them as video codes: code^2.4 within 0.01 stop (codes 2..254).
  Other curves bit-identical; the learned route untouched; the v5e checks stand (the fill acts near white, where
  the two readings agree within 0.01 stop).

## Not in 1.3 (decided 23 Nov)

- An sRGB choice in Reconstruct > Source (Ahmed, 24 Nov: 1.3.1): an analytic master that lines up with an SDR
  read as sRGB in Nuke (FXTD's setup: Input Transform sRGB Encoded Rec.709 (sRGB), view ACES 2.0 SDR 100 nits).
  1.3.0 reads the viewer's SDR side that way by default and the user guide says which reading matches which route.
- The OFX plug-in with all three routes, for Resolve and Nuke (Ahmed, 24 Nov): the next release, started
  after 1.3.0 ships. The plug-in drives the installed app's engines as separate programs (no model inside
  the host; Nuke ships its own LibTorch): the analytic route first (native/ofx, in the code since 20 Oct,
  never built for Windows nor opened in a real host), then the learned ACES 2.0 route through
  `rudra-native aces-v2-serve`, then the AI fill as a "Generate AI fill" step per clip (the app's job in
  the background, stored, marked out of date when a setting changes). Each tried in Resolve and Nuke.
- On-viewer area selection for the reconstruction and the fill (sliders, a drawn box, click to pick a
  highlight): revisit after v5's results and the first 1.3 feedback. If v5 still invents things or
  overdoes lamps, click-to-pick becomes the first item of 1.4.
- AI upscaling: needs its own commercially clean model.
- Full colour-profile (ICC) support.
- The highlight fill on fireworks (research).

## Needs others

- Code signing: a Microsoft Store developer account (free) and the Apple Developer Program ($99 a year).
- The OFX plug-in in real hosts: Resolve or Nuke on the studio PC.
- Dolby Vision checked with Dolby's own tools.
- A Mac for the hardware pass.
- 20-30 held-out studio shots (lamps, windows, sun, water, night) for the wider evidence base.
