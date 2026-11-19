# RUDRA 1.2.6

RUDRA 1.2.6 is shaped by your feedback on 1.2.5 and by a review against what a VFX and grading
facility expects. The Deliver tab is easier to read, a Media panel keeps your clips and their
grades, every render gets a QC report, AI fill shows exactly where it draws, and RUDRA now ships
for Linux as well.

## Opening it the first time

- **Windows**: 1.2.6 is not signed yet, so Windows may ask once: **More info** › **Run anyway**.
  Signing through SignPath Foundation follows once SignPath has approved RUDRA
  ([code signing policy](CODE_SIGNING_POLICY.md)).
- **macOS**: right-click RUDRA.app › **Open** › **Open** (or System Settings › Privacy & Security ›
  **Open Anyway**).
- **Linux (new)**: unpack `RUDRA-1.2.6-linux-x86_64.tar.gz` anywhere and run `RUDRA` inside it.
  `./install-desktop-entry.sh` adds it to your application menu. It runs on Rocky, Alma and RHEL 8
  or newer (glibc 2.28, the VFX Reference Platform's floor) and every newer distribution.

Download it only from this GitHub page; the checksums are listed with each release.

## Deliver, easier to read

- One crowded card is now **Format**, **The file**, **Extras**, **AI fill**, **Analytic master** and
  **Render**. Cards that only report (The file, This frame) look different from the ones you set,
  and carry a Read-only tag.
- Long explanations moved into tooltips; frame number, timecode and reel fold behind one button.
- Choices a shot cannot take (a movie format for a still, say) are hidden, not greyed out.
- On a 1080p screen at 100 to 150 % scaling no text is cut off: the controls over the picture wrap
  to a second row.
- Over the picture, three buttons (**Overlays**, **Colour**, **gamma · brackets**) open what used to
  be a long row of switches. Nothing was removed and the keys are unchanged.
- **Region exposure** (the old Grade tab) is now a fold at the end of Reconstruct, on the analytic
  route where it applies.
- The monitor offers **Scene** and **HDR 1000** (SDR 100 showed the source again), and RUDRA starts
  on the SDR | RUDRA wipe at HDR 1000.

## The Media panel

A new panel on the left keeps your clips:

- **Import** files or folders, see them as thumbnails or a list, and drag one onto the picture to
  open it. Several stills dropped together open as one shot.
- **Bins** you make, and **smart bins** that fill themselves: *Clipped highlights*, *Crushed shadows*
  and *Delivered*. Each clip is scanned in the background as it is imported.
- **Each clip keeps its own grade.** Leaving a clip stores its grade, view and Deliver settings;
  opening it brings them back. Right-click › **Settings** keeps versions, saves a new one, and copies
  or pastes settings across a selection or a whole bin. Versions keep painted masks too.
- Right-click › **Add to batch queue** renders clips with the Deliver settings without opening them.
- The library is saved with the project and between sessions.

## JPEG and 8-bit movies

Highlights rebuilt from a JPEG showed white blotches and coloured dots: RUDRA read the
compression noise near white as detail. JPEG, WebP, AVIF and HEIC stills, and 8-bit H.264, HEVC,
VP9, AV1 and similar movies, now get a highlight fill that smooths that noise and keeps the hue,
and a fully clipped pixel stays fully clipped. Lossless and 10-bit sources are unchanged.

## QC report with every render

Beside every master RUDRA writes `<name>.qc.html` and `<name>.qc.json` (Render › *Write a QC
report*, on by default):

- **Pass** or **Review**, with the reasons: over 10,000 nits or the PQ peak, more than 5 % of a frame
  invented, one-frame MaxFALL jumps, the export differing from the viewer, the learned model's own
  checks, and more than 1 % of a frame outside P3-D65.
- MaxCLL and MaxFALL per frame, the source, the route and every setting the render used, as they
  were when it started.
- **Route check**: a source graded on a curve other than ACES, rendered on the learned ACES 2.0
  route, gets a Review, and Deliver offers **Use the analytic route**.

## AI fill

- **Where it fills.** The card says what the fill will draw on this frame ("fills 0.3 % · the model
  sees 8.6 %"), or why it draws nothing, and **Where it fills** shows it on the picture. Isolated
  clipped specks are no longer sent to the model.
- **A progress bar** in the card and over the viewer, and the preview counts its seconds, so a
  model that is loading no longer looks stuck.
- **v3 or v4.3** (whole-region re-render) is a switch in the card.
- **Suggest a prompt from the shot**: Florence-2 (MIT, about 0.5 GB, runs on your computer)
  describes the frame and its bright area ("bright hazy sky, soft even white light, seen through an
  open cargo door"). Check this computer offers it with the other models, and AI fill setup has a
  **Download Florence-2** button.
- **Offline install**: AI fill setup › **Install from a folder or zip…** installs Wan2.1-VACE or
  Florence-2 with no internet. Every model, downloaded or installed, is checked file by file against
  a pinned revision and its SHA-256.
- **Disclosure**: every fill writes `disclosure.json` beside its frames: which fill, the model and
  LoRA with their checksums, prompts, seed, steps, route, and the share generated per frame.

## Masters

- **ACEScg and linear EXR masters are lossless ZIP**: a third to a half smaller, read by Nuke,
  Resolve and every OpenEXR reader. ACES 2065-1 stays the uncompressed SMPTE ST 2065-4 container.
- **Fixed**: a float32 ACES 2065-1 master carried the ST 2065-4 container flag, which only half-float
  files may carry. It is labelled instead.
- **Fixed**: on the learned route, "Clipped in source" always read 0.00 %.

## Fixed

- Opening several stills or closing a clip could lose its grade; each still now keeps its own as
  you step between them.
- The QC report could describe the grade or clip on screen when the render ended rather than the
  one it rendered.
- Fire and tungsten JPEGs: flat yellow-green patches where red and green clipped together.

## For developers

- **ONNX Runtime comes from NuGet.** `github.com/microsoft/onnxruntime` and its release downloads
  have answered 404 since 17-18 November 2026. `tools/fetch_onnxruntime.sh` takes
  Microsoft.ML.OnnxRuntime 1.24.4 from NuGet (pinned SHA-256) and lays it out as the old archive;
  CI, `native_app.sh` and `native_gate_a.sh` use it. NuGet's headers have no
  `coreml_provider_factory.h`, so on macOS Core ML is appended by name.
- **Linux package**: `scripts/package_linux.sh` (Qt, OpenColorIO and ONNX Runtime bundled with
  `$ORIGIN` paths; refuses anything needing more than glibc 2.28 / GLIBCXX 3.4.25, anything linked
  outside the package, and model weights). `release.yml` builds it in `almalinux:8` with
  gcc-toolset-13 and publishes the tar.gz with the others.
- **ZIP EXR**: `deliver/src/exr.cpp` writes ZIP and reads ZIP and ZIPS, with a 2 GiB decode budget
  and a per-block size check. zlib comes from vcpkg (`native/vcpkg.json`) or the system; without it
  masters are written uncompressed. `rudra/delivery/exr.py` still reads uncompressed files only.
- **Pinned models**: `rudra/genfill/model_manifest.json` (Wan2.1-VACE at `ec4d2cb0`, Florence-2 base
  at `00921df6`, every file's SHA-256); `rudra.genfill.setup` has `--make-manifest`, `--from`,
  `--florence`, `--florence-from`, `--florence-only`, `--verify` and `--allow-unverified`.
  `tools/make_models_json.py` has kind `florence` and `--upstream REPO@REVISION`.
- New files: `app/media_library`, `app/media_scan`, `app/qc_report`, `core/fill_region`,
  `tools/fetch_onnxruntime.sh`, `scripts/package_linux.sh`, `theme/qc_report.css`.
- MSVC builds `rudra_app_tests` and `rudra_app_ui` with `/bigobj`.
- Dependencies: actions/setup-python 7.1.0, actions/cache 6.1.0, upload-artifact 7.0.2,
  download-artifact 8.0.2; diffusers 0.41.0 and tokenizers 0.23.3 for AI fill.
