# RUDRA 1.1 vs a cloud SDR-to-HDR workflow: gap audit (5 Nov 2026)

Audit only, no code changed. Baseline: RUDRA 1.1.0 (native app, `native/`), spec
`docs/composite.spec.md` rev 1. HAVE = exists, PART = exists but not as asked, MISS = absent.

## Already there (do not duplicate)

| Feature | Where |
|---|---|
| Recovery mode All / Highlights / Shadows / Off, one Strength | `core/composite.hpp` `RecoveryMode`, spec §2 step 4 |
| Highlight / shadow priors `hp = sigmoid(24(y-0.82))`, `sp = sigmoid(24(0.10-y))*w_s` | spec §2 step 3; `core/src/composite.cpp:146`, `render/shaders/composite.frag:120`, `ui/compositor.js`, Python oracle |
| Preserve outside the model's masks (default on) | spec §2 step 6 |
| Region EV bands in nits, softness in stops, up to 4 painted masks (brush size/softness/flow, invert) | `core/masks.hpp`, `app/region_editor.*`, spec §3 |
| Clip fill (partial / large / small regions), per-region peak override | `core/clip_fill.hpp` |
| AI fill v3: tau_high / tau_low, seed, steps, brightness, highlight + shadow prompts, review A/B/mask, flicker flags, swap-in with backup | `app/genfill_*` |
| `rudra.reconstructed` channel in the master EXR | `deliver/exr.hpp` |
| HDR viewer (scRGB / HDR10 / EDR), View peak, B flip, W wipe, False colour, Difference, Invented layer, scopes, probe | `render/viewer_window.hpp`, `app/main_window.cpp` |
| HDR10 / HLG / ProRes delivery, Dolby Vision + HDR10+ sidecars, flicker report | `deliver/*` |

## Checklist status

### 1. Regional reconstruction control
| Item | Status | Note |
|---|---|---|
| Luminance mask with adjustable highlight / shadow thresholds | **MISS** | The reconstruction gate is fixed at 0.82 / 0.10, slope 24, in four implementations. Region EV is a grade, not a reconstruction gate. |
| Isolate a luminance range for reconstruction | **MISS** | Region EV bands do this for the grade only. |
| Mask softness / feather / edge refinement | PART | Gate slope fixed; painted masks have brush softness, no edge-aware refine. |
| Pixel-level preservation outside the mask | PART | Sigmoid gate is never exactly 0, and `expm1(log1p(bL))/L` is not bit-identical to `b`: "preserved" pixels drift by fp32 roundoff. |
| Independent highlight and shadow controls | PART | Mode picks one or both; one Strength for both. |

### 2. Prompt-directed reconstruction
| Item | Status |
|---|---|
| Optional prompts, semantic instructions | HAVE (AI fill, highlight + shadow prompts) |
| Only inside the mask | HAVE (r_safe_blend in the fill region) |
| Preserve composition | HAVE by contract (no SDR code moves); adherence not measured |
| Prompt strength | PART: `--guidance` in `rudra/genfill/run.py`, not in `GenFillOptions` |
| Reconstruction intensity | HAVE (brightness offset), shape gain not exposed |
| Optional and separate | HAVE |

### 3. Independent matte outputs
| Item | Status |
|---|---|
| Highlight matte / shadow matte | MISS: one `rudra.reconstructed`, one `rudra.generated` |
| Editable masks as EXR | PART: painted masks as `<master>.masks.png` (RGBA8) |
| Separate passes for comp | PART |
| Mattes match the final image | unverified by test |

### 4. HDR preview and exposure brackets
| Item | Status |
|---|---|
| -2 / 0 / +2 EV preview renders | MISS (fill review is -3 EV only) |
| Immediate HDR preview | HAVE |
| SDR / HDR synced compare | HAVE (B, W wipe) |
| Preview-to-export consistency check | PART (Hold steady matches export; no automated check) |
| Show transfer, primaries, reference white | PART (swapchain name only) |

### 5. Reconstruction quality
| Item | Status |
|---|---|
| Region visualization | PART (Invented, Difference) |
| Original / preserved / reconstructed / generated separately | MISS |
| Confidence visualization | MISS |
| Independent AI vs analytic controls | HAVE (Use the model, clip fill, AI fill) |
| Fallback on AI artifacts | PART (review + backup; nothing automatic) |

### 6. Validation: all MISS as automated tests on moving footage.

## Priority 1: landed 5 Nov 2026 (spec revision 2)

Section 1 items 1, 2, 4, 5 are now HAVE: *Limit to a luminance region* on the
Reconstruct card (highlights + shadows with independent threshold, softness,
strength; or a band), bit-exact baseline outside (gate exactly 0), in the
viewer (composite.frag), masters, movie exports, projects, sidecar and EXR
(`rudra:reconstructionRegion`). Off is revision 1 bit for bit. Refused with
Keep super-whites (headroom v1). Item 3 (edge-aware refinement) is still
PART: the region is luma-only; painted masks remain the spatial control.
Verified: rudra_tests 377 pass (8 new `ReconstructionRegion.*`; 3 `Queue.*`
fail in the build container only, they need the Python queue's checkpoint),
6 new pytest, composite.frag compiles (glslang), app layout.cpp /
main_window.cpp compile against Qt 6.4 headers. Not run: the Qt app tests and
GPU parity (Windows / RTX).

## The workflow gaps closed in code (5 Nov 2026)

1. **Live region view**: the viewer tints the reconstruction region (display.frag binding 5,
   `region_overlay_rgba8`, the same 8-bit weights in core render_view and both GPU paths);
   the Reconstruct card has a luma histogram with the region's weight and "Reaches x % of this
   frame" (`region_histogram`), recomputed per frame; "Show on the viewer" switch. Tested: the
   tint is exactly where the composite's gate is > 0, nothing else moves.
2. **Mattes for movies**: per frame in `<name>_mattes/` (`MasterRequest::mattes_dir`), never in
   the frames folder. Found on the way: `list_linear_frames` (rudra-native deliver, the movie
   encoder) took `<name>.mattes.exr` as a frame, so a folder rendered with mattes encoded every
   matte as an extra (black) frame (priority 3 bug; the selftest's HDR10 file had 2 frames for
   1). Fixed there and in the fill's `list_frames`; tested.
3. **Region + prompts on one route**: AI fill now runs after an analytic EXR render in ACES
   2065-1 (`ai_fill_possible`), with this render's frames only (glob from master_targets' names).
   Still not: the learned ACES route does not apply the luminance region to its own
   reconstruction (its engine is separate; the region gates its AI fill only).
4. **Containers**: analytic ACEScg (AP1) and linear Rec.709 beside ACES 2065-1 and linear
   Rec.2020 (master, sidecar, chromaticities, Deliver gamut, projects, pipe). Tested: float
   masters in both read back to the preview with 0 code difference.
- Verified: rudra_tests 390 pass (4 new `Gaps.*`; the 3 container-only `Queue.*`); 19 pytest;
  display.frag compiles; the render files have no errors beyond the Qt 6.4 API differences of
  the container's headers; app files + `test_app.cpp` (1 new test) compile; the validation
  selftest unchanged (still F1 on the sky). Not run: app tests, GPU parity, the viewer on screen.

## Finding F1: decided (5 Nov 2026): the delivery keeps a saturated pixel's luminance

- **Cause, measured.** Not the partial-channel stage: with `max_partial_ev` at 0, 2 or 4 the
  synthetic sky's fill is identical (stage 1 lifts nothing there; no clean pixels to read a
  colour from). A blue channel at 255 decodes to the analytic ceiling itself, ~1,150 nits in
  Rec.2020, on every partly clipped sky pixel while its luminance is ~140 to 250 nits (the
  5,131-nit master peak is the fully clipped band, whose hue comes from that saturated edge).
  At a 1,000-nit peak, `master_to_peak` scaled the whole pixel down until blue fit, and HLG's
  scene-light step did the same harder: that is the 0.16 / 0.70 stop.
- **Options weighed.** A lower `max_partial_ev` cannot help (above). Lowering the analytic
  ceiling would change every master. So: a saturation-aware fit in the delivery only.
- **Fix.** `PeakFit::KeepLuminance` (`core/hdr10.hpp fit_to_peak`): a pixel whose luminance
  fits the peak but one channel does not moves toward its own Rec.2020 luminance just far
  enough that the channel fits. Luminance and hue direction kept, saturation given up, only on
  pixels the display cannot show at that peak; everything inside the peak is untouched, and a
  pixel whose luminance reaches the peak is still scaled. Applied in the shoulder and in HLG's
  scene-light step (the OOTF then gives the display luminance back).
- **Where.** `deliver_linear_frames` (the app's movie export, the headroom export,
  `rudra-native deliver`): `SequenceEncodeOptions::peak_fit`, default KeepLuminance;
  `rudra-native deliver --peak-fit scale` gives the Python's fit and report byte for byte
  (the report says `"peak_fit": "luminance"` otherwise). Masters are not changed.
  The parity functions (`master_to_peak`, `encode_master`, `encode_sequence_frame`, and so
  `video`'s learned-route movie path) keep `PeakFit::Scale` by default: the Python goldens hold.
- **Measured** (validation selftest, synthetic sky, 1,000 nits): SDR-range shift against the
  master **-0.16 -> -0.009 stop (HDR10), -0.70 -> -0.016 stop (HLG)**. Other shots unchanged.
  The sky's C6 *fidelity* check still FAILs (0.025 / 0.037 stop median on neutral pixels,
  delivered at 6,400 nits where no fit happens): that is 10-bit 4:2:0 Y'CbCr on a saturated
  gradient, not F1.
- Tests: `PeakFit.TheDeliveryKeepsASaturatedPixelsLuminance`,
  `PeakFit.HlgKeepsTheSceneLuminanceOfASaturatedPixel`; rudra_tests 392 pass (the 3
  container-only `Queue.*` aside); the CLI report parity run passes with `--peak-fit scale`.
- Not done: the Python `rudra deliver` and `video` still scale (port when the Python
  delivery is next touched); the real sky plates should still be run.

 (5 Nov 2026, after the priorities; cloud tools have one)

- `rudra/genfill/prompts.py`: 9 presets (8 highlight, 1 shadow) and `suggest()`, a heuristic
  over the fully clipped components (area, place, rectangularity, circularity, the colour of
  the ring just outside, and with several frames the warm area's change) that ranks presets and
  says why. No model; torch-free. `precache` embeds the presets once (GENFILL.ps1
  -PrecachePrompts) so a preset needs no umT5 load when the fill starts.
- App: Deliver > AI fill > "Suggest a prompt..." runs it on the frame on screen and opens a menu:
  the suggestions (with reason, "ready" when cached) and All presets; picking one fills the
  highlight or shadow prompt.
- Verified: 9 pytest (each synthetic case lands on its preset: blue / sunset / overcast sky,
  sun, lamp, window, glints, a moving flame, shadows only with tau_low); the selftest shots of
  the validation harness all get the right first suggestion; the app's JSON reader run against
  the module's real output; app files + test compile against Qt 6.4.
- Not done: suggestions are not validated on real footage (add them to the validation shots),
  and the prompts themselves were never part of the fill's v3 gates (prompts stay optional).

## Priority 6: harness landed 5 Nov 2026; the footage runs are yours

- `tools/validate_rudra_1_1.py` (docs/VALIDATION_RUDRA_1_1.md): C1 preservation outside the
  region (bit-exact), C2 mattes vs the master, C3 brackets preview vs export, C4 the default
  chain outside the region (INFO), C5 temporal on moving shots, C6 HDR10 / HLG fidelity and
  preview-vs-deliverable (ffprobe tags, decoded back to light), C7 the AI fill contracts plus
  prompt adherence (MANUAL contact sheet, CLIP score if available).
- `rudra-native master analytic <image>`: the analytic master with no model package (what the
  harness and any script needs; `cli/master.cpp render_analytic_master`).
- Selftest on synthetic shots: everything PASS except C6 on the sky, which found **F1**: a sky
  with only blue clipped gets blue rebuilt to +4 EV by the clip fill, and the 1,000-nit
  HDR10 / HLG delivery then darkens the sky's SDR range by 0.16 / 0.70 stop against the preview.
  Decided and fixed 5 Nov 2026: see "Finding F1: decided" below.
- Still to run on the GPU machine with real plates: every category, moving shots, and C7
  (prompt adherence, geometry) with the fill weights.

## Priority 5: landed 5 Nov 2026

- **Pixel classes** (`core classify_pixels`): 0 preserved (bit-identical to the baseline:
  the source decoded), 1 follows (< 0.1 stop), 2 reconstructed (moved where the SDR has
  detail), 3 estimated (moved where it is clipped / crushed), 4 generated (AI fill). Class 0
  equals `rudra.changed == 0` exactly (tested). Builds on compare.hpp's invented /
  reinterpreted test, not a second definition.
- **Confidence** (`rudra.confidence`): a constraint score, stated as not a probability: 1 for
  0/1, 1/(1+|d|) reconstructed, 0.5/(1+|d|) estimated, 0 generated.
- **Visualization**: `<name>.regions.png` with the mattes (classes over grey, the luminance
  region's edge in white), `rudra.class` / `rudra.confidence` in the mattes EXR, class shares
  in the sidecar (`pixel_classes`). Viewer: the existing Invented layer already shows the
  reconstructed / estimated split live; no duplicate layer was added.
- **Independent AI vs analytic controls**: already HAVE (Use the model / clip fill / AI fill).
- **Fallback** (`fill_master(fallback='spikes')`, `--fallback spikes`, preview `fallback`,
  Deliver "Drop flickering fill frames", on by default in the app, off in the CLI so v3's
  command line is unchanged): a frame whose fill jumps > 0.25 stop beyond the master's own
  change both into and out of it is written as the master's frame; onset / end jumps (one
  side) are kept. A non-finite fill always falls back. Frames are written one behind so the
  next frame can decide; a stop still leaves only whole frames (existing test passes).
  `fell_back` per frame and `fallback_frames` in genfill.json.
- Not done: the class map for the AI fill's own output inside the app (the fill writes
  `rudra.generated`; `classify_pixels` takes it, nothing calls it with it yet); a per-pixel
  (not per-frame) artifact detector.
- Verified: rudra_tests 386 pass (2 new `PixelClasses.*`; the 3 container-only `Queue.*`);
  35 pytest on the tiny Wan (2 new fallback tests); the regions picture checked by eye on the
  golden frame (sun magenta = estimated, region edges white); app files + `test_app.cpp`
  (1 new test) compile against Qt 6.4.

## Priority 4: landed 5 Nov 2026

- **-2 / 0 / +2 EV brackets** (`core exposure_bracket`, `kBracketStops`): the viewer's SDR
  view at view peak 203 and viewer EV -2, 0, +2. Deliver "Write EV brackets" (analytic EXR
  stills / sequences) writes `<name>.ev-2.png`, `.ev+0.png`, `.ev+2.png`, rendered from the
  EXR **read back from disk**; `brackets` in the request JSON, `rudra:brackets` in the EXR.
- **Preview-to-export check**: the same brackets from the in-memory picture the viewer shows,
  compared code for code (`code_delta`), in the sidecar as
  `brackets.preview_vs_export` (max code delta, % pixels differing). Measured on the golden
  frame: float32 masters 0 codes in both containers; half masters at most 1 code (3.0 % /
  3.8 % of pixels, the half rounding). Tested as such.
- **Display signal**: `display_signal_text` in a new `pipeDisplay` segment of the pipe bar:
  transfer, primaries, reference white 203 nits, display peak (or, on SDR, the view peak
  simulated).
- Already HAVE (no change): immediate HDR preview (scRGB / HDR10 / EDR swapchains), SDR / HDR
  compare in sync (B flip, W wipe, same frame, same playback).
- Not done: brackets for movie exports and the learned ACES route; validating the HLG / PQ
  movie masters against the preview (Priority 6, needs ffmpeg decode of the delivered file).
- Verified: rudra_tests 384 pass (3 new `Brackets.*`; the 3 container-only `Queue.*` as
  before); app files and `test_app.cpp` (1 new test) compile against Qt 6.4 headers.

## Priority 3: landed 5 Nov 2026

- `core reconstruction_mattes`: `rudra.highlight`, `rudra.shadow` (the gate's priors as
  applied: mode, region, strength, recovery mask), `rudra.changed` (exact: 1 where the
  ungraded reconstruction differs from the baseline), `rudra.region`, plus `rudra.paint.N`.
  With the region on, both mattes 0 implies the baseline bit for bit (tested).
- Written as `<master>.mattes.exr` (float32, `deliver write_mattes_exr`), Deliver
  "Write mattes (EXR)" for analytic EXR stills / sequences, `mattes` in the request JSON,
  `rudra:mattes` in the master EXR, `mattes` in the sidecar. `read_exr` now keeps every
  other channel by name (`ExrImage::other`).
- Found and fixed on the way: `publish_master` wrote `<name>.masks.png` into its staging
  folder and deleted it with the folder, so a published master never had its painted masks
  beside it. It now places `.masks.png` and `.mattes.exr` with the master, never
  overwriting, and removes what it placed if one refuses.
- AI fill: `rudra.generated.highlight` / `.shadow` channels, summing to `rudra.generated`
  exactly (tested on the tiny model).
- Not done: mattes for movie exports (a movie has no channels; Keep the EXR masters is the
  learned route's, which keeps `rudra.reconstructed`); editing a matte and reading it back
  as a painted mask.
- Verified: rudra_tests 381 pass (4 new `ReconstructionMattes.*`; the 3 container-only
  `Queue.*` as before), 33 pytest, the app files compile against Qt 6.4 headers.

## Priority 2: landed 5 Nov 2026

- **Prompt strength** (`FillSettings.prompt_strength`, `--prompt-strength`, Deliver
  "Prompt strength" 0..2): a region prompt's prediction is blended in at
  `w * strength`; 1 is v3's formula exactly (same code path), 0 ignores the prompt,
  > 1 extrapolates from the base prompt (experimental: v3's gates were at 1). No
  extra model pass, no umT5 load.
- **Detail strength** = the existing `shape_gain`, now in the app (Deliver, 0..2) and
  per request in the preview server.
- **Prompts only inside the selected region**: the grade's reconstruction region is
  passed to the fill (`--region`, the preview request's `region`, batch jobs). The
  model's fill regions are cut to it and the blend weight is scaled by it
  (`rudra.delivery.region.fill_region_weight`), so pixels outside are the master's
  bit for bit; a region with nothing to fill skips the model entirely.
- Defaults send nothing new: the command line is unchanged.
- Still open from section 2: measured prompt adherence and geometry preservation
  on real footage (needs the GPU and v3 weights; section 6).
- Verified: 32 genfill + region pytest on a tiny random Wan VACE (CPU, torch 2.15,
  diffusers 0.42, OCIO 2.6), including the master staying bit-identical outside the
  region (this test caught a variable-shadowing bug before it shipped); the app files
  and `test_app.cpp` compile against Qt 6.4 headers. Not run: the app tests, the
  real model.

## Priority 1 design (as built)

Spec rev 2, §2 step 3: `hp = gate_up(y; h_at, h_soft) * h_strength`,
`sp = gate_down(y; s_at, s_soft) * s_strength * w_s`, optional band `[r_lo, r_hi]`.
Defaults reproduce rev 1 bit for bit (sigmoid, 0.82 / 0.10, slope 24, strengths 1).
A custom gate uses a compact smoothstep, and where the gate is exactly 0 the
output is `b` itself (new step 5b), so preserved pixels are bit-identical.
Lands in: spec, Python oracle, `composite.cpp`, `master.cpp`, `composite.frag`,
`ui/compositor.js`, goldens (`tools/emit_composite_golden.py`), session/project
JSON, sidecar, Reconstruct card UI, tests.
