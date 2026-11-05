# RUDRA 1.1 vs a SwitchHDR-style workflow: gap audit (5 Nov 2026)

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

## Priority 1 design (as built)

Spec rev 2, §2 step 3: `hp = gate_up(y; h_at, h_soft) * h_strength`,
`sp = gate_down(y; s_at, s_soft) * s_strength * w_s`, optional band `[r_lo, r_hi]`.
Defaults reproduce rev 1 bit for bit (sigmoid, 0.82 / 0.10, slope 24, strengths 1).
A custom gate uses a compact smoothstep, and where the gate is exactly 0 the
output is `b` itself (new step 5b), so preserved pixels are bit-identical.
Lands in: spec, Python oracle, `composite.cpp`, `master.cpp`, `composite.frag`,
`ui/compositor.js`, goldens (`tools/emit_composite_golden.py`), session/project
JSON, sidecar, Reconstruct card UI, tests.
