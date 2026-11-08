# RUDRA models: licence and status

One table for every model RUDRA has published. The README, the Hugging Face
model card, the user guide and `models.json` must say the same thing; if they
disagree, this file is the one to correct them from.

Updated 7 November 2026 (RUDRA 1.1.0; RUDRA ACES 2.0 release candidate).

## In the app (downloaded, not bundled)

From 1.0.3 the Windows and macOS packages carry **no model weights**. Each
model below is downloaded in **File › Model packages**, shows its licence
first, and is checked file by file (SHA-256) before use. The analytic
reconstruction, which needs no model, is the default.

| Model | Files (SHA-256) | Licence | Commercial use | Status |
|---|---|---|---|---|
| **RUDRA ACES 2** (learned ACES 2.0 reconstruction) | `rudra_g1.ts` `fb30e3df748f4fa9…`<br>`rudra_g1.onnx` `8ba97cb0ed36f503…` | Apache 2.0 | Yes | **Preview.** See "Evidence" below |
| **AI fill v3** (Wan2.1-VACE-1.3B LoRA + brightness estimator) | `rudra_genfill.safetensors` `9fff7eb8492ae626…`<br>`brightness.json` `34892fb840de138f…` | Apache 2.0; base model Wan2.1-VACE-1.3B is Apache 2.0 (Wan-AI) | Yes | **Experimental**, off by default. Passed its pre-registered gates on 25 held-out Poly Haven scenes (30 Oct 2026; `registration_v3.json`, `report_v3.json`) |

Training data for both: Poly Haven HDRIs (CC0), FXTD Studios' own footage and
renders, procedurally generated scenes, CC0 plates. No non-commercial data,
no LTX or DiffHDR code, weights or outputs.

## Not distributed in packages (research and Beta 2)

| Model | SHA-256 | Licence | Commercial use | Status |
|---|---|---|---|---|
| `sdr2hdr_image_v8` | `20130171…` | Non-commercial (`checkpoints/LICENSE`) | No | Retired. **Bundled by mistake in packages built by `release.yml` up to 1.0.2** (confirmed in the published 1.0.1 Windows ZIP; the 1.0.2 tag was never published as a release); not from 1.0.3 |
| `sdr2hdr_shadow_v1` | `6b7f73f0…` | Non-commercial | No | Retired (Beta 2 web app's model) |
| `sdr2hdr_shadow_s2`, `_s3`, `sdr2hdr_image_v5`, `_v6`, `sdr2hdr_temporal_v1` | `checkpoints/SHA256SUMS` | Non-commercial | No | Research only, kept for reproducing the paper's tables |

If you installed 1.0.0 or 1.0.1: the bundled `sdr2hdr_image_v8` is under the
non-commercial licence wherever it is. Update to 1.0.3, or delete the
`models/sdr2hdr_image_v8` folder from the install, before commercial work.

## Evidence: RUDRA ACES 2

**Published today (1.0, preview):** the checkpoint behind `rudra_g1.ts`
`fb30e3df…` fails the 0 EV grading gate (0.121 against 0.0604, 29 Oct 2026).
Its manifest says `release_authorized: true`, set by hand; no committed report
names its checkpoint. It stays preview.

**Release candidate (2.0): w4 + lock_dqs050g.** The checkpoint
`research_bin_hinge_w4_20261102` step 1600, with the lock (stage 6b,
`training/aces_partial_tighten.py`, native `core/aces_lock.hpp`) at strength
0.5 toward the dequantized source, and the appearance guard.

- **Crew holdout, run once as registered** (7 Nov 2026,
  `configs/experiments/holdout_crew_eval_20261107.json`,
  `docs/competitive/RESULT_CREW_HOLDOUT_2026_11_07.md`): passes every gate.
  0 EV RMS 0.311 code against a limit of 0.342 (analytic inverse 0.242 + 0.1);
  −3 EV, +3 EV and highlight colour better than the analytic inverse; no new
  negative luminance; no pixel more than 1 code off.
- **What it does not show:** at the delivered exposure it is *worse* than the
  plain inverse in every case (56 of 56), by less than a tenth of a code. The
  claim is "within 0.1 code RMS of the plain ACES 2.0 inverse at the delivered
  exposure", never "better". One production (The Crew), one camera (ARRI
  ALEXA LogC3), single frames.
- **Native lock:** bit-exact with the Python oracle on its golden
  (`native/tests/test_aces_lock.cpp`); on real frames through the real view,
  `native/tools/aces_v2_parity` with `--lock`.
- **The package:** contract major 3 (stages plus the lock). Apps before the
  lock (1.1.0 and older) refuse it rather than run the model without its
  lock. It needs RUDRA 1.2.0.

It leaves preview only through `tools/promote_aces2_release.py`, which
refuses unless every one of these is on file: the holdout verdict, native
parity with the lock, LibTorch against ONNX Runtime DirectML and CPU, Core ML
(or a stated reason it was not measured), the provenance record
(`docs/models/RUDRA_ACES_2.provenance.json`), and a signed visual review.
`scripts/RELEASE_ACES2_W4.ps1` runs every step that needs no person.
