# RUDRA models: licence and status

One table for every model RUDRA has published. The README, the Hugging Face
model card, the user guide and `models.json` must say the same thing; if they
disagree, this file is the one to correct them from.

Updated 1 November 2026 (RUDRA 1.0.3).

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

What the repository can show today:

- **The export.** `tools/export_native_v2.py` writes `parity_pass` from its
  own LibTorch/ONNX check and always writes `release_authorized: false`. The
  published manifest says `release_authorized: true`; that was set by hand, and
  no committed report names the checkpoint behind it.
- **The last committed gate result** (29 Oct 2026, `archive/docs/RELEASE_V1.md`,
  `docs/STATUS.md`): the learned model **fails the 0 EV grading check, 0.121
  against a limit of 0.0604**. On the fresh FXTD grade set v2 (42 cases, 18
  scenes, 26 Oct) the release candidate improved grading at −3 EV and +3 EV and
  highlight accuracy, but its highlight pixels sat about 0.47 code RMS from the
  reference at the delivered exposure against about 0.24 for the plain inverse.
- **Not yet measured in CI:** the shipped files on LibTorch, ONNX Runtime CPU,
  DirectML and Core ML against each other (`docs/RELEASE_NOTES_1.0.3.md`).

Until a gate report that names this checkpoint is committed, RUDRA ACES 2 is
labelled **preview** everywhere. To close it, on the training machine:

    python tools/model_provenance.py dist\models\RUDRA_ACES_2 ^
        --reports G:\rudra_training reports ^
        --out docs\models\RUDRA_ACES_2.provenance.json

and commit the record. Exit 2 means no report names the checkpoint.
