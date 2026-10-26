# RUDRA — ACES 2.0 model status

**Date:** 25 October 2026 · **Author:** Claude (Cowork) for ahmed · **Scope:** the learned SDR → ACEScg model and its native pipeline

**In short:** the model passes every check except one grading check. On 25 Oct the 0 EV grading rule was relaxed (approved by ahmed) to "within 0.1 display code (RMS) of the analytic conversion". Under that rule, two new fine-tune arms are now training to fix the last failing check (the older-scene highlight cap). Nothing has been frozen or released yet, and none of this work is on GitHub yet.

---

## 1. Model in one paragraph

Input: Rec.709 display codes (8 or 16 bit). Output: an ACEScg scene-linear, signed, unclamped 32-bit float master, plus a `rudra.reconstructed` mask channel. The colour basis is the exact inverse of the official ACES 2.0 SDR 100-nit view (OpenColorIO 2.5.2). Shadows are analytic (nothing invented). Partially and fully clipped highlights each have their own learned route. Fully clipped interiors are smoothed, not textured, and every output displays within 1 code of the source.

## 2. Current results (development data, not holdout)

Lower is better.

| Check | Limit | Best candidate (no fine-tune) | Fine-tune, −3/0/+3 EV loss, weight 0.5 |
|---|---|---|---|
| New-scene highlight error | ≤ 0.29235 | 0.28844 ✓ | 0.29125 ✓ |
| Older-scene highlight error | ≤ 0.38252 | 0.38117 ✓ | **0.38539 ✗** |
| Detail in clipped areas | ≤ analytic | tie ✓ | tie ✓ |
| Highlight detail | ≤ 0.14308 | 0.10063 ✓ | 0.12143 ✓ |
| Motion consistency | ≤ 0.01037 | 0.00865 ✓ | 0.01014 ✓ |
| Video flicker (48 frames) | ≤ 0.020 EV | 0.0154 ✓ | not yet run |
| Grading −3 EV | ≤ 18.62 | 10.45 ✓ | 10.28 ✓ |
| Grading +3 EV | ≤ 0.00250 | 0.00249 ✓ | 0.00247 ✓ |
| Grading 0 EV, **old rule** | ≤ 0.0604 | 0.1210 ✗ | 0.0833 ✗ |
| Grading 0 EV, **new rule** (RMS + 0.1 code) | ≤ 0.1195 | 0.1210 ✗ (just) | 0.0833 ✓ |

All four fine-tunes trained so far:

| Arm | 0 EV | +3 EV | Older-scene | Notes |
|---|---|---|---|---|
| Plain (photo loss only) | 0.0907 | 0.00524 ✗ | 0.37530 ✓ | |
| 0 EV display loss | 0.0732 | 0.00322 ✗ | 0.38391 ✗ | |
| −3/0/+3 relative, weight 0.5 | 0.0833 | 0.00247 ✓ | 0.38539 ✗ | closest |
| −3/0/+3 relative, weight 2 | 0.0986 | 0.00286 ✗ | 0.38305 ✗ | |

Partially clipped pixels at 0 EV: error went from 0.186 to 0.133 code² (analytic 0.086). In RMS that is 0.43 → 0.36 code against 0.29, all below half a display code.

## 3. Rules agreed on 25 Oct

Registration: `configs/experiments/claude_partial_tolerance_20261025.json` (written before the new arms ran).

- **0 EV grading:** pass if √(error) ≤ √(analytic) + 0.1 code. The limit is 0.1195 code² on the old-scene development set.
- **Restated check:** "outside fully clipped exact" is replaced, because partial-route training changes partial pixels by design. The replacement is:
  - unchanged outside highlights;
  - fully clipped pixels bit-exact with the parent (`tools/check_full_support_unchanged.py`, verified on the weight-0.5 arm);
  - all non-trained weights bit-exact.
- **All other checks unchanged:** −3/+3 EV, detail, the older-scene cap and video flicker.
- **Selection:** among arms passing everything, the lowest 0 EV error.
- **Honesty note:** the allowance was chosen after seeing development results. Development passes are therefore exploratory; only a fresh holdout is confirmatory.

## 4. In progress now

| Item | State |
|---|---|
| Fine-tune arms, display weights 0.25 and 0.35 (aim: older-scene ≤ 0.38252 while keeping 0 EV and +3 EV) | training (cloud CPU), then the full checks; about 2–3 h |
| Push all work to a new GitHub branch `aces-v2` | in progress; GitHub is reachable from the machine, push access still to be confirmed |

## 5. Still to do before the model is complete

1. **A passing arm.** If neither new arm passes, the next step is GPU research, or keeping the analytic master (v1) for release.
2. **Video check** (Hunyuan 48 frames) on the passing arm, then **freeze it**: hash, package (TorchScript + ONNX + manifest with the smoothing contract), and a C++ = Python parity check.
3. **Fresh holdout:** new stills and clips nobody has looked at, through the same frozen rules. This is the only confirmatory test.
4. **Broader video:** the 18-clip generated review, longer clips, moving objects, dark scenes.
5. **Real 10-bit and compressed inputs:** all test clips so far are 8-bit.
6. **Head-to-head with the old v8 model** (Hugging Face) and the analytic conversion, on the same frames.
7. **Banding and compression artefacts:** no tested fix yet; fix it or document it as a limitation.
8. **Weights paperwork:**
   - Fine-tune data is all CC0 (351 rows).
   - Confirm the full ancestry before choosing a weights licence.
   - Replay the first training run to prove its fresh start.

## 6. App side (done, needs Windows to run)

- **Native ACES v2 pipeline:**
  - C++ stages, an OpenColorIO 2.5.2 view, LibTorch and ONNX Runtime/DirectML runners.
  - A v2 package and manifest, a background master job, and `rudra-native aces-v2`.
- **Parity:**
  - C++ = Python bit for bit on 8 real frames.
  - The ONNX route is within 0.002 code on real frames.
- **Tests:** 8 unit tests pass (cloud, and on the machine's sandbox).
- **Run on Windows:** `scripts\NATIVE_ACES_V2.ps1` (needs OCIO 2.5.2 via `-OcioPrefix` or `VCPKG_ROOT`).
- **Masters available now:**
  - 32-bit ACEScg EXR with mask;
  - 16-bit PQ BT.2020 HDR video (FFV1).
- **LogC:** LogC4 exists in the app, on the old model. LogC3 and LogC4 from the new model come by converting the EXR with OpenColorIO (studio config).

## 7. Decisions still open for you

- Release plan: ship v1 (analytic) first, or wait for the learned model.
- The weights licence (the new model may be commercially licensable; v8 is not).
- Fixing the Hugging Face card: README licence (Apache 2.0) vs repo tag (non-commercial), the retired `shadow_v1`, and a local `G:\` path in the v8 config.
- Signing stays the last step.

## Evidence

- Fine-tune results: `G:\rudra_training\claude_partial_finetune_20261025\`
- Smoothing results: `G:\rudra_training\claude_clipped_smoothing_20261024\`
- Provenance archive: `G:\rudra_training\provenance_guarded_projected600_20261025\`
- Handoff: `docs\CURRENT_HANDOFF.md`
- Licences (draft): `docs\THIRD_PARTY_LICENSES.md`
