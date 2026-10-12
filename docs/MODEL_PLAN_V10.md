# The model after 1.0: v10 and what follows

Written 13 Oct 2026. RUDRA 1.0 ships the analytic reconstruction:
- the Source picker, with BT.1886 by default;
- calibration and reference match;
- the clip fill;
- the artist's grade.

The model is optional (STATUS line G). This page sets out how a model earns a
place in it, and what v10 changes.

## What the earlier models got wrong

| | finding | evidence |
|---|---|---|
| baseline | Every model, v4 to v9, learned a correction to the ACES inverse. On real SDR, BT.1886 alone is +14 dB over that inverse (R3). The model was fixing the wrong starting point. | docs/RESULTS.md, "Against the market" |
| selection | best.pt was chosen on 256-px crops scored in log-PSNR, which does not predict the bench. v9's best.pt never left step 0, because its validation set had no exposure push. | STATUS line E; 13 Oct v9 entry |
| validation | The real-SDR validation title, Cosmos Laundromat, is an animation; the real test title, Meridian, is live action. | pipeline/fetch_netflix_pairs.py |
| shadows | v8 and v9 are 1.6 to 2.4 stops worse than the inverse inside crushed pixels, and better on 0 of 287 frames. | cp8_results_v8/v9s20 |
| clip | v9 beats the clip fill at +0 EV (+0.116 stops) but loses at +1 and +2 EV. | cp8_results_v9s20 |

## v10 (built 13 Oct 2026; run with `scripts\run_v10_2026_10_13.ps1`)

### Baseline: `--baseline-source`

The residual rides on the Source picker's curve. `rudra/source_baseline.py` is a Python copy of `native/core/src/source_curve.cpp`; a test checks it against the native knots to within 1e-5.

Each sample's curve comes from `effective_baseline_curve`:

| sample | baseline curve |
|---|---|
| real SDR | BT.1886 |
| degraded | BT.1886 |
| rendered | its own curve |
| exposure push | ACES |
| rendered or pushed, in training (`--baseline-dropout 0.5`) | half of them on BT.1886 anyway: the artist who never picks |

The model's weights are unchanged, so v9 loads as v10's starting point.

### Shadows: `--crushed-hold 1.0`

Changing a crushed black (max code ≤ 2/255) is penalised.

### Validation: `pipeline/build_v10_manifest.py`

- Held-out shots of Nocturne and Sparks (live action) are the real validation set.
- Cosmos Laundromat moves to train.
- Rendered and test rows are unchanged.

### Selection: `--best-metric bench_gain` (`training/bench_val.py`)

Each evaluation scores full frames (long side 960):
- PU21 on real and on rendered validation frames;
- the error inside clipped pixels at +1 and +2 EV;
- the error inside crushed pixels.

All four are measured against the picker's baseline and combined into one score:

`real + min(0, rendered) + 4 × clip + 4 × min(0, crushed)`

### Gate: `training/cp10_verdicts.py`, written before the run

| row | condition |
|---|---|
| real (Meridian) | above the BT.1886 baseline, PU21 and CVVDP intervals above zero |
| aces | not worse than the ACES inverse |
| oog, mix | not worse than the BT.1886 baseline |
| clipped pixels at +0, +1 and +2 EV | better than the clip fill |
| crushed pixels, everywhere | not worse |

Bench trees: `export_bench_pairs.py --baseline-curve` and `--picker-baseline`.

### Order

1. `-Stage smoke`
2. `-Stage train`
3. `-Stage bench -ClipOnly`: the cheapest gate to fail first
4. `-Stage bench`

## If v10 passes: the app

The app runs the network on the model's own baseline, while the residual is applied over the picker's curve. A model trained over the picker's curve has to see it at inference too. Two changes:
- The exported tile graph takes the effective curve parameters: the picker's 256 knots plus the model's head, which is `effective_curve_params` in native.
- The engine re-runs inference when the picker changes.

Neither is worth doing before a model passes the gate.

## Next, in order

1. **v10**, as above.
2. **More real pairs.** These are what moved real SDR:
   - HDRTV4K and other licensed graded pairs;
   - our own footage, graded once.
3. **Clip-fill failures.** Training on over-exposed frames, sparks and fireworks: the clip fill's weak cases.
4. **A real-time model for playback.** Distilled from the passing model; target 1080p ≤ 20 ms with TensorRT fp16 (roadmap R1 and R4). Run it only on tiles that clip or crush.
5. **Commercial weights** without the HdM data (roadmap 4.4).
