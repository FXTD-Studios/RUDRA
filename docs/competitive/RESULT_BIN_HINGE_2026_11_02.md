# Result: the 0 EV training kit, bin hinge w4 and w16 (2 Nov 2026)

Registrations `claude_bin_hinge_w{4,16}_20261102.json` (before training). Parent d6400.
Evidence `G:\rudra_training\claude_bin_hinge_w{4,16}_20261102\`.

**Concluded: neither arm passes its registered rule (the fresh 0 EV tolerant rule fails for both). Not promoted.**

| | d6400 (parent) | **w4** | w16 | limit |
|---|---|---|---|---|
| aligned residual | 0.00851 | **0.00906 pass** | 0.00928 pass | ≤ 0.010367 |
| paired checks | — | **all pass** | all pass | |
| old24 grade −3 EV | — | **9.51 pass** | 10.08 pass | ≤ 18.62 |
| old24 grade 0 EV | 0.0648 fail | **0.0537 pass** | 0.0622 fail | ≤ 0.0604 |
| old24 grade +3 EV | 0.00330 fail | **0.00241 pass** | 0.00243 pass | ≤ 0.00250 |
| Hunyuan48 | pass | **pass** (0.0151) | pass (0.0150) | ≤ 0.020 |
| fresh 0 EV RMS | 0.473 | 0.453 **fail** | 0.450 fail | ≤ 0.338 |
| fresh −3 / +3 / H | 485.8 / 0.333 / 0.0770 | 490.4 / 0.319 / 0.0774 | 498.8 / 0.317 / 0.0786 | ≤ analytic 526.5 / 0.393 / 0.0832 |
| pixels over 1 code (first → last 100 steps) | — | 0.055 → 0.076 | 0.057 → 0.086 | should fall |

What it shows:
1. **w4 is the first checkpoint to pass every original development gate**, including
   old24 0 EV and +3 EV, which d6400 failed. It fails only the fresh FXTD 0 EV rule.
2. **The hinge itself did not work.** Its own diagnostic rose during training for both
   arms (the share of pixels a code or more off went up, not down). So w4's development
   gains cannot be credited to the hinge: they may come from 1600 more steps of the 25 Oct
   recipe with EV weights 1,1,1. Without a hinge-free control (bin weight 0, same seed) the
   cause is unknown. Likely reason the hinge lost: PCGrad projects a conflicting objective's
   gradient away, and the hinge conflicts with the 0 EV display term (the hinge pulls
   toward the source codes, the display term toward the reference's view).
3. **The fresh set's 0 EV error barely moves with any training change** (0.473 → 0.453 /
   0.450 RMS). Development and fresh sets disagree consistently: the fresh FXTD footage has
   partially clipped highlights the training data does not represent.

Next, in order of cost:
- a hinge-free control (bin weight 0, otherwise identical to w4): attributes w4's gains;
- the gate policy decision (is a sub-code 0 EV deviation a release blocker?), in writing
  before more data, since on every set every other number beats the analytic inverse;
- real-footage training data like the fresh set's (FXTD footage kept out of every
  evaluation set), the only lever the fresh 0 EV has not been tried with.
