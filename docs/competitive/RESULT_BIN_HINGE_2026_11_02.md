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

## Addendum: the hinge-free control w0 (same recipe, seed and schedule; registered after w4)

| | w0 (no hinge) | w4 (hinge 4) | limit |
|---|---|---|---|
| aligned residual | 0.00783 | 0.00906 | ≤ 0.010367 |
| old24 grade −3 EV | 11.47 | **9.51** | ≤ 18.62 |
| old24 grade 0 EV | 0.0642 fail | **0.0537 pass** | ≤ 0.0604 |
| old24 grade +3 EV | 0.00281 fail | **0.00241 pass** | ≤ 0.00250 |
| Hunyuan48 | pass | pass | |
| fresh 0 EV RMS | 0.456 | 0.453 | ≤ 0.338 |
| fresh −3 / +3 / H | 485.6 / 0.317 / 0.0768 | 490.4 / 0.319 / 0.0774 | ≤ analytic |

- **On the development sets the hinge is what made the difference.** Training longer
  without it (w0) fails old24 0 EV and +3 EV like d6400 did; with it (w4) both pass.
  The cost is motion consistency: aligned residual 0.0078 → 0.0091, still inside its
  limit. The rising training diagnostic does not contradict this: it is measured on
  training crops, and w0 (which logs none) may have drifted further.
  One seed each and old24 is 14 frames, so the size of the effect is uncertain.
- **On the fresh set the hinge changes nothing** (0.456 vs 0.453). Every variant tried
  lands at 0.45-0.47 code there, 42 of 42 cases worse than the analytic inverse at 0 EV
  and better at −3 EV and on H.
- **The fresh set is 41 of 42 cases from one project** (IRAH Manchale song; one ELMATAREED
  case). Its verdict is about one production's footage, which limits how far it
  generalises in either direction. The confirmatory holdout must be stratified across
  projects and cameras (HOLDOUT_KIT.md does this) to say anything general.
