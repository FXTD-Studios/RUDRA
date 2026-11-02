# Decision: is a sub-code 0 EV deviation a release blocker? (2 Nov 2026)

**For: Ahmed (FXTD Studios). Status: open. To be decided in writing before the
confirmatory holdout is built or any new data is looked at.**

## The question

The learned ACES 2.0 model is labelled preview because of one gate: at the
delivered exposure (0 EV), its highlight pixels must sit no further from the
reference than the plain analytic inverse does. Every attempt so far fails it
on the fresh FXTD set, while every other measurement beats the analytic inverse.
Should that gate block release as it stands?

## What has been measured (all development evidence)

| | analytic | d6400 (shipped preview) | w4 (bin hinge) |
|---|---|---|---|
| fresh 0 EV, RMS in 8-bit codes | 0.24 | 0.47 | 0.45 |
| fresh −3 EV grade (lower is better) | 526.5 | 485.8 | 490.4 |
| fresh +3 EV grade | 0.393 | 0.333 | 0.319 |
| fresh H (highlight colour) | 0.0832 | 0.0770 | 0.0774 |
| old24 0 EV (limit 0.0604) | 0.0604 | 0.0648 | **0.0537** |
| aligned residual (limit 0.0104) | | 0.0085 | 0.0091 |

Hard guarantee independent of all this: the appearance constraint keeps every
pixel of every master within 1 code of the source through the ACES 2.0 SDR
view (tested in CI on the published model, stills and video).

Tried and failed to close the fresh 0 EV gap: routing (no fully clipped pixels to
route), inference-time tightening (breaks motion consistency), a training hinge
(did not act). Untried: real-footage training data.

## The case for keeping the gate as it is

- **It is the honest promise.** RUDRA's pitch against generative tools is
  fidelity: at the delivered exposure the result should be no worse than doing
  nothing. 0.45 code against 0.24 is nearly twice the deviation.
- **Grading moves the error.** A colourist who lifts the master by a stop or two
  re-exposes those highlights; a sub-code deviation at 0 EV is not guaranteed to stay
  sub-code after a grade. (The −3 / +3 EV grades bound this, and the model wins
  there, but they test two exposures, not every grade.)
- **Changing a gate after seeing results is how evaluation goes wrong.** This
  gate was set before any of these models existed.
- **The development and fresh sets disagree.** That is a sign the model has not
  learned something about real footage; relaxing the gate hides it.

## The case for changing it

- **The deviation is below one 8-bit step.** At the delivered exposure 0.45 code
  RMS is not visible on an SDR display; 1 code is the smallest difference an 8-bit
  pipeline can show.
- **The constraint already bounds the worst case.** No pixel can be more than 1 code
  off; the gate is about the mean, not about any visible pixel.
- **The strict form is near-impossible by construction.** At 0 EV the source is the
  quantised view of the reference, so the analytic inverse sits almost exactly on
  it. Any model that changes partially clipped pixels at all pays for it at 0 EV,
  while that change is what buys the −3 EV, +3 EV and H wins.
- **The trade users get is better everywhere they will look:** stopped down,
  pushed, highlight colour.

## Options

1. **Keep the gate.** The learned model stays preview; the analytic route stays
   the default; the next work is real-footage training data.
2. **A perceptual 0 EV rule**, decided now, applied to the holdout once: mean 0 EV
   deviation ≤ **0.5 code RMS** (below one 8-bit step), with every other gate
   unchanged and the 1-code per-pixel constraint as the hard bound. On present
   evidence w4 (0.45) and d6400 (0.47) would pass the fresh set; the holdout decides.
3. **Ship both, named for what they are:** the analytic route as the fidelity
   default; the learned model as "highlight recovery" with its 0 EV figure
   published in the model card, not called a fix.

Whatever is chosen: the rule is written into a registration **before** the holdout
is built, with this note as its reason, and the holdout is run once.

## Decision

- [ ] Option 1   - [ ] Option 2   - [ ] Option 3   - [ ] Other: ____________

Decided by: ____________   Date: ____________   Reason in one line: ____________
