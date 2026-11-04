# Result: the learned shadow branch, edge supervision, 5 seeds (run Oct 2026, read 4 Nov 2026)

Registration `configs/experiments/aces_shadow_edge_seeds_20261024.json`; summary
`G:\rudra_training\aces_shadow_edge_seeds_20261024_summary.json`.

**Concluded: no effect. The learned shadow branch is not pursued.**

- Edge over control, natural shadow MSE, mean of 5 seeds: **0.9998** (registered bar: 5 % below).
- Edge wins on 3 of 5 seeds, by tiny margins (bar: 4 of 5).
- **0 of 10 runs** pass every candidate gate; which check fails differs from seed to seed.
- The 24 Oct single-seed 5 % gain was seed noise. The development set has only 7
  shadow-eligible frames from 3 scenes, too few to separate the arms.

Shadow work instead (4 Nov 2026, Ahmed: option a): **Clean shadows**, an optional,
off-by-default edge-preserving denoise inside the shadow band of the learned route,
held to the 1-code appearance constraint (no invention; recovering crushed detail
stays with AI fill's τ low, experimental).
