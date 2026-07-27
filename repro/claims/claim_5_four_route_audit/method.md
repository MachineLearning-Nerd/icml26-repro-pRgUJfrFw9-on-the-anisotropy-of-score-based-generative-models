# Four-route method

1. **Exact-setup CPU feasibility.** Construct each exact author iDDPM config,
   use the paper batch size 500, and time a warm-up plus one optimizer update
   on a same-shape deterministic tensor. This tests execution capability only:
   it uses no real dataset, transform, complete horizon, generation, or metric.
2. **Primary public-artifact recovery.** Audit the author repository releases,
   tags, tracked files, and Hugging Face searches for checkpoints or raw
   five-seed outputs. Absence cannot establish a scientific verdict.
3. **Independent paper-result audit.** Recompute the direction and magnitude
   of each published Figure 9 `W_min` versus `I` difference from the primary
   figure. These are paper evidence, not newly observed evidence, and lack
   seed-level uncertainty.
4. **Falsification route.** Restate the exact assumptions and search the
   primary aggregate values for a contradicting dataset. A valid
   falsification would need `W_min >= I` (or direct regenerated contradictory
   evidence) under the exact setup. An implementation failure or unavailable
   checkpoint does not count.

The independent checker recomputes projections and published-bar comparisons,
checks all route labels, and requires the deliberately invalid “one update
verifies Claim 5” control to be rejected.
