# Method — paper-scale exact paired endpoint aggregation

The directions are taken from the accepted one-million-network geometry
estimator, formed by summing two independently generated 500,000-network
shards. The ten frozen terminal summaries are grouped by predeclared paired
seed and endpoint. The verifier requires one largest and one smallest endpoint
for each of seeds 0–4, matching paired seed identifiers, unique run IDs, finite
metrics, and successful exact-setup and independent-checker flags.

For each metric it computes largest minus smallest. Uncertainty is not
Monte Carlo: the verifier enumerates all `5^5 = 3,125` ordered paired bootstrap
resamples and takes the linearly interpolated 2.5th and 97.5th percentiles.
Acceptance requires all five observed differences and both interval lower
bounds to be positive.

The negative control reverses largest and smallest labels. It must fail the
same positive-effect rule. A separate checker independently parses the frozen
inputs and reconstructs every difference, mean, interval, and control result
without importing the primary verifier.

The leading direction passes the preregistered cross-shard individual-vector
stability gate. The smallest full-estimator direction does not. The experiment
uses that direction without outcome-dependent reselection and records the
instability as a limitation rather than silently replacing it.
