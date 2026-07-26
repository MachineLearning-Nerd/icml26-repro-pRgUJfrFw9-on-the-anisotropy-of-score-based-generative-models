# Claims 1/4 geometry limitations

- The paper used one million initialized networks; this calibrated estimator
  uses 8,000. Split-half and checkpoint stability quantify the resulting
  uncertainty but cannot make the sample counts equivalent.
- Individual eigenvectors are not identifiable inside a nearly degenerate
  eigenspace. Band-subspace overlap is therefore reported alongside vector
  overlap.
- Stratified timesteps preserve the exact uniform probe expectation but differ
  from iid timestep sampling in estimator variance.
- This node establishes an empirical SAD basis for the exact architecture. It
  cannot verify generation preference until networks are trained and sampled
  on the committed directions.
