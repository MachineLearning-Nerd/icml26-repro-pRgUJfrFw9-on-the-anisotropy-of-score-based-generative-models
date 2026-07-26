# Claims 1/4 equal-stratum geometry calibration

Status: **GEOMETRY CALIBRATED, NOT ACCEPTED FOR GENERATION**. Claims 1 and 4
remain BLOCKED.

Run `f310f5f6-4268-4295-b3ee-489a4aaf71aa`, Git
`2884f6f48c077c8497543d0d030e8bc0af5f79ea`, Hugging Face `cpu-upgrade`.
Estimated 32 cores / 4–10 minutes; actual 64 logical / 32 physical CPUs,
affinity 64. Full job 430 seconds; geometry checker 408.896 seconds; estimator
383.818 seconds.

The exact `D=256` iDDPM outer-product geometry from 8,000 initializations has
`lambda_max=116.876082`, `lambda_min=0.0168951391`, condition number
`6917.73`, orthonormality error `2.55e-15`, and selected eigen residual
`2.21e-13`. Independent Torch eigenvalues agree within `6.08e-16`; random
directions are rejected with median relative residual `0.936`.

The estimator is not adequate for training-direction selection. Independent
split-half vector overlap is `0.657` at the largest eigenvalue and `0.877` at
the smallest; most interior vectors are unstable. Equal allocation provides
only eight models at each timestep even though low-noise conditional outputs
dominate the estimator variance. A new independent-pilot variance allocation
is required.
