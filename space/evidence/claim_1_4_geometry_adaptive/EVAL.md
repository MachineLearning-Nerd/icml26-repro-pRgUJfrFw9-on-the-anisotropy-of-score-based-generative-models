# Adaptive iDDPM geometry evaluation

This is a direction-selection gate, not a generation result and not a verdict
for Claims 1 or 4.

The exact D=256 author iDDPM architecture was evaluated with a discarded
2,000-network allocation pilot and 16,000 fresh final networks. The largest
and smallest SADs passed the preregistered endpoint stability gate:

| Audit | Largest endpoint | Smallest endpoint |
|---|---:|---:|
| Eigenvalue | 133.58986350488536 | 0.09507780044162173 |
| Split-half vector overlap | 0.9984578458723848 | 0.9005352938586038 |
| Five-vector band mean cosine | 0.988422061376246 | 0.9735476866409851 |

The independent Torch spectrum check had relative error
`5.318837201552464e-16`. A random-direction control was correctly rejected
with median relative eigen residual `0.8969067557074499`.

HF `cpu-upgrade` supplied 64 logical / 32 physical CPUs (affinity 64). The
estimator ran for `1657.9766` seconds and the complete checker for
`1852.1567` seconds. No GPU was used.

The complete machine-readable matrix, selected vectors, allocations,
bootstraps, checker result, and negative control are in
[`raw_geometry.json`](raw_geometry.json).
