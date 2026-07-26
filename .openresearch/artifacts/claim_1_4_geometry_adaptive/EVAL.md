# Adaptive iDDPM geometry evaluation

Status: **DIRECTIONS READY FOR TRAINING**, not a verdict for Claims 1 or 4.

The exact author D=256 iDDPM architecture and uniform default probe were
evaluated with a discarded 2,000-network allocation pilot followed by 16,000
fresh networks. The final estimate used at least four networks at every one of
1,000 timesteps and four independent quarters.

- Largest SAD split-half vector overlap: `0.9984578458723848`
- Smallest SAD split-half vector overlap: `0.9005352938586038`
- Largest/smallest endpoint band mean cosine: `0.988422061376246` /
  `0.9735476866409851`
- Eigenvalue range: `133.58986350488536` to `0.09507780044162173`
- Independent Torch eigenspectrum relative error: `5.318837201552464e-16`
- Random-direction negative-control median eigen residual: `0.8969067557074499`
- HF cpu-upgrade allocation: 64 logical / 32 physical CPUs, affinity 64
- Estimator runtime: `1657.9766111620702` seconds
- Whole checker runtime: `1852.1566860929597` seconds

Only the stable endpoint vectors are authorized for the first generation
round. The intermediate individual eigenvectors did not meet comparable
split-half stability and are not treated as validated directions.
