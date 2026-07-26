# Claims 1/4 CPU calibration

Status: **CALIBRATION ONLY**. Claims 1 and 4 remain BLOCKED.

Successful run `43284fe9-76ff-4c27-9ba3-c99edb6f578f` at Git
`54469a6a583e14dddfaf344f79c0dc4c76743104`, Hugging Face `cpu-upgrade`.
The pre-run estimate was 32 cores and 3–10 minutes. The job reported 64
logical / 32 physical CPUs and affinity 64; Torch used 32 intra-op and 32
inter-op threads. Full job runtime was 44 seconds, profile checker 19.295439
seconds, and cumulative suite 31.068959 seconds.

Measured author-exact `D=256` iDDPM costs:

- one fresh initialized network plus default-probe forward: median
  `0.0222868 s`, projecting to `6.191 h` for the paper's one million;
- one batch-1000 DSM optimizer step: `3.618896 s`, projecting to `2.010 h`
  for 2,000 steps;
- one batch-1000 epsilon forward: `0.903503 s`, projecting to `2.510 h` for
  the 10,000 calls needed for 10,000 samples at 1,000 reverse steps;
- peak RSS: `2528.4 MB`.

The estimates plan compute only and are not evidence for anisotropy or
generation preference.
