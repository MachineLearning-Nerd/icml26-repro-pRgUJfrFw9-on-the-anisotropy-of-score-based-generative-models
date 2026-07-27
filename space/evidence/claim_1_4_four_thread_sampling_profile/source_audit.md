# Source and scope audit

This calibration follows the Figure 6 setup in arXiv:2510.22899v1:
`1×16×16` iDDPM, 1,000 diffusion steps, batch 1,000, and 10,000 generated
samples. It executes one complete batch and projects the cost of ten batches.

The model is intentionally untrained because this node measures execution
cost, not generation quality. Architecture, tensor shapes, and the reverse
loop are exact. No SW2/MSW2 value is computed, and Claims 1 and 4 remain
BLOCKED.
