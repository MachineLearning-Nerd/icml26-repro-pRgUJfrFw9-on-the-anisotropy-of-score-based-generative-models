# Claims 1/4 geometry source audit

The paper defines the average geometry as the output **outer-product**
expectation

`G_F(P,Theta)=E[F_theta(x_sigma,sigma)F_theta(x_sigma,sigma)^T]`.

Under the default probe it fixes `x_sigma=0`, draws the noise level uniformly,
and draws network parameters from the architecture's default initialization.
Conjecture 1 orders the eigenvectors in ascending eigenvalue order as SADs and
hypothesizes that small-eigenvalue-aligned data is modeled better.

Figure 6 uses the paper's iDDPM U-Net at `1×16×16` (`D=256`), 1000 linear
diffusion steps, base channels 32, channel multipliers 1/1, and residual depth
1. The paper estimates geometry with one million random networks.

This node preserves the exact architecture, dimension, initialization, score
normalization, zero-input probe, and uniform timestep marginal. It substitutes
8,000 networks, using timestep stratification as unbiased variance reduction.
The substitution is accepted only as a geometry calibration and is accompanied
by convergence and split-half audits. It does not itself verify the
generation-preference conjecture.
