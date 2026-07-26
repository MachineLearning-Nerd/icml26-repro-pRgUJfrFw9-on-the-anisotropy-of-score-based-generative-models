# Claim 6 method

The route has three independent layers.

1. Reconstruct each proof from conditional means, covariances, output-channel
   exchangeability, and the zero-mean iid contraction
   `E[(W a)_r (W c)_s]=delta_rs variance(W) a^T c`.
2. Execute actual batched PyTorch layers with nonlinear MLP and convolutional
   outputs and a shared-token linear transformer output. Accumulate the raw
   output outer products, project them onto the claimed structured matrix
   families, and report block-level Monte Carlo uncertainty. The transformer
   comparison uses the independently accumulated hidden-state second moment
   and declared parameter variances.
3. Independently enumerate complete finite Rademacher parameter domains for
   small MLP, circular-convolution, and shared-token layers. This checker does
   not call the primary Monte Carlo implementation.

Controls deliberately break a necessary assumption: one MLP/CNN output
coordinate receives a different scale, and transformer weights receive a
nonzero mean. The same structural projections must then show a large residual.

Sample counts, dimensions, tolerances, and seeds are committed before the run;
none is selected from an observed residual or from a formula-derived budget.
