# Adaptive geometry source audit

The estimand is unchanged from Definition 2 and Conjecture 1:

`G_F=E_theta,t[F_theta(0,t)F_theta(0,t)^T]`,

where `theta` is the author iDDPM default initialization and `t` is uniform
over all 1000 linear-schedule diffusion levels. The output is the epsilon
prediction divided by `sqrt(1-alpha_bar_t)`, matching the paper's score
normalization.

The paper uses one million iid initialized networks. This estimator uses an
independent 2,000-network allocation pilot and 16,000 fresh final networks.
It estimates every timestep-conditional expectation separately and gives each
exact weight `1/1000`; therefore its unequal conditional sample counts do not
change the source statement's uniform marginal.

The gate only decides whether endpoint directions are stable enough to enter
the Figure 6 training experiment. It does not use generation outcomes and
cannot itself verify the conjecture.
