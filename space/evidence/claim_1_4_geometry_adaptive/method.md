# Variance-allocated geometry method

The equal-stratum 8,000-network pilot showed that geometry variance, not matrix
algebra, limited direction stability. This replacement uses two stages.

First, two independent initialized networks are evaluated at each of the 1000
timesteps. Their outer-product difference norm supplies an empirical
conditional variability proxy. These 2,000 pilot outputs are discarded.

Second, 16,000 fresh networks are allocated in groups of four. Every timestep
receives at least four; the remaining groups follow the pilot proxy by
deterministic largest-remainder allocation. Each of four independent quarters
receives exactly the same per-timestep count. Within a quarter, the conditional
outer products are averaged at each timestep; the 1000 conditional means are
then averaged with equal `1/1000` weights. Unequal sample allocation therefore
changes variance but not the exact uniform-probe estimand.

Alternating quarters form independent split halves. Readiness is predeclared
only for the top and bottom vectors/bands because the paper's key endpoint
claim can be tested without pretending individual vectors inside
near-degenerate interior eigenspaces are identifiable.
