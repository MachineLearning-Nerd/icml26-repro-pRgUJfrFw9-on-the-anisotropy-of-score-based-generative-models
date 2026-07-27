# Method — CPU thread scaling

The verifier evaluates intra-op thread counts 1, 2, 4, 8, 16, and 32 in two
oppositely ordered sweeps. Every trial reconstructs the same exact author
iDDPM, uses the same largest-SAD batch and model/DSM seeds, and performs three
Adam updates. The first update warms the thread pool and is excluded; the last
two are timed.

A thread count is eligible only if both repetitions have the exact same final
model-state SHA-256 as the 32-thread reference and every loss differs by at
most `1e-7` relatively. The fastest eligible count is selected by the median
of its four timed updates. An independent NumPy checker reconstructs all
medians, equivalence decisions, and the selection from raw records without
importing the primary verifier.

The negative control proposes zero intra-op threads and must be rejected. The
diagnostic never samples a model, computes SW2/MSW2, or assigns a claim result.
