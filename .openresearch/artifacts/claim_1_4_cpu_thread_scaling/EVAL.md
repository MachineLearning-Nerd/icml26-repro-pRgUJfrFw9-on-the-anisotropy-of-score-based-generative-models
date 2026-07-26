# Evaluator note — CPU thread-scaling diagnostic

Status: **resource diagnostic only; Claims 1 and 4 remain BLOCKED**.

On the same 64-logical/32-physical-CPU HF host, median exact DSM update times
were 5.604, 3.513, 2.752, 3.780, 20.307, and 47.890 seconds for 1, 2, 4, 8,
16, and 32 intra-op threads, respectively. Four threads were 17.40× faster
than 32.

Both repetitions within every thread setting produced identical final model
SHA-256 hashes. Different thread settings changed reduction order: every
non-32 loss differed from the 32-thread reference by at most `6.79e-8`
relatively, but state hashes differed. The strict cross-thread equivalence
contract therefore admitted only 32 threads. The independent checker passed
all 12 integrity checks and rejected the zero-thread negative control.

The next resource round adopts four threads identically for both scientific
endpoints; it will never compare a four-thread result with a 32-thread result.
