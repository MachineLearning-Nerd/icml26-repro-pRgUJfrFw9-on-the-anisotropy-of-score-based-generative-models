# Limitations

- Two timed steps characterize a large order-of-magnitude effect but are not a
  general CPU benchmark.
- Condition order is fixed and therefore may retain modest thermal/frequency
  order effects; the sought mechanism requires at least a twofold effect.
- Flushing subnormals changes floating-point semantics for values below the
  normal float32 range. It is acceptable for a full rerun only if losses and
  updates remain numerically indistinguishable at the preregistered tolerance.
- This diagnostic does not test generation quality and cannot change Claims 1
  or 4 from BLOCKED.
