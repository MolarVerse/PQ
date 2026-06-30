- New `mace_mode` input option for MACE QM runs: `accurate` (default, the
  exact e3nn reference) or `fast` (cuequivariance-accelerated kernels —
  substantially faster for MD, at the cost of not being bit-identical to the
  e3nn reference). `fast` requires the `cuequivariance` and
  `cuequivariance-torch` packages.
