- New `mace_mode` input option for MACE QM runs: `accurate` (default, the
  exact e3nn reference) or `fast` (cuequivariance-accelerated kernels —
  substantially faster for MD, at the cost of not being bit-identical to the
  e3nn reference). `fast` requires `cuequivariance`, `cuequivariance-torch`
  and the matching CUDA ops package `cuequivariance-ops-torch-cuXX` (not pulled
  in automatically), where `XX` matches the CUDA build of the installed `torch`.
