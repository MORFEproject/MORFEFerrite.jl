# Reference provenance

- `R_coefficients_ref.csv` — historic FULL run (z ≤ 11, θ ≤ 7) from the original
  MORFE ex07 code, a different session's Arpack gauge. Gauge-invariant and
  low-order rows match the current code at 1e-8..1e-10; near-resonant high-order
  rows deviate up to ~2e-4 through order-by-order gauge-noise amplification
  (same phenomenon as the documented Kármán raw-R rule). Compare against it
  only with rtol ≥ 1e-3.
- `R_coefficients_fast_z5_t3.csv` — blessed from the current general
  ParametricStructural pipeline (MORFE_FAST=1, z ≤ 5, θ ≤ 3), the code whose
  kernels are machine-precision-validated against the analytic arch series and
  whose pipeline reproduces example 04's blessed reference bit-identically.
  validate.jl compares FAST runs against this at tight tolerance.
