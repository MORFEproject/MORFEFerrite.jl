# Reference provenance

> **The two entries below are SUPERSEDED.** They describe the deleted
> `ParametricStructural` module, which no longer exists. `R_coefficients_fast_z5_t3.csv`
> was regenerated on 2026-08-25 from `ParametricGeometry` — see the dated entry at the
> end of this file. `R_coefficients_ref.csv` (FULL) has **not** been regenerated and is
> still historic; compare against it only with `rtol ≥ 1e-3`.

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

## R_coefficients_fast_z5_t3.csv — regenerated 2026-08-25

- profile: **FAST** (z ≤ 5, θ ≤ 3)
- code: commit `cb4d910`
- validated by: `test/ParametricGeometry/test_moved_mesh_fom.jl` (3D) and
  `test_moved_mesh_fom_2d.jl`, both passing at the time of blessing — the
  parametric transform against a FOM whose geometry actually moved.
- this file is a CHANGE DETECTOR. It proves reproducibility, not
  correctness; the gate above is what establishes correctness.

## R_coefficients_fast_z5_t3.csv — regenerated 2026-08-25

- profile: **FAST** (z ≤ 5, θ ≤ 3)
- code: commit `cb4d910`
- validated by: `test/ParametricGeometry/test_moved_mesh_fom.jl` (3D) and
  `test_moved_mesh_fom_2d.jl`, both passing at the time of blessing — the
  parametric transform against a FOM whose geometry actually moved.
- this file is a CHANGE DETECTOR. It proves reproducibility, not
  correctness; the gate above is what establishes correctness.
