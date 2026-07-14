# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MORFEFerrite.jl provides the **Ferrite.jl FEM backends and high-level UIs for
MORFE.jl** (Direct Parametrisation of Invariant Manifolds). MORFE is
FEM-backend-agnostic — it owns only the abstract `FEMMultilinearMap` interface;
this package implements it for concrete physics domains as first-class
submodules (no `Base.get_extension`).

Both packages are unregistered: dev them by path (`test/bootstrap.jl` helper;
example drivers bootstrap on missing `Manifest.toml`). MORFE is expected as a
sibling checkout at `../../MORFE_jl`.

## Submodules

| Submodule | Contents |
|---|---|
| `Common` | `load_comsol_grid` (COMSOL `.mphtxt` → Ferrite grid), `ParaviewExport` stubs (`write_paraview_*`) |
| `StructuralSVK` | St. Venant-Kirchhoff backend (`FerriteGeometricNonlinearity`, `svk_nonlinearity`, `svk_assemble_KM!`) + "mesh → ROM" UI: `SVKMaterial`, `RayleighDamping`, `HarmonicForcing`, `mechanical_model`, `parametrise(::AssembledMechanicalModel)` (extends `MORFE.parametrise`), `real_dynamics`, `print_equations`, `save_rom` |
| `ParametricStructural` | **General multivariate parametric formulation**: additive map `x(θ,x₀) = x₀ + Σᵢ θᵢψᵢ(x₀)`, θ-series over a MORFE `MultiindexSet` **box (per-parameter truncation)** — `ThetaBasis`, `det_adj_series`, `reciprocal_series`, `ParametricGeometricNonlinearity{N_input,Nθ}`, `assemble_parametric_K_M!`, `build_{K,C,M}_corrections`. Geometry providers: analytic `geom(x₀)` or FE-field `geom(x₀, cell, cv, q)` |
| `FluidNavierStokes` | Incompressible cylinder-flow DPIM: `setup_fem` (P2/P1 Taylor-Hood), `solve_steady_state` (Newton), `assemble_linear_operators`, `FluidConvection`, `make_param_coupling`, `make_base_forcing`, energy-Gram/lift helpers. Mesh *generation* (Gmsh) stays example-local |

VTK export is the `MORFEFerriteWriteVTKExt` extension, activated by `using WriteVTK`.

## Tests & gates

```bash
julia --project -e 'using Pkg; Pkg.test()'   # small in-memory SVK Gate A/B
# Full-mesh SVK equivalence gates (run via the example env, NOT Pkg.test —
# Arpack noise floor on some machines is ~2e-9 vs the 1e-10 gate threshold):
julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl
```

Validation invariants (do not regress):
- SVK: high-level `parametrise` ≡ low-level pipeline (Gate A) — migrated code was
  verified bit-identical to the pre-split MORFE.
- ParametricStructural: N_θ=1 ≡ the proven arch kernels at machine precision;
  ex07 reproduces its reference; ex04's **corrected** two-parameter reference is
  `examples/04_parametric_clamped_beam/reference_data/` (the old MORFE ex04
  used buggy total-degree θ-truncation — never compare against it).
- Fluid: gauge-invariant Kármán quantities (λ, c₁₀₁, Im/Re of c₂₁₀) vs
  `examples/05_karman_vortex_street/reference_data/` (raw R is NOT
  run-comparable across Arpack runs).

## Examples

`examples/{01,03}` (StructuralSVK), `{04,07}` (ParametricStructural — 04 is the
corrected two-parameter beam, 07 the single-parameter arch), `05` (Kármán,
FluidNavierStokes), `08` (MEMS micromirror). Drivers accept
`MORFE_MAXZ`/`MORFE_MAXT` env overrides where noted.

## Known limitations

- The ParametricStructural θ-series kernel allocates in its inner loop; full
  high-order solves are slow (one-sweep-all-coefficients + input cache already
  landed; an SVK-O5-style allocation pass is the pending follow-up).
- `FluidNavierStokes` hardcodes the Turek–Schäfer channel geometry constants
  (must match the example's mesh generator).

## Coding conventions

Same as MORFE: SciML style (JuliaFormatter), UK English (`parametrise`,
`normalise`), `!` suffix for in-place, no aligned `=`.
