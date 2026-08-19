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
| `Common` | `load_comsol_grid` (COMSOL `.mphtxt` → Ferrite grid), `ParaviewExport` stubs (`write_paraview_*`), the `AbstractAssembledModel`/`build_model` contract, and `write_summary` — one run-summary writer owning the shared skeleton (sizes, order, and every `*_time_s` stage found in `case.info`/`rom.info`), extended per physics by adding a `summary_entries(case, rom)` method. Same arrangement as MORFE's `print_setup`: a module contributes rows, it does not reimplement the writer |
| `StructuralSVK` | St. Venant-Kirchhoff backend (`FerriteGeometricNonlinearity`, `svk_nonlinearity`, `svk_assemble_KM!`) + "mesh → ROM" UI: `SVKMaterial`, `RayleighDamping`, `HarmonicForcing`, `mechanical_model`, `parametrise(::AssembledMechanicalModel)` (extends `MORFE.parametrise`), `real_dynamics`, `print_equations`, `save_rom` |
| `ParametricGeometry` | **Physics-blind parametric mesh coordinate transforms.** Additive map `x(θ,x₀) = x₀ + Σᵢ θᵢψᵢ(x₀)`, expanded as power series in the Jacobian's determinant, adjugate and inverse determinant over a MORFE `MultiindexSet` **box (per-parameter truncation)**: `GeometryParameterBasis`, `det_adj_series`, `reciprocal_series`, `PullbackCache`. **Connects to a physics module through `AbstractPullbackKernel`** (`det_weight_power`, `qp_prepare`, `qp_integrand!`, `linear_qp_series!`) — it names no material, stress law or strain measure. Owns the assembly driver (`ParametricMap`, `sweep_all!`, `multilinear_maps`), the arity-generic `build_linear_corrections`, and the `build_model` contract. Geometry providers: analytic `geom(x₀)` or FE-field `geom(x₀, cell, cv, q)`. **Included before the physics modules — they depend on it, not the reverse.** |
| `FluidNavierStokes` | Incompressible cylinder-flow DPIM, full "mesh → ROM" UI. Backend: `setup_fem` (P2/P1 Taylor-Hood), `solve_steady_state` (Newton), `assemble_linear_operators`, `FluidConvection`, `make_param_coupling`, `make_base_forcing`, energy-Gram helpers. UI: `fluid_model` (assembles, and applies the `−D` scaling of `K_visc`/`h₀`), `build_model` (ORD = 1; Hopf eigensolve, model, `SpectralData`, derived `[2,1,3]` permutation), `parametrise(::AssembledFluidModel)` → `FluidROM`, `lift_functional`/`lift_polynomial`. `solve_hopf_eigenproblem` is standard KLU shift-invert Arpack; its **gauge is explicit** — `normalisation` (`SymmetricBiorthogonal` default / `LeftBiorthogonal` / `NoNormalisation`) and `scale` (`1e-2` for Kármán). Mesh *generation* (Gmsh) stays example-local |

VTK export is the `MORFEFerriteWriteVTKExt` extension, activated by `using WriteVTK`.

## Local hardware limits

**Do not run `examples/05` without `MORFE_FAST=1` on the development machine**, and do not
run heavy Julia jobs concurrently. The FULL Kármán profile is 57,860 free DOFs at order 9
(`W` ≈ 220 MB plus the solver working set); several such runs in parallel froze the machine
and required a reboot. FAST is 14,436 DOFs at order 3, `W` ≈ 11 MB, ~1 minute.

The package suite, the MORFE suite and the SVK gates are each light — run them **one at a
time** rather than backgrounding several. Order-9 / full-mesh work belongs on other
hardware.

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
- ParametricGeometry: the physics-blind driver + `SVKPullbackKernel` ≡ the previous
  hardcoded-SVK kernels bit-for-bit — ex07's R coefficients agree to **0.0**, and
  the unit gates cover a curved J₀ ≠ I with per-form bases as well as J = I ≡
  non-parametric StructuralSVK. ex04's **corrected** two-parameter reference is
  `examples/04_parametric_clamped_beam/reference_data/` (the old MORFE ex04
  used buggy total-degree θ-truncation — never compare against it).
- ⚠ **BOTH parametric references are stale.** Neither ex04 nor ex07 reproduces its
  blessed reference, and this is PRE-EXISTING: the previous ParametricStructural
  pipeline, reconstructed and re-run, deviates by *exactly* the same amount as the
  current code (new ≡ old at 0.0 for both examples). Do not treat either reference
  as authoritative until re-baselined.
  - ex07: max rel dev 9.305635786432396e-5, all 16 rows shared.
  - ex04: 6 ref-only rows, max rel dev 1.0. The disagreement obeys a clean
    **parity rule in θ₂**: all 12 θ₂-ODD monomials are numerically zero in the
    current code (≤6e-14) while the reference has 10 of them nonzero at 1e-6…1e-11;
    all 12 θ₂-EVEN monomials agree, at healthy 1e-2 magnitudes.
    ψ₂ = ∇φ₁ is the antisymmetric first bending mode of a SYMMETRIC straight beam,
    so odd powers of it should integrate to zero — i.e. the current zeros look
    physically right and the reference's small values look like spurious residue.
    Confirm that symmetry argument before re-blessing.
  - The recorded provenance is not traceable: the archived runs name
    `morfe_commit 273fab5`, which is not a valid object in MORFE_jl's history.
- Fluid: gauge-invariant Kármán quantities (λ, c₁₀₁, Im/Re of c₂₁₀) vs
  `examples/05_karman_vortex_street/reference_data/` (raw R is NOT
  run-comparable across Arpack runs).
- Fluid **conjugate symmetry**: `Φ[:, σ(r)] = conj(Φ[:, r])` and likewise `Ψ`, asserted inside
  every `build_model`. Never call `left_eigenvector` for both halves of a pair — `eigs` pins the
  left vector only up to a scalar, so the second solve returns a phase-rotated partner and the
  reduction then exploits a symmetry that is not there, **silently**: λ and everything linear in
  the master coordinate stay exact while the nonlinear coefficients keep their modulus and rotate
  in phase. Fill the partner by conjugation instead. `test/FluidNavierStokes/test_conjugate_pairing.jl`
  covers the two helpers mesh-free.
- Fluid **degenerate real modes**: most near-real Kármán modes have `|ψᵀBφ| ≲ 1e-11` and cannot
  carry a master coordinate (the gauge divides by √α); they sit in the essential spectrum, where
  Arpack's left and right vectors for the same λ are not the same mode. `left_eigenvector` warns
  below 1e-8 and names the mode — that warning means "drop it from `master`", not noise. At
  Re₀ = 49.03 only λ ≈ −2.11 is comfortably promotable.

## Examples

`examples/{01,03}` (StructuralSVK), `{04,07}` (ParametricGeometry + StructuralSVK — 04 is the
corrected two-parameter beam, 07 the single-parameter arch), `05` (Kármán,
FluidNavierStokes), `08` (MEMS micromirror). Drivers accept
`MORFE_MAXZ`/`MORFE_MAXT` env overrides where noted.

**ex05 has ONE results directory and ONE reference.** `RESULTS_DIR` lives in `config.jl` (not
`main.jl`, which is where it drifted from `validate.jl` and caused a days-old directory to be
validated silently); it is `results/Re<Re₀>_ord<N>/`, untagged, so `validate.jl`,
`solve_rom.jl` and `fom_reference.jl` all agree.

The reference is `reference_data/karman_invariants_ref.txt` — **physical invariants, not raw
R**: λ = σ+iω, c₁₀₁, the effective Landau coefficient c₂₁₀ and its gauge-free ratio
Im/Re. It validates a run **with or without promoted outer modes**, because `invariants.jl`
SLAVES the promoted coordinates (solving R_k = 0; R is exactly affine in them) before reading
the coefficients off. Zeroing them instead drops the mean-flow distortion and misreports
criticality — `test_invariants.jl` proves the invariance analytically. `R_coefficients_ref.csv`
is kept only as the archived order-9 output the invariants were derived from; nothing reads it.

`validate.jl --bless` rewrites the reference from the current run, and refuses to do so from
the FAST profile (different mesh). It also warns when the artefact predates `src/FluidNavierStokes/`.

**Two analyses.** `MORFE_PROMOTE=1` promotes the resonant outer modes into the master set at
first order and writes to `…_promoted/`; without it only the Hopf pair is master. Promotion is
a change of coordinates, so both must give the same physics — `compare_runs.jl` checks that via
`invariants.jl`. λ at the bifurcation must be **exact** between them (the promoted coordinates
slave to zero at ρ→0, η′=0); `c₁₀₁` and `c₂₁₀` agree only up to the `PROMOTED_CORE_ORD`
truncation of `y_k`, and that residual is the result of interest. Criticality must match.

**Not every resonant mode can be promoted.** `α = ψᵀB₁φ` is the normalisation denominator
(both sides ÷ √α) and `B₁` is **singular** — 6636 of 57860 rows are P1 pressure DOFs, so this is
a descriptor pencil whose null-space vectors are algebraic incompressibility modes with α = 0.
`main.jl`'s Pass 1.5 measures α for every candidate and drops those below
`PROMOTE_ALPHA_RTOL · |α_Hopf|` (relative, since |α_Hopf| ≈ 6e-6 depends on mesh and Arpack
scaling), logging a full table. A conjugate pair always shares |α|, so the filter cannot split
one. `compare_orders.py` emits **one figure per run** and refuses to mix meshes.

## Known limitations

- The ParametricGeometry series kernel allocates in its inner loop; full
  high-order solves are slow (one-sweep-all-coefficients + input cache already
  landed; an SVK-O5-style allocation pass is the pending follow-up).
- `FluidNavierStokes` hardcodes the Turek–Schäfer channel geometry constants
  (must match the example's mesh generator).

## Coding conventions

Same as MORFE: SciML style (JuliaFormatter), UK English (`parametrise`,
`normalise`), `!` suffix for in-place, no aligned `=`.
