# 05 — Kármán vortex street

## Model

Incompressible Navier-Stokes flow past a cylinder (2D, Ferrite P2/P1 Taylor-Hood,
Turek–Schäfer channel geometry). The steady base flow is computed first; then DPIM
parametrises the unstable spectral submanifold associated with the Hopf bifurcation
(Re_c ≈ 49) that gives rise to the Kármán vortex street, with the Reynolds-number
offset `η′ = 1/Re − 1/Re₀` as an extra parametric coordinate. Parameters are set in
`config.jl` (default: Re₀ = 49.03, expansion order 9).

## Workflow

```bash
cd examples/05_karman_vortex_street

# 1. One DPIM run at order MAX_ORD
julia --project=. main.jl

# 2. ROM limit-cycle branch vs Re, one rom_branch_ord<N>.csv per truncation order
julia --project=. solve_rom.jl

# 3. Order comparison: max|lift| and period-averaged TKE vs Re per truncation
python3 compare_orders.py     # needs numpy + matplotlib

# 4. (optional) FOM reference for the Hopf tube: converge the actual full-order
#    periodic orbit at each Re in FOM_REF_RE and record the same observables;
#    rerunning step 3 then overlays the points as black stars
julia --project=. fom_reference.jl
```

A single run at `MAX_ORD` suffices for the whole order-convergence study: the
cohomological solve is graded (degree-N coefficients never depend on degrees > N), so
the order-N ROM is the exact truncation of the order-9 one (verified bit-exact).
Lower orders are obtained by truncating (1) the reduced dynamics `R` when tracing the
bifurcation diagram and (2) the observables — lift polynomial and TKE Gram — when
evaluating the physical quantities, per `TRUNC_ORDERS` in config.jl.

Measured runtime (Apple Silicon, ~58k free DOFs): ≈ 7 min — shared stages ≈ 35 s +
order-9 cohomological solve ≈ 344 s. Steps 2 and 3 take seconds.

Step 4 is the expensive ground truth (~10–25 min per Re point): each point runs
undamped spin-up, a lift-autocorrelation period probe (gated to [0.5, 2]×T_ROM),
and Picard orbits of the full perturbation NSE until the mass-norm anchor
residual drops below FOM_REF_TOL; the reported period carries a final
tangent-projection correction, so it is FOM-measured rather than the ROM seed.
Seeding uses the order-FOM_REF_SEED_ORD branch (default 7: the order-9 branch
folds back before Re 56; the seed only sets the transient length, never the
converged orbit).

Periodicity is then VERIFIED by its definition on a two-period measurement:
`periodicity_max` = max_t ‖v(t+T) − v(t)‖ / max_t ‖v(t)‖ (velocity mass norm,
uniform over the whole period — anchor closure alone would miss transient
growth within the period), alongside `floquet_mu` (the measured transversal
Floquet multiplier), `dist_rel_est` = rel/(1−μ) (the distance-to-cycle bound
that a single anchor residual under-reports by 1/(1−μ), ≈ 30× at Re 49.5), and
`lift_amp_drift` (observable stationarity between the two periods). Pressure
is excluded from the state norm: under Crank–Nicolson the algebraic pressure
mode rings at the Nyquist frequency without decay, inflating raw pressure-lift
extrema — lift observables therefore come from the harmonic projection of the
one-period signal (j ≤ 10), the `lift_ringing` column reports the removed
pollution amplitude, and `data/fom_lift_Re*.csv` stores raw vs projected
traces per point. Near Re_c the ROM seed
is most accurate exactly where Floquet contraction is slowest, so the orbit
counts stay moderate. Seeding from the ROM does not bias the reference — Picard
converges to the FOM's own attracting cycle.

## Outputs

```text
results/
  Re49.03_ord9/                — the single DPIM run directory
    summary.log                — tee'd run log
    summary.txt                — structured key:value summary
    data/
      W.jls, R.jls             — parametrisation + reduced dynamics (serialised)
      reduced_dynamics.txt     — realified Stuart-Landau coefficients
      R_coefficients.csv       — complex reduced dynamics, one monomial per row
      L_coefficients.csv       — pressure-lift polynomial L(z) (+ base-flow constant)
      lift_polynomial.jls      — same, serialised
      tke_gram_{re,im}.csv     — kinetic-energy Gram matrix G = WᵀM_velW/|Ω|
      tke_avector.csv          — monomial exponents for G
      rom_branch_ord{3,5,7,9}.csv — (from solve_rom.jl)  eta, Re, rho, omega, T
      fom_reference.csv        — (from fom_reference.jl)  FOM orbit per Re point
      linear_ops.jls           — (from fom_reference.jl)  cached B₀, B₁, K_visc, h₀
      vtk_data.jls             — mesh + mode bundle for ParaView export
  comparison/                  — (from compare_orders.py)
    comparison.csv             — order, eta, Re, rho, omega, T, avg_TKE, max_abs_lift
    lift_vs_Re.png, tke_vs_Re.png
```

## Files

| File | Purpose |
| ---- | ------- |
| `main.jl` | Step 1 — single DPIM run at MAX_ORD |
| `solve_rom.jl` | Step 2 — ROM limit-cycle branch (PALC) per truncation order |
| `compare_orders.py` | Step 3 — truncation-order lift / avg-TKE comparison |
| `fom_reference.jl` | Step 4 — FOM periodic-orbit reference per Re in FOM_REF_RE |
| `config.jl` | All parameters (Re₀, MAX_ORD, TRUNC_ORDERS, mesh, eigensolver, branch) |
| `fem/mesh.jl` | Gmsh channel-with-cylinder mesh generation (example-local; needs Gmsh) |
| `solver/eigensolver.jl` | Shift-invert ARPACK Hopf eigensolver |
| `solver/rom_palc.jl` | Pseudo-arclength continuation toolkit for the ROM branch |
| `solver/time_integration.jl` | IMEX θ-method FOM integrator (perturbation NSE) |
| `solver/picard_orbit.jl` | ROM-seeded Picard iteration for FOM periodic orbits |
| `validation/average_tke.py` | TKE evaluation library (used by compare_orders.py) |
| `validation/run_tke.py` | Single-orbit TKE runner (cross-check) |
| `validation/validate_tke.jl` | Independent FOM-space TKE check |
| `validation/generate_matlab.py` | Optional matcont/COCO export (`EXPORT_MATLAB = true`) |

The fluid FEM layer itself lives in the package as
[`MORFEFerrite.FluidNavierStokes`](../../src/FluidNavierStokes/)
(`setup_fem`, `solve_steady_state`, `assemble_linear_operators`,
`FluidConvection`/`make_param_coupling`/`make_base_forcing`, energy-Gram and
lift helpers) — the driver imports it instead of carrying its own
`fem/{fem_setup,linear_operators,fluid_maps,energy_gram}.jl` +
`solver/steady_state.jl`.

## Validation

- `julia --project=. validation/validate_tke.jl results/Re49.03_ord9` recomputes the
  period-averaged TKE by direct integration in the full DOF space (independent of the
  Gram-matrix path).
- `python3 validation/run_tke.py --data-dir results/Re49.03_ord9/data --orbit <csv> --eta <η′>`
  evaluates a single orbit's TKE.
- Setting `EXPORT_MATLAB = true` in `config.jl` additionally emits
  `vec_fields_karman.m` / `lift_karman.m` per run for matcont/COCO cross-checks.
