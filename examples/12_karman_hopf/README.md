# 12 — Kármán vortex street (Hopf)

This notebook computes an invariant-manifold ROM for two-dimensional incompressible
flow past a cylinder. The flow loses stability at `Re_c ≈ 49` through a Hopf
bifurcation, the Kármán vortex street, and the reduction turns the 57,860-free-DOF
Taylor-Hood model into a single complex Stuart–Landau equation. The Reynolds number
rides along as a parametric coordinate, so one run at the expansion point describes
the whole bifurcation neighbourhood.

The example is intentionally limited to the public API:

```julia
order = 3 # Change to 9 for the reference calculation.
case = NSE.fluid_model("cylinder_flow.msh"; Re = 49.03)
spectrum = NSE.solve_hopf_eigenproblem(case.B; ...)
(; model, spectral, meta) = build_model(case, spectrum;
    master = master, outer = outer, expansion_order = order, scale = 1e-2)
W, R = parametrise(model, spectral, order;
    resonance = ResonanceConfig(style = :complex_normal_form, tol_relative = 0.1,
        outer_targets = true))
branch = normal_form_branch(R; parameter = 1, sheet = :primary, ...)
l_free, L0 = NSE.lift_functional(case)          # the one fluid-specific observable
L_coeffs, mset_L = NSE.lift_polynomial(W, l_free)
MORFE.save_rom(results_dir, W, R) # optional
```

There is no example-specific mesh generation, assembly, eigensolver, cohomological
solver, ROM wrapper or convergence diagnostic. The limit-cycle branch is MORFE's own
`normal_form_branch`, which needs nothing but `R`: with a single conjugate pair in
complex normal form the substitution `z₁ = ρ·exp(iθ)` clears the phase from the first
row, leaving `ρ̇ = Re R₁(ρ, ρ, η′)` and `Ω = Im R₁(ρ, ρ, η′)/ρ`, and at fixed `ρ` that
first equation is a polynomial in `η′` solved by companion matrix. Only the lift is
fluid-specific.

For the outer-mode promotion study (mode promotion, near-resonance tables, fold
analysis, Padé-resummed branches and the DNS reference) see
[`05_karman_vortex_street/`](../05_karman_vortex_street/).

## Run

From the repository root, initialise this example's environment once:

```bash
julia --project=examples/12_karman_hopf -e \
  'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

This uses the MORFEFerrite source in the current checkout and downloads the
registered MORFE release. The generated `Manifest.toml` stays local and is not
committed, so it contains no machine-specific paths in the repository. CairoMakie
comes with it: the notebook plots the branch itself, since MORFE returns data and
draws nothing.

Open and execute [`karman_hopf.ipynb`](karman_hopf.ipynb). Its committed outputs use
order 3 so that the demonstration remains quick; the mesh is committed too, so
nothing has to be generated first.

From a shell:

```bash
cd examples/12_karman_hopf
jupyter nbconvert --execute --to notebook --inplace karman_hopf.ipynb
julia --project=. validate.jl
```

Change `order = 3` to `order = 9` in the notebook to reproduce the reference. Order 3
is an exact graded truncation of order 9, and every invariant `validate.jl` checks is
of degree 3 or less, so either result validates against
`reference_data/karman_invariants_ref.txt`. Order 9 takes about seven minutes for the
cohomological solve and `W` reaches ~193 MB; run it on its own.

The optional save cell creates:

```text
results/
  summary.txt
  data/
    W.jls
    R.jls
    R_coefficients.csv
    branch.csv
  figures/
```

`branch.csv` is one row per branch point (`order,eta,Re,rho,omega,St,max_abs_lift`), in
amplitude order because orders 5 and 9 fold. Its order-9 form is committed in the MORFE
repository as `website/tutorials/assets/karman/branch.v1.csv`, which is what the tutorial
page plots.

`sheet = :primary` is what keeps that file single-valued. `normal_form_branch` returns every
real root in the window, and from order 9 the `η′`-polynomial has a second one at large
amplitude; `:primary` follows the sheet born at the Hopf point and stops where it ends.

[`reference_data/PROVENANCE.md`](reference_data/PROVENANCE.md) records how the
reference was produced.

## Reynolds number as a parameter

The reduction expands about the base flow at `Re₀ = 49.03` in the coordinate

```text
η′ = 1/Re − 1/Re₀
```

which enters linearly because the viscosity does: `ν = D/Re`, with `D` the cylinder
diameter. `η′` is a **frozen** external state, with its own equation `η̇′ = 0`, so
`N_EXT = 1` and the reduced system has `NVAR = ROM + N_EXT = 3` variables: `z₁`, `z̄₁`
and `η′`. That is why a single solve covers a range of Reynolds numbers rather than
one operating point: `R`'s dependence on `η′` is expanded to the same order as its
dependence on the amplitude.

`η′` is real and therefore its own conjugate, a pairing the usual adjacent-pairs
formula cannot express, so `build_model` derives the permutation with
`full_conjugate_permutation` rather than a literal. It comes out `[2, 1, 3]`: the
Hopf pair swaps, `η′` maps to itself.

## Mode selection

Which modes span the manifold is a modelling decision and belongs in the driver, so
the fluid `build_model` takes `master` and `outer` as index vectors and selects
nothing itself. Two points are worth knowing.

**The spectrum comes back closed under conjugation.** `solve_hopf_eigenproblem`
shifts at a complex `σ`, so ARPACK returns only the modes near `σ`; the Kármán mode's
conjugate sits near `σ̄` and is never computed. The eigensolve therefore appends the
missing halves itself, which is exact because the operators are real, and
`conjugate_index` names the true partner. Pass `close_conjugates = false` to get
exactly what ARPACK produced.

**`nev` and `outer_targets` buy a diagnostic, not accuracy.** `outer_targets = true`
flags monomials that are near-resonant with modes left off the manifold, but the
cohomological solve reads the master block alone, so `W` and `R` do not depend on how
much of the spectrum was computed. `nev = 40` is what the reference run used and it
gives the off-manifold warning something to work with.

The mode gauge does matter. `scale = 1e-2` multiplies both eigenvector sides, and
`c210` scales as `|scale|²`, so a run made at a different scale will fail
`validate.jl` even though the physics is unchanged. `λ`, `c101` and the `Im/Re` ratio
of `c210` are gauge-free.
