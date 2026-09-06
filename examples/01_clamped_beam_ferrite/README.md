# 01 — Clamped beam

This notebook computes an invariant-manifold ROM for a conservative,
clamped–clamped St. Venant–Kirchhoff beam and reads its **backbone curve** off
the reduced dynamics. The quadratic Ferrite mesh has approximately 5,000 free
degrees of freedom.

The example is intentionally limited to the public, physics-independent API:

```julia
order = 9
case = SVK.mechanical_model("clamped_clamped_beam.msh"; ...)
(; model, spectral, meta) = build_model(case;
    master = [1], expansion_order = order)
W, R = parametrise(model, spectral, order;
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))

u = observable_polynomial(W, SVK.probe_dof(case, 289, 2))
b = normal_form_branch(R; amplitudes = range(0, 85; length = 341))
a = cycle_amplitude.(Ref(u), b.amplitude)   # physical amplitude at the probe

MORFE.save_rom(results_dir, W, R; drop_below = 0.0)
```

The solve is graded, so orders 3, 5 and 7 come from truncating this one order-9
result with `restrict_ReducedDynamics_to_degree` and
`restrict_polynomial_to_degree`; they are not separate runs. The four curves on
the [tutorial page](https://morfeproject.github.io/tutorials/structural_svk.html)
are exactly that.

There is no example-specific assembly, eigensolver, cohomological solver, ROM
wrapper, or equation printer. Backend information remains available separately
in `meta`.

## Run

From the repository root, initialise this example's environment once:

```bash
julia --project=examples/01_clamped_beam_ferrite -e \
  'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

This uses the MORFEFerrite source in the current checkout and downloads the
registered MORFE release. The generated `Manifest.toml` stays local and is not
committed, so it contains no machine-specific paths in the repository.

Open and execute [`clamped_beam.ipynb`](clamped_beam.ipynb). It ships at
`order = 9`, the conservative reference: about 35 s of solve, and under a minute
end to end. The string `dirichlet = "Dirichlet"` selects the facet group named
`Dirichlet` in the Gmsh mesh and fixes all displacement components on those
facets, producing the two clamped ends.

From a shell:

```bash
cd examples/01_clamped_beam_ferrite
python3 -m nbconvert --execute --to notebook --inplace clamped_beam.ipynb
MORFE_FAST=1 julia --project=. validate.jl
```

`validate.jl` compares the fresh `R` against
`reference_data/R_coefficients_ref.csv` row by row, and the backbone against the
committed probe row of `W` through `cycle_amplitude`, which is invariant to the
eigenvector gauge.

The save cell creates:

```text
results/
  summary.txt
  data/
    W.jls
    R.jls
    R_coefficients.csv
    backbone.csv                # ρ, Ω and probe amplitude at orders 3, 5, 7, 9
    W_probe_coefficients.csv    # the W row the backbone amplitude comes from
  figures/
```

[`reference_data/PROVENANCE.md`](reference_data/PROVENANCE.md) records how the
committed notebook output and the order-9 reference were produced.

## Harmonic forcing

`HarmonicForcing(mode = 1, amplitude = 0.03)` adds a load `f(t) = amplitude · M·ϕ₁ · cos(Ωt)`
shaped like mode 1 and oscillating at mode 1's natural frequency (Ω defaults to `|λ₁|`;
pass `Ω = ...` to detune). It is a **`build_model` keyword**, not a `parametrise` one:
each forcing appends a conjugate pair of external states with eigenvalues ±iΩ, so
`N_EXT = 2`, and `parametrise` stays the same physics-independent call as above.

```julia
(; model, spectral, meta) = build_model(case; master = [1], expansion_order = order,
    forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.03))
W, R = parametrise(model, spectral, order;
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))
```

Pass a **vector** for multi-harmonic excitation, `f(t) = Σₖ aₖ · M·ϕ_{pₖ} · cos(Ωₖ t)`.
Each forcing gets its own ±iΩₖ pair, so `N_EXT = 2 · length(forcing)` and forcing `k`
occupies reduced coordinates `ROM+2k-1`, `ROM+2k`:

```julia
ω₁ = abs(SVK.eigenfrequencies(case; nev = 10)[1])   # pair p sits at entries 2p-1, 2p
(; model, spectral, meta) = build_model(case; master = [1], expansion_order = order,
    forcing = [SVK.HarmonicForcing(mode = 1, amplitude = 0.03),
               SVK.HarmonicForcing(mode = 1, amplitude = 0.01, Ω = 3.0 * ω₁)])
```

The forcing mode only supplies the load shape and need not be a master mode. Separately,
`parametrise` warns when any monomial's `s = ⟨λ, α⟩` lands near an eigenvalue left off the
manifold — that direction is then solved through a near-singular operator. The test is on
**frequency alone**, so it fires for autonomous models too and shaping the load away from
the offending mode is no protection; detune the forcing, add damping, or add the mode to
`master`. `meta` reports the resulting `N_EXT`, `Ω` and `forcings`.
