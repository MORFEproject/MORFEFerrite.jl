# Reference provenance

`karman_invariants_ref.txt` holds the physical invariants of the reduced dynamics:
the growth rate and frequency of the Hopf mode, the linear Reynolds coupling
`c101`, and the Landau coefficient `c210`. It was derived from the blessed order-9
run of `05_karman_vortex_street` (un-promoted, `NVAR = 3`) and carried over
unchanged, because the two examples solve the same problem at the same expansion
point on the same mesh.

The settings the values depend on:

```text
mesh            cylinder_flow.msh, n_free = 57860 (Turek-Schäfer channel, ⌀0.1 m cylinder)
Re₀             49.03
master modes    the Hopf pair only (ROM = 2), η′ = 1/Re − 1/Re₀ as the single external state
mode scale      1e-2
normalisation   SymmetricBiorthogonal
resonance       :complex_normal_form, tol_relative = 0.1, eigenvalue_projection = :full
```

The original blessing passed the equivalent **absolute** `tol = 1.6859169502334799`.
Both inner targets are the Hopf pair with `|λ| = 16.859169…`, and `resolve_tolerances`
turns `tol_relative` into `tol_relative · |λ_master[r]|`, so the two spellings give the
same inner resonance set and the values are directly comparable.

**The mode scale is part of the reference.** `c210` scales as `|scale|²`, so a run at
a different gauge deviates from these numbers while describing identical physics.
`σ`, `ω`, `c101` and `c210_ratio` are gauge-free.

Invariants are compared instead of raw `R` coefficients because ARPACK fixes the
eigenvector gauge only up to a phase, which varies run to run; the residual spread on
these quantities is ~1e-7 relative, which is what `validate.jl`'s tolerances are set
above.

## Regenerating

Set `order = 9` in `karman_hopf.ipynb`, execute the notebook, then from this example's
directory:

```julia
using MORFEFerrite
const NSE = MORFEFerrite.FluidNavierStokes
inv = NSE.rom_invariants(NSE.load_rom_poly(joinpath("results", "data", "R_coefficients.csv")))
NSE.write_invariants(joinpath("reference_data", "karman_invariants_ref.txt"), inv)
```

Order 3 is an exact graded truncation of order 9 and every quantity here is of degree
3 or less — `c101` is monomial `(1,0,1)`, `c210` is `(2,1,0)` — so the committed
order-3 notebook validates against this order-9 file. Restore `order = 3` and
re-execute before committing, so the notebook's stored output stays the cheap one.
