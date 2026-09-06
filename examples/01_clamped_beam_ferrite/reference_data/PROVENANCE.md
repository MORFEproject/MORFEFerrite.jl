# Reference provenance

Both files are written by `clamped_beam.ipynb`, which ships at `order = 9`, so the
committed notebook *is* the reference run. Regenerating means executing it and
copying `results/data/` over this directory:

```bash
python3 -m nbconvert --execute --to notebook --inplace clamped_beam.ipynb
cp results/data/R_coefficients.csv       reference_data/R_coefficients_ref.csv
cp results/data/W_probe_coefficients.csv reference_data/W_node289_y_coefficients_ref.csv
MORFE_FAST=1 julia --project=. validate.jl
```

`R_coefficients_ref.csv` is exported with `drop_below = 0.0`, so the degree-9
resonant coefficient, of order `1e-17`, is retained rather than cut by the
`1e-14` default. The zero Rayleigh damping is part of the reference:
conservative coefficients have zero real part up to roundoff.

`W_node289_y_coefficients_ref.csv` holds the position-map coefficients at Ferrite
node 289 in the transverse `y` direction (global DOF 2468, free DOF 2405). Node
289 is at `(499.9999999998496, 3.333333333329796, 24.0)`, mid-span on the top
surface, where the first bending mode peaks.

**Neither file is compared coefficient by coefficient except through
`compare_rom_coefficients`' own tolerance.** A rescaling of the master
eigenvector by `exp(iψ)` multiplies the `(a,b)` coefficient by `exp(i(a-b)ψ)`, so
the raw rows are reproducible only while the eigensolver keeps picking the same
phase. `validate.jl` therefore checks the probe file through `cycle_amplitude`,
half the peak-to-peak displacement over a full cycle, which is a rigid shift of
that signal in the phase and so gauge-free.

## The tutorial's backbone data

The tutorial page plots `backbone.v1.csv`, a copy of the order-9
`results/data/backbone.csv` this notebook writes. `results/` is gitignored, so
the copy lives in the MORFE repository next to the chart generator:

```bash
cp results/data/backbone.csv \
   ../../../MORFE_jl/website/tutorials/assets/structural_svk/backbone.v1.csv
cd ../../../MORFE_jl
python3 website/tutorials/assets/structural_svk/generate_assets.py
```

`generate_assets.py` renders; it computes no physics. It reads that CSV for the
curves and asks Julia only for the first mode shape, which is FOM-sized and has
no business in a committed file.

All four curves come from the single order-9 solve.
`restrict_ReducedDynamics_to_degree` truncates `R` and
`restrict_polynomial_to_degree` truncates the projected displacement, which is
exact because the cohomological solve is graded; they are not four separate runs.

Two things the backbone depends on that the coefficient tables do not. The
**amplitude range** `ρ ∈ [0, 85]`, sampled at 341 points, which reaches about
1.1 × the beam thickness at the probe; and the **mode gauge**, which sets the
scale of `ρ` and so the modal panel's axis, though not the frequency ratio or
the physical displacement, which are gauge-free.
