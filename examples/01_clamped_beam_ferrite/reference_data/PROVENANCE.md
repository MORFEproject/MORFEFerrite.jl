# Reference provenance

`R_coefficients_ref.csv` — order-9 run of the conservative notebook workflow,
regenerated 2026-08-26 with Julia 1.12.6. It was exported with
`drop_below = 0.0`, so the small degree-9 resonant coefficient is retained.
The zero Rayleigh damping is part of the reference: conservative coefficients
have zero real part up to roundoff.

`W_node289_y_coefficients_ref.csv` contains the position-map coefficients at
Ferrite node 289 in the transverse `y` direction (global DOF 2468, free DOF
2405). Node 289 is at `(499.9999999998496, 3.333333333329796, 24.0)`. The
website backbone uses half the peak-to-peak displacement at this node over one
phase cycle.

The reference is produced by setting `order = 9` in `clamped_beam.ipynb` and
executing the notebook. The committed notebook uses `order = 3` and validates
against the shared rows of this order-9 coefficient table.

The website's order 3, 5, 7, and 9 curves are all produced by truncating these
same order-9 `W` and `R` coefficients; they are not separate solves.
