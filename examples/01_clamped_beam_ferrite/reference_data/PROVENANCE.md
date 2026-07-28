# Reference provenance

`R_coefficients_ref.csv` — FULL run (order 9) of the current high-level pipeline.

Blessed 2026-07-17, replacing the CSV that had been committed with the example
before MORFE's **orthogonality-blocks refactor** (MORFE_jl `1a973fb`,
2026-07-08: sesquilinear φᴴB𝒲 = 0, J built from conjugated left blocks, no
λ/Λ folding, and a `dot()` bug fixed). That refactor leaves the eigenvalues
untouched but changes the nonlinear coefficients — e.g. the leading backbone
row (2,1) went from 6.8716e-5·i (old, buggy) to 2.4472e-5·i (corrected).

The current numbers are the self-consistent ones: the high-level and low-level
paths reproduce each other at the Arpack noise floor (~2e-9, Gate A in
`test/StructuralSVK/run_gates.jl`), and the in-process small gates agree at
~4e-11.
