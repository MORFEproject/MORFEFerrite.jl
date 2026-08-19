# Reference data — example 05 (Kármán vortex street)

## `karman_invariants_ref.txt` — THE reference

The one file `validate.jl` compares against. It stores **physical invariants**, not raw
coefficients, so a single file validates a run **with or without promoted outer modes**:

| quantity | meaning | invariant under |
|---|---|---|
| `σ`, `ω` | Hopf eigenvalue λ at η′ = 0 | gauge, promotion |
| `c101_re/im` | ∂λ/∂η′, linear Reynolds sensitivity | gauge, promotion |
| `c210_re/im` | effective Landau coefficient, after slaving | promotion (scales as \|c\|² under z → cz, so fixed `MODE_SCALE`) |
| `c210_ratio` | Im/Re of c₂₁₀ | gauge, promotion — the strict check |

Why not raw R:

- The Arpack eigenvector gauge differs run to run (z → e^{iφ}z), so raw coefficients were
  never comparable — the old check had to hand-pick "gauge-invariant" rows.
- Promoting resonant outer modes into the master set is a **change of coordinates**: the ROM
  gains columns and shares no monomials with an un-promoted one. Both describe the same
  physics and must validate against the same file.

`invariants.jl` bridges the two by **slaving** the promoted coordinates — on the slow
manifold they solve `R_k(ρ,ρ,y,η) = 0`, and R is exactly affine in them, so it is one small
linear solve. Zeroing them instead drops the mean-flow distortion, the dominant stabilising
term in c₂₁₀, and reports a supercritical Hopf as subcritical. `../test_invariants.jl`
proves the invariance on a synthetic pair of ROMs that are identical up to promotion.

Re-bless with:

```bash
julia --project=. main.jl            # FULL profile — see the hardware note below
julia --project=. validate.jl --bless
```

It refuses to bless from `MORFE_FAST=1`, whose coarser mesh is a different discretisation.

## `R_coefficients_ref.csv` — archived provenance

The blessed order-9, un-promoted (NVAR = 3) raw output the invariants above were derived
from. **Nothing reads it**; it is kept so the invariants can be re-derived or audited.

Its recorded provenance is not traceable — the archived run names `morfe_commit 273fab5`,
which is not a valid object in MORFE_jl's history. Its `c₂₁₀ = −0.11101 + 0.07065i` was
independently corroborated in 2026-08: a post-fix FAST run (coarse mesh, order 3) reproduces
it to 0.68%, i.e. to discretisation.

## Hardware

The FULL profile is 57,860 DOFs at order 9 and must not be run on the development machine
(it freezes it). `fom_reference.jl` is heavier still — direct IMEX time integration at
`dt = 5e-4` over nine Re points. See `CLAUDE.md` → *Local hardware limits*.
