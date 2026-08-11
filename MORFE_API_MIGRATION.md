# MORFE API migration — what changed, what I edited here, what is left

Written while landing MORFE's `parametrise` / spectral-layer refactor (MORFE_jl commits
`e0247d3` … `fac3fd1`). **Nothing in this repository has been committed or staged** — the
tree holds your own uncommitted work, so every edit below is sitting in the working tree
for you to review.

---

## 1. What MORFE deleted

| Gone | Replacement |
|---|---|
| `parametrise(model, order, spectrum; mset = …)` | `parametrise(model, spectral, expansion_order)` |
| `solve_cohomological_problem(model, mset, λ, Ψ, ℓ, rset; master_modes_derivatives, left_modes_derivatives, …)` | `solve_cohomological_problem(model, mset, spectral, rset; …)` |
| `select_master_modes_by_hand / _by_sorting / _by_target_frequency` | `master_by_sorting(n)` / `master_by_target_frequency(sp, freqs, tol)` — **pure**, they return indices |
| `get_eigenpairs(sp)` | read `sp.eigenvalues`, `sp.eigenmodes`, `sp.left_eigenmodes` |
| `build_resonance_set(model, style, mset, sp, tol, conjugacy_map)` | `build_resonance_set(model, mset, spectral, ResonanceConfig(...))` |

`Spectrum` is now **immutable** and no longer has `master_modes` or `external_modes`.
Which modes are master is a property of the `SpectralData` you build from it, not a flag
written back onto the spectrum.

The five spectral arrays become one object. Two ways to build it:

```julia
# from a solved Spectrum — slices and, when the model's ORD is higher, reconciles
sd = SpectralData(model, sp; master = master_by_sorting(ROM))

# from raw arrays: physical slices plus their companion blocks (new keywords)
sd = SpectralData(; eigenvalues = master_eigenvalues,
    right_modes = master_modes,   right_derivatives = master_modes_derivatives,
    left_modes  = left_eigenmodes, left_blocks      = left_modes_derivatives,
    conjugate_permutation = [2, 1])          # ROM-length master block, optional
```

`right_derivatives` / `left_blocks` are new, and exist precisely because this repository's
call sites already held that shape. They keep the mirrored convention in **one** place:
right physical is block 1, left physical is block `ORD`. A swap there is type-correct and
compiles silently, so it is worth not restating at every call site.

A conjugate permutation spanning external variables (`NVAR`-length, as examples 04/07 use)
does **not** go to the constructor — that one validates a ROM-length master block. Pass it
to `parametrise` / `solve_cohomological_problem` as `conjugate_permutation = …`, where it
is used verbatim.

---

## 2. Edits already applied here (uncommitted)

| File | Change |
|---|---|
| `test/StructuralSVK/test_structural_svk.jl` | reference solve → `SpectralData`; dropped the vestigial `select_master_modes_by_sorting` |
| `test/StructuralSVK/test_master_selection.jl` | same |
| `examples/01_clamped_beam_ferrite/low_level.jl` | same |
| `examples/04_parametric_clamped_beam/main.jl` | `parametrise(model, sd, mset; resonance, conjugate_permutation)`; `PERMUTATION` moved to the call |
| `examples/07_parametric_arch/main.jl` | same |
| `examples/10_turbine_blade/Blade/MainLorenz.jl` | `SpectralData`; left blocks built with `left_eigenmode_orders_from_slice` (the old call passed none) |

Not touched: `src/`. `StructuralSVK/parametrise.jl` already goes through
`parametrise(model, sd, mset)`, which is unchanged.

> ⚠️ `MainLorenz.jl` previously passed **no** `left_modes_derivatives`. It now gets the
> blocks rebuilt from the physical left slice against `model.linear_terms`, which is what
> every other caller does. Its numbers can therefore move; re-baseline deliberately.

---

## 3. Left to do — Change 2c: delete this repo's duplicate outer-resonance warning

MORFE's own warning is now cheap (a constant ~64 B when nothing is flagged, against the
246 kB it used to cost) and reports **one warning per conjugate pair**, naming both the
physical mode number and the spectrum entries. So
`src/StructuralSVK/parametrise.jl` can lose:

- `_warn_outer_resonances` (≈ L124-166) and the `resonances` helper (≈ L22-48);
- the `warn_outer = false` in the `ResonanceConfig` at ≈ L306-308.

**But it cannot be deleted as it stands**, and the spec testset
`test/StructuralSVK/test_structural_svk.jl` ("outer-resonance diagnostic") is what would
catch it:

`build_model` passes a **ROM-length master block** to `SpectralData`
(`parametrise.jl:258-260`). MORFE's `_resolve_permutation(::AbstractVector{Int}, …)`
widens that to the whole spectrum by leaving every **outer** entry *self-paired* — the
honest reading of "the caller stated the master pairing, not the spectrum's". `_mode_numbers`
then numbers spectrum entry 3 as mode 2 and entry 4 as mode 3, so the beam's second bending
pair reports as two separate modes and warns twice. `occursin("mode pair 2", …)` and
`length(warns) == 1` both fail.

Give `SpectralData` the pairing for the whole spectrum and it all lines up. Two ways:

1. **Preferred — pass the full involution.** The Rayleigh solver returns adjacent conjugate
   pairs, so `σ = reduce(vcat, [[2p, 2p-1] for p in 1:n_pairs])` over *all* `n_eigs` entries
   is exactly true. MORFE's `_resolve_permutation` currently accepts only a ROM-length
   vector, so this needs a small additive change there: treat a vector of length `n_eigs`
   as the spectrum-wide involution. The master restriction it derives is then identical to
   today's, so the solve stays **bit-identical** and the SVK gates cannot move.
2. `conjugate_permutation = :detect`. No MORFE change, but it verifies eigenvectors on every
   run (`O(FOM·ORD·ROM)`) and, if that check fails, silently proceeds with *no* conjugate
   symmetry — which would move Gate A/B. Not worth it here.

Do this together with the SVK rewrite rather than before it.

---

## 4. Verification

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl
```

Gate A and Gate B must still read **1.0149555905989023e-9** and **6.501646250144121e-10**.
They have not moved through any of MORFE's refactor commits; if they move now, something
real changed — investigate rather than adjust the threshold.
