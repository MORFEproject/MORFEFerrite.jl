# How MORFEFerrite's physics modules should be implemented

The division of labour, in one line:

> **MORFE owns `parametrise`. MORFEFerrite owns `build_model` — one per multiphysics
> module — and may overload a small, named set of things (`show`, the spectral layer, the
> FEM term interface, the expansion policy).**

Everything below follows from that. This document is the plan each module is measured
against, with the current state noted so the gap is visible.

---

## 1. The boundary

```
        MORFEFerrite                          │              MORFE
                                              │
  mesh ─▶ FE spaces ─▶ assembled operators    │
       ─▶ AbstractAssembledModel              │
       ─▶ build_model ──────────────────────▶ (NthOrderModel, SpectralData)
                                              │        │
                                              │        ▼
                                              │   parametrise(model, sd, order)
                                              │        │
       ◀────────────────────────────── (W, R) │        ▼
  postprocess: real_dynamics, VTK export,     │   mset → resonance → solve
  backbone curves, ROM wrappers               │
```

**MORFE's half is physics-blind.** It sees an `NthOrderModel` — linear operator tuple,
nonlinear terms, external system — and a `SpectralData`. It does not know whether the model
came from a beam, a fluid or a MEMS actuator, and nothing in it should ever need to.

**MORFEFerrite's half is model-building.** Meshing, quadrature, assembly, boundary
conditions, damping models, forcing shapes, mode selection conventions, and the packaging of
`(W, R)` into whatever the user of that physics wants back.

The single point of contact is `build_model`.

---

## 2. The `build_model` contract

Declared in [`src/common/assembled_model.jl`](src/common/assembled_model.jl).

```julia
abstract type AbstractAssembledModel end

build_model(case::AbstractAssembledModel; kwargs...) -> (; model, spectral, meta)
```

**Return a `NamedTuple`, not a positional tuple.** `StructuralSVK.build_model` already
returns three things (`model, sd, meta`) while the docstring promises two — that mismatch is
exactly the failure mode the MORFE refactor spent fifteen commits removing (five positional
spectral arrays nobody could keep straight). Named fields grow without breaking callers:

```julia
(; model, spectral) = build_model(case; master = [1], forcing = f)
W, R = parametrise(model, spectral, order)
```

`meta` is **backend-private**: whatever that physics needs to report afterwards (timings,
the raw spectrum, forcing records, DOF maps). MORFE never sees it.

### Every implementation must

1. Return an `NthOrderModel` that is **complete** — linear terms, every nonlinear term, and
   the `ExternalSystem` the forcing introduces. Not "mostly built, the caller adds forcing".
2. Return a `SpectralData` **reconciled against that model's order**. Use
   `SpectralData(model, spectrum; master = …)`, which slices when the orders match and
   extends only when the model's `ORD` is higher (augmented `(K, C, M, 0)` fed by a
   second-order eigenproblem). Do not hand-roll the extension: getting `λ · last block`
   versus a fresh `λ^{k-1}ψ` wrong is silent.
3. Apply conditioning tweaks (mode scaling, unit changes) to the **raw arrays before**
   constructing the bundle — `SpectralData` deliberately has no `scale` field, so such
   tweaks stay visible at the call site. See `examples/05_karman_vortex_street/main.jl`,
   which scales the modes by `1e-2`.
4. Build any conjugate permutation with `full_conjugate_permutation(master_block, sys)` —
   never a literal. A literal bakes in a pairing that a re-based external system invalidates,
   and cannot express an odd `N_EXT` (the Kármán case: `[2, 1, 3]`, where `η′` is real and
   therefore its own conjugate).

### Every implementation must NOT

- Build a `MultiindexSet`, a `ResonanceSet`, or a resonance *policy* — those are arguments to
  `parametrise`, chosen by the caller. A backend may offer a helper that *suggests* one (see
  §4.4), but `build_model` returns a model, not a reduction.
- Validate the monomial set, derive conjugate closure, or warn about resonances. MORFE does
  all three, and doing them here means doing them twice, differently (see §5).
- Solve anything.

---

## 3. Anatomy of a physics module

Canonical layout, one directory under `src/`:

```
src/<Physics>/
    <Physics>.jl        module, imports, includes, exports  — nothing else
    types.jl            material/damping/forcing structs, the AbstractAssembledModel
                        subtype, the ROM wrapper, and every Base.show
    <fem>_assembly.jl   the FEMMultilinearMap subtypes and their element kernels
    linear_operators.jl assembly of K/C/M (or B₀/B₁), boundary conditions, DOF maps
    <physics>_model.jl  the user-facing assembly entry point (`mechanical_model`, …)
    build_model.jl      the ONE build_model method, and nothing else
                        (there is NO parametrise.jl — see §4.5)
    eigensolver.jl      optional AbstractEigensolver subtype for this physics
    postprocess.jl      real_dynamics, printing, export, backbone extraction
```

Required pieces, in dependency order:

| # | Piece | Rule |
|---|---|---|
| 1 | `struct Assembled<Physics>Model <: AbstractAssembledModel` | holds FE spaces + assembled operators + the physics' own settings. Immutable. |
| 2 | `Base.show(io, ::MIME"text/plain", ::Assembled<Physics>Model)` | header line, 2-space-indented `key : value` rows, `print` (not `println`) on the last row — the house style at [`types.jl:71`](src/StructuralSVK/types.jl#L71) |
| 3 | assembly entry point, e.g. `fluid_model(grid; …)` | takes a mesh + physics settings, returns (1). No reduction concepts in its signature. |
| 4 | `FEMMultilinearMap` subtypes | one per nonlinear degree, implementing the interface in §4.3 |
| 5 | `build_model(case; …)` | §2. Thin: it wires (3) and (4) into an `NthOrderModel` and builds the bundle. |
| 6 | ROM wrapper + its `show` | a **result container the caller packages**, not an entry point: holds `(W, R)` plus what postprocessing needs from `meta` |
| 7 | optional `summary_entries(case, meta)` | rows this physics adds to `Common.write_summary` — see §4.6 |

---

## 4. The sanctioned overload points

These are the places a backend may legitimately extend MORFE. Anything not on this list
should be a pull request against MORFE instead of a method here.

### 4.1 `Base.show`

Every backend type gets one. MORFE has its own for `NthOrderModel`, `MultilinearMap`,
`SpectralData`, `ResonanceSet` and `ResonanceConfig`; those are about the *reduced* problem
and are not to be shadowed. A backend's `show` describes the *physics*: mesh size, material,
damping, boundary sets, forcing.

### 4.2 The spectral layer

This is the "some spectral things" half of the division, and it is deliberately narrow:

```julia
struct RayleighEigensolver <: MORFE.AbstractEigensolver …
MORFE.SpectralDecomposition.eigensolve(model::NthOrderModel, s::RayleighEigensolver)
MORFE.SpectralDecomposition.eigensolve_left(model::NthOrderModel, s::RayleighEigensolver)
```

as [`src/StructuralSVK/rayleigh_solver.jl`](src/StructuralSVK/rayleigh_solver.jl) already
does. A physics that can exploit structure MORFE cannot see — symmetry, a shift-invert
target, a modal-damping form — belongs here.

The interface contract is the important part: **both must return `FOM × ORD × n` arrays
holding all companion order-blocks**, right blocks satisfying `ψ_{k+1} = λ ψ_k` and left
blocks the sesquilinear adjoint. A solver that naturally produces only the physical left
slice rebuilds the rest with `left_eigenmode_orders_from_slice`.

> ⚠️ `rayleigh_solver.jl:57` reaches into `MORFE.SpectralDecomposition._structural_left_eigenmode_orders`
> — a private function. Either it becomes public in MORFE (with a docstring stating the
> contract), or this call moves to the exported `left_eigenmode_orders_from_slice`. A leading
> underscore across a package boundary is a rename waiting to break the build.

Also legitimate: extending `spectrum` for a physics-specific operator pair (SVK's
`spectrum(K, M, solver)`), and offering the module's own master-selection helper — SVK's
"pair `p` means spectrum entries `2p-1, 2p`" is a convention of that discretisation, not of
MORFE. Keep such helpers **pure**: they return indices, mirroring MORFE's
`master_by_sorting` / `master_by_target_frequency`. Nothing writes a selection back onto a
`Spectrum`; it is immutable precisely so a spectrum means one thing for its whole lifetime.

### 4.3 The FEM term interface

For each nonlinear term, subtype `FEMMultilinearMap{ORD}` and implement `fem_elements`,
`fem_n_qp`, `fem_ndofs_per_cell`, `scatter_qp!`, `accumulate_qp!`, `assemble_element!`,
`fem_getdetJdV`, `fem_qp_buffer`, `fem_reinit!`. This is what buys batched RHS assembly.
`FerriteGeometricNonlinearity` and `FluidConvection` are the two reference implementations.

### 4.4 The expansion policy — available, but **not** a place to put a backend type

`build_multiindex_set(expansion_order, nvar)` is a dispatch seam in MORFE: a new expansion
policy is a new method, never a branch. `parametrise` leaves its third argument untyped and
delegates here, so an `Integer` (total degree) and a `MultiindexSet` (stated explicitly) both
work out of the box.

An earlier revision proposed a `ThetaBoxExpansion` type in this repository for the
anisotropic `z`-total × `θ`-box sets of examples 04 and 07. **That was dropped.** MORFE
already ships enough constructors (`all_multiindices_up_to`, `all_multiindices_in_box`,
`indices_in_box_with_bounded_degree`), and the two examples' policies are not actually the
same — 04 caps only the `z` group while 07 also caps the joint total — so a single type would
have had to carry that difference as a field and would still have been a thin wrapper around a
comprehension. The truncation is a **modelling choice**: callers state their own `mset` and
pass it to `parametrise`, which validates it once against MORFE's five-clause contract.

### 4.5 `print_setup` — and **no** backend `parametrise`

`print_setup(io, model, spectral, mset, resonance)` is a normal function; add a method for a
richer banner if a physics wants one. The simpler route is to `show` your own assembled model
before calling `parametrise` — the information a backend has (mesh, material, boundary sets)
is not visible from an `NthOrderModel` anyway.

> **There is one `parametrise` and it is MORFE's.** A backend must not add
> `parametrise(case::Assembled<Physics>Model; …)`.

An earlier revision of this document sanctioned such a method provided its body was *only*
`build_model` → `parametrise` → wrap. Both backends that took it up immediately exceeded that
budget — SVK's built the monomial set and a `ResonanceConfig`, the fluid's built the set and a
whole `ResonanceSet` by hand — which is precisely the "logic that has escaped" this section
warned about. A rule that is violated by both of its two users is the wrong rule, so the
allowance is withdrawn rather than re-policed.

Every caller reduces the same way:

```julia
(; model, spectral, meta) = build_model(case; master = …, expansion_order = order)
W, R = parametrise(model, spectral, order; resonance = ResonanceConfig(…))
rom  = <PhysicsROM>(W, R, meta; master = …, order = order)   # optional packaging
```

Two consequences worth stating, because they are what made the sugar tempting:

- **The resonance policy is the caller's, and must be expressible as a `ResonanceConfig`.**
  If a physics needs a policy the config cannot state, extend `ResonanceConfig` in MORFE —
  do not hand-build a `ResonanceSet` in a backend. The fluid's frequency-only Hopf detection
  became `eigenvalue_projection = :imaginary_part_only` for exactly this reason.
- **`build_model` must be able to size its caches without a `MultiindexSet`.** Give it
  `expansion_order` and let it *count* (`binomial(nvar + order, order) - 1`); reserve
  `n_monomials` for callers with a custom set.

A `<Physics>ROM` type is still fine — as a **result container** the caller packages, holding
what postprocessing needs and `build_model`'s `meta` cannot lose. It is not an entry point.

---

### 4.6 The run summary

`Common.write_summary(io, path, case; rom, title, metadata)` owns the skeleton every backend
wants — problem size, order, and **every `*_time_s` entry** it finds in `case.info` and the
result's info. That last part is the convention worth knowing: a stage that records
`eig_time_s` is reported as "eig", so a backend gets its stages timed by recording them, with
no reporting code at all.

`summary_entries(case, result) -> Vector{Pair{String,Any}}` is the seam. Add a method to
contribute the rows only your physics has; the skeleton is not restated. Later rows override
earlier ones (skeleton → physics → caller `metadata`), so a backend can correct a shared row
as well as append.

Record the things that change **what `W` and `R` mean** — a mode gauge, a resonance policy,
a continuation parameter — not just what is convenient. Coefficients computed in two
different gauges are not comparable, and a summary that omits the gauge cannot say so.

### 4.7 The parametric pullback kernel

`ParametricGeometry` is a physics-blind *lower* layer (§6), and
`AbstractPullbackKernel{DEG}` is how a physics plugs into it:

```julia
det_weight_power(k)                              # the p in (1/det J)^p
qp_prepare(k, ctx, ∇u_adj::NTuple{DEG})          # per-QP precompute, test-function-independent
qp_integrand!(integ, k, ctx, state, ∇N_adj)      # the θ-series of the integrand
linear_qp_series!(a_ser, b_ser, k, ctx, i, j)    # for a linear-operator kernel
```

Gradients arrive **already contracted with the adjugate series** (`∇u·adj J(θ)`), so a kernel
writes its weak form exactly as it would in the reference configuration — which is why
`StructuralSVK.SVKPullbackKernel` agrees with the non-parametric `accumulate_qp!` at `J = I`,
and why that is a test worth having.

`ParametricGeometry` owns everything around it: the geometry cache, the cell loops, the
all-coefficient sweep, the shared-input cache, the `MultilinearMap` wrapping, the
arity-generic `build_linear_corrections`, and `build_model`.

It also **measures**, never assumes, which θ-multiindices contribute nothing —
`report_zero_coefficients` emits one `@info` naming them, split by whether the *geometry*
never reaches that degree (shrink the basis, free) or the operators/forms annihilate it (a
property of the physics, worth understanding). Do not encode a symmetry rule here: one that
is right for one family of shape fields is silently wrong for the next.

---

## 5. What must never be duplicated

Resonance detection and its warnings, `mset` validation, conjugate-permutation derivation,
the graded solve, and progress/banner output are MORFE's. The live example of what happens
otherwise is in this repo:

`src/StructuralSVK/parametrise.jl` carries `_warn_outer_resonances` and a `resonances`
helper that re-implement detection MORFE now does — because, when they were written, MORFE
warned once per *eigenvalue* while SVK needed once per conjugate *pair*. MORFE now groups by
pair itself and costs a constant ~64 B when nothing is flagged. The duplicate should go; the
one blocker and the two ways to remove it are written up in
[`MORFE_API_MIGRATION.md`](MORFE_API_MIGRATION.md) §3.

The rule that would have prevented it: **when MORFE's behaviour is nearly right, fix MORFE.**
A backend-local reimplementation is invisible to every other backend and drifts silently.

---

## 6. Module-by-module: state and plan

### `Common` — the shared Ferrite layer ✅ mostly there

Owns `AbstractAssembledModel`, `build_model`, mesh IO (`load_comsol_grid`), DOF lookup, and
VTK export. **Plan:** keep it physics-free. Anything referenced by exactly one physics module
belongs in that module, not here.

### `StructuralSVK` — the reference implementation ✅ aligned

Types + `show`, `FerriteGeometricNonlinearity`, `RayleighEigensolver`, `mechanical_model`,
`build_model`, `SVKPullbackKernel` (§4.7), postprocessing, and the equivalence gates. All
three alignment items are done:

1. `build_model` returns the NamedTuple of §2, from its own `build_model.jl`.
2. `_warn_outer_resonances` deleted — `build_model` now hands `SpectralData` the
   **spectrum-wide** involution, so MORFE's own warner groups by conjugate pair and names
   the physical mode. The `warn_outer = false` suppression is gone with it.
3. `rayleigh_solver.jl` uses the public `left_eigenmode_orders_from_slice`.

It also carries **no `parametrise`** (§4.5): `InvariantManifoldROM` is a result container the
caller packages, and `RayleighEigensolver` holds a `RayleighDamping` rather than a loose
`(α, β)` pair, so the solver's `ξ` and the model's `C` cannot drift apart.

### `ParametricGeometry` — a physics-blind LOWER layer ✅ (was `ParametricStructural`)

Not a physics module at all, and the dependency runs the other way: the physics modules
depend on it. It implements the general parametric mesh coordinate transform
`x(θ,x₀) = x₀ + Σᵢ θᵢψᵢ(x₀)` as power series in the Jacobian's determinant, adjugate and
inverse determinant, and expands **any** physics' multilinear maps over it. Included before
the physics modules in `MORFEFerrite.jl`.

It names no material, stress law or strain measure. The seam is `AbstractPullbackKernel`
(§4.7): the physics supplies the QP integrand, this module owns the geometry cache, the cell
loops, the all-coefficient sweep, the `MultilinearMap` wrapping, the arity-generic
`build_linear_corrections`, and the `build_model` contract.

`AssembledParametricModel` + `build_model` produce the augmented `NthOrderModel` (`ORD`
**derived** from the operators' arities, not assumed), the reconciled `SpectralData`, and a
`full_conjugate_permutation`-derived pairing. Examples 04 and 07 are now
assemble → `build_model` → `parametrise` → save, and their `PERMUTATION` literals are gone.

`ThetaBoxExpansion` (an earlier §4.4 proposal) was **dropped**: MORFE has enough multiindex
constructors and the θ-box truncation is a modelling choice, so callers state their own
`mset`.

### `FluidNavierStokes` — complete ✅ `fluid_model` + `build_model`

Backend (`setup_fem`, `solve_steady_state`, `assemble_linear_operators`, `FluidConvection`,
`make_param_coupling`, `make_base_forcing`, energy Gram) plus the mesh → ROM layer:
`fluid_model` assembles and `build_model` reduces-ready. `examples/05` no longer wires it.

This physics differs from the structural ones in ways the module must own, not the example:

- **First order.** `NthOrderModel((B₀, B₁), …)` — `ORD = 1`, so there are no derivative
  blocks and `SpectralData` takes plain `FOM × ROM` matrices.
- **A real external state.** The Reynolds-continuation variable `η′` has a real eigenvalue,
  so `N_EXT` is odd and the conjugate permutation is `[2, 1, 3]` — expressible only through
  `full_conjugate_permutation`, never `[[2p, 2p-1] …]`.
- **Mode scaling** (`1e-2`) for conditioning, applied to raw arrays pre-bundle.
- **Arpack gauge.** Its eigenvectors are not reproducible across runs, so anything comparing
  raw `R` coefficients between runs is meaningless; compare invariants (see the
  `project_karman_demo_validation` memory).

**Plan:** `AssembledFluidModel <: AbstractAssembledModel` (mesh, FE spaces, base flow,
`Re₀`), `fluid_model(mesh; Re, …)`, and one `build_model(case; master, continuation = …)`
that encapsulates all four points above. Example 05 keeps its narration but drops the
assembly.

### Electromechanical / MEMS — future modules

`examples/08_mems_micromirror` and `examples/09_electromechanical` are the next physics.
When either grows past "an example that calls SVK", it becomes `src/Electromechanical/`
with the same seven pieces. The dielectric actuator in MORFE's own
`examples/06` is pure-Julia (Hermite beam, no Ferrite) and stays there.

`examples/10_turbine_blade` is an SVK *application*, not a physics: it should end up calling
`build_model` + `parametrise`, with nothing model-shaped in `Blade/`.

---

## 7. Conventions

- **UK English** for identifiers: `parametrise`, `normalise`, `normaliser!`.
- **Exports:** each module exports its own public API; `MORFEFerrite.jl` re-exports
  deliberately, naming any deliberate omission and why (as it already does for `save_rom`).
- **No `Base.get_extension`** for the physics modules: they are first-class submodules. VTK
  export stays an extension because `WriteVTK` is genuinely optional.
- **In-place functions end with `!`** and return `nothing` unless a value is meaningful.
- **Formatting:** the pinned formatter in MORFE's `format/` environment governs both repos'
  style; run it before committing.

---

## 8. Testing

Each module carries a `test/<Physics>/` group with three layers:

1. **Unit** — assembly kernels against hand-computed element values; operators against
   analytic limits (a linear beam's eigenfrequencies, a Stokes flow's drag).
2. **Contract** — `build_model` returns a complete model and a bundle whose ORD matches;
   `check_biorthogonality(sd, model) ≈ I`, which is the numerical guard on the mirrored
   right/left convention and catches any block swap loudly.
3. **Equivalence gate** — the high-level path ≡ the hand-rolled low-level path, on the same
   mesh, to a stated tolerance. This is what `test/StructuralSVK/run_gates.jl` does (Gate A
   **1.0149555905989023e-9**, Gate B **6.501646250144121e-10**), and it is the single most
   valuable test in the repo: it is what proves a refactor moved nothing.

Environment notes that cost time when forgotten:

- Run gates through the **example environment**, not `Pkg.test` —
  `julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl`
  (see the `project_svk_gate_env_failure` memory).
- **Never `git add -A` in this repo**: the working tree carries in-flight work, and
  `examples/` holds large meshes and result archives (it also times out).

---

## 9. Migration order

Each step leaves both repos green and is independently revertible.

1. **Contract** — `build_model` returns a NamedTuple; update `Common`'s docstring and SVK's
   method and its two callers. Mechanical, no numerics.
2. **SVK tidy** — split `build_model.jl` from `parametrise.jl`; move the private spectral
   call onto public API. Gates A/B must not move.
3. **SVK duplicate warner** — with the involution fix, delete it and drop
   `warn_outer = false`. The outer-resonance testset is the specification and must pass
   **unchanged** (`MORFE_API_MIGRATION.md` §3).
4. **ParametricStructural** — assembled model + `build_model` + `ThetaBoxExpansion`. Gate it
   by re-running examples 04 and 07 and comparing `(W, R)` against the archived results
   bit-for-bit; the assembly is unchanged, so anything else is a bug.
5. **FluidNavierStokes** — assembled model + `build_model`. Gate against the FOM reference
   orbits, not raw coefficients (Arpack gauge).
6. **Applications** — examples 08/09/10 onto whichever module then covers them.

The invariant to check after every step: **grep the physics modules for `mset`,
`ResonanceSet`, `resonance_set_from_`, `conjugate_permutation` literals, and
`validate_multiindex_set`.** In a finished module, `build_model` mentions none of them.
