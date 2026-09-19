# Module responsibilities — what `src/` owns and what `examples/` owns

An empirical audit of the current tree: for each physics module under `src/`, what it
actually holds today, what the matching example still holds, and which of those
example-side responsibilities could move.

This is the **current-state** companion to [`MODULE_ARCHITECTURE.md`](MODULE_ARCHITECTURE.md),
which is the normative spec ("how a physics module *should* be implemented"). Where the
two disagree about what exists, this document is the newer one.

- Audited at commit `9d02635` ("mesh") plus the working-tree changes present alongside it.
- Every `.jl` file under `src/` and `examples/` was read, plus both notebooks.
- Line counts in [§8](#8-appendix--sizes).

---

## 0. The boundary that already exists

```
        MORFEFerrite                          │              MORFE
                                              │
  mesh ─▶ FE spaces ─▶ assembled operators    │
       ─▶ AbstractAssembledModel              │
       ─▶ build_model ──────────────────────▶ (NthOrderModel, SpectralData)
                                              │        │
       ◀────────────────────────────── (W, R) │   parametrise(model, spectral, order)
```

The contract is declared once, in [`src/common/assembled_model.jl`](src/common/assembled_model.jl):

```julia
build_model(m::AbstractAssembledModel; kwargs...) -> (; model, spectral, meta)
```

**This boundary is now real, and it holds.** Four assembled-model types and four
`build_model` methods exist:

| Type | Declared in | `build_model` |
|---|---|---|
| `AssembledMechanicalModel` | [`src/StructuralSVK/types.jl:73`](src/StructuralSVK/types.jl#L73) | [`StructuralSVK/build_model.jl:72`](src/StructuralSVK/build_model.jl#L72) |
| `AssembledParametricModel` | [`src/ParametricGeometry/types.jl:35`](src/ParametricGeometry/types.jl#L35) | [`ParametricGeometry/build_model.jl`](src/ParametricGeometry/build_model.jl) |
| `AssembledFluidModel` | [`src/FluidNavierStokes/types.jl:33`](src/FluidNavierStokes/types.jl#L33) | [`FluidNavierStokes/build_model.jl:45`](src/FluidNavierStokes/build_model.jl#L45) |
| `AssembledParametricFluidModel` | [`src/FluidNavierStokes/parametric_fluid.jl:19`](src/FluidNavierStokes/parametric_fluid.jl#L19) | [`parametric_fluid.jl:288`](src/FluidNavierStokes/parametric_fluid.jl#L288) |

And — worth stating because it was the previous plan's open item — **no backend
`parametrise` wrapper survives anywhere in `src/`**. `grep -rn "function parametrise" src/`
returns nothing. The only `parametrise` is MORFE's. Examples 03 and 08 show the intended
three-step shape verbatim:

```julia
model_case = SVK.mechanical_model(MESH; material, damping, dirichlet, …)
(; model, spectral, meta) = build_model(model_case; master, forcing, expansion_order)
W, R = parametrise(model, spectral, ORDER; resonance = ResonanceConfig(…))
```

So the question this document answers is no longer "does the boundary exist" but
**"which responsibilities are still on the wrong side of it, and does that matter"**.

---

## 1. Scorecard

| `src/` module | LOC | Examples | Responsibility split |
|---|---:|---|---|
| `Common` | 1639 | (all) | ✅ Clean. Physics-free: contract, mesh IO, DOF lookup, VTK, summary skeleton. |
| `ParametricGeometry` | 2085 | 04, 07 (via SVK) | ✅ Clean. Physics-blind; examples hold only the case. |
| `StructuralSVK` | 1641 | 01, 03, 08 | ✅ Clean — examples are a CASE block plus a 3-line pipeline. |
| ″ | | **10** | ❌ **Example 10 re-implements the module.** 1241 lines, ~90 % duplicated or superseded. |
| `FluidNavierStokes` | 4602 | 05 | 🟡 Mostly resolved. ~350 lines of modelling policy remain in the notebook. |
| ″ | | **11** | ❌ **A second fluid module lives inside the example.** 5185 lines. |

Legend: ✅ responsibilities sit where they belong · 🟡 defensible residue, some movable ·
❌ library-shaped code in an example.

---

## 2. `Common` — the shared Ferrite layer

**What `src/` owns.** The `build_model` contract itself
([`assembled_model.jl`](src/common/assembled_model.jl)); the `MeshIO` submodule
(COMSOL `.mphtxt` → Ferrite grid, plus Abaqus↔Gmsh↔COMSOL converters, 1222 lines);
node→DOF lookup; the Paraview export stubs whose implementations live in the
`WriteVTK` extension; and — newer — [`summary.jl`](src/common/summary.jl), which owns a
shared `write_summary` skeleton with `summary_entries(case, rom)` as the per-physics
dispatch seam.

**What the examples own.** `examples/mesh_import/` (580 lines) is a *demonstration* of
`MeshIO`, not an implementation — the right relationship.

**Verdict: nothing to move.** `summary.jl` is the model to copy elsewhere: it is exactly
the pattern of "shared skeleton in the library, one short method per physics", and it
retired a per-example summary writer in the process. `Common` names no material, no
stress law, no Reynolds number, which is the invariant that keeps it reusable.

---

## 3. `ParametricGeometry` — the physics-blind transform

**What `src/` owns.** The whole geometric-parameter machinery: the additive map
`x(θ,x₀) = x₀ + Σᵢθᵢψᵢ(x₀)`, its Jacobian det/adjugate/reciprocal series over a
per-parameter multiindex box, the `AbstractPullbackKernel` seam a physics implements,
the assembly driver, the arity-generic linear corrections, `build_model`, and
[`diagnostics.jl`](src/ParametricGeometry/diagnostics.jl) — which measures the
expansion's validity radius (`|det J − 1| < 1`) rather than assuming it.

It names no material, stress law or strain measure. `StructuralSVK.SVKPullbackKernel`
and `FluidNavierStokes`'s P2/P1 pullback are the two implementations, which is the
proof the seam works.

**What examples 04 and 07 own.** Their `main.jl` files are *textually identical* — the
generic pipeline — and everything problem-specific is in `config.jl`: the mesh, material,
damping, the shape-field providers, the θ-basis bounds, and the monomial set. Example 04's
config carries a 35-line commentary deriving why the box is `[8, 2] / [8, 3] / [8, 4]`
rather than square (θ₂ is exactly isochoric so its degrees are lossless; θ₁ drives a
genuine `1/det J` geometric series and needs the terms). That reasoning is problem
knowledge and belongs exactly where it is.

**Verdict: nothing substantive to move.** Two small things could:

- `main.jl` lines 54–68 build the FE space and Dirichlet set by hand — 15 lines
  duplicated verbatim between 04 and 07. A `Common` helper (`grid → (dh, cv, free)` for a
  single vector field with one clamped facetset) would remove it, but the duplication is
  visible and inert.
- `BUILD_MSET()` in both configs hand-rolls an anisotropic z-total × θ-box monomial set.
  See finding **F5** — this is the one genuinely shared policy still written per example.

---

## 4. `StructuralSVK` — examples 01, 03, 08 (and 10)

**What `src/` owns.** The Ferrite SVK backend (`FerriteGeometricNonlinearity` for the
quadratic and cubic forms, `assemble_KM!`, and an `AbstractStress` layer covering both
isotropic Lamé and full anisotropic Voigt); the material/damping/forcing types
(`SVKMaterial`, `AnisotropicMaterial`, `CubicCrystal`, `RayleighDamping`,
`HarmonicForcing`); `mechanical_model` as the mesh→case entry point;
`build_model`, including the harmonic-forcing external system and
the spectrum-wide conjugate involution; and inspection helpers (`spectrum`,
`eigenfrequencies`, `print_mode_table`, `resonances`, `print_resonances`).

**What examples 01, 03, 08 own.** A CASE block — mesh path, material, damping, Dirichlet
set, FE order, master pairs, expansion order, forcing — and the three-step pipeline.
Example 01 is now a notebook whose *entire* code content is 16 lines. Example 08 adds
`mesh.jl` (CAD → Gmsh), which is problem geometry and belongs there.

**Verdict for 01/03/08: this is the reference state.** Nothing to move in either
direction.

### 4.1 Example 10 (turbine blade) — ❌ the outlier

Example 10 does not use `StructuralSVK` at all. It carries its own copy of the module:

| Example 10 file | Duplicates | Note |
|---|---|---|
| [`setup/assembly.jl`](examples/10_turbine_blade/Blade/setup/assembly.jl) (326 lines) | [`src/StructuralSVK/ferrite_assembly.jl`](src/StructuralSVK/ferrite_assembly.jl) (388) | A fork from before the `AbstractStress` split — isotropic-only, `λ`/`μ` passed loose. |
| [`setup/mesh.jl`](examples/10_turbine_blade/Blade/setup/mesh.jl) (134) | [`src/common/MeshIO/comsol_grid.jl`](src/common/MeshIO/comsol_grid.jl) (147) | Same `.mphtxt` reader, renamed `load_arch_mesh`. |
| [`setup/logging.jl`](examples/10_turbine_blade/Blade/setup/logging.jl) (168) | [`src/common/summary.jl`](src/common/summary.jl) | Predates the shared summary writer. |

[`MainLorenz.jl`](examples/10_turbine_blade/Blade/MainLorenz.jl) then hand-assembles
everything `build_model` exists to assemble: `K`/`M`/`C`, the eigenproblem, the mode
derivative blocks, `SpectralData` with its left blocks, the resonance set — and, at
line 36, a **hardcoded `conjugate_permutation = [2, 1, 3, 5, 4]` literal**, exactly the
failure mode the derived-permutation rule exists to prevent.

**Verdict: this is the largest single correction available.** Deleting `setup/` and
routing through `mechanical_model` + `build_model` would remove ~630 lines outright.

One caveat that makes it real work rather than mechanical: example 10's external system
is a **Lorenz oscillator** — a `DensePolynomial` external system with a nonlinear,
re-based linear part ([`MainLorenz.jl:155`](examples/10_turbine_blade/Blade/MainLorenz.jl#L155)).
`StructuralSVK.build_model` currently builds only the diagonal `±iΩ` external system that
`HarmonicForcing` implies. So the honest sequence is: (1) swap the three `setup/` files
for the library equivalents — pure deletion, no numerics change; (2) extend
`build_model` with an `external_system = …` keyword so a caller can supply its own; then
(3) the literal permutation becomes `full_conjugate_permutation(master_block, ext_sys)`.

---

## 5. `FluidNavierStokes` — examples 05 and 11

This is the module the original concern was about, and the story has two halves.

### 5.1 What already moved out of example 05

The module is now 4602 lines and explicitly records its own provenance. Comparing against
the pre-promotion layout, the following were **example files and are now library files**:

| Now in `src/FluidNavierStokes/` | Was |
|---|---|
| `fem_setup.jl`, `steady_state.jl`, `linear_operators.jl`, `fluid_maps.jl`, `energy_gram.jl` | `examples/05/fem/*.jl`, `solver/steady_state.jl` |
| `eigensolver.jl` (+ the `AbstractModeNormalisation` gauge types) | `examples/05/solver/eigensolver.jl` |
| `fom_orbit.jl` (599) — IMEX θ-integrator, Picard periodic orbit, `measure_orbit` | `examples/05/solver/{time_integration,picard_orbit}.jl` |
| `rom_analysis.jl` (1144) — slaving, PALC continuation, `rom_invariants`, Domb–Sykes and ratio-test diagnostics | `examples/05/solver/rom_palc.jl` + `invariants.jl` |
| `resummation.jl` (297) — Padé summation past the radius of convergence | (new) |
| `observables.jl` (188) — lift functional/polynomial, VTK bundle, CSV exports | `examples/05/exports.jl` |
| `types.jl`, `fluid_model.jl`, `build_model.jl` | ~200 lines of hand-wiring in `examples/05/main.jl` |

The comment at [`rom_analysis.jl:5`](src/FluidNavierStokes/rom_analysis.jl#L5) names the
motive plainly: *"three copies of the same algorithm had drifted apart"*. And
[`fluid_model.jl`](src/FluidNavierStokes/fluid_model.jl) absorbed the `−D` Reynolds
scaling that used to be applied in the driver, where *"the convention lived in a comment
and depended on two copies of `D` agreeing"*.

**On the FOM-side tooling specifically** (the time integrator, Picard orbits, PALC
continuation, the TKE observable): all four are now in `src/`, and that is the right
call. Each takes every parameter as an argument and holds no example configuration;
`fom_orbit.jl` records that it moved *verbatim* for exactly that reason. The DNS
bookkeeping that remains in the notebook — which Re values to sweep, seeding from the
branch, subtracting the steady-lift offset — is per-study choice and belongs there.

### 5.2 What example 05 still holds — 🟡

The notebook is 920 code lines across 16 cells. Classifying them:

| Cell | Lines | Content | Belongs |
|---|---:|---|---|
| §1 config | 162 | Re₀, orders, mesh sizes, gauge, promotion targets, tolerances — with the reasoning | **Example** ✅ |
| §2 mesh | 85 | Gmsh Turek–Schäfer generator | **Example** ✅ (module docstring says so explicitly) |
| §3–4 | 33 | `fluid_model`, `solve_hopf_eigenproblem` calls | **Example** ✅ |
| §5 | 10 | prints `homological_denominators` | **Example** ✅ |
| §6 mode selection | 79 | α-pairing gate, denominator gate, **conjugation closure of the promoted set** | 🟡 **movable** |
| §7 monomial set | 67 | the promoted-coordinate `mset` rule + activity/boundedness assertions | 🟡 **movable** |
| §8–13 | ~310 | saving, diagnostics printing, branch tracing, validation, figures | **Example** ✅ |
| §14 DNS | 175 | per-Re sweep bookkeeping around `fom_orbit` kernels | **Example** ✅ |

Two things are library-shaped:

- **Conjugation closure of a promoted master set** (§6, ~15 lines): given a set of modes
  to promote, add each complex mode's conjugate partner so the master set stays
  conjugation-invariant. This is a correctness invariant of the method, not a study
  choice — get it wrong and the ROM has no real realisation. `build_model` already
  enforces the *adjacent* case via `_master_conjugate_pairing`; this is the same
  invariant one step earlier, at selection time.
- **The promoted-coordinate monomial rule** (§7, ~20 lines): "a promoted coordinate rides
  along with a core monomial if the mixing is shallow (`core ≤ PROMOTED_CORE_ORD`) **or**
  the companion is pure parameter". That is a genuine expansion policy with a derived
  justification, and it will be needed by any promoted run. See **F5**.

The `α`-ratio and denominator *gates* around them should stay: they are thresholds this
study chose, and the notebook argues for each value at length.

**Verdict on the original concern.** Example 05's physics-specific code has largely been
promoted already; what remains is ~35 lines of genuine policy plus configuration and
narration. The premise was right, and it has mostly been acted on.

### 5.3 Example 11 (parametric Kármán profile) — ❌ where the pattern re-formed

5185 lines of Julia in three files. `main.jl` is a 56-line command dispatcher; the
substance is:

- **[`model.jl`](examples/11_parametric_karman_profile/model.jl) (1981)** — five nested
  modules: `…Geometry` (Joukowski profile, chord/camber metrics, boundary-simplicity
  certification), `…Mesh` (Gmsh generation, 429 lines), `…Pipeline` (787),
  `…Publication`, `…Coordinates`.
- **[`analysis.jl`](examples/11_parametric_karman_profile/analysis.jl) (3148)** — fifteen
  stage modules (`SetupCentered`, `ContinueHopf`, `RefineHopf`, `VerifyHopf`,
  `FixedGeometryROMs`, `BuildBranches`, `PeriodicFOM`, `PublicationValidation`, …).

Most of this is a **research campaign** — restartable stages, publication data, figure
manifests, cleanup — and campaigns belong in examples. The exception is
`ParametricKarmanPipeline`, which is a fluid module wearing an example's clothes. From
[`model.jl:824–1033`](examples/11_parametric_karman_profile/model.jl#L824) alone:

| Function | What it is |
|---|---|
| `eigen_residual` | Scale-independent generalised-eigenproblem residual |
| `mass_overlap`, `_phase_align`, `_midmass_action` | Mass-weighted mode correlation and phase alignment across two grids |
| `select_initial_mode`, `select_anchor_mode`, `select_tracked_mode` | Mode selection and continuation-tracking heuristics |
| `spectral_separation` | Distance to the nearest non-conjugate eigenvalue |
| `_hopf_at`, `find_midpoint_hopf`, `find_circle_hopf`, `continue_hopf_geometry` | Warm-started steady continuation + Hopf bracketing and bisection |
| `_steady_residual_norm` | Newton-residual verification of a base state |
| `outlet_energy_diagnostics`, `outlet_backflow_diagnostic`, `boundary_flux_diagnostic` | Outflow-boundary health checks |
| `geometry_provider`, `setup_profile_fem`, `profile_freestream_bc` | Thin adapters over `ParametricGeometry` / `setup_fem` |

None of it mentions a Joukowski profile. **Hopf-point continuation in a physical
parameter, mode tracking across a moving mesh, and base-state verification are exactly
the fluid-DPIM operations `FluidNavierStokes` exists to own** — and the first is the
single most reusable thing in the example, since every new geometry needs its neutral
point found before anything else can run.

**Verdict:** promote the mode-tracking group (`mass_overlap`, `_phase_align`,
`select_tracked_mode`, `spectral_separation`) and the Hopf-continuation group
(`_hopf_at`, `find_midpoint_hopf`, `continue_hopf_geometry`, `_steady_residual_norm`)
into `FluidNavierStokes`. Roughly 300 of `Pipeline`'s 787 lines. Leave the geometry, the
mesh generator, the coordinate normalisation (`χ`), and all fifteen analysis stages in
the example — those are the study.

The rest of `Pipeline` is adapters (`setup_profile_fem`, `geometry_provider`) — a sign
the module's own entry points are slightly too narrow for a non-cylinder obstacle, worth
noting but not urgent.

---

## 6. Findings, ranked

**F1 · Example 10 re-implements `StructuralSVK` and `MeshIO`.** ~630 lines of duplicated
backend plus a hardcoded conjugate-permutation literal. The duplicate `assembly.jl` is a
*fork*: it predates the `AbstractStress` split, so it silently cannot express anisotropic
materials. Highest value, and steps (1)–(2) are independent.
→ Delete `setup/{assembly,mesh,logging}.jl`; add an `external_system` keyword to
`StructuralSVK.build_model` for the Lorenz forcing; derive the permutation.

**F2 · Example 11's `ParametricKarmanPipeline` is a second fluid module.** ~300 lines of
geometry-independent fluid-DPIM operations — Hopf continuation, mode tracking, base-state
verification — with no home in `src/`. They will be needed verbatim by the next profile.
→ Promote those two groups into `FluidNavierStokes`.

**F3 · Two Hopf-mode selection strategies now exist, in different places.**
`FluidNavierStokes.solve_hopf_eigenproblem` selects by smallest `|Re λ|` (or a target
frequency); example 11 selects by frequency band *plus* mass-weighted overlap against the
previous grid, because the first heuristic picks a damped branch when the boundary
conditions change. The second is strictly better-informed and is invisible to every other
caller.
→ Fold the overlap-tracking selector into the module as an alternative strategy.

**F4 · The reference length is defined twice.** `_CYL_D = 0.1` in
[`src/FluidNavierStokes/fem_setup.jl`](src/FluidNavierStokes/fem_setup.jl) (as the
`reference_length` default) and again in example 05's mesh-generation cell. They must
agree or `ν = D/Re` is wrong, and nothing checks it. Partly mitigated already — the
`−D` scaling moved into `fluid_model` — but the constant itself is still duplicated.
→ Have the example pass `reference_length` explicitly from its own mesh constants, so
one value flows from the geometry that produced the mesh.

**F5 · The expansion policy is hand-rolled in every parametric example.** `BUILD_MSET()`
in examples 04 and 07, the `mset_exps` loop in example 05 §7, and example 11's own
variant. MORFE offers `build_multiindex_set(policy, nvar)` as a dispatch seam, and
`MODULE_ARCHITECTURE.md` §4.4 already proposes `ThetaBoxExpansion`.
→ Add two policy types — a θ-box for the parametric structural pair, and a
promoted-coordinate policy for the fluid runs. Then `parametrise(model, spectral,
ThetaBoxExpansion(9, [4, 4]))` replaces a comprehension in each config.

**F6 · [`examples/common/results_io.jl`](examples/common/results_io.jl) is dead.** 52
lines; the only reference to it is its own usage comment. Its `save_rom` was superseded by
`MORFE.save_rom` and its `write_summary` by `Common.write_summary`.
→ Delete. (Note `src/MORFEFerrite.jl` no longer carries the export-clash comment that
used to reference it — the cleanup is already half done.)

**F7 · Examples 04 and 07 duplicate 15 lines of FE-space boilerplate.** Grid → `ip`/`qr`/
`cv` → `DofHandler` → `ConstraintHandler` → `free`. Inert, but it is the one part of the
"textually identical" pipeline that is copied rather than shared.
→ Optional `Common` helper.

**F8 · Example 09 is an unopened `.7z` archive.** No Julia. Either a future
`Electromechanical` module or it should be removed from `examples/`.
→ Decide; it currently reads as an example that does not run.

---

## 7. What should *not* move

Worth stating, because "move physics into the library" has a natural overshoot:

- **Mesh generation.** Examples 05, 08 and 11 each generate their own Gmsh geometry.
  `FluidNavierStokes`'s docstring makes this explicit — `setup_fem` reads a `.msh`,
  it does not create one. Geometry is the problem, not the physics.
- **Case constants and their reasoning.** Example 05 §1 and example 04's `config.jl`
  contain long derivations of why a tolerance or a box bound has its value. That
  commentary is worth more next to the number than in a docstring.
- **Study orchestration.** Example 11's fifteen restartable stages, publication manifests
  and figure scripts are a campaign, not an API.
- **Validation thresholds.** Each example's tolerances encode what "unchanged" means for
  *that* reference — including example 05's note that a promoted run carries a legitimate
  extra deviation the linear tolerances must still not absorb.
- **Narrative.** The notebooks' markdown sections are the reason they read well. Promoting
  code out of a notebook should not promote the explanation with it.

---

## 8. Appendix — sizes

`src/` (Julia, 9967 lines total):

| Module | Lines |
|---|---:|
| `FluidNavierStokes` | 4602 |
| `ParametricGeometry` | 2085 |
| `StructuralSVK` | 1641 |
| `common` | 1639 |

`examples/` (Julia, excluding `results/`):

| Example | Lines | Notebook code | Python |
|---|---:|---:|---:|
| 01 clamped beam | 36 | 16 | — |
| 03 arch (COMSOL) | 360 | — | — |
| 04 parametric beam | 746 | — | 254 |
| 05 Kármán | 0 | 920 | 569 |
| 07 parametric arch | 281 | — | — |
| 08 MEMS micromirror | 219 | — | — |
| 09 electromechanical | 0 | — | — |
| 10 turbine blade | 1241 | — | — |
| 11 parametric Kármán | 5185 | — | ✓ |
| `common` | 52 | — | — |
| `mesh_import` | 580 | — | — |

The two ratios that summarise the audit: example 10 carries 1241 lines against a
1641-line module it does not use, and example 11 carries 5185 lines against the
4602-line module it partially reimplements.
