# Migration Plan — Move the Ferrite code out of MORFE.jl into MORFEFerrite

## Goal

Extract every Ferrite-specific piece of `MORFE.jl` into the new package
`MORFEFerrite` (`../MORFEFerrite/MORFEFerrite.jl`). After the split:

- `MORFE` stays a **FEM-backend-agnostic** core: it defines the abstract
  `FEMMultilinearMap` interface and the DPIM solver, and knows nothing about
  Ferrite.
- `MORFEFerrite` depends on MORFE and is an **umbrella package organised into
  per-physics submodules** (`StructuralSVK`, `FluidNavierStokes`, …) sharing a
  common Ferrite backend layer (mesh IO, VTK export, qp helpers). It exports its
  UI **directly** — no more `Base.get_extension`.

## Decisions (settled)

1. **Remove** `ferrite_nonlinearity` / `ferrite_assemble_KM!` / `write_paraview_*`
   stubs from MORFE entirely. MORFEFerrite owns those generic functions.
2. MORFEFerrite's `parametrise` **adds methods to `MORFE.parametrise`** dispatching
   on the domain model types — no second `parametrise`, no export clash.
3. Ferrite examples to move: **01, 03, 04, 05, 06, 07, 08**. Only `02` (Gridap) and
   the Gmsh mesh-import demos stay in MORFE.
4. MORFEFerrite is split into **domain submodules**: `StructuralSVK`,
   `FluidNavierStokes`, … each with its own FEM terms / model / `parametrise`
   methods, over a shared `common` backend layer.

---

## 1. Inventory — where Ferrite appears today

### 1a. The extension seam that MUST stay in MORFE (backend-agnostic)

`src/FullOrderModel/MultilinearMaps.jl`:
- `abstract type FEMMultilinearMap{ORD} <: AbstractMultilinearMap{ORD}`
- interface generics (exported from `src/MORFE.jl`): `fem_elements`, `fem_n_qp`,
  `fem_ndofs_per_cell`, `scatter_qp!`, `accumulate_qp!`, `assemble_element!`,
  `fem_getdetJdV`, `fem_qp_buffer`, `fem_reinit!`

Consumed by the batched RHS-C assembly path (`FullOrderModel.jl`, the
cohomological solve). **Not Ferrite — keep.** This is the contract every
MORFEFerrite domain submodule extends. Both `FerriteGeometricNonlinearity{2}`
(structural) and `FluidConvection{1}` (fluid) are concrete subtypes of it.

### 1b. Ferrite-specific `ext/` extensions → MOVE (structural + VTK)

| Extension entry | Weakdeps | Implementation | LoC |
|---|---|---|---|
| `ext/MORFEFerriteExt.jl` | `Ferrite` | `ext/FerriteBackend/ferrite_assembly.jl` | 16 + 316 |
| `ext/MORFEStructuralSVK.jl` | `Ferrite, FerriteGmsh, Arpack, LinearMaps` | `ext/StructuralSVK/{types,rayleigh_solver,mechanical_model,parametrise,postprocess}.jl` + `ext/FEMUtility/comsol_ferrite.jl` | 31 + ~594 |
| `ext/MORFEWriteVTKExt.jl` | `Ferrite, WriteVTK` | `ext/Paraview/write_vtk.jl` | 13 + 338 |

- `ferrite_assembly.jl` → `FerriteGeometricNonlinearity <: MORFE.FEMMultilinearMap{2}`
  (SVK geometric nonlinearity) + `ferrite_nonlinearity`, `ferrite_assemble_KM!`.
  **These belong to the `StructuralSVK` submodule** (they are structural-specific;
  the generic-sounding `ferrite_nonlinearity` name is really "SVK nonlinearity").
- `StructuralSVK/*` → the high-level UI (`SVKMaterial`, `RayleighDamping`,
  `HarmonicForcing`, `mechanical_model`, `parametrise`, `real_dynamics`,
  `print_equations`, `save_rom`).
- `comsol_ferrite.jl` (COMSOL `.mphtxt` → Ferrite `Grid`) → **`common`** layer
  (mesh IO, usable by any domain).
- `write_vtk.jl` → **`common`** VTK export (Ferrite-grid node permutations).

### 1c. Fluid Navier–Stokes code — currently ONLY in example 05 → PROMOTE to a submodule

Example 05 (Kármán vortex street) has no extension; its reusable FEM library lives
in `examples/05_karman_vortex_street/fem/`:
- `fluid_maps.jl` (314) — `FluidConvection <: FEMMultilinearMap{1}` (quadratic
  convective term) + `make_param_coupling` / `make_base_forcing` (`MultilinearMap`
  terms for the Re-parametric viscous coupling and base-flow forcing).
- `fem_setup.jl` (125), `linear_operators.jl` (278), `energy_gram.jl` (126),
  `mesh.jl` (146) — Ferrite setup, linear operators, TKE Gram, mesh.
- `solver/*` (steady_state, picard_orbit, rom_palc, time_integration, eigensolver;
  ~1040 LoC) — analysis/validation drivers. **Judgement call**: the reusable
  primitives (steady_state Newton, Picard orbit) could join `FluidNavierStokes`;
  bespoke study scripts stay example-local.

`FluidNavierStokes` = the reusable subset of 05's `fem/` (+ selected `solver/`).

### 1d. `src/` core stubs → MOVE

- `src/MORFE.jl`: `function ferrite_nonlinearity end`, `function
  ferrite_assemble_KM! end` + exports (declared also in `MultilinearMaps.jl`).
  → `StructuralSVK` submodule.
- `src/Export/ParaviewExport.jl` (the `ParaviewExport` stub submodule +
  `write_paraview_*` docstrings). → MORFEFerrite `common`.

### 1e. Ambiguous: `FEMUtility` mesh converters (NOT Ferrite — leave in MORFE)

`src/FEMUtility.jl` + `ext/MORFEGmshExt.jl` + `ext/FEMUtility/{AbaqusToGmsh,
ComsolToGmsh,GmshToComsol}.jl` are **Gmsh**-based. Leave in MORFE. Only
`comsol_ferrite.jl` from that folder moves (to `common`). Do **not** move the
folder wholesale.

### 1f. NOT Ferrite (leave untouched)

`MORFEArpackExt`, `MORFEPardisoExt`, `MORFEBifurcationKitExt`, `MORFEPlotsExt`,
`MORFESymbolicsExt`, the Gmsh/Abaqus/Comsol converters.

### 1g. Tests

- `test/StructuralSVK/{test_structural_svk.jl,run_gates.jl}` → MORFEFerrite
  (`test/StructuralSVK/`). Gate A: `main.jl` ≡ `low_level.jl` @1e-10; Gate B:
  zero-amplitude forcing ≡ autonomous @1e-10.
- Add `test/FluidNavierStokes/` for the promoted fluid module (see §5).
- `test/runtests.jl`: remove the `structural_svk` group and Ferrite lines from
  the `examples` group in MORFE; recreate in MORFEFerrite.
- `test/FEMUtility/test_gmsh_to_comsol.jl` (Gmsh) **stays** in MORFE. MORFE core
  tests otherwise do not use Ferrite (grep-verified).

### 1h. Examples

| Example | Domain / Ferrite use | Destination |
|---|---|---|
| `01_clamped_beam_ferrite` | SVK UI + `ferrite_nonlinearity` | MORFEFerrite (StructuralSVK) |
| `03_arch_comsol_wedge` | SVK + `write_paraview_*` | MORFEFerrite (StructuralSVK) |
| `08_mems_micromirror` | SVK backend | MORFEFerrite (StructuralSVK) |
| `04_parametric_clamped_beam` | local parametric-structural FEM | MORFEFerrite (candidate `ParametricStructural`, else example-local) |
| `07_parametric_arch` | local parametric-arch FEM + `write_paraview_*` | MORFEFerrite (candidate `ParametricStructural`) |
| `06_dielectric_elastomer_actuator` | local dielectric FEM | MORFEFerrite (candidate `DielectricElastomer`, else example-local) |
| `05_karman_vortex_street` | fluid `FluidConvection` etc. | MORFEFerrite (**FluidNavierStokes**) — source of the module |
| `02_clamped_beam_gridap` | **Gridap** | stays in MORFE |

04/06/07 build bespoke `FEMMultilinearMap` subtypes against the MORFE interface
directly. Recommended: move them to MORFEFerrite/examples for a backend-neutral
MORFE, but only promote their `fem/` into named submodules if reuse warrants it;
otherwise keep that assembly example-local under `MORFEFerrite/examples/…/fem/`.

---

## 2. Target shape of MORFEFerrite (umbrella + submodules)

```
MORFEFerrite.jl/
├── Project.toml              # deps: MORFE, Ferrite, FerriteGmsh, Arpack, LinearMaps,
│                             #       LinearAlgebra, SparseArrays, Serialization,
│                             #       Printf, StaticArrays, Statistics
│                             # weakdeps: WriteVTK  (VTK export optional)
├── src/
│   ├── MORFEFerrite.jl       # umbrella: using MORFE; include common + submodules;
│   │                         #   re-export chosen public names
│   ├── common/               # shared Ferrite backend layer
│   │   ├── Common.jl         # module MORFEFerrite.Common
│   │   ├── mesh_comsol.jl    (from ext/FEMUtility/comsol_ferrite.jl)
│   │   ├── paraview.jl       (ParaviewExport stubs, now owned here)
│   │   └── qp_helpers.jl     (shared qp-buffer / scatter utilities if extracted)
│   ├── StructuralSVK/
│   │   ├── StructuralSVK.jl  # module MORFEFerrite.StructuralSVK
│   │   ├── ferrite_assembly.jl   (FerriteGeometricNonlinearity, ferrite_nonlinearity)
│   │   ├── types.jl  rayleigh_solver.jl  mechanical_model.jl
│   │   ├── parametrise.jl        # methods on MORFE.parametrise(::MechanicalModel; …)
│   │   └── postprocess.jl
│   └── FluidNavierStokes/
│       ├── FluidNavierStokes.jl  # module MORFEFerrite.FluidNavierStokes
│       ├── maps.jl               (FluidConvection, param_coupling, base_forcing)
│       ├── fem_setup.jl  linear_operators.jl  energy_gram.jl  mesh.jl
│       └── parametrise.jl        # methods on MORFE.parametrise(::FluidModel; …)  (optional)
├── ext/
│   └── MORFEFerriteWriteVTKExt.jl  (from ext/Paraview/write_vtk.jl)
├── test/
│   ├── runtests.jl
│   ├── StructuralSVK/{test_structural_svk.jl,run_gates.jl}
│   └── FluidNavierStokes/…
└── examples/                 # 01,03,04,05,06,07,08 (retargeted)
```

Umbrella `src/MORFEFerrite.jl` sketch:
```julia
module MORFEFerrite
using MORFE
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using LinearAlgebra, SparseArrays, Serialization, Printf, StaticArrays, Statistics
include("common/Common.jl");                    using .Common
include("StructuralSVK/StructuralSVK.jl");      using .StructuralSVK
include("FluidNavierStokes/FluidNavierStokes.jl"); using .FluidNavierStokes
# re-export the public UI (or leave namespaced: MORFEFerrite.StructuralSVK.parametrise)
end
```
Submodules `import MORFE` and add methods to `MORFE.parametrise`, plus own their
domain generic functions (e.g. `StructuralSVK.ferrite_nonlinearity`).

Public API (no `get_extension`):
```julia
using MORFE, MORFEFerrite
const SVK = MORFEFerrite.StructuralSVK
beam = SVK.mechanical_model(msh; material=..., damping=..., ...)
rom  = SVK.parametrise(beam; master=[1], order=9)   # dispatches MORFE.parametrise
```

---

## 3. Execution plan

### Phase 0 — Prep & safety net
1. Capture golden references on the MORFE side: SVK Gate A/B CSVs (order 9 + order 5)
   and the Kármán order-9 R/lift outputs. These are the acceptance oracle.
2. Branch both repos (`feat/extract-ferrite` in MORFE, `feat/import-ferrite` in
   MORFEFerrite). Delete nothing from MORFE until MORFEFerrite is green.

### Phase 1 — Skeleton
3. Fill `MORFEFerrite/Project.toml`: hard deps as above (MORFE via `Pkg.develop`
   path `../../MORFE_jl`), `[weakdeps]`/`[extensions]` WriteVTK, `[compat]`
   mirroring MORFE (Ferrite=1, FerriteGmsh=1, Arpack=0.5.3, LinearMaps=3,
   WriteVTK=1, julia≥1.10).
4. Create the umbrella module + empty `Common`, `StructuralSVK`,
   `FluidNavierStokes` submodules so the package precompiles.

### Phase 2 — Common backend layer
5. Move `comsol_ferrite.jl` → `common/mesh_comsol.jl`; move `ParaviewExport`
   submodule → `common/paraview.jl`; move `ext/Paraview/write_vtk.jl` →
   `ext/MORFEFerriteWriteVTKExt.jl` retargeting `MORFE.ParaviewExport.*` →
   `MORFEFerrite.Common.ParaviewExport.*`. Qualify all MORFE-owned names.

### Phase 3 — StructuralSVK submodule
6. Move `ferrite_assembly.jl` + `StructuralSVK/*`. Fold `MORFEFerriteExt.jl`'s two
   glue one-liners into the submodule; define `ferrite_nonlinearity` /
   `ferrite_assemble_KM!` here. Make `parametrise` a **method on `MORFE.parametrise`**
   dispatching on the mechanical-model type. Qualify `MORFE.FEMMultilinearMap`,
   `MORFE.fem_elements`, etc.
7. Export the SVK UI names from the submodule; decide umbrella re-exports.

### Phase 4 — FluidNavierStokes submodule
8. Promote example 05's `fem/{fluid_maps,fem_setup,linear_operators,energy_gram,
   mesh}.jl` into `FluidNavierStokes/`. Turn the ad-hoc `include`s into a proper
   module; qualify MORFE interface names. Decide which `solver/*` primitives are
   library vs example-local. Optionally add `parametrise(::FluidModel; …)` methods.

### Phase 5 — Tests
9. Move `test/StructuralSVK/*`; fix `EX` path to MORFEFerrite's own `examples/01_…`.
   Add `test/FluidNavierStokes/` (at minimum a smoke/regression test of the order-9
   Kármán ROM against the Phase-0 golden, tolerance-aware per §5). Write
   `test/runtests.jl` running both.

### Phase 6 — Examples
10. Move examples 01,03,04,05,06,07,08 to `MORFEFerrite/examples/`. In each driver:
    - `using MORFE, MORFEFerrite`; drop `Base.get_extension`; `SVK.foo` →
      `MORFEFerrite.StructuralSVK.foo`; fluid `include("fem/…")` → `using
      MORFEFerrite: FluidNavierStokes` (or keep example-local for bespoke bits).
    - update the `Pkg.develop`/`Pkg.add` bootstrap to develop **both** MORFE and
      MORFEFerrite by path; add Ferrite/FerriteGmsh/Arpack/LinearMaps/StaticArrays
      (and Gmsh/WriteVTK where used).
    - regenerate each `Manifest.toml`. Ensure `using WriteVTK` at
      `write_paraview_*` call sites (03, 07) to activate the VTK ext.

### Phase 7 — Strip Ferrite from MORFE (only after MORFEFerrite is green)
11. Delete: `ext/MORFEFerriteExt.jl`, `ext/FerriteBackend/`,
    `ext/MORFEStructuralSVK.jl`, `ext/StructuralSVK/`,
    `ext/FEMUtility/comsol_ferrite.jl`, `ext/MORFEWriteVTKExt.jl`, `ext/Paraview/`,
    `src/Export/ParaviewExport.jl`.
12. `src/MORFE.jl`: remove `include("Export/ParaviewExport.jl")`, `using
    .ParaviewExport`, `write_paraview_*` exports, `ferrite_nonlinearity` /
    `ferrite_assemble_KM!` stubs + exports. **Keep** `FEMMultilinearMap` + the
    `fem_*` interface exports.
13. `src/FullOrderModel/MultilinearMaps.jl`: remove the two `ferrite_*` generic
    declarations/docstrings/exports; keep the `FEMMultilinearMap` interface;
    repoint the "see `ext/FerriteBackend/`" docstring to MORFEFerrite.
14. `Project.toml`: `[weakdeps]` drop `Ferrite`, `FerriteGmsh`, and `WriteVTK`
    (unused elsewhere); keep `Gmsh`. `[extensions]` remove the three Ferrite exts.
    `[extras]`/`[targets]` drop `Ferrite`, `FerriteGmsh` (keep `Arpack`/`LinearMaps`
    only if the Arpack eigensolver ext test needs them — verify). `[compat]` drop
    `Ferrite`, `FerriteGmsh`, `WriteVTK`.
15. `test/runtests.jl`: remove `structural_svk` group and Ferrite `examples` lines.
16. Docs: rewrite CLAUDE.md's "High-level API (MORFEStructuralSVK)" section to
    describe MORFEFerrite; keep the FEM-backend-interface paragraph but point at
    MORFEFerrite's `StructuralSVK`/`FluidNavierStokes` as reference
    implementations. Move/delete `HIGH_LEVEL_API_PLAN.md`. Run `graphify update .`
    in both repos.

### Phase 8 — Verify
17. MORFE: `Pkg.test("MORFE")` passes **with no Ferrite installed** (key proof of
    a clean cut).
18. MORFEFerrite: `test/runtests.jl` → SVK Gate A/B @1e-10; Fluid regression within
    tolerance; smoke-run one moved example per domain (01 + 05).

---

## 4. What stays vs moves (one-glance)

| Item | Verdict |
|---|---|
| `FEMMultilinearMap` + `fem_*` interface | **Stay** in MORFE (the seam) |
| DPIM solver, Multiindices, Polynomials, eigen, resonance, cohomology | **Stay** |
| Gmsh/Abaqus/COMSOL→Gmsh converters (`FEMUtility`, `MORFEGmshExt`) | **Stay** |
| Arpack/Pardiso/BifurcationKit/Plots/Symbolics exts | **Stay** |
| Example 02 (Gridap), Gmsh mesh-import demos | **Stay** |
| `FerriteGeometricNonlinearity`, `ferrite_nonlinearity`, SVK UI | **Move** → StructuralSVK |
| `FluidConvection` + fluid `fem/` (ex-05) | **Move/promote** → FluidNavierStokes |
| `comsol_ferrite.jl`, `write_vtk.jl`, `ParaviewExport` | **Move** → Common (+VTK ext) |
| `ferrite_*` / `write_paraview_*` stubs in `src/` | **Delete** from MORFE |
| Examples 01,03,04,05,06,07,08 | **Move** → MORFEFerrite/examples |

---

## 5. Risks & watch-items

- **Arpack cross-run gauge noise** (project memory): raw R coefficients are NOT
  bit-comparable across Arpack runs; SVK Gate A already borderline on this machine
  (2.8e-9 vs 1e-10). Run gates via the **example env**
  (`--project=examples/01_…`), not `Pkg.test`; **re-baseline** goldens inside
  MORFEFerrite rather than diffing against MORFE-side CSVs. The Kármán regression
  test must use a scale-invariant / physical-observable check (lift, avg-TKE, or
  FD invariance error), **not** raw R equality — the ROM's Arpack eigenvector
  gauge makes raw coefficients non-comparable across runs.
- **`parametrise` collision**: MORFE exports `parametrise` (low-level) and each
  domain adds methods — extending the same generic function avoids a "both export
  parametrise" warning under `using MORFE, MORFEFerrite`.
- **Submodule ownership of the `FEMMultilinearMap` interface**: each domain must
  `import MORFE` and add methods to `MORFE.fem_elements` etc. — verify no domain
  accidentally defines a *new* local `fem_elements` (would break the solver's
  `which(fem_elements, …)` dispatch).
- **Fluid promotion scope creep**: example 05's `solver/*` (Picard orbit, PALC,
  time integration) is ~1040 LoC of analysis. Keep the module lean — promote only
  reusable primitives; leave study drivers example-local. See project memory on
  the Kármán FOM-reference toolkit preserved at commit 383d51a.
- **Qualified references** everywhere: ext files currently run inside modules that
  `using Ferrite`+`using MORFE`; after the move they `using MORFE` (not *are*
  MORFE), so MORFE-owned names must be qualified.
- **`comsol_ferrite.jl` vs `MORFEGmshExt`** share `ext/FEMUtility/` but activate
  via different exts — move only the Ferrite file.
- **Manifest churn**: every moved example needs a fresh `Manifest.toml`
  (dev-path to two unregistered local packages).
- **Docstring/path references** in `MultilinearMaps.jl`, CLAUDE.md,
  `HIGH_LEVEL_API_PLAN.md` point at `ext/FerriteBackend/` /
  `examples/02_clamped_beam_gridap` — repoint to MORFEFerrite.

---

## 6. Definition of done

- MORFE has zero `Ferrite`/`FerriteGmsh` references (grep clean), still exports the
  `FEMMultilinearMap` interface, and `Pkg.test("MORFE")` passes with no Ferrite
  installed.
- MORFEFerrite: umbrella + `Common`/`StructuralSVK`/`FluidNavierStokes` submodules
  precompile; SVK Gate A/B pass; Kármán regression passes; examples run against
  `MORFE` + `MORFEFerrite` dev-paths.
- Docs / CLAUDE.md / graphify updated in both repos.
