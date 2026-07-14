# Graph Report - MORFEFerrite.jl  (2026-07-14)

## Corpus Check
- 59 files · ~37,997 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 487 nodes · 495 edges · 59 communities (45 shown, 14 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `115b1a8a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_3. Execution plan|3. Execution plan]]
- [[_COMMUNITY_fom_reference.jl|fom_reference.jl]]
- [[_COMMUNITY_nonlinearity.jl|nonlinearity.jl]]
- [[_COMMUNITY_low_level.jl|low_level.jl]]
- [[_COMMUNITY_main.jl|main.jl]]
- [[_COMMUNITY_validate_tke.jl|validate_tke.jl]]
- [[_COMMUNITY_theta_series.jl|theta_series.jl]]
- [[_COMMUNITY_main.jl|main.jl]]
- [[_COMMUNITY_StructuralSVK|StructuralSVK]]
- [[_COMMUNITY_exports.jl|exports.jl]]
- [[_COMMUNITY_main.jl|main.jl]]
- [[_COMMUNITY_visualise_structure_and_modes.jl|visualise_structure_and_modes.jl]]
- [[_COMMUNITY_test_structural_svk.jl|test_structural_svk.jl]]
- [[_COMMUNITY_compare_orders.py|compare_orders.py]]
- [[_COMMUNITY_solve_rom.jl|solve_rom.jl]]
- [[_COMMUNITY_energy_gram.jl|energy_gram.jl]]
- [[_COMMUNITY_01 — Clamped-clamped beam (Ferrite.jl backend)|01 — Clamped-clamped beam (Ferrite.jl backend)]]
- [[_COMMUNITY_MORFEFerriteWriteVTKExt|MORFEFerriteWriteVTKExt]]
- [[_COMMUNITY_steady_state.jl|steady_state.jl]]
- [[_COMMUNITY_03 — Arch COMSOL wedge|03 — Arch COMSOL wedge]]
- [[_COMMUNITY_eigensolver.jl|eigensolver.jl]]
- [[_COMMUNITY_08 — MEMS micromirror (gimbal)|08 — MEMS micromirror (gimbal)]]
- [[_COMMUNITY_fluid_maps.jl|fluid_maps.jl]]
- [[_COMMUNITY_FluidNavierStokes|FluidNavierStokes]]
- [[_COMMUNITY_linear.jl|linear.jl]]
- [[_COMMUNITY_CLAUDE|CLAUDE.md]]
- [[_COMMUNITY_main.jl|main.jl]]
- [[_COMMUNITY_main.jl|main.jl]]
- [[_COMMUNITY_main.jl|main.jl]]
- [[_COMMUNITY_05 — Kármán vortex street|05 — Kármán vortex street]]
- [[_COMMUNITY_rom_palc.jl|rom_palc.jl]]
- [[_COMMUNITY_results_io.jl|results_io.jl]]
- [[_COMMUNITY_MORFEFerrite|MORFEFerrite]]
- [[_COMMUNITY_linear_operators.jl|linear_operators.jl]]
- [[_COMMUNITY_ParaviewExport|ParaviewExport]]
- [[_COMMUNITY_load_mesh.jl|load_mesh.jl]]
- [[_COMMUNITY_node_dof_table.jl|node_dof_table.jl]]
- [[_COMMUNITY_arch_geometry.jl|arch_geometry.jl]]
- [[_COMMUNITY_RayleighEigenSolver|RayleighEigenSolver]]
- [[_COMMUNITY_time_integration.jl|time_integration.jl]]
- [[_COMMUNITY_mesh.jl|mesh.jl]]
- [[_COMMUNITY_Common|Common]]
- [[_COMMUNITY_mesh_comsol.jl|mesh_comsol.jl]]
- [[_COMMUNITY_fem_setup.jl|fem_setup.jl]]
- [[_COMMUNITY_postprocess.jl|postprocess.jl]]
- [[_COMMUNITY_types.jl|types.jl]]
- [[_COMMUNITY_mesh.jl|mesh.jl]]
- [[_COMMUNITY_picard_orbit.jl|picard_orbit.jl]]
- [[_COMMUNITY_bootstrap.jl|bootstrap.jl]]
- [[_COMMUNITY_run_gates.jl|run_gates.jl]]
- [[_COMMUNITY_validate.jl|validate.jl]]
- [[_COMMUNITY_README|README.md]]
- [[_COMMUNITY_ParametricStructural.jl|ParametricStructural.jl]]
- [[_COMMUNITY_mechanical_model|mechanical_model]]
- [[_COMMUNITY_runtests.jl|runtests.jl]]

## God Nodes (most connected - your core abstractions)
1. `StructuralSVK` - 14 edges
2. `3. Execution plan` - 10 edges
3. `MORFEFerriteWriteVTKExt` - 9 edges
4. `Migration Plan — Move the Ferrite code out of MORFE.jl into MORFEFerrite` - 9 edges
5. `1. Inventory — where Ferrite appears today` - 9 edges
6. `01 — Clamped-clamped beam (Ferrite.jl backend)` - 9 edges
7. `FluidNavierStokes` - 8 edges
8. `03 — Arch COMSOL wedge` - 8 edges
9. `08 — MEMS micromirror (gimbal)` - 8 edges
10. `process_branch()` - 6 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (59 total, 14 thin omitted)

### Community 0 - "3. Execution plan"
Cohesion: 0.07
Nodes (26): 1. Inventory — where Ferrite appears today, 1a. The extension seam that MUST stay in MORFE (backend-agnostic), 1b. Ferrite-specific `ext/` extensions → MOVE (structural + VTK), 1c. Fluid Navier–Stokes code — currently ONLY in example 05 → PROMOTE to a submodule, 1d. `src/` core stubs → MOVE, 1e. Ambiguous: `FEMUtility` mesh converters (NOT Ferrite — leave in MORFE), 1f. NOT Ferrite (leave untouched), 1g. Tests (+18 more)

### Community 1 - "fom_reference.jl"
Cohesion: 0.13
Nodes (19): DelimitedFiles, Ferrite, FerriteGmsh, Gmsh, KLU, LinearAlgebra, MORFE, MORFE.Polynomials (+11 more)

### Community 2 - "nonlinearity.jl"
Cohesion: 0.17
Nodes (15): ∇adj_series(), _all_cubic!(), _all_quadratic!(), E_nl_adj_series(), _ensure_cubic!(), _ensure_quadratic!(), _expand_multiindex(), Ferrite (+7 more)

### Community 3 - "low_level.jl"
Cohesion: 0.12
Nodes (15): AbstractEigensolver, Arpack, Ferrite, FerriteGmsh, Float64, Int, LinearAlgebra, LinearMaps (+7 more)

### Community 4 - "main.jl"
Cohesion: 0.12
Nodes (15): Arpack, DelimitedFiles, Ferrite, FerriteGmsh, Gmsh, LinearAlgebra, LinearMaps, MORFE (+7 more)

### Community 5 - "validate_tke.jl"
Cohesion: 0.12
Nodes (11): Ferrite, FerriteGmsh, LinearAlgebra, MORFE, MORFE.Polynomials, MORFEFerrite.FluidNavierStokes, Pkg, Printf (+3 more)

### Community 6 - "theta_series.jl"
Cohesion: 0.20
Nodes (14): det_adj_series(), _entry(), inv_det_power(), LinearAlgebra, MORFE, StaticArrays, Tensors, nterms() (+6 more)

### Community 7 - "main.jl"
Cohesion: 0.13
Nodes (14): BeamParametricGeom, Arpack, Ferrite, FerriteGmsh, LinearAlgebra, LinearMaps, MORFE, MORFEFerrite (+6 more)

### Community 8 - "StructuralSVK"
Cohesion: 0.13
Nodes (12): Arpack, Common, Ferrite, FerriteGmsh, LinearAlgebra, LinearMaps, MORFE, Printf (+4 more)

### Community 9 - "exports.jl"
Cohesion: 0.15
Nodes (7): DelimitedFiles, Printf, Serialization, TeeIO, to_gb(), write_summary(), IO

### Community 10 - "main.jl"
Cohesion: 0.14
Nodes (12): Arpack, Ferrite, FerriteGmsh, LinearAlgebra, LinearMaps, MORFE, MORFEFerrite, Pkg (+4 more)

### Community 11 - "visualise_structure_and_modes.jl"
Cohesion: 0.15
Nodes (12): Arpack, Ferrite, FerriteGmsh, LinearAlgebra, LinearMaps, MORFE, MORFEFerrite, Pkg (+4 more)

### Community 12 - "test_structural_svk.jl"
Cohesion: 0.15
Nodes (11): Arpack, Ferrite, FerriteGmsh, LinearAlgebra, LinearMaps, MORFE, MORFEFerrite, MORFEFerrite.StructuralSVK (+3 more)

### Community 13 - "compare_orders.py"
Cohesion: 0.35
Nodes (11): lift_series(), load_fom_reference(), load_gram(), load_lift(), main(), process_branch(), process_run(), (Re, max_abs_lift, avg_TKE, converged) rows from fom_reference.jl, if present. (+3 more)

### Community 14 - "solve_rom.jl"
Cohesion: 0.22
Nodes (10): LinearAlgebra, MORFE, MORFE.Polynomials, Pkg, Printf, Serialization, StaticArrays, process_dir() (+2 more)

### Community 15 - "energy_gram.jl"
Cohesion: 0.24
Nodes (9): assemble_velocity_mass_full(), domain_area(), DelimitedFiles, Ferrite, LinearAlgebra, MORFE, SparseArrays, prepare_energy_gram() (+1 more)

### Community 16 - "01 — Clamped-clamped beam (Ferrite.jl backend)"
Cohesion: 0.20
Nodes (9): 01 — Clamped-clamped beam (Ferrite.jl backend), Expected outputs, Harmonic forcing, How to run (high-level UI), Measured runtime, Model, Notes, Reference results (+1 more)

### Community 17 - "MORFEFerriteWriteVTKExt"
Cohesion: 0.20
Nodes (6): Ferrite, MORFE, MORFEFerrite, Printf, WriteVTK, MORFEFerriteWriteVTKExt

### Community 18 - "steady_state.jl"
Cohesion: 0.27
Nodes (8): _assemble_element!(), assemble_steady_nse!(), Ferrite, KLU, LinearAlgebra, SparseArrays, solve_steady_state(), _split_free_solution()

### Community 19 - "03 — Arch COMSOL wedge"
Cohesion: 0.22
Nodes (8): 03 — Arch COMSOL wedge, Expected outputs, Historical results, How to run, Measured runtime, Model, Reference results, Subdirectories

### Community 20 - "eigensolver.jl"
Cohesion: 0.22
Nodes (7): Arpack, KLU, LinearAlgebra, LinearMaps, Printf, SparseArrays, StaticArrays

### Community 21 - "08 — MEMS micromirror (gimbal)"
Cohesion: 0.22
Nodes (8): 08 — MEMS micromirror (gimbal), Expected outputs, Model, Notes, Step 1 — export a STEP from the `.SLDPRT`, Step 2 — mesh it, Step 3 — build the ROM, Why the mesh is not committed

### Community 22 - "fluid_maps.jl"
Cohesion: 0.25
Nodes (5): FluidConvection(), Ferrite, LinearAlgebra, MORFE, SparseArrays

### Community 23 - "FluidNavierStokes"
Cohesion: 0.22
Nodes (8): FluidNavierStokes, DelimitedFiles, Ferrite, FerriteGmsh, KLU, LinearAlgebra, MORFE, SparseArrays

### Community 24 - "linear.jl"
Cohesion: 0.22
Nodes (4): Ferrite, MORFE, SparseArrays, Tensors

### Community 25 - "CLAUDE.md"
Cohesion: 0.25
Nodes (6): Coding conventions, Examples, Known limitations, Project Overview, Submodules, Tests & gates

### Community 26 - "main.jl"
Cohesion: 0.25
Nodes (7): Arpack, Ferrite, FerriteGmsh, LinearMaps, MORFE, MORFEFerrite, Pkg

### Community 27 - "main.jl"
Cohesion: 0.25
Nodes (7): Arpack, Ferrite, FerriteGmsh, LinearMaps, MORFE, MORFEFerrite, Pkg

### Community 28 - "main.jl"
Cohesion: 0.25
Nodes (7): Arpack, Ferrite, FerriteGmsh, LinearMaps, MORFE, MORFEFerrite, Pkg

### Community 29 - "05 — Kármán vortex street"
Cohesion: 0.29
Nodes (6): 05 — Kármán vortex street, Files, Model, Outputs, Validation, Workflow

### Community 30 - "rom_palc.jl"
Cohesion: 0.62
Nodes (6): rom_hopf_eta(), rom_palc_step(), rom_palc_tangent(), rom_po_frequency(), rom_po_residual(), _rom_R1()

### Community 31 - "results_io.jl"
Cohesion: 0.29
Nodes (3): LinearAlgebra, Printf, Serialization

### Community 32 - "MORFEFerrite"
Cohesion: 0.29
Nodes (6): FluidNavierStokes, ParametricStructural, Common, MORFE, MORFEFerrite, StructuralSVK

### Community 33 - "linear_operators.jl"
Cohesion: 0.33
Nodes (4): _assemble_linear_element!(), assemble_linear_operators(), Ferrite, SparseArrays

### Community 36 - "load_mesh.jl"
Cohesion: 0.40
Nodes (3): Arpack, FerriteGmsh, LinearMaps

### Community 37 - "node_dof_table.jl"
Cohesion: 0.40
Nodes (3): Ferrite, Pkg, Printf

### Community 38 - "arch_geometry.jl"
Cohesion: 0.50
Nodes (3): arch_jacobian(), arch_jacobian_pair(), Tensors

### Community 39 - "RayleighEigenSolver"
Cohesion: 0.40
Nodes (4): AbstractEigensolver, Float64, Int, RayleighEigenSolver

### Community 42 - "Common"
Cohesion: 0.50
Nodes (3): ParaviewExport, Common, Ferrite

### Community 43 - "mesh_comsol.jl"
Cohesion: 0.83
Nodes (3): load_comsol_grid(), QuadraticWedge, _read_mesh()

### Community 46 - "types.jl"
Cohesion: 1.00
Nodes (3): HarmonicForcing(), RayleighDamping(), SVKMaterial()

## Knowledge Gaps
- **269 isolated node(s):** `Pkg`, `MORFE`, `MORFEFerrite`, `MORFEFerrite.StructuralSVK`, `Ferrite` (+264 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Pkg`, `MORFE`, `MORFEFerrite` to the rest of the system?**
  _270 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `3. Execution plan` be split into smaller, more focused modules?**
  _Cohesion score 0.07407407407407407 - nodes in this community are weakly interconnected._
- **Should `fom_reference.jl` be split into smaller, more focused modules?**
  _Cohesion score 0.12631578947368421 - nodes in this community are weakly interconnected._
- **Should `low_level.jl` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Should `main.jl` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Should `validate_tke.jl` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Should `main.jl` be split into smaller, more focused modules?**
  _Cohesion score 0.13333333333333333 - nodes in this community are weakly interconnected._