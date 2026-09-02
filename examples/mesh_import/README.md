# mesh_import — FEM mesh format conversion utilities

Demonstrates converting Abaqus and COMSOL mesh files to GMSH format via the
`Common.MeshIO` subsystem exported from MORFEFerrite.

**Entry scripts:**
- `Abaqus/demo_abaqus_to_gmsh.jl` — Abaqus `.inp` → GMSH `.msh`
- `Comsol/demo_comsol_to_gmsh.jl` — COMSOL `.mphtxt` → GMSH `.msh`

Reusable `.inp` and `.mphtxt` source fixtures are included. Generated `.msh`
files are written to temporary directories by the demos and tests.

## How to run

```bash
julia --project=. examples/mesh_import/Abaqus/demo_abaqus_to_gmsh.jl
julia --project=. examples/mesh_import/Comsol/demo_comsol_to_gmsh.jl
```
