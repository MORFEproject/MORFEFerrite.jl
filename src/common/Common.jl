"""
`MORFEFerrite.Common` — shared Ferrite backend layer.

- `load_comsol_grid` — COMSOL `.mphtxt` → Ferrite `Grid` (quadratic prism cells).
- `node_dof`, `free_dofs_at_nodes` — mesh node + direction → DOF index.
- `ParaviewExport`   — `write_paraview_*` stubs; implementations live in the
  `MORFEFerriteWriteVTKExt` extension (activated by `using WriteVTK`).
"""
module Common

using Ferrite

include("assembled_model.jl")
include("mesh_comsol.jl")
include("dof_lookup.jl")
include("paraview.jl")

using .ParaviewExport

export AbstractAssembledModel, build_model
export load_comsol_grid, node_dof, free_dofs_at_nodes
export write_paraview_mesh, write_paraview_modes,
	write_paraview_manifold, write_paraview_deformation

end # module Common
