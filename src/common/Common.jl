"""
`MORFEFerrite.Common` — shared Ferrite backend layer.

- `MeshIO` — COMSOL/Abaqus/Gmsh mesh loading and conversion.
- `node_dof`, `free_dofs_at_nodes` — mesh node + direction → DOF index.
- `ParaviewExport`   — `write_paraview_*` stubs; implementations live in the
  `MORFEFerriteWriteVTKExt` extension (activated by `using WriteVTK`).
"""
module Common

using Ferrite

include("assembled_model.jl")
include("summary.jl")
include("MeshIO/MeshIO.jl")
include("dof_lookup.jl")
include("paraview.jl")

using .MeshIO
using .ParaviewExport

export AbstractAssembledModel, build_model
export write_summary, summary_entries, stage_timings
export MeshIO,
       load_comsol_grid,
       abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol,
       node_dof, free_dofs_at_nodes
export write_paraview_mesh, write_paraview_modes,
	write_paraview_manifold, write_paraview_deformation, write_paraview_p2p1,
	write_paraview_p2p1_phase_animation

end # module Common
