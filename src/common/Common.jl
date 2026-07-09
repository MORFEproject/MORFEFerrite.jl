"""
`MORFEFerrite.Common` — shared Ferrite backend layer.

- `load_comsol_grid` — COMSOL `.mphtxt` → Ferrite `Grid` (quadratic prism cells).
- `ParaviewExport`   — `write_paraview_*` stubs; implementations live in the
  `MORFEFerriteWriteVTKExt` extension (activated by `using WriteVTK`).
"""
module Common

using Ferrite

include("mesh_comsol.jl")
include("paraview.jl")

using .ParaviewExport

export load_comsol_grid
export write_paraview_mesh, write_paraview_modes,
	write_paraview_manifold, write_paraview_deformation

end # module Common
