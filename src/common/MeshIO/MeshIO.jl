"""
`MORFEFerrite.Common.MeshIO` — mesh loading and conversion utilities.

The subsystem owns direct COMSOL-to-Ferrite loading and conversions between
COMSOL, Abaqus, and Gmsh mesh formats.
"""
module MeshIO

using Ferrite

include("comsol_grid.jl")
include("AbaqusToGmsh.jl")
include("ComsolToGmsh.jl")
include("GmshToComsol.jl")

using .AbaqusToGmsh: abaqus_to_gmsh, abaqus_to_gmsh_linear
using .ComsolToGmsh: comsol_to_gmsh, comsol_to_gmsh_linear
using .GmshToComsol: gmsh_to_comsol

export load_comsol_grid,
       abaqus_to_gmsh, abaqus_to_gmsh_linear,
       comsol_to_gmsh, comsol_to_gmsh_linear,
       gmsh_to_comsol

end # module MeshIO
