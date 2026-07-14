using FerriteGmsh
using Arpack
using LinearMaps

function load_arch_mesh(mesh_path::AbstractString)
    return MORFEFerrite.load_comsol_grid(mesh_path, Set([1, 11]))
end
