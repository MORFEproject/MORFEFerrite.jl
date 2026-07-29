# Node → DOF lookups. ROM post-processing works on free-DOF indices, while the
# physical quantities of interest ("the z-displacement of the mirror tip") are
# named by mesh node and direction; these two helpers bridge the two.

"""
	node_dof(dh, node_id, direction) -> Int

Global DOF index of `direction` (1 = x, 2 = y, 3 = z) at mesh node `node_id`.
Assumes a single vector field with one DOF per direction per node, which is what
[`mechanical_model`](@ref) builds.
"""
function node_dof(dh::Ferrite.DofHandler, node_id::Int, direction::Int)
    1 <= direction <= 3 ||
        throw(ArgumentError("direction must be 1 (x), 2 (y) or 3 (z); got $direction"))
    for cell_id in 1:Ferrite.getncells(dh.grid)
        k = findfirst(==(node_id), dh.grid.cells[cell_id].nodes)
        k === nothing && continue
        return Ferrite.celldofs(dh, cell_id)[3 * (k - 1) + direction]
    end
    throw(ArgumentError("node $node_id is not part of any cell of the grid"))
end

"""
	free_dofs_at_nodes(dh, free_to_local, node_ids, directions) -> Vector{Int}

Free-DOF indices (the row indices of `K`, `M` and of the parametrisation `W`)
for the given mesh nodes and directions. `free_to_local` is the map built by
[`mechanical_model`](@ref) and available as `model.info.free_to_local`.

Throws if a requested node/direction is constrained, since it then has no row.
"""
function free_dofs_at_nodes(
        dh::Ferrite.DofHandler,
        free_to_local::Dict{Int, Int},
        node_ids::AbstractVector{Int},
        directions::AbstractVector{Int}
)
    length(node_ids) == length(directions) ||
        throw(ArgumentError("node_ids and directions must have equal length"))
    return map(zip(node_ids, directions)) do (n, d)
        g = node_dof(dh, n, d)
        haskey(free_to_local, g) ||
            throw(ArgumentError("node $n direction $d (global DOF $g) is constrained"))
        free_to_local[g]
    end
end
