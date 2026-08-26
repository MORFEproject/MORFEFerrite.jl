module MORFEFerriteWriteVTKExt

# Implementations of MORFEFerrite.Common.ParaviewExport.write_paraview_* for
# Ferrite grids. Activated when WriteVTK is loaded.

using MORFEFerrite
using MORFE
using Ferrite
using WriteVTK
using Printf

const PVExport = MORFEFerrite.Common.ParaviewExport

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Ferrite Lagrange{RefPrism,2} stores nodes in this order (see reference_coordinates):
#   1-6  : corners  (3 bottom, 3 top)
#   7    : mid(0-1) bottom        10: mid(1-2) bottom    8: mid(0-2) bottom
#   9    : mid(0-3) vertical      11: mid(1-4) vertical  12: mid(2-5) vertical
#   13   : mid(3-4) top           15: mid(4-5) top        14: mid(3-5) top
#   16-18: quad-face centres
#
# VTK_BIQUADRATIC_QUADRATIC_WEDGE (type 32) expects:
#   0-5  : corners (same mapping)
#   6    : mid(0-1) bottom   7: mid(1-2) bottom   8: mid(2-0) bottom
#   9    : mid(3-4) top     10: mid(4-5) top      11: mid(5-3) top
#   12   : mid(0-3) vert    13: mid(1-4) vert     14: mid(2-5) vert
#   15-17: quad-face centres in same order
#
# VTK position p needs Ferrite index PERM_FERRITE_TO_VTK[p]:
const PERM_FERRITE_TO_VTK = [1, 2, 3, 4, 5, 6,   # corners — identical
                              7, 10, 8,            # bottom edge mids: VTK 7,8,9
                              13, 15, 14,          # top    edge mids: VTK 10,11,12
                              9, 11, 12,           # vert   edge mids: VTK 13,14,15
                              16, 18, 17]          # face centres:     VTK 16,17,18

# Ferrite Lagrange{RefHexahedron, 2} (27-node) vs VTK_QUADRATIC_HEXAHEDRON (20-node, type 25):
# Ferrite: corners 1-8, bottom-edges 9-12, top-edges 13-16, vert-edges 17-20, face-centers/body 21-27
# VTK:     corners 0-7, bottom-edges 8-11, top-edges 12-15, vert-edges 16-19
# Face-center and body-center nodes (Ferrite 21-27) become orphaned points — invisible in Surface mode.
const PERM_FERRITE_TO_VTK_HEX20 = [1, 2, 3, 4, 5, 6, 7, 8,   # corners
                                    9, 10, 11, 12,              # bottom edges
                                    13, 14, 15, 16,             # top edges  (VTK pos 13-16)
                                    17, 18, 19, 20]             # vert edges (VTK pos 17-20)

"""
    _vtk_geometry(grid) -> (points, cells)

Extract a 3×n_nodes coordinate matrix and a `MeshCell` array from a
Ferrite 3-D grid.  Dispatches on the number of nodes per cell:
  - 18 nodes → VTK_BIQUADRATIC_QUADRATIC_WEDGE (18-node quadratic prism, type 32)
               Applies PERM_FERRITE_TO_VTK to reorder Ferrite nodes to VTK convention.
  - 6  nodes → VTK_WEDGE (linear wedge, type 13) — no reordering needed.
"""
function _vtk_geometry(grid::Ferrite.Grid{3})
    points = reduce(hcat, [collect(n.x) for n in grid.nodes])
    cells  = map(grid.cells) do cell
        nn = length(cell.nodes)
        if nn == 18
            MeshCell(VTKCellTypes.VTK_BIQUADRATIC_QUADRATIC_WEDGE,
                     [cell.nodes[i] for i in PERM_FERRITE_TO_VTK])
        elseif nn == 6
            MeshCell(VTKCellTypes.VTK_WEDGE, collect(cell.nodes))
        elseif nn == 27
            MeshCell(VTKCellTypes.VTK_QUADRATIC_HEXAHEDRON,
                     [cell.nodes[i] for i in PERM_FERRITE_TO_VTK_HEX20])
        else
            error("_vtk_geometry: unsupported cell size $nn (expected 6, 18, or 27).")
        end
    end
    return points, cells
end

"""Return a ParaView-friendly, z-padded representation of a 2-D Ferrite grid."""
function _vtk_geometry(grid::Ferrite.Grid{2})
    points = zeros(Float64, 3, length(grid.nodes))
    for (i, node) in enumerate(grid.nodes)
        points[1, i] = node.x[1]
        points[2, i] = node.x[2]
    end
    cells = map(grid.cells) do cell
        nn = length(cell.nodes)
        if nn == 3
            MeshCell(VTKCellTypes.VTK_TRIANGLE, collect(cell.nodes))
        elseif nn == 4
            MeshCell(VTKCellTypes.VTK_QUAD, collect(cell.nodes))
        elseif nn == 6
            MeshCell(VTKCellTypes.VTK_QUADRATIC_TRIANGLE, collect(cell.nodes))
        elseif nn == 8
            MeshCell(VTKCellTypes.VTK_QUADRATIC_QUAD, collect(cell.nodes))
        else
            error("_vtk_geometry: unsupported 2-D cell size $nn " *
                  "(expected 3, 4, 6, or 8).")
        end
    end
    return points, cells
end

"""
    _scatter_to_nodes(u_full, dh) -> 3 × n_nodes matrix

Scatter a full DOF vector (length `ndofs(dh)`) to a component-major
nodal array, assuming a single 3-component `:u` field with node-major
DOF ordering (standard for `Lagrange{RefX, p}()^3`).
"""
function _scatter_to_nodes(u_full::AbstractVector, dh::Ferrite.DofHandler)
    n_nodes = Ferrite.getnnodes(dh.grid)
    out     = zeros(eltype(u_full), 3, n_nodes)
    visited = falses(n_nodes)
    for cell in CellIterator(dh)
        gdofs  = celldofs(cell)
        cnodes = cell.nodes
        for (i, node) in enumerate(cnodes)
            visited[node] && continue
            visited[node] = true
            for c in 1:3
                out[c, node] = u_full[gdofs[(i - 1) * 3 + c]]
            end
        end
    end
    return out
end

"""
    _build_node_dof_map(dh) -> 3 × n_nodes Int matrix

For each node and each spatial component (1=x, 2=y, 3=z), return the
global DOF index (1-indexed, same as the full system row/column number).
Assumes a single 3-component `:u` field with node-major DOF ordering.
"""
function _build_node_dof_map(dh::Ferrite.DofHandler)
    n_nodes = Ferrite.getnnodes(dh.grid)
    out     = zeros(Int, 3, n_nodes)
    visited = falses(n_nodes)
    for cell in CellIterator(dh)
        gdofs  = celldofs(cell)
        cnodes = cell.nodes
        for (i, node) in enumerate(cnodes)
            visited[node] && continue
            visited[node] = true
            for c in 1:3
                out[c, node] = gdofs[(i - 1) * 3 + c]
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# write_paraview_mesh
# ---------------------------------------------------------------------------

function PVExport.write_paraview_mesh(
    filename::AbstractString,
    grid::Ferrite.Grid{2};
    dh::Union{Ferrite.DofHandler, Nothing} = nothing,
    prescribed_dofs = nothing,
    node_positions::Union{AbstractMatrix, Nothing} = nothing,
)
    isnothing(dh) || throw(ArgumentError(
        "DOF metadata for a 2-D mixed space is ambiguous; use " *
        "write_paraview_p2p1 for Taylor--Hood velocity-pressure data."))
    isnothing(prescribed_dofs) || throw(ArgumentError(
        "prescribed_dofs requires dh, which is not supported by the generic 2-D mesh writer."))
    mkpath(dirname(filename))
    points, cells = _vtk_geometry(grid)
    if !isnothing(node_positions)
        size(node_positions, 2) == size(points, 2) || throw(DimensionMismatch(
            "node_positions has $(size(node_positions, 2)) nodes; expected $(size(points, 2))"))
        points = size(node_positions, 1) == 2 ?
            vcat(Float64.(node_positions), zeros(1, size(node_positions, 2))) :
            Float64.(node_positions)
        size(points, 1) == 3 || throw(DimensionMismatch(
            "node_positions must have two or three rows"))
    end
    vtk_grid(filename, points, cells) do vtk
        vtk_point_data(vtk, Int32.(1:size(points, 2)), "node_id")
        vtk_point_data(vtk, points[1, :], "x")
        vtk_point_data(vtk, points[2, :], "y")
    end
    println("  Mesh → $(filename).vtu")
    return filename * ".vtu"
end

function PVExport.write_paraview_mesh(
    filename::AbstractString,
    grid::Ferrite.Grid{3};
    dh::Union{Ferrite.DofHandler, Nothing} = nothing,
    prescribed_dofs = nothing,
    node_positions::Union{Matrix{Float64}, Nothing} = nothing,
)
    mkpath(dirname(filename))
    _pts, cells = _vtk_geometry(grid)
    points = isnothing(node_positions) ? _pts : node_positions
    n_nodes = Ferrite.getnnodes(grid)

    if isnothing(dh)
        vtk_grid(filename, points, cells) do _
        end
    else
        pset = isnothing(prescribed_dofs) ? Set{Int}() : Set{Int}(prescribed_dofs)
        dof_map = _build_node_dof_map(dh)

        # Build free-DOF index (same convention as the reduced K/M vectors and
        # node_dof_table.txt: 1…n_free, -1 for constrained directions).
        n_dofs = Ferrite.ndofs(dh)
        free = sort(setdiff(1:n_dofs, pset))
        free_to_local = Dict(d => i for (i, d) in enumerate(free))

        node_ids = Int32.(1:n_nodes)
        xs = isnothing(node_positions) ?
             Float64[grid.nodes[n].x[1] for n in 1:n_nodes] : points[1, :]
        ys = isnothing(node_positions) ?
             Float64[grid.nodes[n].x[2] for n in 1:n_nodes] : points[2, :]
        zs = isnothing(node_positions) ?
             Float64[grid.nodes[n].x[3] for n in 1:n_nodes] : points[3, :]
        dof_x = Int32[get(free_to_local, dof_map[1, n], -1) for n in 1:n_nodes]
        dof_y = Int32[get(free_to_local, dof_map[2, n], -1) for n in 1:n_nodes]
        dof_z = Int32[get(free_to_local, dof_map[3, n], -1) for n in 1:n_nodes]

        vtk_grid(filename, points, cells) do vtk
            vtk_point_data(vtk, node_ids, "node_id")
            vtk_point_data(vtk, xs,       "x")
            vtk_point_data(vtk, ys,       "y")
            vtk_point_data(vtk, zs,       "z")
            vtk_point_data(vtk, dof_x,    "dof_x")
            vtk_point_data(vtk, dof_y,    "dof_y")
            vtk_point_data(vtk, dof_z,    "dof_z")
        end
    end
    println("  Mesh → $(filename).vtu")
end

# ---------------------------------------------------------------------------
# write_paraview_p2p1 — 2-D Taylor--Hood velocity-pressure fields
# ---------------------------------------------------------------------------

function _p2p1_visualization_grid(grid::Ferrite.Grid{2}, dh::Ferrite.DofHandler,
                                  dof_range_u, dof_range_p)
    length(dof_range_u) == 12 || throw(ArgumentError(
        "P2 vector velocity must have 12 local DOFs; got $(length(dof_range_u))"))
    length(dof_range_p) == 3 || throw(ArgumentError(
        "P1 pressure must have 3 local DOFs; got $(length(dof_range_p))"))

    refs = Ferrite.reference_coordinates(Lagrange{RefTriangle, 2}())
    length(refs) == 6 || error("unexpected Ferrite P2 triangle node count")
    velocity_key_to_node = Dict{Int, Int}()
    points = NTuple{3, Float64}[]
    velocity_dofs = NTuple{2, Int}[]
    pressure_dofs = NTuple{3, Int}[]
    pressure_weights = NTuple{3, Float64}[]
    visualization_nodes = NTuple{6, Int}[]
    cells = MeshCell[]

    for cell in CellIterator(dh)
        length(cell.nodes) == 3 || throw(ArgumentError(
            "P2/P1 export currently requires linear three-node triangle geometry"))
        gdofs = celldofs(cell)
        vdofs = gdofs[dof_range_u]
        pdofs = gdofs[dof_range_p]
        X = ntuple(i -> grid.nodes[cell.nodes[i]].x, 3)
        vtk_nodes = Vector{Int}(undef, 6)

        for a in 1:6
            ux, uy = vdofs[2a - 1], vdofs[2a]
            node = get(velocity_key_to_node, ux, 0)
            if node == 0
                ξ = refs[a]
                N = (Float64(ξ[1]), Float64(ξ[2]), Float64(1 - ξ[1] - ξ[2]))
                x = N[1] * X[1] + N[2] * X[2] + N[3] * X[3]
                push!(points, (Float64(x[1]), Float64(x[2]), 0.0))
                push!(velocity_dofs, (ux, uy))
                push!(pressure_dofs, (pdofs[1], pdofs[2], pdofs[3]))
                push!(pressure_weights, N)
                node = length(points)
                velocity_key_to_node[ux] = node
            end
            vtk_nodes[a] = node
        end
        push!(visualization_nodes, Tuple(vtk_nodes))
        push!(cells, MeshCell(VTKCellTypes.VTK_QUADRATIC_TRIANGLE, vtk_nodes))
    end

    point_matrix = zeros(Float64, 3, length(points))
    for (i, point) in enumerate(points)
        point_matrix[:, i] .= point
    end
    return (; points=point_matrix, cells, velocity_dofs,
              pressure_dofs, pressure_weights, visualization_nodes)
end

function _p2p1_nodal_fields(viz, state, mode_full;
                            normalize_mode::Bool, align_mode_phase::Bool)
    n = length(viz.velocity_dofs)
    base_velocity = isnothing(state) ? nothing : zeros(Float64, 3, n)
    base_pressure = isnothing(state) ? nothing : zeros(Float64, n)
    mode_velocity = isnothing(mode_full) ? nothing : zeros(ComplexF64, 3, n)
    mode_pressure = isnothing(mode_full) ? nothing : zeros(ComplexF64, n)

    for a in 1:n
        ux, uy = viz.velocity_dofs[a]
        pdofs = viz.pressure_dofs[a]
        weights = viz.pressure_weights[a]
        if !isnothing(state)
            base_velocity[1, a] = state[ux]
            base_velocity[2, a] = state[uy]
            base_pressure[a] = sum(weights[i] * state[pdofs[i]] for i in 1:3)
        end
        if !isnothing(mode_full)
            mode_velocity[1, a] = mode_full[ux]
            mode_velocity[2, a] = mode_full[uy]
            mode_pressure[a] = sum(weights[i] * mode_full[pdofs[i]] for i in 1:3)
        end
    end

    phase_rotation = 0.0
    raw_velocity_scale = 1.0
    mode_multiplier = 1.0 + 0.0im
    if !isnothing(mode_velocity)
        amplitude = vec(sqrt.(sum(abs2, mode_velocity; dims=1)))
        raw_velocity_scale = maximum(amplitude)
        raw_velocity_scale > 0 || throw(ArgumentError("mode has zero velocity amplitude"))
        rotation = 1.0 + 0.0im
        if align_mode_phase
            hotspot = argmax(amplitude)
            component = abs(mode_velocity[1, hotspot]) >= abs(mode_velocity[2, hotspot]) ? 1 : 2
            phase_rotation = angle(mode_velocity[component, hotspot])
            rotation = cis(-phase_rotation)
        end
        scale = normalize_mode ? raw_velocity_scale : 1.0
        mode_multiplier = rotation / scale
        mode_velocity .*= mode_multiplier
        mode_pressure .*= mode_multiplier
    end

    return (; base_velocity, base_pressure, mode_velocity, mode_pressure,
              phase_rotation, raw_velocity_scale, mode_multiplier)
end

function _p2p1_nodal_vorticity(viz, grid, dh, dof_range_u, state, mode_full,
                                mode_multiplier)
    n = size(viz.points, 2)
    base_vorticity = isnothing(state) ? nothing : zeros(Float64, n)
    mode_vorticity = isnothing(mode_full) ? nothing : zeros(ComplexF64, n)
    weights = zeros(Float64, n)
    refs = Ferrite.reference_coordinates(Lagrange{RefTriangle, 2}())
    point_values = PointValues(Lagrange{RefTriangle, 2}()^2,
                               Lagrange{RefTriangle, 1}())

    for (cell, vtk_nodes) in zip(CellIterator(dh), viz.visualization_nodes)
        gdofs = celldofs(cell)
        velocity_dofs = gdofs[dof_range_u]
        coordinates = getcoordinates(cell)
        x1, x2, x3 = coordinates
        area = abs((x2[1]-x1[1])*(x3[2]-x1[2]) -
                   (x2[2]-x1[2])*(x3[1]-x1[1])) / 2
        base_element = isnothing(state) ? nothing : state[velocity_dofs]
        mode_element = isnothing(mode_full) ? nothing : mode_full[velocity_dofs]
        for a in 1:6
            reinit!(point_values, coordinates, refs[a])
            node = vtk_nodes[a]
            weights[node] += area
            if !isnothing(base_element)
                gradient = function_gradient(point_values, base_element)
                base_vorticity[node] += area * (gradient[2,1] - gradient[1,2])
            end
            if !isnothing(mode_element)
                gradient = function_gradient(point_values, mode_element)
                mode_vorticity[node] += area * (gradient[2,1] - gradient[1,2])
            end
        end
    end
    all(>(0), weights) || error("vorticity recovery found an unvisited P2 node")
    isnothing(base_vorticity) || (base_vorticity ./= weights)
    if !isnothing(mode_vorticity)
        mode_vorticity ./= weights
        mode_vorticity .*= mode_multiplier
    end
    return (; base_vorticity, mode_vorticity)
end

function _full_mode(mode, mode_dofs, n_dofs::Int)
    isnothing(mode) && return nothing
    if length(mode) == n_dofs
        return ComplexF64.(mode)
    end
    isnothing(mode_dofs) && throw(DimensionMismatch(
        "a reduced mode requires mode_dofs"))
    length(mode) == length(mode_dofs) || throw(DimensionMismatch(
        "mode has length $(length(mode)); expected $n_dofs (full) or " *
        "$(length(mode_dofs)) (reduced)"))
    all(d -> 1 <= d <= n_dofs, mode_dofs) || throw(ArgumentError(
        "mode_dofs contains an index outside 1:$n_dofs"))
    full = zeros(ComplexF64, n_dofs)
    full[mode_dofs] .= mode
    return full
end

function _vtk_scalar_cell_data(vtk, value, name::AbstractString, ncells::Int)
    if value isa Real
        vtk_cell_data(vtk, fill(Float64(value), ncells), name)
    elseif value isa Complex
        vtk_cell_data(vtk, fill(Float64(real(value)), ncells), name * "_real")
        vtk_cell_data(vtk, fill(Float64(imag(value)), ncells), name * "_imag")
    else
        throw(ArgumentError("cell datum '$name' must be a real or complex scalar"))
    end
end

function PVExport.write_paraview_p2p1(
    filename::AbstractString,
    grid::Ferrite.Grid{2},
    dh::Ferrite.DofHandler,
    dof_range_u,
    dof_range_p;
    state=nothing,
    mode=nothing,
    mode_dofs=nothing,
    normalize_mode::Bool=true,
    align_mode_phase::Bool=true,
    visual_amplitude::Real=0.1,
    phase::Union{Nothing,Real}=nothing,
    reynolds=nothing,
    eigenvalue=nothing,
    diagnostics=NamedTuple(),
)
    n_dofs = ndofs(dh)
    if !isnothing(state)
        length(state) == n_dofs || throw(DimensionMismatch(
            "state has length $(length(state)); expected $n_dofs"))
        eltype(state) <: Real || throw(ArgumentError("state must be real-valued"))
    end
    visual_amplitude >= 0 || throw(ArgumentError("visual_amplitude must be nonnegative"))
    mode_full = _full_mode(mode, mode_dofs, n_dofs)
    viz = _p2p1_visualization_grid(grid, dh, dof_range_u, dof_range_p)
    fields = _p2p1_nodal_fields(viz, state, mode_full;
        normalize_mode, align_mode_phase)
    vorticity = _p2p1_nodal_vorticity(viz, grid, dh, dof_range_u,
        state, mode_full, fields.mode_multiplier)
    phase_total_vorticity = if isnothing(phase) ||
        isnothing(vorticity.base_vorticity) || isnothing(vorticity.mode_vorticity)
        nothing
    else
        vorticity.base_vorticity .+ visual_amplitude .*
            real.(cis(Float64(phase)) .* vorticity.mode_vorticity)
    end
    ncells = length(viz.cells)
    mkpath(dirname(filename))

    vtk_grid(filename, viz.points, viz.cells) do vtk
        vtk_point_data(vtk, Int32.(1:size(viz.points, 2)), "node_id")
        vtk_point_data(vtk, viz.points[1, :], "x")
        vtk_point_data(vtk, viz.points[2, :], "y")
        if !isnothing(fields.base_velocity)
            speed = vec(sqrt.(sum(abs2, fields.base_velocity; dims=1)))
            vtk_point_data(vtk, fields.base_velocity, "base_velocity")
            vtk_point_data(vtk, speed, "base_speed")
            vtk_point_data(vtk, fields.base_pressure, "base_pressure")
            vtk_point_data(vtk, vorticity.base_vorticity, "base_vorticity")
        end
        if !isnothing(fields.mode_velocity)
            mode_re = real.(fields.mode_velocity)
            mode_im = imag.(fields.mode_velocity)
            amplitude = vec(sqrt.(sum(abs2, fields.mode_velocity; dims=1)))
            vtk_point_data(vtk, mode_re, "mode_velocity_real")
            vtk_point_data(vtk, mode_im, "mode_velocity_imag")
            vtk_point_data(vtk, amplitude, "mode_velocity_amplitude")
            vtk_point_data(vtk, real.(fields.mode_pressure), "mode_pressure_real")
            vtk_point_data(vtk, imag.(fields.mode_pressure), "mode_pressure_imag")
            vtk_point_data(vtk, abs.(fields.mode_pressure), "mode_pressure_amplitude")
            mode_vorticity_re = real.(vorticity.mode_vorticity)
            mode_vorticity_im = imag.(vorticity.mode_vorticity)
            vtk_point_data(vtk, mode_vorticity_re, "mode_vorticity_real")
            vtk_point_data(vtk, mode_vorticity_im, "mode_vorticity_imag")
            vtk_point_data(vtk, abs.(vorticity.mode_vorticity),
                "mode_vorticity_amplitude")
            if !isnothing(fields.base_velocity)
                vtk_point_data(vtk,
                    fields.base_velocity .+ visual_amplitude .* mode_re,
                    "base_plus_mode_real")
                vtk_point_data(vtk,
                    fields.base_velocity .+ visual_amplitude .* mode_im,
                    "base_plus_mode_imag")
                vtk_point_data(vtk,
                    vorticity.base_vorticity .+ visual_amplitude .* mode_vorticity_re,
                    "total_vorticity_real")
                vtk_point_data(vtk,
                    vorticity.base_vorticity .+ visual_amplitude .* mode_vorticity_im,
                    "total_vorticity_imag")
                if !isnothing(phase)
                    phase_factor = cis(Float64(phase))
                    phase_velocity = real.(phase_factor .* fields.mode_velocity)
                    vtk_point_data(vtk,
                        fields.base_velocity .+ visual_amplitude .* phase_velocity,
                        "total_velocity")
                    vtk_point_data(vtk, phase_total_vorticity, "total_vorticity")
                end
            end
        end
        _vtk_scalar_cell_data(vtk, visual_amplitude, "visual_amplitude", ncells)
        _vtk_scalar_cell_data(vtk, fields.raw_velocity_scale,
            "mode_raw_velocity_scale", ncells)
        isnothing(phase) || _vtk_scalar_cell_data(vtk, Float64(phase),
            "visualization_phase", ncells)
        isnothing(reynolds) || _vtk_scalar_cell_data(vtk, reynolds, "Reynolds", ncells)
        isnothing(eigenvalue) || _vtk_scalar_cell_data(vtk, eigenvalue, "eigenvalue", ncells)
        for (name, value) in pairs(diagnostics)
            _vtk_scalar_cell_data(vtk, value, String(name), ncells)
        end
    end

    output = filename * ".vtu"
    println("  P2/P1 fields → $output")
    return (; file=output, n_points=size(viz.points, 2), n_cells=ncells,
              phase_rotation=fields.phase_rotation,
              raw_velocity_scale=fields.raw_velocity_scale,
              base_vorticity_extrema=isnothing(vorticity.base_vorticity) ?
                  nothing : extrema(vorticity.base_vorticity),
              mode_vorticity_max=isnothing(vorticity.mode_vorticity) ?
                  nothing : maximum(abs, vorticity.mode_vorticity),
              total_vorticity=phase_total_vorticity)
end

function PVExport.write_paraview_p2p1_phase_animation(
    filename::AbstractString, fom;
    state,
    mode,
    phase_frames::Integer=24,
    visual_amplitude::Real=0.1,
    diagnostics=NamedTuple(),
    kwargs...)
    phase_frames >= 2 || throw(ArgumentError("phase_frames must be at least two"))
    visual_amplitude >= 0 || throw(ArgumentError(
        "visual_amplitude must be nonnegative"))
    mkpath(dirname(filename))
    entries = Tuple{Float64,String}[]
    infos = NamedTuple[]
    for k in 0:(phase_frames-1)
        phase = 2π * k / phase_frames
        stem = filename * @sprintf("_phase_%03d", k)
        info = PVExport.write_paraview_p2p1(stem, fom;
            state, mode, visual_amplitude, phase,
            diagnostics=(; diagnostics..., phase_index=k,
                phase_fraction=k/phase_frames,
                visualization_is_linear_mode=true), kwargs...)
        push!(entries, (k/phase_frames, info.file))
        push!(infos, info)
    end
    pvd = filename * ".pvd"
    open(pvd, "w") do io
        println(io, "<?xml version=\"1.0\"?>")
        println(io, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">")
        println(io, "  <Collection>")
        for (time, file) in entries
            @printf(io, "    <DataSet timestep=\"%.16g\" group=\"\" part=\"0\" file=\"%s\"/>\n",
                time, basename(file))
        end
        println(io, "  </Collection>")
        println(io, "</VTKFile>")
    end
    println("  P2/P1 phase animation → $pvd")
    return (; pvd, files=last.(entries), frames=phase_frames,
        raw_velocity_scale=first(infos).raw_velocity_scale,
        visual_amplitude=Float64(visual_amplitude))
end

function PVExport.write_paraview_p2p1(filename::AbstractString, fom; kwargs...)
    required = (:grid, :dh, :dof_range_u, :dof_range_p)
    all(name -> hasproperty(fom, name), required) || throw(ArgumentError(
        "fom must provide grid, dh, dof_range_u, and dof_range_p"))
    defaults = NamedTuple()
    if !haskey(kwargs, :mode_dofs)
        dofs = hasproperty(fom, :free_dpim) ? fom.free_dpim :
               (hasproperty(fom, :free) ? fom.free : nothing)
        defaults = (; mode_dofs=dofs)
    end
    return PVExport.write_paraview_p2p1(filename, fom.grid, fom.dh,
        fom.dof_range_u, fom.dof_range_p; defaults..., kwargs...)
end

# ---------------------------------------------------------------------------
# write_paraview_modes
# ---------------------------------------------------------------------------

function PVExport.write_paraview_modes(
    outdir::AbstractString,
    grid::Ferrite.Grid{3},
    dh::Ferrite.DofHandler,
    eigenvalues::AbstractVector{<:Complex},
    Y::AbstractArray{<:Complex, 3},
    free::AbstractVector{Int};
    n_modes::Int = 10,
    prefix::AbstractString = "mode",
    node_positions::Union{Matrix{Float64}, Nothing} = nothing,
    re_only::Bool = false,
)
    mkpath(outdir)
    _pts, cells = _vtk_geometry(grid)
    points = isnothing(node_positions) ? _pts : node_positions
    n_dofs = Ferrite.ndofs(dh)
    n_eig  = size(Y, 3)
    n_out  = min(n_modes, n_eig)

    pvd = paraview_collection(joinpath(outdir, "modes"))

    println("\n  Mode shapes → $(outdir)/")
    println("  " * "-"^58)
    @printf("  %-6s  %-22s  %s\n", "Mode", "Freq (Hz)", "File")
    println("  " * "-"^58)

    for k in 1:n_out
        fname = joinpath(outdir, @sprintf("%s_%02d", prefix, k))

        u_full_re = zeros(Float64, n_dofs)
        u_full_im = zeros(Float64, n_dofs)
        u_full_re[free] .= real.(Y[:, 1, k])
        u_full_im[free] .= imag.(Y[:, 1, k])

        u_re = _scatter_to_nodes(u_full_re, dh)
        u_im = _scatter_to_nodes(u_full_im, dh)

        # pvd[time] = vtk must be inside the do-block (WriteVTK API requirement)
        vtk_grid(fname, points, cells) do vtk
            vtk_point_data(vtk, u_re, "Re_u")
            re_only || vtk_point_data(vtk, u_im, "Im_u")
            pvd[Float64(k)] = vtk
        end

        freq_hz = abs(eigenvalues[k]) / (2π)
        @printf("  %-6d  %-22.6g  %s.vtu\n", k, freq_hz, basename(fname))
    end

    println("  " * "-"^58)
    vtk_save(pvd)
    println("  Collection → $(joinpath(outdir, "modes.pvd"))")
end

# ---------------------------------------------------------------------------
# write_paraview_deformation
# ---------------------------------------------------------------------------

function PVExport.write_paraview_deformation(
    outdir::AbstractString,
    grid::Ferrite.Grid{3},
    dh::Ferrite.DofHandler,
    W::MORFE.ParametrisationMethod.Parametrisation,
    free::AbstractVector{Int},
    r_amplitude::Real;
    theta::Real = 0.0,
    extra_states::AbstractVector = ComplexF64[],
    label::AbstractString = "deformation",
    node_positions::Union{Matrix{Float64}, Nothing} = nothing,
)
    C    = MORFE.ParametrisationMethod.coefficients(W)   # (FOM, ORD, L)
    mset = MORFE.ParametrisationMethod.multiindex_set(W)
    W_disp = MORFE.Polynomials.DensePolynomial(@view(C[:, 1, :]), mset)

    z1 = r_amplitude * cis(float(theta))
    sv = ComplexF64[z1, conj(z1), extra_states...]

    u_free     = MORFE.Polynomials.evaluate(W_disp, sv)
    A_lin      = MORFE.Polynomials.linear_matrix_of_polynomial(W_disp)  # FOM × NVAR
    u_free_lin = A_lin * sv
    u_free_nl  = u_free .- u_free_lin

    n_dofs = Ferrite.ndofs(dh)
    function to_vtk(u_cplx)
        buf = zeros(Float64, n_dofs)
        buf[free] .= real.(u_cplx)
        return _scatter_to_nodes(buf, dh)   # (3, n_nodes)
    end

    mkpath(outdir)
    _pts, cells = _vtk_geometry(grid)
    points = isnothing(node_positions) ? _pts : node_positions
    fname = joinpath(outdir, label)
    vtk_grid(fname, points, cells) do vtk
        vtk_point_data(vtk, to_vtk(u_free),    "u_total")
        vtk_point_data(vtk, to_vtk(u_free_nl), "u_nonlinear")
    end
    println("  Deformation → $(fname).vtu")
end

# ---------------------------------------------------------------------------
# write_paraview_manifold
# ---------------------------------------------------------------------------

function PVExport.write_paraview_manifold(
    outdir::AbstractString,
    grid::Ferrite.Grid{3},
    dh::Ferrite.DofHandler,
    W::MORFE.ParametrisationMethod.Parametrisation,
    free::AbstractVector{Int};
    alpha_indices = nothing,
)
    mkpath(outdir)
    points, cells = _vtk_geometry(grid)
    n_dofs = Ferrite.ndofs(dh)

    mset    = MORFE.ParametrisationMethod.multiindex_set(W)
    C       = MORFE.ParametrisationMethod.coefficients(W)   # FOM × ORD × L
    exps    = mset.exponents
    n_mono  = length(exps)
    indices = isnothing(alpha_indices) ? (1:n_mono) : alpha_indices

    pvd = paraview_collection(joinpath(outdir, "manifold"))

    println("\n  Manifold slices → $(outdir)/")

    for m in indices
        alpha = exps[m]
        label = join(alpha, "_")
        fname = joinpath(outdir, "manifold_$(label)")

        v = C[:, 1, m]   # position slice (first ORD level)

        u_full_re = zeros(Float64, n_dofs)
        u_full_im = zeros(Float64, n_dofs)
        u_full_re[free] .= real.(v)
        u_full_im[free] .= imag.(v)

        u_re = _scatter_to_nodes(u_full_re, dh)
        u_im = _scatter_to_nodes(u_full_im, dh)

        vtk_grid(fname, points, cells) do vtk
            vtk_point_data(vtk, u_re, "Re_W_alpha")
            vtk_point_data(vtk, u_im, "Im_W_alpha")
            pvd[Float64(m)] = vtk
        end
    end

    vtk_save(pvd)
    println("  Collection → $(joinpath(outdir, "manifold.pvd"))")
end

end # module MORFEFerriteWriteVTKExt
