"""Reference-surface pressure vectors."""

function _restrict_pressure(case, full)
    f = zeros(eltype(full), case.info.n_dofs)
    for (global_dof, free_dof) in case.info.dof_to_free
        f[free_dof] = full[global_dof]
    end
    return f
end

function _surface_pressure_vector(case, facets, pressure)
    info = case.info
    dh = info.dof_handler
    cell = Ferrite.getcells(dh.grid, 1)
    RefShape = _refshape(cell)
    fv = FacetValues(FacetQuadratureRule{RefShape}(info.quad_order),
        info.displacement_interpolation, info.geometry_interpolation)
    full = zeros(Float64, Ferrite.ndofs(dh))
    fe = zeros(Float64, getnbasefunctions(fv))
    for facet in FacetIterator(dh, facets)
        reinit!(fv, facet)
        fill!(fe, 0)
        for q in 1:getnquadpoints(fv)
            traction = -pressure * getnormal(fv, q)
            dΓ = getdetJdV(fv, q)
            for i in 1:getnbasefunctions(fv)
                fe[i] += dot(traction, shape_value(fv, q, i)) * dΓ
            end
        end
        assemble!(full, celldofs(facet), fe)
    end
    return _restrict_pressure(case, full)
end

"""
    surface_pressure_vector(case, load::SurfacePressure)

Assemble the free-DOF pressure vector on the named Ferrite/Gmsh facet set in
`load.boundary`. Positive pressure acts inward, opposite to the outward normal.
"""
function surface_pressure_vector(case::AssembledMechanicalModel,
        load::SurfacePressure{<:AbstractString})
    facets = getfacetset(case.info.dof_handler.grid, load.boundary)
    return _surface_pressure_vector(case, facets, load.pressure)
end

"""
    surface_pressure_vector(case, mesh_path, load; scale = 1.0)

Assemble the free-DOF vector for a uniform *dead* pressure on the COMSOL
boundary entities in `load.boundary`. Positive pressure acts opposite each
surface element's stored normal. The result is a force vector, not an external
time law, so callers may couple it to harmonic, chaotic, or measured signals.
"""
function surface_pressure_vector(case::AssembledMechanicalModel,
        mesh_path::AbstractString, load::SurfacePressure{<:Set}; scale::Real = 1.0)
    endswith(mesh_path, ".mphtxt") || throw(ArgumentError("surface pressure currently requires a COMSOL .mphtxt mesh"))
    raw = Common.MeshIO.ComsolToGmsh._read_mesh(mesh_path)
    nn, xyz, tnodes, tids, qnodes, qids = raw[1:6]
    info = case.info; dh = info.dof_handler
    node_dofs = Dict{Int,NTuple{3,Int}}()
    for c in 1:Ferrite.getncells(dh.grid)
        nodes, dofs = dh.grid.cells[c].nodes, Ferrite.celldofs(dh, c)
        for (i, n) in enumerate(nodes)
            node_dofs[n] = (dofs[3i-2], dofs[3i-1], dofs[3i])
        end
    end
    f = zeros(Float64, info.n_dofs)
    point(n) = scale .* [xyz[3n-2], xyz[3n-1], xyz[3n]]
    function addface!(nodes, area_normal)
        # Equal nodal lumping preserves the resultant uniform-pressure force.
        nodal = -load.pressure .* area_normal ./ length(nodes)
        for n in nodes
            ds = node_dofs[n]
            for k in 1:3
                haskey(info.dof_to_free, ds[k]) && (f[info.dof_to_free[ds[k]]] += nodal[k])
            end
        end
    end
    for i in eachindex(tids)
        tids[i] in load.boundary || continue
        ns = tnodes[(6i-5):(6i)]
        addface!(ns, cross(point(ns[2])-point(ns[1]), point(ns[3])-point(ns[1])) / 2)
    end
    for i in eachindex(qids)
        qids[i] in load.boundary || continue
        ns = qnodes[(9i-8):(9i)]
        a, b, c, d = point(ns[1]), point(ns[2]), point(ns[3]), point(ns[4])
        addface!(ns, (cross(b-a, c-a) + cross(c-a, d-a)) / 2)
    end
    f
end
