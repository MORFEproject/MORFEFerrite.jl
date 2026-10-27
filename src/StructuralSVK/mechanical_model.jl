# Ferrite cells are parametrised by their reference shape: `AbstractCell{refshape}`.
_refshape(::Ferrite.AbstractCell{RS}) where {RS} = RS

"""
    mechanical_model(grid::Ferrite.Grid, constrained_nodes::Set{Int};
                     material, damping, fe_order = 2, quad_order = fe_order + 1)
    mechanical_model(grid::Ferrite.Grid; material, damping, dirichlet,
                     fe_order = 2, quad_order = fe_order + 1)
    mechanical_model(mesh_path::AbstractString; material, damping, dirichlet,
                     scale = 1.0, fe_order = 2, quad_order = fe_order + 1)

Assemble a three-dimensional `AssembledMechanicalModel` from a Ferrite grid or
mesh file. The displacement interpolation is `Lagrange{RefShape,fe_order}()^3`;
the returned `K`, `M`, and `C = αM + βK` are restricted to free DOFs, while the
model retains the Ferrite objects needed to construct its quadratic and cubic
SVK terms lazily.

`material` is an [`SVKMaterial`](@ref) or [`AnisotropicMaterial`](@ref), and
`damping` is a [`RayleighDamping`](@ref). All three displacement components are
fixed on the selected nodes or facets:

- The `constrained_nodes` overload accepts an already computed set of node indices.
- For a Ferrite grid or Gmsh `.msh` path, `dirichlet` is the name of a facet set.
- For a COMSOL `.mphtxt` path, `dirichlet` is a `Set{Int}` of 1-based boundary
  entity IDs (the raw COMSOL IDs plus one). `scale` rescales COMSOL node
  coordinates before assembly; it is ignored for Gmsh input.

The returned model's `info` carries the discretisation, for post-processing and for
building terms by hand with [`svk_nonlinearity`](@ref):

- `dof_handler`: the Ferrite `DofHandler` of the displacement field `:u`, over all DOFs.
- `cell_values`: the Ferrite `CellValues` of the displacement interpolation and the
  quadrature rule of order `quad_order`.
- `dof_to_free`: `Dict{Int, Int}` from a global DOF number to its index in the free state
  vector, which is the row index of `K`, `M` and of the parametrisation `W`; constrained
  DOFs are absent.
- `n_dofs`, `n_dofs_total`: the number of free DOFs and of all DOFs.
"""
function mechanical_model(grid::Ferrite.Grid, constrained_nodes::Set{Int};
        material::Union{SVKMaterial, AnisotropicMaterial},
        damping::RayleighDamping,
        fe_order::Int = 2,
        quad_order::Int = fe_order + 1)
    _mechanical_model(grid, constrained_nodes,
        "$(length(constrained_nodes)) constrained nodes";
        material, damping, fe_order, quad_order)
end

function mechanical_model(grid::Ferrite.Grid;
        material::Union{SVKMaterial, AnisotropicMaterial},
        damping::RayleighDamping,
        dirichlet::String,
        fe_order::Int = 2,
        quad_order::Int = fe_order + 1)
    _mechanical_model(grid, getfacetset(grid, dirichlet), dirichlet;
        material, damping, fe_order, quad_order)
end

function mechanical_model(mesh_path::AbstractString; dirichlet, scale::Real = 1.0, kwargs...)
    if endswith(mesh_path, ".mphtxt")
        # `scale` converts the mesh units to the material's (e.g. 1e-3 for a mm
        # mesh against SI constants); ignored for Gmsh meshes.
        grid, constrained = load_comsol_grid(mesh_path, dirichlet; scale = scale)
        return mechanical_model(grid, constrained; kwargs...)
    else
        return mechanical_model(FerriteGmsh.togrid(mesh_path); dirichlet = dirichlet, kwargs...)
    end
end

# The common body: `clamped` is a node set or a facet set, which Ferrite's `Dirichlet`
# accepts alike, and `dirichlet_label` is what `info` records about it.
function _mechanical_model(grid::Ferrite.Grid, clamped, dirichlet_label;
        material, damping, fe_order, quad_order)
    cell1 = Ferrite.getcells(grid, 1)
    RefShape = _refshape(cell1)
    displacement_interpolation = Lagrange{RefShape, fe_order}()^3
    quadrature_rule = QuadratureRule{RefShape}(quad_order)
    geometry_interpolation = Ferrite.geometric_interpolation(typeof(cell1))
    cell_values = CellValues(quadrature_rule, displacement_interpolation,
        geometry_interpolation)

    dof_handler = DofHandler(grid)
    add!(dof_handler, :u, displacement_interpolation)
    close!(dof_handler)

    constraints = ConstraintHandler(dof_handler)
    add!(constraints, Dirichlet(:u, clamped, (x, t) -> zeros(3), [1, 2, 3]))
    close!(constraints)
    update!(constraints, 0.0)

    K_full = allocate_matrix(dof_handler)
    M_full = allocate_matrix(dof_handler)
    svk_assemble_KM!(K_full, M_full, dof_handler, cell_values, material)

    free = sort(setdiff(1:ndofs(dof_handler), constraints.prescribed_dofs))
    dof_to_free = Dict(d => i for (i, d) in enumerate(free))
    n_free = length(free)

    K = K_full[free, free]
    M = M_full[free, free]
    C = damping.α * M + damping.β * K

    function factory(deg::Int)
        svk_nonlinearity(deg, dof_handler, cell_values, dof_to_free, n_free, material)
    end

    # (K, C, M): DERIVATIVE order, not the order they were assembled in.
    return AssembledMechanicalModel((K, C, M), factory, (2, 3), material, damping,
        (n_dofs = n_free, n_dofs_total = ndofs(dof_handler), backend = "Ferrite",
            fe_order = fe_order, quad_order = quad_order, dirichlet = dirichlet_label,
            dof_handler = dof_handler, cell_values = cell_values,
            dof_to_free = dof_to_free))
end
