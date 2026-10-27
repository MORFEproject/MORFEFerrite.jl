# =====================================================================
# Ferrite P2/P1 Taylor-Hood setup for the cylinder-flow mesh: `setup_fem` builds the
# discretisation every other file of this module reads (called `fom` throughout).
# =====================================================================

using Ferrite
using FerriteGmsh

# Mean inflow velocity (Ū = 1 → parabolic max = 1.5)
const U_MEAN = 1.0
const U_MAX = 1.5 * U_MEAN  # max of Poiseuille profile

# Turek–Schäfer benchmark geometry (channel 2.2 × 0.41 m, Ø 0.1 m cylinder).
# MUST match the mesh generator that produced `meshfile` — the Kármán example's
# `fem/mesh.jl` defines the same constants and generates this exact channel.
# _CHANNEL_H shapes the Poiseuille inlet; _CYL_D is the reference length in
# ν = _CYL_D/Re₀ (steady state + linearised operators).
const _CHANNEL_L = 2.2
const _CHANNEL_H = 0.41
const _CYL_R = 0.05
const _CYL_D = 2.0 * _CYL_R

"""Base type for velocity boundary-condition policies accepted by [`setup_fem`](@ref)."""
abstract type AbstractFlowBoundaryConditions end

"""
    PoiseuilleChannelBC(; mean_velocity=1.0, channel_height=0.41,
                         inlet_tag="Inlet", wall_tag="Walls")

Legacy Turek--Schäfer channel conditions: parabolic inlet and no-slip horizontal
walls. This remains the implicit `setup_fem` default so existing examples are
bit-for-bit compatible at the constraint level.
"""
struct PoiseuilleChannelBC <: AbstractFlowBoundaryConditions
    mean_velocity::Float64
    channel_height::Float64
    inlet_tag::String
    wall_tag::String
    function PoiseuilleChannelBC(mean_velocity::Real, channel_height::Real,
            inlet_tag::AbstractString, wall_tag::AbstractString)
        mean_velocity > 0 || throw(ArgumentError("mean_velocity must be positive"))
        channel_height > 0 || throw(ArgumentError("channel_height must be positive"))
        new(Float64(mean_velocity), Float64(channel_height),
            String(inlet_tag), String(wall_tag))
    end
end
function PoiseuilleChannelBC(; mean_velocity::Real = U_MEAN,
        channel_height::Real = _CHANNEL_H, inlet_tag::AbstractString = "Inlet",
        wall_tag::AbstractString = "Walls")
    PoiseuilleChannelBC(mean_velocity, channel_height, inlet_tag, wall_tag)
end

"""
    UniformFreestreamBC((1.0, 0.0); inlet_tag="Inlet", farfield_tag="Farfield")

External-flow conditions with a fixed, spatially uniform velocity on the inlet
and the top/bottom far-field boundary. The outlet is intentionally absent from
this policy and therefore retains the natural traction condition of the weak
form.
"""
struct UniformFreestreamBC <: AbstractFlowBoundaryConditions
    velocity::NTuple{2, Float64}
    inlet_tag::String
    farfield_tag::String
    function UniformFreestreamBC(velocity,
            inlet_tag::AbstractString, farfield_tag::AbstractString)
        length(velocity) == 2 || throw(ArgumentError(
            "freestream velocity must have exactly two components"))
        v = (Float64(velocity[1]), Float64(velocity[2]))
        all(isfinite, v) || throw(ArgumentError("freestream velocity must be finite"))
        hypot(v...) > 0 || throw(ArgumentError("freestream velocity must be nonzero"))
        new(v, String(inlet_tag), String(farfield_tag))
    end
end
function UniformFreestreamBC(velocity = (1.0, 0.0);
        inlet_tag::AbstractString = "Inlet", farfield_tag::AbstractString = "Farfield")
    UniformFreestreamBC(velocity, inlet_tag, farfield_tag)
end

function _domain_bounds(grid::Ferrite.Grid{2})
    xs = (node.x[1] for node in grid.nodes)
    ys = (node.x[2] for node in grid.nodes)
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    return (xmin = Float64(xmin), xmax = Float64(xmax),
        ymin = Float64(ymin), ymax = Float64(ymax))
end

function _bc_signature(bc::PoiseuilleChannelBC)
    join(
        (
            "PoiseuilleChannelBC", repr(bc.mean_velocity), repr(bc.channel_height),
            bc.inlet_tag, bc.wall_tag),
        '|')
end
function _bc_signature(bc::UniformFreestreamBC)
    join(
        (
            "UniformFreestreamBC", repr(bc.velocity[1]), repr(bc.velocity[2]),
            bc.inlet_tag, bc.farfield_tag),
        '|')
end

function _model_fingerprint(grid, bc, obstacle, reference_length, quadrature_order)
    b = _domain_bounds(grid)
    payload = join(
        (_bc_signature(bc), obstacle,
            repr(Float64(reference_length)), string(Int(quadrature_order)),
            repr(b.xmin), repr(b.xmax), repr(b.ymin), repr(b.ymax)),
        '|')
    return bytes2hex(sha256(Vector{UInt8}(codeunits(payload))))
end

function _add_full_velocity_constraints!(constraints, grid, bc::PoiseuilleChannelBC,
        obstacle)
    H = bc.channel_height
    Umax = 1.5 * bc.mean_velocity
    add!(constraints,
        Dirichlet(:u, getfacetset(grid, bc.inlet_tag),
            (x, _) -> Vec{2}((Umax * 4.0 * x[2] * (H - x[2]) / H^2, 0.0))))
    add!(constraints, Dirichlet(:u, getfacetset(grid, bc.wall_tag),
        (x, _) -> Vec{2}((0.0, 0.0))))
    add!(constraints, Dirichlet(:u, getfacetset(grid, obstacle),
        (x, _) -> Vec{2}((0.0, 0.0))))
end

function _add_full_velocity_constraints!(constraints, grid, bc::UniformFreestreamBC,
        obstacle)
    velocity = Vec{2}(bc.velocity)
    add!(constraints, Dirichlet(:u, getfacetset(grid, bc.inlet_tag),
        (x, _) -> velocity))
    add!(constraints, Dirichlet(:u, getfacetset(grid, bc.farfield_tag),
        (x, _) -> velocity))
    add!(constraints, Dirichlet(:u, getfacetset(grid, obstacle),
        (x, _) -> Vec{2}((0.0, 0.0))))
end

function _add_homogeneous_velocity_constraints!(constraints, grid,
        bc::PoiseuilleChannelBC, obstacle)
    for tag in (bc.inlet_tag, bc.wall_tag, obstacle)
        add!(constraints, Dirichlet(:u, getfacetset(grid, tag),
            (x, _) -> Vec{2}((0.0, 0.0))))
    end
end

function _add_homogeneous_velocity_constraints!(constraints, grid,
        bc::UniformFreestreamBC, obstacle)
    for tag in (bc.inlet_tag, bc.farfield_tag, obstacle)
        add!(constraints, Dirichlet(:u, getfacetset(grid, tag),
            (x, _) -> Vec{2}((0.0, 0.0))))
    end
end

"""
    setup_fem(meshfile_or_grid; obstacle_tag = "Cylinder", reference_length = 0.1,
              quadrature_order = 6, channel_height = 0.41,
              boundary_conditions = nothing) -> NamedTuple

Load `meshfile`, or use an already-loaded two-dimensional Ferrite grid, and
build all FEM objects for the P2/P1 Taylor-Hood cylinder-flow problem. The grid
overload permits topology-preserving geometry continuation without serialising
one mesh file per parameter value.

The returned NamedTuple, called `fom` across this module, holds:

- `grid`: the Ferrite grid, linear triangles.
- `dof_handler`: the Ferrite `DofHandler` with the velocity field `:u` (P2 vector) first
  and the pressure field `:p` (P1) second, over all DOFs.
- `velocity_interpolation`, `pressure_interpolation`: the Ferrite interpolations of the
  two fields.
- `quadrature_rule`: the cell `QuadratureRule` of order `quadrature_order`.
- `velocity_cell_values`, `pressure_cell_values`: the Ferrite `CellValues` of each field
  with that rule, on the linear geometry.
- `dof_range_u`, `dof_range_p`: the local positions of the velocity and pressure DOFs
  within a cell's DOF vector, as `Ferrite.dof_range` returns them; `n_vel_dofs_per_cell`
  is the length of the first.
- `base_flow_constraints`: `ConstraintHandler` with the boundary values of
  `boundary_conditions` (inflow profile, walls, obstacle), for the steady-state solve.
- `perturbation_constraints`: `ConstraintHandler` with zero velocity on the same
  boundaries, for the linearised problem and the reduction.
- `free`, `dof_to_free`, `n_free`: the global DOFs left free by `base_flow_constraints`,
  the `Dict{Int, Int}` from a global DOF number to its index among them (constrained DOFs
  absent), and their number.
- `free_dpim`, `dof_to_free_dpim`, `n_free_dpim`: the same for
  `perturbation_constraints`, which defines the state vector of the reduction. Both
  constraint sets prescribe the same DOFs, so the two index sets coincide.
- `obstacle_tag`, `reference_length`, `quadrature_order`, `channel_height`,
  `boundary_conditions`, `domain_bounds`, `model_fingerprint`: the case parameters.
"""
function setup_fem(meshfile::AbstractString; kwargs...)
    return setup_fem(togrid(String(meshfile)); kwargs...)
end

function setup_fem(grid::Ferrite.Grid{2};
        obstacle_tag::AbstractString = "Cylinder",
        reference_length::Real = _CYL_D,
        quadrature_order::Integer = 6,
        channel_height::Real = _CHANNEL_H,
        boundary_conditions::Union{Nothing, AbstractFlowBoundaryConditions} = nothing)
    reference_length > 0 || throw(ArgumentError("reference_length must be positive"))
    quadrature_order >= 1 || throw(ArgumentError("quadrature_order must be positive"))
    channel_height > 0 || throw(ArgumentError("channel_height must be positive"))
    obstacle = String(obstacle_tag)
    bc = isnothing(boundary_conditions) ?
         PoiseuilleChannelBC(; channel_height) : boundary_conditions
    @info "Grid: $(getncells(grid)) cells, $(getnnodes(grid)) nodes"

    # ── Interpolations ────────────────────────────────────────────────────
    # Geometric mapping: linear (mesh has 3-node triangles)
    geometry_interpolation = Lagrange{RefTriangle, 1}()

    # Velocity: P2 vector (6 scalar nodes × 2 components = 12 DOFs/cell)
    velocity_interpolation = Lagrange{RefTriangle, 2}()^2

    # Pressure: P1 scalar (3 corner nodes = 3 DOFs/cell)
    pressure_interpolation = Lagrange{RefTriangle, 1}()

    # ── Quadrature ────────────────────────────────────────────────────────
    # Order 6 is the parametric-profile production default: convection is degree
    # five even before pullback factors are introduced. The keyword also permits
    # the mandatory 4/6/8 quadrature convergence study.
    quadrature_rule = QuadratureRule{RefTriangle}(Int(quadrature_order))

    # Sub-parametric CellValues: P2/P1 fields on linear-triangle geometry
    velocity_cell_values = CellValues(quadrature_rule, velocity_interpolation, geometry_interpolation)
    pressure_cell_values = CellValues(quadrature_rule, pressure_interpolation, geometry_interpolation)

    # ── DofHandler: :u first, :p second ──────────────────────────────────
    dof_handler = DofHandler(grid)
    add!(dof_handler, :u, velocity_interpolation)
    add!(dof_handler, :p, pressure_interpolation)
    close!(dof_handler)

    dof_range_u = dof_range(dof_handler, :u)   # local DOF indices for :u in a cell
    dof_range_p = dof_range(dof_handler, :p)   # local DOF indices for :p in a cell
    n_vel_dofs_per_cell = length(dof_range_u)
    n_pres_dofs_per_cell = length(dof_range_p)

    @info "DOFs: $(ndofs(dof_handler)) total"
    @info "  Velocity per cell : $n_vel_dofs_per_cell  (range $dof_range_u)"
    @info "  Pressure per cell : $n_pres_dofs_per_cell (range $dof_range_p)"

    # ── Inhomogeneous BCs (for steady-state Newton solve) ─────────────────
    base_flow_constraints = ConstraintHandler(dof_handler)
    _add_full_velocity_constraints!(base_flow_constraints, grid, bc, obstacle)
    close!(base_flow_constraints)
    update!(base_flow_constraints, 0.0)

    # ── Homogeneous BCs (for perturbation / DPIM eigenproblem) ───────────
    # Inlet, walls and cylinder — all where the base flow has a Dirichlet BC.
    # The inflow profile is imposed (u = u₀), so the perturbation must vanish
    # there (u' = 0); leaving the inlet free would inject spurious inlet velocity
    # into the modes and the convective quadratic. Matches the reference code.
    perturbation_constraints = ConstraintHandler(dof_handler)
    _add_homogeneous_velocity_constraints!(perturbation_constraints, grid, bc, obstacle)
    close!(perturbation_constraints)
    update!(perturbation_constraints, 0.0)

    # ── Free DOFs ─────────────────────────────────────────────────────────
    # free      : for steady-state Newton solve (inlet prescribed to Poiseuille)
    # free_dpim : for DPIM operators (inlet frozen → u' = 0 for perturbation)
    # Both exclude inlet + walls + cylinder, so they coincide as index sets.
    free = sort(setdiff(1:ndofs(dof_handler), base_flow_constraints.prescribed_dofs))
    dof_to_free = Dict{Int, Int}(d => i for (i, d) in enumerate(free))
    n_free = length(free)

    free_dpim = sort(setdiff(1:ndofs(dof_handler), perturbation_constraints.prescribed_dofs))
    dof_to_free_dpim = Dict{Int, Int}(d => i for (i, d) in enumerate(free_dpim))
    n_free_dpim = length(free_dpim)

    @info "  Free DOFs (steady state) : $n_free"
    @info "  Free DOFs (DPIM)         : $n_free_dpim  (inlet frozen)"

    bounds = _domain_bounds(grid)
    model_fingerprint = _model_fingerprint(grid, bc, obstacle,
        reference_length, quadrature_order)
    return (;
        grid, dof_handler, base_flow_constraints, perturbation_constraints,
        velocity_cell_values, pressure_cell_values,
        velocity_interpolation, pressure_interpolation, quadrature_rule,
        free, dof_to_free, n_free,
        free_dpim, dof_to_free_dpim, n_free_dpim,
        dof_range_u, dof_range_p, n_vel_dofs_per_cell,
        obstacle_tag = obstacle, reference_length = Float64(reference_length),
        quadrature_order = Int(quadrature_order),
        channel_height = Float64(channel_height), boundary_conditions = bc,
        domain_bounds = bounds, model_fingerprint
    )
end
