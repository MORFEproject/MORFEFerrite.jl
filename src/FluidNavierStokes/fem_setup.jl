"""
	fem_setup.jl — Ferrite P2/P1 Taylor-Hood setup for the cylinder-flow mesh.

Returns a NamedTuple with:
  grid, dh, ch_full, ch_hom, cv_vel, cv_pres,
  free, free_to_local, n_free, dof_range_u, dof_range_p, n_vel_dofs_per_cell

Two ConstraintHandlers use the same prescribed-DOF set:
  ch_full  — values from the selected base-flow boundary policy;
             used for the steady-state Newton solve
  ch_hom   — zero perturbations on every prescribed base-flow boundary;
             used for the linearised (perturbation / DPIM) problem.

Consequently 'free' == 'free_dpim' as index sets:
  free = setdiff(all, ch_full)
  free_dpim = setdiff(all, ch_hom)
"""

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
PoiseuilleChannelBC(; mean_velocity::Real=U_MEAN,
	channel_height::Real=_CHANNEL_H, inlet_tag::AbstractString="Inlet",
	wall_tag::AbstractString="Walls") =
	PoiseuilleChannelBC(mean_velocity, channel_height, inlet_tag, wall_tag)

"""
    UniformFreestreamBC((1.0, 0.0); inlet_tag="Inlet", farfield_tag="Farfield")

External-flow conditions with a fixed, spatially uniform velocity on the inlet
and the top/bottom far-field boundary. The outlet is intentionally absent from
this policy and therefore retains the natural traction condition of the weak
form.
"""
struct UniformFreestreamBC <: AbstractFlowBoundaryConditions
	velocity::NTuple{2,Float64}
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
UniformFreestreamBC(velocity=(1.0, 0.0);
	inlet_tag::AbstractString="Inlet", farfield_tag::AbstractString="Farfield") =
	UniformFreestreamBC(velocity, inlet_tag, farfield_tag)

function _domain_bounds(grid::Ferrite.Grid{2})
	xs = (node.x[1] for node in grid.nodes)
	ys = (node.x[2] for node in grid.nodes)
	xmin, xmax = extrema(xs)
	ymin, ymax = extrema(ys)
	return (xmin=Float64(xmin), xmax=Float64(xmax),
		ymin=Float64(ymin), ymax=Float64(ymax))
end

_bc_signature(bc::PoiseuilleChannelBC) = join((
	"PoiseuilleChannelBC", repr(bc.mean_velocity), repr(bc.channel_height),
	bc.inlet_tag, bc.wall_tag), '|')
_bc_signature(bc::UniformFreestreamBC) = join((
	"UniformFreestreamBC", repr(bc.velocity[1]), repr(bc.velocity[2]),
	bc.inlet_tag, bc.farfield_tag), '|')

function _model_fingerprint(grid, bc, obstacle, reference_length, quadrature_order)
	b = _domain_bounds(grid)
	payload = join((_bc_signature(bc), obstacle,
		repr(Float64(reference_length)), string(Int(quadrature_order)),
		repr(b.xmin), repr(b.xmax), repr(b.ymin), repr(b.ymax)), '|')
	return bytes2hex(sha256(Vector{UInt8}(codeunits(payload))))
end

function _add_full_velocity_constraints!(ch, grid, bc::PoiseuilleChannelBC,
	obstacle)
	H = bc.channel_height
	Umax = 1.5 * bc.mean_velocity
	add!(ch, Dirichlet(:u, getfacetset(grid, bc.inlet_tag),
		(x, _) -> Vec{2}((Umax * 4.0 * x[2] * (H - x[2]) / H^2, 0.0))))
	add!(ch, Dirichlet(:u, getfacetset(grid, bc.wall_tag),
		(x, _) -> Vec{2}((0.0, 0.0))))
	add!(ch, Dirichlet(:u, getfacetset(grid, obstacle),
		(x, _) -> Vec{2}((0.0, 0.0))))
end

function _add_full_velocity_constraints!(ch, grid, bc::UniformFreestreamBC,
	obstacle)
	velocity = Vec{2}(bc.velocity)
	add!(ch, Dirichlet(:u, getfacetset(grid, bc.inlet_tag),
		(x, _) -> velocity))
	add!(ch, Dirichlet(:u, getfacetset(grid, bc.farfield_tag),
		(x, _) -> velocity))
	add!(ch, Dirichlet(:u, getfacetset(grid, obstacle),
		(x, _) -> Vec{2}((0.0, 0.0))))
end

function _add_homogeneous_velocity_constraints!(ch, grid,
	bc::PoiseuilleChannelBC, obstacle)
	for tag in (bc.inlet_tag, bc.wall_tag, obstacle)
		add!(ch, Dirichlet(:u, getfacetset(grid, tag),
			(x, _) -> Vec{2}((0.0, 0.0))))
	end
end

function _add_homogeneous_velocity_constraints!(ch, grid,
	bc::UniformFreestreamBC, obstacle)
	for tag in (bc.inlet_tag, bc.farfield_tag, obstacle)
		add!(ch, Dirichlet(:u, getfacetset(grid, tag),
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
"""
function setup_fem(meshfile::AbstractString; kwargs...)
	return setup_fem(togrid(String(meshfile)); kwargs...)
end

function setup_fem(grid::Ferrite.Grid{2};
	obstacle_tag::AbstractString = "Cylinder",
	reference_length::Real = _CYL_D,
	quadrature_order::Integer = 6,
	channel_height::Real = _CHANNEL_H,
	boundary_conditions::Union{Nothing,AbstractFlowBoundaryConditions}=nothing)
	reference_length > 0 || throw(ArgumentError("reference_length must be positive"))
	quadrature_order >= 1 || throw(ArgumentError("quadrature_order must be positive"))
	channel_height > 0 || throw(ArgumentError("channel_height must be positive"))
	obstacle = String(obstacle_tag)
	bc = isnothing(boundary_conditions) ?
		PoiseuilleChannelBC(; channel_height) : boundary_conditions
	@info "Grid: $(getncells(grid)) cells, $(getnnodes(grid)) nodes"

	# ── Interpolations ────────────────────────────────────────────────────
	# Geometric mapping: linear (mesh has 3-node triangles)
	ip_geo = Lagrange{RefTriangle, 1}()

	# Velocity: P2 vector (6 scalar nodes × 2 components = 12 DOFs/cell)
	ip_vel = Lagrange{RefTriangle, 2}()^2

	# Pressure: P1 scalar (3 corner nodes = 3 DOFs/cell)
	ip_pres = Lagrange{RefTriangle, 1}()

	# ── Quadrature ────────────────────────────────────────────────────────
	# Order 6 is the parametric-profile production default: convection is degree
	# five even before pullback factors are introduced. The keyword also permits
	# the mandatory 4/6/8 quadrature convergence study.
	qr = QuadratureRule{RefTriangle}(Int(quadrature_order))

	# Sub-parametric CellValues: P2/P1 fields on linear-triangle geometry
	cv_vel = CellValues(qr, ip_vel, ip_geo)
	cv_pres = CellValues(qr, ip_pres, ip_geo)

	# ── DofHandler: :u first, :p second ──────────────────────────────────
	dh = DofHandler(grid)
	add!(dh, :u, ip_vel)
	add!(dh, :p, ip_pres)
	close!(dh)

	dof_range_u = dof_range(dh, :u)   # local DOF indices for :u in a cell
	dof_range_p = dof_range(dh, :p)   # local DOF indices for :p in a cell
	n_vel_dofs_per_cell = length(dof_range_u)
	n_pres_dofs_per_cell = length(dof_range_p)

	@info "DOFs: $(ndofs(dh)) total"
	@info "  Velocity per cell : $n_vel_dofs_per_cell  (range $dof_range_u)"
	@info "  Pressure per cell : $n_pres_dofs_per_cell (range $dof_range_p)"

	# ── Inhomogeneous BCs (for steady-state Newton solve) ─────────────────
	ch_full = ConstraintHandler(dh)
	_add_full_velocity_constraints!(ch_full, grid, bc, obstacle)
	close!(ch_full)
	update!(ch_full, 0.0)

	# ── Homogeneous BCs (for perturbation / DPIM eigenproblem) ───────────
	# Inlet, walls and cylinder — all where the base flow has a Dirichlet BC.
	# The inflow profile is imposed (u = u₀), so the perturbation must vanish
	# there (u' = 0); leaving the inlet free would inject spurious inlet velocity
	# into the modes and the convective quadratic. Matches the reference code.
	ch_hom = ConstraintHandler(dh)
	_add_homogeneous_velocity_constraints!(ch_hom, grid, bc, obstacle)
	close!(ch_hom)
	update!(ch_hom, 0.0)

	# ── Free DOFs ─────────────────────────────────────────────────────────
	# free      : for steady-state Newton solve (inlet prescribed to Poiseuille)
	# free_dpim : for DPIM operators (inlet frozen → u' = 0 for perturbation)
	# Both exclude inlet + walls + cylinder, so they coincide as index sets.
	free = sort(setdiff(1:ndofs(dh), ch_full.prescribed_dofs))
	free_to_local = Dict{Int, Int}(d => i for (i, d) in enumerate(free))
	n_free = length(free)

	free_dpim = sort(setdiff(1:ndofs(dh), ch_hom.prescribed_dofs))
	free_to_local_dpim = Dict{Int, Int}(d => i for (i, d) in enumerate(free_dpim))
	n_free_dpim = length(free_dpim)

	@info "  Free DOFs (steady state) : $n_free"
	@info "  Free DOFs (DPIM)         : $n_free_dpim  (inlet frozen)"

	bounds = _domain_bounds(grid)
	model_fingerprint = _model_fingerprint(grid, bc, obstacle,
		reference_length, quadrature_order)
	return (;
		grid, dh, ch_full, ch_hom, cv_vel, cv_pres, ip_vel, ip_pres, qr,
		free, free_to_local, n_free,
		free_dpim, free_to_local_dpim, n_free_dpim,
		dof_range_u, dof_range_p, n_vel_dofs_per_cell,
		obstacle_tag = obstacle, reference_length = Float64(reference_length),
		quadrature_order = Int(quadrature_order),
		channel_height = Float64(channel_height), boundary_conditions=bc,
		domain_bounds=bounds, model_fingerprint,
	)
end
