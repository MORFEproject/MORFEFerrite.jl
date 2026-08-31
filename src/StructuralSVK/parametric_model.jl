# =====================================================================
# StructuralSVK's entry point into ParametricGeometry.
#
# Takes an SVK case and a geometry provider and returns the physics-blind
# `AssembledParametricModel` that `build_model` consumes. Everything here is
# SVK's: which kernels exist, what arity each linear operator has, and how
# Rayleigh damping combines the stiffness and mass series.
# =====================================================================

using ..ParametricGeometry: PullbackCache, ParametricDiscretisation, ParametricMap,
	ParametricOperator, AssembledParametricModel, assemble_linear_series!,
	GeometryParameterBasis, nterms, AbstractInverseDeterminant, PowerSeriesInverseDet,
	free_dof_map

"""
	parametric_model(dh, cv, geometry; geometry_parameter_basis, material, damping,
					 free = nothing, base = nothing,
					 inverse_determinant = PowerSeriesInverseDet())
		-> AssembledParametricModel

Assemble the parameter expansion of a three-dimensional SVK structure over a
parametric mesh coordinate transform.

`geometry` is a provider returning `(J₀, ∇ψ₁, …, ∇ψ_Nθ)` per quadrature point,
either analytically (`geom(x₀)`) or from an FE field (`geom(x₀, cell, cv, q)`).

`geometry_parameter_basis` is either one [`GeometryParameterBasis`](@ref), shared
by every form, or a `NamedTuple` with a required `linear` entry and optional
`quadratic` and `cubic` entries. A missing `quadratic` basis falls back to
`linear`; a missing `cubic` basis falls back to `quadratic`, then `linear`.
Separate bases avoid over-expanding low-degree forms—for example, an isochoric
affine arch has stiffness degree at most two but cubic-form degree at most four.

`material` is an [`SVKMaterial`](@ref) or [`AnisotropicMaterial`](@ref).
`damping` is a [`RayleighDamping`](@ref) and defines the assembled series
`C(θ) = αM(θ) + βK(θ)`. If `free` is omitted, every DOF in `dh` is retained;
otherwise it supplies the global free-DOF indices. `base` is stored in the
returned case for physics-side bookkeeping. `inverse_determinant` selects how
the pullback cache constructs the series for `1/det(J)`.

This function returns an assembled parametric case, not an `NthOrderModel`.
Calling [`build_model`](@ref) on it produces `ORD = 3`: a parameter-dependent
mass is a correction on the highest derivative of the original second-order
system, so the augmented representation needs one additional, zero linear block.
"""
function parametric_model(dh, cv, geometry;
	geometry_parameter_basis,
	material,
	damping::RayleighDamping,
	free::Union{Nothing, AbstractVector{Int}} = nothing,
	base = nothing,
	inverse_determinant::AbstractInverseDeterminant = PowerSeriesInverseDet())
	bases = _geometry_parameter_bases(geometry_parameter_basis)
	stress = stress_model(material)
	ρ = Float64(material.ρ)

	freedofs = free === nothing ? collect(1:ndofs(dh)) : collect(free)
	# The parametric assembly path indexes a dense vector (one lookup per DOF per
	# cell per θ-coefficient); `info` keeps the `Dict` form, which is what
	# `free_dofs_at_nodes` and the VTK extension consume.
	free_to_local = Dict(d => i for (i, d) in enumerate(freedofs))
	f2l = free_dof_map(ndofs(dh), freedofs)
	n_free = length(freedofs)

	# One geometry cache per DISTINCT θ-basis, carrying every inverse-determinant
	# power the forms sharing that basis will ask for: the quadratic form weights
	# by (1/det J)², the cubic by (1/det J)³, the linear operators by the raw
	# 1/det J series (which every cache holds). When all three bases coincide —
	# the usual case — this is one cache and one sweep of the geometry.
	caches = _pullback_caches(dh, cv, geometry, bases, inverse_determinant)
	pd_lin = ParametricDiscretisation(dh, cv, f2l, n_free, caches.linear)

	# ── Linear operators: assemble K(θ) and M(θ), then form C(θ) from them ──
	L = nterms(bases.linear)
	K_full = [allocate_matrix(dh) for _ in 1:L]
	M_full = [allocate_matrix(dh) for _ in 1:L]
	assemble_linear_series!(K_full, M_full, pd_lin, SVKPullbackKernel{0}(stress, ρ))
	K_arr = [Kf[freedofs, freedofs] for Kf in K_full]
	M_arr = [Mf[freedofs, freedofs] for Mf in M_full]
	C_arr = [damping.α * M_arr[i] + damping.β * K_arr[i] for i in 1:L]

	operators = ParametricOperator[
		ParametricOperator(K_arr, (1, 0, 0)),
		ParametricOperator(C_arr, (0, 1, 0)),
		ParametricOperator(M_arr, (0, 0, 1)),
	]

	# ── Nonlinear forms ────────────────────────────────────────────────────
	pd_q = ParametricDiscretisation(dh, cv, f2l, n_free, caches.quadratic)
	pd_c = ParametricDiscretisation(dh, cv, f2l, n_free, caches.cubic)
	maps = [ParametricMap(pd_q, SVKPullbackKernel{2}(stress, ρ)),
		ParametricMap(pd_c, SVKPullbackKernel{3}(stress, ρ))]
	map_arities = [(2, 0, 0), (3, 0, 0)]

	info = (; n_dofs = n_free, n_dofs_total = ndofs(dh), backend = "Ferrite/SVK",
		material = material, damping = damping, dh = dh, cellvalues = cv,
		free_to_local = free_to_local, free = freedofs)

	return AssembledParametricModel(pd_lin, base, operators, maps, map_arities;
		info = info)
end

# Build one PullbackCache per distinct θ-basis, each carrying the union of the
# inverse-determinant powers the forms over it need. Bases are compared by
# identity: sharing one object is how a caller says "same truncation".
function _pullback_caches(dh, cv, geometry, bases, inverse_determinant)
	needed = [(bases.linear, Int[]), (bases.quadratic, [2]), (bases.cubic, [3])]
	uniq = Tuple{GeometryParameterBasis, Vector{Int}}[]
	for (b, pw) in needed
		i = findfirst(u -> u[1] === b, uniq)
		i === nothing ? push!(uniq, (b, copy(pw))) : append!(uniq[i][2], pw)
	end
	built = [(b, PullbackCache(dh, cv, geometry, b; det_powers = unique(pw),
				  inverse_determinant = inverse_determinant))
			 for (b, pw) in uniq]
	pick(b) = built[findfirst(u -> u[1] === b, built)][2]
	return (; linear = pick(bases.linear), quadratic = pick(bases.quadratic),
		cubic = pick(bases.cubic))
end

# One basis for everything, or one per form.
_geometry_parameter_bases(b::GeometryParameterBasis) = (; linear = b, quadratic = b, cubic = b)
function _geometry_parameter_bases(b::NamedTuple)
	haskey(b, :linear) || throw(ArgumentError(
		"geometry_parameter_basis NamedTuple needs a `linear` entry; got keys $(keys(b))"))
	return (; linear = b.linear,
		quadratic = get(b, :quadratic, b.linear),
		cubic = get(b, :cubic, get(b, :quadratic, b.linear)))
end

"""
	base_operators(m::AssembledParametricModel) -> (K, M)

Return the `θ = 0` stiffness and mass coefficient matrices of an assembled
parametric SVK case. These are the base-configuration operators used to solve
the eigenproblem supplied to `build_model`.
"""
base_operators(m::AssembledParametricModel) =
	(m.operators[1].arrays[1], m.operators[3].arrays[1])
