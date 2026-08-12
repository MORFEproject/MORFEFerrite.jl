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
	GeometryParameterBasis, nterms

"""
	parametric_model(dh, cv, geometry; geometry_parameter_basis, material, damping,
					 free = …, base = nothing) -> AssembledParametricModel

Expand an SVK structure over a parametric mesh coordinate transform.

`geometry` is a provider returning `(J₀, ∇ψ₁, …, ∇ψ_Nθ)` per quadrature point,
either analytically (`geom(x₀)`) or from an FE field (`geom(x₀, cell, cv, q)`).

`geometry_parameter_basis` is either one [`GeometryParameterBasis`](@ref) used for every form, or a
`NamedTuple` `(; linear, quadratic, cubic)` when the forms have different exact
polynomial degrees — the arch, for instance, has `K(θ)` of degree ≤ 2 but a
cubic form of degree ≤ 4, and truncating them all at the largest is wasted work.

`damping` is a [`RayleighDamping`](@ref), not a loose `(α, β)` pair: the
parametric damping series `C(θ) = αM(θ) + βK(θ)` and the model's own `C` must
describe the same structure, and passing the object makes that one value rather
than several kept in step by hand.

The returned model is `ORD = 3` — a parametric mass is a correction on the
highest derivative, which only exists one order up, so the fourth linear block
is zero.
"""
function parametric_model(dh, cv, geometry;
	geometry_parameter_basis,
	material,
	damping::RayleighDamping,
	free::Union{Nothing, AbstractVector{Int}} = nothing,
	base = nothing)
	bases = _geometry_parameter_bases(geometry_parameter_basis)
	stress = stress_model(material)
	ρ = Float64(material.ρ)

	freedofs = free === nothing ? collect(1:ndofs(dh)) : collect(free)
	free_to_local = Dict(d => i for (i, d) in enumerate(freedofs))
	n_free = length(freedofs)

	# One geometry cache per DISTINCT θ-basis, carrying every inverse-determinant
	# power the forms sharing that basis will ask for: the quadratic form weights
	# by (1/det J)², the cubic by (1/det J)³, the linear operators by the raw
	# 1/det J series (which every cache holds). When all three bases coincide —
	# the usual case — this is one cache and one sweep of the geometry.
	caches = _pullback_caches(dh, cv, geometry, bases)
	pd_lin = ParametricDiscretisation(dh, cv, free_to_local, n_free, caches.linear)

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
	pd_q = ParametricDiscretisation(dh, cv, free_to_local, n_free, caches.quadratic)
	pd_c = ParametricDiscretisation(dh, cv, free_to_local, n_free, caches.cubic)
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
function _pullback_caches(dh, cv, geometry, bases)
	needed = [(bases.linear, Int[]), (bases.quadratic, [2]), (bases.cubic, [3])]
	uniq = Tuple{GeometryParameterBasis, Vector{Int}}[]
	for (b, pw) in needed
		i = findfirst(u -> u[1] === b, uniq)
		i === nothing ? push!(uniq, (b, copy(pw))) : append!(uniq[i][2], pw)
	end
	built = [(b, PullbackCache(dh, cv, geometry, b; det_powers = unique(pw)))
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

The θ⁰ stiffness and mass of a parametric SVK model — the base configuration's
operators, which is what the eigenproblem is solved on.
"""
base_operators(m::AssembledParametricModel) =
	(m.operators[1].arrays[1], m.operators[3].arrays[1])
