# =====================================================================
# The parametric mesh coordinate transform, as a θ-power series.
#
# This file is PHYSICS-BLIND. It knows the reference map and nothing else:
# given the additive Jacobian series J(θ,x₀) = J₀(x₀) + Σᵢ θᵢ ∇ψᵢ(x₀), it
# precomputes, once per (cell, quadrature point), the three series every
# pullback needs —
#
#     det J(θ)        volume factor            (mass-like terms)
#     adj J(θ)        cofactor transpose       (gradient pullback, ∇u·adj J)
#     (1/det J(θ))^p  inverse-determinant      (weak-form weighting)
#
# — so that a physics kernel never recomputes geometry, and every kernel over
# the same (geometry, basis) shares one cache.
# =====================================================================

using Ferrite

# --- geometry provider dispatch --------------------------------------
# A provider is called per quadrature point and must return either
#   (J₀, ∇ψ₁, …, ∇ψ_{Nθ})            — the affine map x₀ + Σᵢ θᵢψᵢ(x₀), or
#   [α => J_α, …]                    — the general polynomial x = Σ_α x_α θ^α
# of `Tensor{2,dim}` for ANY dim; `jacobian_series` dispatches on which. Two
# signatures are supported:
#   geom(x₀)                 — analytic shape fields (e.g. the sinusoidal arch)
#   geom(x₀, cell, cv, q)    — FE-field shape modes (e.g. a bending eigenmode,
#                              whose gradient needs the element/QP context)
# `cv` is already reinit!-ed for `cell` at every call site.
@inline _geom_at(geom, x₀, cell, cv, q) =
	applicable(geom, x₀, cell, cv, q) ? geom(x₀, cell, cv, q) : geom(x₀)

"""
	PullbackCache(dh, cv, geom, basis; det_powers = Int[])

Per-(cell, quadrature-point) θ-series of the coordinate transform.

`geom` is a geometry provider (see `_geom_at`); `basis` is the [`GeometryParameterBasis`](@ref)
the series are truncated to. `det_powers` lists the inverse-determinant powers the
kernels will ask for — each is expanded eagerly at construction, because computing
`(1/det J)^p` inside an assembly sweep would put a `nterms(basis)`-long convolution
in the hot loop.

Fields are indexed `[cell][qp]`:

- `adj[ci][q]`     — `Vector{Tens3}`, the adj J series
- `det[ci][q]`     — `Vector{Float64}`, the det J series
- `inv_det[ci][q]` — `Vector{Float64}`, the 1/det J series
- `inv_det_pow[p]` — the `(1/det J)^p` series, same `[cell][qp]` shape

One cache serves every kernel over the same geometry: the quadratic and cubic
forms of a structural physics differ only in which `det_powers` entry they read.
"""
struct PullbackCache{Nθ, TT}
	basis::GeometryParameterBasis{Nθ}
	adj::Vector{Vector{Vector{TT}}}
	det::Vector{Vector{Vector{Float64}}}
	inv_det::Vector{Vector{Vector{Float64}}}
	inv_det_pow::Dict{Int, Vector{Vector{Vector{Float64}}}}
end

"""
	adj_tensor_type(cache) -> Type

The tensor type of the `adj J` coefficients, i.e. `Tensor{2,dim,Float64,dim²}`.
The driver sizes its gradient buffers from this rather than from a hardcoded
3D alias, which is what makes the assembly loops dimension-general.
"""
adj_tensor_type(::PullbackCache{Nθ, TT}) where {Nθ, TT} = TT

function PullbackCache(dh::DofHandler, cv::CellValues, geom,
	basis::GeometryParameterBasis{Nθ}; det_powers::AbstractVector{Int} = Int[],
	inverse_determinant::AbstractInverseDeterminant = PowerSeriesInverseDet()) where {Nθ}
	# The dimension is whatever the geometry provider returns — probe it once
	# rather than assuming 3, so a 2D physics needs no change here.
	TT = _probe_tensor_type(dh, cv, geom, basis)
	adj = Vector{Vector{TT}}[]
	det = Vector{Vector{Float64}}[]

	for cell in CellIterator(dh)
		reinit!(cv, cell)
		coords = getcoordinates(cell)
		acell = Vector{TT}[]
		dcell = Vector{Float64}[]
		for q in 1:getnquadpoints(cv)
			x₀ = spatial_coordinate(cv, q, coords)
			J = jacobian_series(_geom_at(geom, x₀, cell, cv, q), basis)
			det_ser, adj_ser = det_adj_series(J, basis)
			push!(acell, adj_ser)
			push!(dcell, det_ser)
		end
		push!(adj, acell)
		push!(det, dcell)
	end

	# THE SEAM between Method 4 (pointwise power series) and Method 3 (auxiliary
	# FE field). Everything above and below is common to both.
	inv = build_inverse_determinant(inverse_determinant, dh, cv, basis, det)

	pow = Dict{Int, Vector{Vector{Vector{Float64}}}}()
	for p in unique(det_powers)
		pow[p] = [[inv_det_power(inv[ci][q], p, basis) for q in eachindex(inv[ci])]
				  for ci in eachindex(inv)]
	end
	return PullbackCache{Nθ, TT}(basis, adj, det, inv, pow)
end

# Ask the provider for one Jacobian tuple and read the dimension off it. Cheap,
# and it keeps `dim` out of every signature in the module.
function _probe_tensor_type(dh::DofHandler, cv::CellValues, geom,
	basis::GeometryParameterBasis)
	for cell in CellIterator(dh)
		reinit!(cv, cell)
		x₀ = spatial_coordinate(cv, 1, getcoordinates(cell))
		return eltype(jacobian_series(_geom_at(geom, x₀, cell, cv, 1), basis))
	end
	throw(ArgumentError("PullbackCache: the DofHandler has no cells"))
end

"""
	inv_det_power_series(cache, p) -> Vector{Vector{Vector{Float64}}}

The precomputed `(1/det J)^p` series, indexed `[cell][qp]`. Hoist this out of an
assembly loop — the `Dict` lookup is not meant for the hot path.
"""
function inv_det_power_series(cache::PullbackCache, p::Int)
	haskey(cache.inv_det_pow, p) || throw(ArgumentError(
		"PullbackCache was built without inverse-determinant power $p; pass " *
		"det_powers = $(sort(collect(union(keys(cache.inv_det_pow), p)))) to the constructor"))
	return cache.inv_det_pow[p]
end

nterms(c::PullbackCache) = nterms(c.basis)

function Base.show(io::IO, ::MIME"text/plain", c::PullbackCache{Nθ}) where {Nθ}
	n_cells = length(c.adj)
	n_qp = n_cells == 0 ? 0 : length(c.adj[1])
	println(io, "PullbackCache{$Nθ}")
	println(io, "  cells      : $n_cells ($n_qp quadrature points each)")
	println(io, "  θ-terms    : $(nterms(c.basis))")
	print(io, "  det powers : $(isempty(c.inv_det_pow) ? "none" :
							  join(sort(collect(keys(c.inv_det_pow))), ", "))")
end
