# =====================================================================
# General parametric geometric nonlinearity (any number of parameters).
#
# For the additive map J(θ,x₀) = J₀(x₀) + Σᵢ θᵢ ∇ψᵢ(x₀), the quadratic and
# cubic St. Venant-Kirchhoff elastic forms are multivariate θ-polynomials.
# Each θ-multiindex coefficient becomes a MORFE `MultilinearMap` whose external
# multiplicity is |α| and whose closure multiplies by the matching product of
# external (θ) states — component i taken aᵢ times (mirrors the single-parameter
# arch and two-parameter beam wrappers, generalised via `_expand_multiindex`).
#
# Weak-form structure (identical cofactor pattern as ex 01/04/07):
#   quadratic integrand carries (1/det J)²,  cubic carries (1/det J)³,
#   with ε_adj(u) = sym(∇₀u · adj J(θ)).  SIGN: internal-force maps are negated.
# =====================================================================

using Ferrite
using Tensors
using MORFE: MultilinearMap

# --- shared continuum QP helpers -------------------------------------
@inline function gather_local!(ue::Vector{T}, u::AbstractVector{T},
	dofs::Vector{Int}, free_to_local::Dict{Int, Int}) where {T}
	@inbounds for (i, d) in pairs(dofs)
		ue[i] = haskey(free_to_local, d) ? u[free_to_local[d]] : zero(T)
	end
	return ue
end

@inline function scatter_local!(res::AbstractVector{T}, re::Vector{T},
	dofs::Vector{Int}, free_to_local::Dict{Int, Int}) where {T}
	@inbounds for (i, d) in pairs(dofs)
		haskey(free_to_local, d) || continue
		res[free_to_local[d]] += re[i]
	end
	return res
end

@inline σ_lame(ε, λ::Float64, μ::Float64) = λ * tr(ε) * one(ε) + 2μ * ε

# ∇u (θ-independent) contracted with the adj-series → a θ-series.
∇adj_series(∇u, adj_ser::Vector) = [∇u ⋅ a for a in adj_ser]

# Green-Lagrange cross term series sym(¼(∇uAᵀ∇uB + ∇uBᵀ∇uA)).
function E_nl_adj_series(A_ser::Vector, B_ser::Vector, basis::ThetaBasis)
	AB = poly_dot([transpose(g) for g in A_ser], B_ser, basis)
	BA = poly_dot([transpose(g) for g in B_ser], A_ser, basis)
	return [symmetric(0.25 * (AB[m] + BA[m])) for m in 1:nterms(basis)]
end

# --- the type --------------------------------------------------------
"""
	ParametricGeometricNonlinearity{N_input, Nθ, G}

`N_input = 2` (quadratic) or `3` (cubic). `geom(x₀_qp)` returns the tuple
`(J₀, ∇ψ₁, …, ∇ψ_{Nθ})` of `Tens3` at a quadrature point.
"""
struct ParametricGeometricNonlinearity{N_input, Nθ, G}
	dh::DofHandler
	cv::CellValues
	λ::Float64
	μ::Float64
	geom::G
	free_to_local::Dict{Int, Int}
	n_free::Int
	basis::ThetaBasis{Nθ}
end

function ParametricGeometricNonlinearity{N_input}(dh, cv, λ, μ, geom,
	free_to_local, n_free, basis::ThetaBasis{Nθ}) where {N_input, Nθ}
	return ParametricGeometricNonlinearity{N_input, Nθ, typeof(geom)}(
		dh, cv, Float64(λ), Float64(μ), geom, free_to_local, n_free, basis)
end

# --- per-QP inv_det^n helper -----------------------------------------
@inline function _qp_series(pgn, x₀)
	Js = pgn.geom(x₀)
	J = jacobian_series(Js, pgn.basis)
	det_ser, adj_ser = det_adj_series(J, pgn.basis)
	inv_det = reciprocal_series(det_ser, pgn.basis)
	return adj_ser, inv_det
end

# --- quadratic θ-coefficient ----------------------------------------
"""
	evaluate_theta_quadratic!(res, pgn, αidx, u₁, u₂)

θ^α coefficient (α = `basis.mset.exponents[αidx]`) of the quadratic elastic
form g(u₁,u₂;θ), accumulated into the free-DOF residual `res`.
"""
function evaluate_theta_quadratic!(res::AbstractVector{T},
	pgn::ParametricGeometricNonlinearity{2}, αidx::Int,
	u₁::AbstractVector{T}, u₂::AbstractVector{T}) where {T}
	fill!(res, zero(T))
	cv = pgn.cv; λ, μ = pgn.λ, pgn.μ; basis = pgn.basis
	nbf = getnbasefunctions(cv)
	nd = ndofs_per_cell(pgn.dh)
	u₁e = zeros(T, nd); u₂e = zeros(T, nd); re = zeros(T, nd)

	for cell in CellIterator(pgn.dh)
		reinit!(cv, cell)
		dofs = celldofs(cell); coords = getcoordinates(cell)
		gather_local!(u₁e, u₁, dofs, pgn.free_to_local)
		gather_local!(u₂e, u₂, dofs, pgn.free_to_local)
		fill!(re, zero(T))
		for q in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q)
			x₀ = spatial_coordinate(cv, q, coords)
			adj_ser, inv_det = _qp_series(pgn, x₀)
			inv_det2 = poly_mul(inv_det, inv_det, basis)

			∇u1 = function_gradient(cv, q, u₁e)
			∇u2 = function_gradient(cv, q, u₂e)
			∇u1a = ∇adj_series(∇u1, adj_ser)
			∇u2a = ∇adj_series(∇u2, adj_ser)
			σ_u1 = [σ_lame(symmetric(g), λ, μ) for g in ∇u1a]
			σ_u2 = [σ_lame(symmetric(g), λ, μ) for g in ∇u2a]
			E12 = E_nl_adj_series(∇u1a, ∇u2a, basis)
			σE12 = [σ_lame(E, λ, μ) for E in E12]

			for I in 1:nbf
				∇NI = shape_gradient(cv, q, I)
				∇NIa = ∇adj_series(∇NI, adj_ser)
				ε_v = [symmetric(g) for g in ∇NIa]
				t1 = poly_contract(ε_v, σE12, basis)
				S2 = [symmetric(A) for A in poly_dot([transpose(g) for g in ∇u1a], ∇NIa, basis)]
				t2 = poly_contract(S2, σ_u2, basis)
				S3 = [symmetric(A) for A in poly_dot([transpose(g) for g in ∇u2a], ∇NIa, basis)]
				t3 = poly_contract(S3, σ_u1, basis)
				integ = [t1[m] + 0.5 * (t2[m] + t3[m]) for m in 1:nterms(basis)]
				re[I] += poly_mul(integ, inv_det2, basis)[αidx] * dΩ₀
			end
		end
		scatter_local!(res, re, dofs, pgn.free_to_local)
	end
	return res
end

# --- cubic θ-coefficient --------------------------------------------
"""
	evaluate_theta_cubic!(res, pgn, αidx, u₁, u₂, u₃)

θ^α coefficient of the cubic elastic form h(u₁,u₂,u₃;θ).
"""
function evaluate_theta_cubic!(res::AbstractVector{T},
	pgn::ParametricGeometricNonlinearity{3}, αidx::Int,
	u₁::AbstractVector{T}, u₂::AbstractVector{T}, u₃::AbstractVector{T}) where {T}
	fill!(res, zero(T))
	cv = pgn.cv; λ, μ = pgn.λ, pgn.μ; basis = pgn.basis
	nbf = getnbasefunctions(cv)
	nd = ndofs_per_cell(pgn.dh)
	u₁e = zeros(T, nd); u₂e = zeros(T, nd); u₃e = zeros(T, nd); re = zeros(T, nd)

	for cell in CellIterator(pgn.dh)
		reinit!(cv, cell)
		dofs = celldofs(cell); coords = getcoordinates(cell)
		gather_local!(u₁e, u₁, dofs, pgn.free_to_local)
		gather_local!(u₂e, u₂, dofs, pgn.free_to_local)
		gather_local!(u₃e, u₃, dofs, pgn.free_to_local)
		fill!(re, zero(T))
		for q in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q)
			x₀ = spatial_coordinate(cv, q, coords)
			adj_ser, inv_det = _qp_series(pgn, x₀)
			inv_det3 = poly_mul(poly_mul(inv_det, inv_det, basis), inv_det, basis)

			∇u1a = ∇adj_series(function_gradient(cv, q, u₁e), adj_ser)
			∇u2a = ∇adj_series(function_gradient(cv, q, u₂e), adj_ser)
			∇u3a = ∇adj_series(function_gradient(cv, q, u₃e), adj_ser)
			σE23 = [σ_lame(E, λ, μ) for E in E_nl_adj_series(∇u2a, ∇u3a, basis)]
			σE13 = [σ_lame(E, λ, μ) for E in E_nl_adj_series(∇u1a, ∇u3a, basis)]
			σE12 = [σ_lame(E, λ, μ) for E in E_nl_adj_series(∇u1a, ∇u2a, basis)]

			for I in 1:nbf
				∇NIa = ∇adj_series(shape_gradient(cv, q, I), adj_ser)
				S1 = [symmetric(A) for A in poly_dot([transpose(g) for g in ∇u1a], ∇NIa, basis)]
				S2 = [symmetric(A) for A in poly_dot([transpose(g) for g in ∇u2a], ∇NIa, basis)]
				S3 = [symmetric(A) for A in poly_dot([transpose(g) for g in ∇u3a], ∇NIa, basis)]
				t1 = poly_contract(S1, σE23, basis)
				t2 = poly_contract(S2, σE13, basis)
				t3 = poly_contract(S3, σE12, basis)
				integ = [(t1[m] + t2[m] + t3[m]) / 3 for m in 1:nterms(basis)]
				re[I] += poly_mul(integ, inv_det3, basis)[αidx] * dΩ₀
			end
		end
		scatter_local!(res, re, dofs, pgn.free_to_local)
	end
	return res
end

# --- external-state factor: expand α into a component list ----------
# α = (a₁,…,a_Nθ) → (1 repeated a₁ times, 2 repeated a₂ times, …).
_expand_multiindex(α) = Tuple(reduce(vcat,
	[fill(i, α[i]) for i in eachindex(α)]; init = Int[]))

# --- fixed-arity closure factories ----------------------------------
const _PS_MAX_EXT = 12   # ≥ largest total θ-degree used

for m in 0:_PS_MAX_EXT
	ext = [Symbol("r$i") for i in 1:m]
	if m == 0
		@eval _ps_quad(::Val{0}, comp, pgn, αidx, buf) =
			(res, u₁, u₂) -> begin
				evaluate_theta_quadratic!(buf, pgn, αidx, u₁, u₂); res .-= buf
			end
		@eval _ps_cube(::Val{0}, comp, pgn, αidx, buf) =
			(res, u₁, u₂, u₃) -> begin
				evaluate_theta_cubic!(buf, pgn, αidx, u₁, u₂, u₃); res .-= buf
			end
		@eval _ps_linK(::Val{0}, comp, Kk) = (res, u) -> (res .-= (Kk * u))
		@eval _ps_linC(::Val{0}, comp, Cc) = (res, v) -> (res .-= (Cc * v))
		@eval _ps_linM(::Val{0}, comp, Mk) = (res, a) -> (res .-= (Mk * a))
	else
		factor = Expr(:call, :*, [:($(ext[s])[comp[$s]]) for s in 1:m]...)
		@eval _ps_quad(::Val{$m}, comp, pgn, αidx, buf) =
			(res, u₁, u₂, $(ext...)) -> begin
				evaluate_theta_quadratic!(buf, pgn, αidx, u₁, u₂)
				res .-= ($factor) .* buf
			end
		@eval _ps_cube(::Val{$m}, comp, pgn, αidx, buf) =
			(res, u₁, u₂, u₃, $(ext...)) -> begin
				evaluate_theta_cubic!(buf, pgn, αidx, u₁, u₂, u₃)
				res .-= ($factor) .* buf
			end
		@eval _ps_linK(::Val{$m}, comp, Kk) =
			(res, u, $(ext...)) -> (res .-= ($factor) .* (Kk * u))
		@eval _ps_linC(::Val{$m}, comp, Cc) =
			(res, v, $(ext...)) -> (res .-= ($factor) .* (Cc * v))
		@eval _ps_linM(::Val{$m}, comp, Mk) =
			(res, a, $(ext...)) -> (res .-= ($factor) .* (Mk * a))
	end
end

# --- wrap θ-coefficients as MultilinearMaps -------------------------
function multilinear_maps(pgn::ParametricGeometricNonlinearity{2})
	maps = MultilinearMap[]
	for (αidx, α) in enumerate(pgn.basis.mset.exponents)
		m = sum(α); comp = _expand_multiindex(α)
		buf = zeros(ComplexF64, pgn.n_free)
		cl = Base.invokelatest(_ps_quad, Val(m), comp, pgn, αidx, buf)
		push!(maps, MultilinearMap(cl, (2, 0, 0), m))
	end
	return maps
end

function multilinear_maps(pgn::ParametricGeometricNonlinearity{3})
	maps = MultilinearMap[]
	for (αidx, α) in enumerate(pgn.basis.mset.exponents)
		m = sum(α); comp = _expand_multiindex(α)
		buf = zeros(ComplexF64, pgn.n_free)
		cl = Base.invokelatest(_ps_cube, Val(m), comp, pgn, αidx, buf)
		push!(maps, MultilinearMap(cl, (3, 0, 0), m))
	end
	return maps
end
