# =====================================================================
# General multivariate θ-series algebra, keyed by a MORFE MultiindexSet.
#
# A θ-series in N_θ parameters θ = (θ₁,…,θ_{N_θ}) is stored as a `Vector{T}`
# aligned to `basis.mset.exponents` (position i ↔ multiindex exponent i, in
# graded-lexicographic order). The parameter set is a per-parameter *box*
# (θ_i up to its own degree bound) built with MORFE's `all_multiindices_in_box`
# — this is the per-parameter truncation the general parametric formulation
# requires (total-degree truncation is the special case of equal bounds).
#
# The affine reference map is  x(θ,x₀) = x₀ + Σ_i θ_i ψ_i(x₀),  giving the
# degree-1 Jacobian series  J(θ,x₀) = J₀(x₀) + Σ_i θ_i ∇ψ_i(x₀).  det J and
# adj J are then exact multivariate polynomials (degree ≤ 3 / ≤ 2), computed
# by the standard 3×3 cofactor formulas over scalar θ-polynomial entries, and
# 1/det J by a graded reciprocal recurrence.
# =====================================================================

using Tensors
using LinearAlgebra
using StaticArrays: SVector
using MORFE: MultiindexSet, all_multiindices_in_box

const Tens3 = Tensor{2, 3, Float64, 9}
const _THETA_ZERO_TOL = 1e-15

"""
	ThetaBasis{Nθ}

Per-parameter box of θ-exponents (a MORFE `MultiindexSet`) plus an
exponent→position lookup. Position 1 is always the zero multiindex
(graded-lex order), i.e. the constant term.
"""
struct ThetaBasis{Nθ}
	mset::MultiindexSet{Nθ}
	index::Dict{SVector{Nθ, Int}, Int}
	prod::Vector{NTuple{3, Int}}   # (i,j,k): exps[i]+exps[j] = exps[k], k in box
end

function ThetaBasis(bounds::AbstractVector{<:Integer})
	mset = all_multiindices_in_box(collect(Int, bounds))
	exps = mset.exponents
	index = Dict(e => i for (i, e) in enumerate(exps))
	# Precompute the truncated-product index table once (keeps the hot poly ops
	# free of Dict lookups — the difference between usable and unusably slow).
	prod = NTuple{3, Int}[]
	L = length(exps)
	for i in 1:L, j in 1:L
		k = get(index, exps[i] + exps[j], 0)
		k == 0 || push!(prod, (i, j, k))
	end
	return ThetaBasis{length(bounds)}(mset, index, prod)
end

nterms(b::ThetaBasis) = length(b.mset.exponents)

# ---------------------------------------------------------------------
# Truncated multivariate products
# ---------------------------------------------------------------------
# C[γ] = Σ_{α+β=γ, α,β,γ ∈ box} op(A[α], B[β]).  `op` is *, ⋅ or ⊡.
@inline function _series_convolve(op, A::AbstractVector, B::AbstractVector,
	basis::ThetaBasis)
	R = typeof(op(A[1], B[1]))
	out = fill(zero(R), length(basis.mset.exponents))
	@inbounds for (i, j, k) in basis.prod
		(iszero(A[i]) || iszero(B[j])) && continue
		out[k] += op(A[i], B[j])
	end
	return out
end

poly_mul(A, B, basis::ThetaBasis) = _series_convolve(*, A, B, basis)
poly_dot(A, B, basis::ThetaBasis) = _series_convolve(⋅, A, B, basis)
poly_contract(A, B, basis::ThetaBasis) = _series_convolve(⊡, A, B, basis)

# Elementwise (same-basis) add / subtract / scale.
padd(A, B) = A .+ B
psub(A, B) = A .- B

# ---------------------------------------------------------------------
# Reciprocal series 1/p(θ), graded recurrence
# ---------------------------------------------------------------------
"""
	reciprocal_series(p, basis) -> Vector{Float64}

Coefficients of `1/p(θ)` truncated to the box, from the graded recurrence
`q[0] = 1/p[0]`, `q[γ] = -(1/p[0]) Σ_{0≠β≤γ} p[β] q[γ-β]`. Graded-lex order
makes each `q[γ-β]` (lower total degree) available before `q[γ]`.
"""
function reciprocal_series(p::AbstractVector{<:Real}, basis::ThetaBasis{Nθ}) where {Nθ}
	exps = basis.mset.exponents
	L = length(exps)
	abs(p[1]) > _THETA_ZERO_TOL || error("reciprocal_series: p(0) = 0, series undefined")
	inv_p0 = 1.0 / p[1]
	q = zeros(Float64, L)
	q[1] = inv_p0
	@inbounds for k in 2:L
		γ = exps[k]
		s = 0.0
		for j in 2:L                      # skip β = 0 (position 1)
			β = exps[j]
			δ = γ - β
			any(<(0), δ) && continue
			m = get(basis.index, δ, 0)
			m == 0 && continue
			s += p[j] * q[m]
		end
		q[k] = -inv_p0 * s
	end
	return q
end

"""
	inv_det_power(inv_det, n, basis) -> Vector{Float64}

`n`-th power of the reciprocal series (`n ≥ 1`), truncated to the box.
"""
function inv_det_power(inv_det::AbstractVector{<:Real}, n::Int, basis::ThetaBasis)
	@assert n ≥ 1
	acc = copy(inv_det)
	for _ in 2:n
		acc = poly_mul(acc, inv_det, basis)
	end
	return acc
end

# ---------------------------------------------------------------------
# Jacobian series and its det / adj (general 3×3, any N_θ)
# ---------------------------------------------------------------------
"""
	jacobian_series(Js, basis) -> Vector{Tens3}

Assemble the degree-1 Jacobian series aligned to `basis` from
`Js = (J₀, ∇ψ₁, …, ∇ψ_{Nθ})`: `J₀` at the zero multiindex, `∇ψ_i` at the
unit multiindex `e_i`. All other coefficients are zero.
"""
function jacobian_series(Js::NTuple{M, Tens3}, basis::ThetaBasis{Nθ}) where {M, Nθ}
	@assert M == Nθ + 1 "expected J₀ plus one ∇ψ per parameter (got $M for $Nθ params)"
	L = nterms(basis)
	J = fill(zero(Tens3), L)
	J[1] = Js[1]                                   # J₀ at exponent 0
	for i in 1:Nθ
		e_i = SVector{Nθ, Int}(ntuple(k -> k == i ? 1 : 0, Nθ))
		J[basis.index[e_i]] = Js[i+1]              # ∇ψ_i at exponent e_i
	end
	return J
end

# scalar-entry series of a Tens3 series: E[(i,j)][m] = J[m][i,j]
@inline _entry(J::AbstractVector{Tens3}, i::Int, j::Int) = Float64[t[i, j] for t in J]

"""
	det_adj_series(J, basis) -> (det_ser, adj_ser)

Exact multivariate coefficients of `det(J(θ))` (scalar series) and
`adj(J(θ))` (Tens3 series) for a degree-1 3×3 Jacobian series `J`, via the
standard 3×3 cofactor formulas evaluated with multivariate polynomial
arithmetic. General for any number of parameters.
"""
function det_adj_series(J::AbstractVector{Tens3}, basis::ThetaBasis)
	E = [_entry(J, i, j) for i in 1:3, j in 1:3]   # 3×3 of scalar series
	m(a, b) = poly_mul(a, b, basis)

	# 2×2 minors M_ij = cofactor sign folded later
	c11 = psub(m(E[2, 2], E[3, 3]), m(E[2, 3], E[3, 2]))
	c12 = psub(m(E[2, 1], E[3, 3]), m(E[2, 3], E[3, 1]))
	c13 = psub(m(E[2, 1], E[3, 2]), m(E[2, 2], E[3, 1]))
	c21 = psub(m(E[1, 2], E[3, 3]), m(E[1, 3], E[3, 2]))
	c22 = psub(m(E[1, 1], E[3, 3]), m(E[1, 3], E[3, 1]))
	c23 = psub(m(E[1, 1], E[3, 2]), m(E[1, 2], E[3, 1]))
	c31 = psub(m(E[1, 2], E[2, 3]), m(E[1, 3], E[2, 2]))
	c32 = psub(m(E[1, 1], E[2, 3]), m(E[1, 3], E[2, 1]))
	c33 = psub(m(E[1, 1], E[2, 2]), m(E[1, 2], E[2, 1]))

	# det = E11 c11 - E12 c12 + E13 c13  (expansion along row 1)
	det_ser = padd(psub(m(E[1, 1], c11), m(E[1, 2], c12)), m(E[1, 3], c13))

	# adj = transpose(cofactor); cofactor_ij = (-1)^{i+j} minor_ij.
	# cof[i][j] holds the signed cofactor series C_ij; adj[i,j] = C_ji.
	cof = ((c11, -1 .* c12, c13),
		(-1 .* c21, c22, -1 .* c23),
		(c31, -1 .* c32, c33))
	L = nterms(basis)
	adj_ser = fill(zero(Tens3), L)
	@inbounds for mrow in 1:L
		adj_ser[mrow] = Tens3((i, j) -> cof[j][i][mrow])   # adj[i,j] = C_ji
	end
	return det_ser, adj_ser
end
