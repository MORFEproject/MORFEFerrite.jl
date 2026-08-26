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
# The reference map is polynomial in θ,  x(θ,x₀) = Σ_α x_α(x₀) θ^α,  giving the
# Jacobian series  J(θ,x₀) = Σ_α J_α(x₀) θ^α  (the affine map x₀ + Σ_i θ_i ψ_i is
# the degree-1 case). det J and adj J are then exact multivariate polynomials,
# bounded by  deg det J ≤ d·deg J  and  deg adj J ≤ (d−1)·deg J, and 1/det J is
# obtained from a graded reciprocal recurrence.
#
# Everything here is DIMENSION-GENERAL: `det` comes from App. A.1's multilinear
# expansion over the Jacobian's columns and `adj` from App. A.2's recurrence, so
# no 3×3 cofactor formula appears anywhere. `d` is carried by the tensor type of
# the Jacobian coefficients, never by this file.
# =====================================================================

using Tensors
using LinearAlgebra
using StaticArrays: SVector
using MORFE: MultiindexSet, all_multiindices_in_box

const Tens3 = Tensor{2, 3, Float64, 9}
# The pulled-back gradients carry the (complex) modal amplitudes, so the series
# the kernels work on are complex even though the geometry itself is real.
const Tens3C = Tensor{2, 3, ComplexF64, 9}
const SymT3C = SymmetricTensor{2, 3, ComplexF64, 6}
const _GEOMETRY_PARAMETER_ZERO_TOL = 1e-15

"""
	GeometryParameterBasis{Nθ}

Per-parameter box of θ-exponents (a MORFE `MultiindexSet`) plus an
exponent→position lookup. Position 1 is always the zero multiindex
(graded-lex order), i.e. the constant term.
"""
struct GeometryParameterBasis{Nθ}
	mset::MultiindexSet{Nθ}
	index::Dict{SVector{Nθ, Int}, Int}
	# The truncated-product table in CSR form, grouped by the LEFT operand and
	# ascending in the right one: for operand position `i`, the pairs live at
	# `prod_ptr[i] : prod_ptr[i+1]-1`, with right index `prod_j[t]` and result
	# position `prod_k[t]`. Grouping is what lets the convolution hoist the
	# `iszero(A[i])` test out of the pair loop and stop early on `j`; the flat
	# `Vector{NTuple{3,Int}}` this replaces forced both tests once PER PAIR, which
	# a profile of example 04 showed to be 23 % of the entire run.
	prod_ptr::Vector{Int32}
	prod_j::Vector{Int32}
	prod_k::Vector{Int32}
	diff::Matrix{Int32}            # diff[i,j]: exps[i]-exps[j] position, 0 if not in box
	bounds::SVector{Nθ, Int}       # the per-parameter box bounds
	pos::Array{Int32, Nθ}          # pos[e .+ 1] = grlex position of exponent e
end

function GeometryParameterBasis(bounds::AbstractVector{<:Integer})
	mset = all_multiindices_in_box(collect(Int, bounds))
	exps = mset.exponents
	index = Dict(e => i for (i, e) in enumerate(exps))
	# Precompute the truncated-product index table once (keeps the hot poly ops
	# free of Dict lookups — the difference between usable and unusably slow).
	# Built row by row in `i`, with `j` ascending, which is exactly the CSR order
	# the convolution walks.
	L = length(exps)
	prod_ptr = Vector{Int32}(undef, L + 1)
	prod_j = Int32[]
	prod_k = Int32[]
	for i in 1:L
		prod_ptr[i] = Int32(length(prod_j) + 1)
		for j in 1:L
			k = get(index, exps[i] + exps[j], 0)
			k == 0 && continue
			push!(prod_j, Int32(j))
			push!(prod_k, Int32(k))
		end
	end
	prod_ptr[L+1] = Int32(length(prod_j) + 1)
	# The DIFFERENCE table, for the graded recurrences (reciprocal series,
	# Method 3's projection). Those run once per quadrature point over the whole
	# mesh, and each step needs the position of γ − β; doing that through the
	# `Dict` costs O(L²) hash lookups PER QUADRATURE POINT — hundreds of millions
	# on a real mesh. An L×L `Int32` table is 2.5 kB for a [4,4] box and stays in
	# cache. Zero encodes "difference has a negative component, or falls outside
	# the box", the two cases the recurrences skip anyway.
	diff = zeros(Int32, L, L)
	for i in 1:L, j in 1:L
		δ = exps[i] - exps[j]
		any(<(0), δ) && continue
		diff[i, j] = Int32(get(index, δ, 0))
	end
	# Dense exponent → position table. `all_multiindices_in_box` generates the
	# COMPLETE Cartesian box, so this has exactly `L` entries and every in-box
	# exponent is present — it is the hash-free counterpart of `index`, for the
	# expansions that run once per quadrature point over the whole mesh.
	Nθ = length(bounds)
	bnd = SVector{Nθ, Int}(bounds)
	pos = zeros(Int32, Tuple(bnd .+ 1))
	for (i, e) in enumerate(exps)
		pos[CartesianIndex(Tuple(e .+ 1))] = Int32(i)
	end
	return GeometryParameterBasis{Nθ}(mset, index, prod_ptr, prod_j, prod_k,
		diff, bnd, pos)
end

nterms(b::GeometryParameterBasis) = length(b.mset.exponents)

"""
	position_of(basis, e) -> Int

Grlex position of exponent `e`, or `0` when it falls outside the box (including
any negative component). The hash-free counterpart of `basis.index[e]`.
"""
@inline function position_of(b::GeometryParameterBasis{Nθ}, e::SVector{Nθ, Int}) where {Nθ}
	@inbounds for i in 1:Nθ
		(e[i] < 0 || e[i] > b.bounds[i]) && return 0
	end
	return @inbounds Int(b.pos[CartesianIndex(Tuple(e .+ 1))])
end

# The complex counterpart of a real Jacobian tensor type — the pulled-back
# gradients carry modal amplitudes, so the driver needs both.
@inline complex_tensor_type(::Type{<:Tensor{2, dim}}) where {dim} =
	Tensor{2, dim, ComplexF64, dim * dim}

# ---------------------------------------------------------------------
# Truncated multivariate products
# ---------------------------------------------------------------------
# C[γ] = Σ_{α+β=γ, α,β,γ ∈ box} op(A[α], B[β]).  `op` is *, ⋅ or ⊡.
@inline function _series_convolve(op, A::AbstractVector, B::AbstractVector,
	basis::GeometryParameterBasis)
	R = typeof(op(A[1], B[1]))
	out = fill(zero(R), length(basis.mset.exponents))
	return _series_convolve!(out, op, A, B, basis)
end

"""
	series_extent(A) -> Int

Position of the last non-zero coefficient of a θ-series, `0` if it is all zero.

This is the theory's degree bound, **measured rather than assumed**. `adj J` has
degree `≤ (d−1)·deg J` and `det J` degree `≤ d·deg J`, and graded-lex order makes
"total degree ≤ k" a contiguous prefix, so everything past this index is exactly
zero and can be skipped without any truncation. It adapts on its own: a
volume-preserving transform (`det J ≡ 1`) reports `1` and its convolutions
collapse to a single term, with no degree bookkeeping threaded anywhere.

Scanning from the end costs `O(L)` `iszero` calls and usually returns at once;
testing the same thing per PAIR costs `O(|prod|)`, which a profile of example 04
showed to be 23 % of the whole run.
"""
@inline function series_extent(A::AbstractVector)
	@inbounds for j in length(A):-1:1
		iszero(A[j]) || return j
	end
	return 0
end

# In-place form: the caller owns `out`, which is zeroed and then accumulated
# into. Every allocating `poly_*` is a one-line wrapper around this, so the two
# can never drift apart — and the assembly hot path calls only this one.
#
# Why it exists: the allocating forms put ~10 freshly allocated length-L series
# in the innermost (cell, quadrature point, basis function) loop. On example 04
# that came to 7.8 G allocations / 4.5 TiB for a single FAST reduction.
#
# `out` is zeroed IN FULL even though only a prefix is written. Bounding the
# `fill!` too would leave stale tail values readable by any consumer whose own
# extent disagrees — a silent-wrong-answer bug of exactly the kind this module
# exists to prevent, and the zeroing is nothing beside the convolution.
@inline function _series_convolve!(out::AbstractVector, op, A::AbstractVector,
	B::AbstractVector, basis::GeometryParameterBasis)
	fill!(out, zero(eltype(out)))
	nA = series_extent(A)
	nB = series_extent(B)
	(nA == 0 || nB == 0) && return out
	ptr, pj, pk = basis.prod_ptr, basis.prod_j, basis.prod_k
	@inbounds for i in 1:nA
		a = A[i]
		iszero(a) && continue                    # once per i, not once per pair
		for t in ptr[i]:(ptr[i+1]-1)
			j = pj[t]
			j > nB && break                      # j ascends within a row
			out[pk[t]] += op(a, B[j])
		end
	end
	return out
end

poly_mul(A, B, basis::GeometryParameterBasis) = _series_convolve(*, A, B, basis)
poly_dot(A, B, basis::GeometryParameterBasis) = _series_convolve(⋅, A, B, basis)
poly_contract(A, B, basis::GeometryParameterBasis) = _series_convolve(⊡, A, B, basis)

poly_mul!(out, A, B, basis::GeometryParameterBasis) = _series_convolve!(out, *, A, B, basis)
poly_dot!(out, A, B, basis::GeometryParameterBasis) = _series_convolve!(out, ⋅, A, B, basis)
poly_contract!(out, A, B, basis::GeometryParameterBasis) =
	_series_convolve!(out, ⊡, A, B, basis)

"""
	convolve_weight_accumulate!(dst, A, w, scale, basis)

`dst[k] += scale · Σ_{α+β=k} A[α] · w[β]` — the determinant weighting fused into
the element accumulation.

The driver's last act per basis function is to multiply the integrand series by
`(1/det J)^p` and add it to the element residual. Doing that as
`poly_mul` followed by a loop materialises a length-L temporary for every single
(cell, quadrature point, basis function); fusing removes it entirely.
"""
@inline function convolve_weight_accumulate!(dst, A::AbstractVector, w::AbstractVector,
	scale, basis::GeometryParameterBasis)
	nA = series_extent(A)
	nw = series_extent(w)
	(nA == 0 || nw == 0) && return dst
	ptr, pj, pk = basis.prod_ptr, basis.prod_j, basis.prod_k
	@inbounds for i in 1:nA
		a = A[i]
		iszero(a) && continue
		sa = scale * a
		for t in ptr[i]:(ptr[i+1]-1)
			j = pj[t]
			j > nw && break
			dst[pk[t]] += sa * w[j]
		end
	end
	return dst
end

# ---------------------------------------------------------------------
# Reciprocal series 1/p(θ), graded recurrence
# ---------------------------------------------------------------------
"""
	reciprocal_series(p, basis) -> Vector{Float64}

Coefficients of `1/p(θ)` truncated to the box, from the graded recurrence
`q[0] = 1/p[0]`, `q[γ] = -(1/p[0]) Σ_{0≠β≤γ} p[β] q[γ-β]`. Graded-lex order
makes each `q[γ-β]` (lower total degree) available before `q[γ]`.

This is **Method 4** of the theory write-up (App. A.3, "Reciprocal of the
Determinant"). The recurrence is the geometric series for `1/(1 − (1 − det J))`
in disguise, so it inherits that series' radius: applied to `det J` it is valid
only where `|det J − 1| < 1`, and it is unbounded in degree because the
reciprocal of a polynomial is not a polynomial. `basis.diff` supplies the
position of `γ − β` without hashing — this runs at every quadrature point.

See [`geometry_validity_report`](@ref) for the measured radius, and
[`AuxiliaryFieldInverseDet`](@ref) for Method 3, the alternative.
"""
function reciprocal_series(p::AbstractVector{<:Real}, basis::GeometryParameterBasis{Nθ}) where {Nθ}
	exps = basis.mset.exponents
	L = length(exps)
	abs(p[1]) > _GEOMETRY_PARAMETER_ZERO_TOL || error("reciprocal_series: p(0) = 0, series undefined")
	inv_p0 = 1.0 / p[1]
	q = zeros(Float64, L)
	q[1] = inv_p0
	D = basis.diff
	@inbounds for k in 2:L
		s = 0.0
		for j in 2:L                      # skip β = 0 (position 1)
			iszero(p[j]) && continue
			m = D[k, j]                   # position of γ − β, 0 if not in the box
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
function inv_det_power(inv_det::AbstractVector{<:Real}, n::Int, basis::GeometryParameterBasis)
	@assert n ≥ 1
	acc = copy(inv_det)
	for _ in 2:n
		acc = poly_mul(acc, inv_det, basis)
	end
	return acc
end

# ---------------------------------------------------------------------
# Jacobian series and its det / adj — dimension-general, any N_θ
# ---------------------------------------------------------------------
"""
	jacobian_series(Js, basis) -> Vector{<:Tensor{2,dim}}

Assemble the degree-1 Jacobian series aligned to `basis` from
`Js = (J₀, ∇ψ₁, …, ∇ψ_{Nθ})`: `J₀` at the zero multiindex, `∇ψ_i` at the
unit multiindex `e_i`. All other coefficients are zero.

This is the AFFINE map `x = x₀ + Σ_i θ_i ψ_i(x₀)`. For the theory's general
polynomial `x = Σ_α x_α θ^α`, pass the coefficients as multiindex ⇒ tensor pairs
instead — see the `AbstractVector{<:Pair}` method.
"""
function jacobian_series(Js::NTuple{M, TT},
	basis::GeometryParameterBasis{Nθ}) where {M, Nθ, TT <: Tensor{2}}
	@assert M == Nθ + 1 "expected J₀ plus one ∇ψ per parameter (got $M for $Nθ params)"
	L = nterms(basis)
	J = fill(zero(TT), L)
	J[1] = Js[1]                                   # J₀ at exponent 0
	for i in 1:Nθ
		e_i = SVector{Nθ, Int}(ntuple(k -> k == i ? 1 : 0, Nθ))
		J[basis.index[e_i]] = Js[i+1]              # ∇ψ_i at exponent e_i
	end
	return J
end

"""
	jacobian_series(Js::AbstractVector{<:Pair}, basis) -> Vector{<:Tensor{2,dim}}

The theory's **general polynomial map** `x(x₀,θ) = Σ_α x_α(x₀) θ^α`, given as
`multiindex => J_α` pairs (App. A, opening). The multiindex may be any
`Nθ`-element integer container; repeated multiindices accumulate, matching the
`Σ_α` they stand for.

A coefficient whose multiindex falls **outside the box throws** rather than being
silently dropped. Truncating the series computed *from* the map is this module's
job; truncating the map itself would change which geometry is being modelled
without saying so — the exact failure mode that made example 04 wrong.

The affine `NTuple` method above is the common case and builds this structure.
"""
function jacobian_series(Js::AbstractVector{<:Pair},
	basis::GeometryParameterBasis{Nθ}) where {Nθ}
	isempty(Js) && throw(ArgumentError("jacobian_series: no Jacobian coefficients given"))
	TT = typeof(last(first(Js)))
	J = fill(zero(TT), nterms(basis))
	for (α, Jα) in Js
		e = SVector{Nθ, Int}(α)
		p = position_of(basis, e)
		p == 0 && throw(ArgumentError(
			"jacobian_series: multiindex $(Tuple(e)) lies outside the θ-box " *
			"$(Tuple(basis.bounds)); widen the GeometryParameterBasis bounds"))
		J[p] += Jα
	end
	return J
end

# --- support of a series: the positions carrying a nonzero coefficient ---
# `det`/`adj` both walk it, and for the usual affine map it holds Nθ+1 of the L
# positions — which is what makes the two expansions below cheap.
function _series_support(J::AbstractVector)
	supp = Int[]
	@inbounds for a in eachindex(J)
		iszero(J[a]) || push!(supp, a)
	end
	return supp
end

"""
	determinant_series(J, basis[, supp]) -> Vector{Float64}

Exact multivariate coefficients of `det(J(θ))`, truncated to the box.

**Theory App. A.1 (Determinant).** The determinant is multilinear in the columns
of `J`, so expanding each column's series independently and collecting the terms
with `α₁+…+α_d = σ` gives `c_σ` directly — one `d×d` scalar determinant per
choice of support multiindex per column. Dimension-general, and no cofactor
formula appears.

Exactly polynomial with `deg det J ≤ d·deg J`, per parameter as well as in total.
"""
function determinant_series(J::AbstractVector{<:Tensor{2, dim, T}},
	basis::GeometryParameterBasis{Nθ},
	supp::Vector{Int} = _series_support(J)) where {Nθ, dim, T}
	out = zeros(T, nterms(basis))
	cols = Vector{Vec{dim, T}}(undef, dim)
	_det_expand!(out, J, basis, supp, cols, 1, zero(SVector{Nθ, Int}), Val(dim))
	return out
end

# One column at a time. `e` is the multiindex accumulated so far; exponents are
# non-negative so it can only grow, and a partial `e` already outside the box
# prunes the whole subtree — which keeps this output-sensitive rather than
# |supp|^d.
function _det_expand!(out, J, basis::GeometryParameterBasis{Nθ}, supp::Vector{Int},
	cols, k::Int, e::SVector{Nθ, Int}, ::Val{dim}) where {Nθ, dim}
	if k > dim
		p = position_of(basis, e)
		p == 0 && return nothing
		@inbounds out[p] += det(Tensor{2, dim}((i, j) -> cols[j][i]))
		return nothing
	end
	exps = basis.mset.exponents
	@inbounds for a in supp
		e2 = e + exps[a]
		position_of(basis, e2) == 0 && continue
		Ja = J[a]
		cols[k] = Vec{dim}(i -> Ja[i, k])
		_det_expand!(out, J, basis, supp, cols, k + 1, e2, Val(dim))
	end
	return nothing
end

"""
	adjugate_series(J, det_ser, basis[, supp]) -> Vector{<:Tensor{2,dim}}

Exact multivariate coefficients of `adj(J(θ))`, truncated to the box.

**Theory App. A.2 (Adjugate).** Matching θ^σ in the defining identity
`J · adj J = det J · I` gives a recurrence that is strictly triangular in
graded-lex order (`β ≤ σ` componentwise with `β ≠ σ` forces `|β| < |σ|`):

	J₀ A_σ + Σ_{0<α≤σ} J_α A_{σ−α} = c_σ I

**The appendix solves this by assuming `J₀ = I`, which this code cannot.** A
curved reference configuration has `J₀ ≠ I` — example 07's arch is exactly that,
`J₀ = I + ∇w ⊗ e₁` — so the leading term must be inverted explicitly:

	A_σ = J₀⁻¹ ( c_σ I − Σ_{0<α≤σ, α ∈ supp J} J_α A_{σ−α} ),   A_0 = adj(J₀)

which reduces to the appendix's form when `J₀ = I`. The same correction applies
to Method 3, where the leading matrix is the `c₀`-weighted mass matrix, not `M`.

The sum runs only over the support of `J` — `Nθ+1` terms for an affine map, not
`L` — so this costs `O(L·|supp J|)` per quadrature point. `basis.diff` supplies
the position of `σ − α` without hashing.

Exactly polynomial with `deg adj J ≤ (d−1)·deg J`. Truncating `det J` to the box
does not corrupt it: `A_σ` reads only `c_σ`, which is in the box whenever `A_σ` is.
"""
function adjugate_series(J::AbstractVector{TT}, det_ser::AbstractVector,
	basis::GeometryParameterBasis,
	supp::Vector{Int} = _series_support(J)) where {TT <: Tensor{2}}
	L = nterms(basis)
	J₀ = J[1]
	d₀ = det(J₀)
	abs(d₀) > _GEOMETRY_PARAMETER_ZERO_TOL || error(
		"adjugate_series: det J₀ = $d₀ — the reference configuration is degenerate")
	invJ₀ = inv(J₀)
	Id = one(TT)
	A = fill(zero(TT), L)
	D = basis.diff
	@inbounds for k in 1:L
		acc = det_ser[k] * Id
		for a in supp
			a == 1 && continue            # α = 0 is the J₀ A_σ term, split off above
			m = D[k, a]                   # position of σ − α; 0 ⟹ outside the box
			m == 0 && continue
			acc -= J[a] ⋅ A[m]
		end
		A[k] = invJ₀ ⋅ acc
	end
	return A
end

"""
	det_adj_series(J, basis) -> (det_ser, adj_ser)

`det(J(θ))` and `adj(J(θ))` together, sharing one scan of `J`'s support.

Both are **exactly polynomial** — App. A.1 and A.2 — which is what makes the
adjugate identity `J⁻¹ = adj J / det J` worth using: it concentrates all the
non-polynomial content of `J⁻¹` into the single scalar `1/det J`, left to
[`reciprocal_series`](@ref) (Method 4) or [`AuxiliaryFieldInverseDet`](@ref)
(Method 3).
"""
function det_adj_series(J::AbstractVector{<:Tensor{2}}, basis::GeometryParameterBasis)
	supp = _series_support(J)
	det_ser = determinant_series(J, basis, supp)
	return det_ser, adjugate_series(J, det_ser, basis, supp)
end
