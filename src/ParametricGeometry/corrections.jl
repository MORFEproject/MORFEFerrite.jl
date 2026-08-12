# =====================================================================
# Linear-operator θ-corrections — physics-blind.
#
# A parametric linear operator A(θ) = Σ_α θ^α A_α contributes its base
# coefficient A_0 to the model's `linear_terms` and every α ≠ 0 as a
# `MultilinearMap` of external multiplicity |α|.
#
# The three builders this replaces (K, C and M corrections) differed only in
# the modal arity tuple they passed: the closures were identical, because a
# linear correction takes exactly ONE modal argument whichever derivative slot
# it occupies. Arity is therefore a parameter, not a new function — which is
# also what lets a first-order physics (Navier-Stokes: ORD = 1, arity `(1,)`)
# use this file unchanged.
# =====================================================================

using SparseArrays: nnz
using MORFE: MultilinearMap

# One factory for every arity: the closure signature is (res, x, r...) regardless
# of which derivative slot `arity` names, so `A * x` is all it ever does.
for mm in 0:_PG_MAX_EXT
	ext = [Symbol("r$i") for i in 1:mm]
	if mm == 0
		@eval _pg_linear(::Val{0}, comp, A) = (res, x) -> (res .-= (A * x))
	else
		factor = Expr(:call, :*, [:($(ext[s])[comp[$s]]) for s in 1:mm]...)
		@eval _pg_linear(::Val{$mm}, comp, A) =
			(res, x, $(ext...)) -> (res .-= ($factor) .* (A * x))
	end
end

"""
	build_linear_corrections(A_arr, basis, arity) -> Vector{MultilinearMap}

`−θ^α · A_α · x` for every α ≠ 0, where `A_arr[αidx]` is the operator's θ^α
coefficient matrix and `arity` names the derivative slot `x` occupies.

The base coefficient (α = 0) is skipped: it belongs in the model's
`linear_terms`, not among its nonlinear terms. Structurally empty coefficients
are skipped too — an all-zero correction is a term the solve would evaluate for
nothing.

For a second-order structure on the augmented `ORD = 3` model:
`(1,0,0)` is a stiffness correction, `(0,1,0)` damping, `(0,0,1)` mass. For a
first-order physics, `(1,)`.
"""
function build_linear_corrections(A_arr::Vector, basis::GeometryParameterBasis,
	arity::NTuple{N, Int}) where {N}
	sum(arity) == 1 || throw(ArgumentError(
		"a linear correction acts on exactly one modal argument, but arity $arity " *
		"sums to $(sum(arity))"))
	length(A_arr) == nterms(basis) || throw(ArgumentError(
		"A_arr has $(length(A_arr)) coefficients but the θ-basis has $(nterms(basis))"))

	corr = MultilinearMap[]
	for (αidx, α) in enumerate(basis.mset.exponents)
		all(iszero, α) && continue
		A = A_arr[αidx]
		A === nothing && continue
		nnz(A) > 0 || continue
		mm = sum(α)
		mm <= _PG_MAX_EXT || throw(ArgumentError(
			"θ-multiindex $α has total degree $mm > _PG_MAX_EXT = $_PG_MAX_EXT"))
		cl = Base.invokelatest(_pg_linear, Val(mm), _expand_multiindex(α), A)
		push!(corr, MultilinearMap(cl, arity, mm))
	end
	return corr
end

"""
	assemble_linear_series!(A_arr, B_arr, pd, kernel)

Fill `A_arr[m]`, `B_arr[m]` with the θ^α coefficient matrices of the two linear
operators `kernel` defines (for a structure: stiffness and mass), at multiindex
`basis.mset.exponents[m]`.

`A_arr` and `B_arr` must each be `nterms(basis)`-long vectors of sparse matrices
sharing the DOF handler's pattern (`allocate_matrix(dh)`). The driver owns the
cell loop and the assembly; the kernel supplies only the per-entry θ-series
through [`linear_qp_series!`](@ref).
"""
function assemble_linear_series!(A_arr::Vector, B_arr::Vector,
	pd::ParametricDiscretisation, kernel::AbstractPullbackKernel)
	b = basis(pd)
	L = nterms(pd)
	@assert length(A_arr) == L&&length(B_arr) == L
	cv = pd.cv
	asmA = [start_assemble(A) for A in A_arr]
	asmB = [start_assemble(B) for B in B_arr]
	nbf = getnbasefunctions(cv)
	ae = [zeros(nbf, nbf) for _ in 1:L]
	be = [zeros(nbf, nbf) for _ in 1:L]
	a_ser = Vector{Float64}(undef, L)
	b_ser = Vector{Float64}(undef, L)

	for (ci, cell) in enumerate(CellIterator(pd.dh))
		for m in 1:L
			fill!(ae[m], 0.0)
			fill!(be[m], 0.0)
		end
		reinit!(cv, cell)
		for q in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q)
			ctx = QPContext(cv, q, b, pd.cache.adj[ci][q], pd.cache.det[ci][q],
				pd.cache.inv_det[ci][q])
			for i in 1:nbf, j in 1:nbf
				linear_qp_series!(a_ser, b_ser, kernel, ctx, i, j)
				for m in 1:L
					ae[m][i, j] += a_ser[m] * dΩ₀
					be[m][i, j] += b_ser[m] * dΩ₀
				end
			end
		end
		for m in 1:L
			assemble!(asmA[m], celldofs(cell), ae[m])
			assemble!(asmB[m], celldofs(cell), be[m])
		end
	end
	return nothing
end
