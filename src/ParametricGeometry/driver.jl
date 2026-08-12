# =====================================================================
# The assembly driver — physics-blind.
#
# Owns the cell loop, the gather/scatter, the all-coefficients sweep, the
# shared-input cache, and the wrapping of each θ-coefficient as a MORFE
# `MultilinearMap`. Every physics-specific quantity comes back through the
# `AbstractPullbackKernel` interface in `kernel.jl`.
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

@inline function scatter_local!(res::AbstractVector{T}, re::AbstractVector{T},
	dofs::Vector{Int}, free_to_local::Dict{Int, Int}) where {T}
	@inbounds for (i, d) in pairs(dofs)
		haskey(free_to_local, d) || continue
		res[free_to_local[d]] += re[i]
	end
	return res
end

"""
	ParametricDiscretisation(dh, cv, free_to_local, n_free, cache)

The FE side of a parametric problem: the DOF handler and quadrature, the
free-DOF restriction, and the [`PullbackCache`](@ref) for this geometry and
θ-basis. Shared by every kernel over the same mesh.
"""
struct ParametricDiscretisation{Nθ, DH, CV}
	dh::DH
	cv::CV
	free_to_local::Dict{Int, Int}
	n_free::Int
	cache::PullbackCache{Nθ}
end

basis(pd::ParametricDiscretisation) = pd.cache.basis
nterms(pd::ParametricDiscretisation) = nterms(pd.cache.basis)

"""
	ParametricMap(pd, kernel)

One physics kernel, expanded over the θ-basis and ready to be turned into
`MultilinearMap`s by [`multilinear_maps`](@ref).

Holds the shared-input cache: one FE sweep computes **all** θ-coefficients for
the same arithmetic cost as one, so the per-α closures share a single sweep per
input tuple. The hit test is an exact content comparison (no hashing), so
results are bit-identical to computing each coefficient separately.
"""
struct ParametricMap{DEG, Nθ, PD, K}
	pd::PD
	kernel::K
	u_store::NTuple{3, Vector{ComplexF64}}
	res_all::Matrix{ComplexF64}      # n_free × nterms(basis)
	valid::Base.RefValue{Bool}
end

function ParametricMap(pd::ParametricDiscretisation{Nθ},
	kernel::AbstractPullbackKernel{DEG}) where {DEG, Nθ}
	u_store = ntuple(_ -> ComplexF64[], 3)
	res_all = zeros(ComplexF64, pd.n_free, nterms(pd))
	return ParametricMap{DEG, Nθ, typeof(pd), typeof(kernel)}(
		pd, kernel, u_store, res_all, Ref(false))
end

Base.ndims(::ParametricMap{DEG}) where {DEG} = DEG

function Base.show(io::IO, ::MIME"text/plain", m::ParametricMap{DEG, Nθ}) where {DEG, Nθ}
	println(io, "ParametricMap{DEG=$DEG, Nθ=$Nθ}")
	println(io, "  kernel   : $(nameof(typeof(m.kernel)))")
	println(io, "  free DOFs: $(m.pd.n_free)")
	print(io, "  θ-terms  : $(nterms(m.pd))")
end

# --- the all-coefficients sweep --------------------------------------
"""
	sweep_all!(out, m::ParametricMap{DEG}, us::NTuple{DEG})

Fill `out[:, α]` with the θ^α coefficient of the kernel's form evaluated at the
`DEG` displacement inputs `us`, for **every** α in the basis.

Computing the whole series costs the same per coefficient as computing one, so
this is the only sweep the per-α closures need.
"""
function sweep_all!(out::Matrix{ComplexF64}, m::ParametricMap{DEG},
	us::NTuple{DEG, <:AbstractVector}) where {DEG}
	fill!(out, zero(ComplexF64))
	pd = m.pd
	cv = pd.cv
	b = basis(pd)
	L = nterms(pd)
	nbf = getnbasefunctions(cv)
	nd = ndofs_per_cell(pd.dh)

	ue = ntuple(_ -> zeros(ComplexF64, nd), DEG)
	re = zeros(ComplexF64, nd, L)
	integ = Vector{ComplexF64}(undef, L)
	# Hoisted out of the cell loop: a Dict lookup has no business in the hot path.
	invpow = inv_det_power_series(pd.cache, det_weight_power(m.kernel))

	for (ci, cell) in enumerate(CellIterator(pd.dh))
		reinit!(cv, cell)
		dofs = celldofs(cell)
		for k in 1:DEG
			gather_local!(ue[k], us[k], dofs, pd.free_to_local)
		end
		fill!(re, zero(ComplexF64))
		for q in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q)
			adj_ser = pd.cache.adj[ci][q]
			wser_det = invpow[ci][q]
			ctx = QPContext(cv, q, b, adj_ser, pd.cache.det[ci][q],
				pd.cache.inv_det[ci][q])

			∇u_adj = ntuple(k -> ∇adj_series(function_gradient(cv, q, ue[k]), adj_ser), DEG)
			state = qp_prepare(m.kernel, ctx, ∇u_adj)

			for I in 1:nbf
				∇N_adj = ∇adj_series(shape_gradient(cv, q, I), adj_ser)
				qp_integrand!(integ, m.kernel, ctx, state, ∇N_adj)
				wser = poly_mul(integ, wser_det, b)
				for mα in 1:L
					re[I, mα] += wser[mα] * dΩ₀
				end
			end
		end
		for mα in 1:L
			scatter_local!(view(out, :, mα), view(re, :, mα), dofs, pd.free_to_local)
		end
	end
	return out
end

@inline function _store!(dst::Vector{ComplexF64}, src::AbstractVector)
	resize!(dst, length(src))
	copyto!(dst, src)
	return dst
end

function _ensure!(m::ParametricMap{DEG}, us::NTuple{DEG, <:AbstractVector}) where {DEG}
	if m.valid[]
		hit = true
		for k in 1:DEG
			m.u_store[k] == us[k] || (hit = false; break)
		end
		hit && return m.res_all
	end
	for k in 1:DEG
		_store!(m.u_store[k], us[k])
	end
	sweep_all!(m.res_all, m, us)
	m.valid[] = true
	return m.res_all
end

# --- external-state factor: expand α into a component list ----------
# α = (a₁,…,a_Nθ) → (1 repeated a₁ times, 2 repeated a₂ times, …).
_expand_multiindex(α) = Tuple(reduce(vcat,
	[fill(i, α[i]) for i in eachindex(α)]; init = Int[]))

# --- fixed-arity closure factories ----------------------------------
# One factory per (field arity, external multiplicity). The external factor
# θ^α = Πₛ r[comp[s]] is built at closure-construction time so the solve path
# does no multiindex work.
const _PG_MAX_EXT = 12   # ≥ largest total θ-degree used

for deg in 2:3, mm in 0:_PG_MAX_EXT
	fields = [Symbol("u$i") for i in 1:deg]
	ext = [Symbol("r$i") for i in 1:mm]
	if mm == 0
		@eval _pg_form(::Val{$deg}, ::Val{0}, comp, pm, αidx) =
			(res, $(fields...)) -> begin
				A = _ensure!(pm, ($(fields...),))
				res .-= view(A, :, αidx)
			end
	else
		factor = Expr(:call, :*, [:($(ext[s])[comp[$s]]) for s in 1:mm]...)
		@eval _pg_form(::Val{$deg}, ::Val{$mm}, comp, pm, αidx) =
			(res, $(fields...), $(ext...)) -> begin
				A = _ensure!(pm, ($(fields...),))
				res .-= ($factor) .* view(A, :, αidx)
			end
	end
end

"""
	multilinear_maps(m::ParametricMap; arity)

One MORFE `MultilinearMap` per θ-multiindex α: the θ^α coefficient of the
kernel's form, with external multiplicity `|α|` so the solve multiplies it by
the matching product of frozen θ states.

`arity` is the modal arity tuple — length `ORD`, with `DEG` in the slot the form
acts on. For a second-order structure with the augmented `ORD = 3` model, a
quadratic displacement form is `(2, 0, 0)`.
"""
function multilinear_maps(m::ParametricMap{DEG}; arity::NTuple{N, Int}) where {DEG, N}
	sum(arity) == DEG || throw(ArgumentError(
		"arity $arity sums to $(sum(arity)) but this kernel has field arity DEG = $DEG"))
	maps = MultilinearMap[]
	for (αidx, α) in enumerate(basis(m.pd).mset.exponents)
		mm = sum(α)
		mm <= _PG_MAX_EXT || throw(ArgumentError(
			"θ-multiindex $α has total degree $mm > _PG_MAX_EXT = $_PG_MAX_EXT"))
		cl = Base.invokelatest(_pg_form, Val(DEG), Val(mm), _expand_multiindex(α), m, αidx)
		push!(maps, MultilinearMap(cl, arity, mm))
	end
	return maps
end
