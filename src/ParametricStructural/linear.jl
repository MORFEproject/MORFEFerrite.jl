# =====================================================================
# General parametric linear stiffness/mass assembly + K/C/M corrections.
#
# For J(θ,x₀) = J₀(x₀) + Σᵢ θᵢ ∇ψᵢ(x₀), the θ-coefficient matrices are
#   K_α = θ^α coeff of  ∫ ε_adj(v) ⊡ σ(ε_adj(u)) · (1/det J) dV₀
#   M_α = θ^α coeff of  ∫ ρ (u·v) · det J           dV₀
# with ε_adj(u) = sym(∇₀u · adj J(θ)).  Each non-base coefficient (α ≠ 0)
# becomes a MultilinearMap correction of external multiplicity |α|.
# =====================================================================

using Ferrite
using Tensors
using SparseArrays: nnz
using MORFE: MultilinearMap

"""
	assemble_parametric_K_M!(K_arr, M_arr, dh, cv, λ, μ, ρ, geom, basis)

Fill `K_arr[m]`, `M_arr[m]` with the θ-coefficient matrices at multiindex
`basis.mset.exponents[m]`. `K_arr`, `M_arr` must each be length `nterms(basis)`
vectors of sparse matrices sharing `dh`'s pattern (`allocate_matrix(dh)`).
`geom(x₀_qp)` returns `(J₀, ∇ψ₁, …, ∇ψ_Nθ)`.
"""
function assemble_parametric_K_M!(K_arr::Vector, M_arr::Vector,
	dh::DofHandler, cv::CellValues,
	λ::Float64, μ::Float64, ρ::Float64, geom,
	basis::ThetaBasis)
	L = nterms(basis)
	@assert length(K_arr) == L && length(M_arr) == L
	asmK = [start_assemble(K) for K in K_arr]
	asmM = [start_assemble(M) for M in M_arr]
	nbf = getnbasefunctions(cv)
	ke = [zeros(nbf, nbf) for _ in 1:L]
	me = [zeros(nbf, nbf) for _ in 1:L]

	for cell in CellIterator(dh)
		for m in 1:L
			fill!(ke[m], 0.0); fill!(me[m], 0.0)
		end
		reinit!(cv, cell)
		coords = getcoordinates(cell)
		for q in 1:getnquadpoints(cv)
			dΩ₀ = getdetJdV(cv, q)
			x₀ = spatial_coordinate(cv, q, coords)
			J = jacobian_series(geom(x₀), basis)
			det_ser, adj_ser = det_adj_series(J, basis)
			inv_det = reciprocal_series(det_ser, basis)
			for i in 1:nbf
				∇Ni = shape_gradient(cv, q, i)
				Ni = shape_value(cv, q, i)
				ε_adj_i = [symmetric(∇Ni ⋅ a) for a in adj_ser]
				for j in 1:nbf
					∇Nj = shape_gradient(cv, q, j)
					Nj = shape_value(cv, q, j)
					ε_adj_j = [symmetric(∇Nj ⋅ a) for a in adj_ser]
					σ_adj_j = [σ_lame(ε, λ, μ) for ε in ε_adj_j]
					bracket = poly_contract(ε_adj_i, σ_adj_j, basis)
					K_ser = poly_mul(bracket, inv_det, basis)
					NiNj = Ni ⋅ Nj
					for m in 1:L
						ke[m][i, j] += K_ser[m] * dΩ₀
						me[m][i, j] += ρ * NiNj * det_ser[m] * dΩ₀
					end
				end
			end
		end
		for m in 1:L
			assemble!(asmK[m], celldofs(cell), ke[m])
			assemble!(asmM[m], celldofs(cell), me[m])
		end
	end
	return nothing
end

# --- correction builders (skip the base α = 0, kept as NDOrderModel terms) ---
"""
	build_K_corrections(K_arr, basis) -> Vector{MultilinearMap}

`−θ^α · K_α · u` for every α ≠ 0 (modal arity (1,0,0), external mult |α|).
"""
function build_K_corrections(K_arr::Vector, basis::ThetaBasis)
	corr = MultilinearMap[]
	for (αidx, α) in enumerate(basis.mset.exponents)
		all(iszero, α) && continue
		Kk = K_arr[αidx]
		nnz(Kk) > 0 || continue
		m = sum(α); comp = _expand_multiindex(α)
		push!(corr, MultilinearMap(Base.invokelatest(_ps_linK, Val(m), comp, Kk), (1, 0, 0), m))
	end
	return corr
end

"""
	build_C_corrections(K_arr, M_arr, α_damp, β_damp, basis) -> Vector{MultilinearMap}

Parametric Rayleigh damping `C(θ) = α_damp M(θ) + β_damp K(θ)`, corrections for α ≠ 0
(modal arity (0,1,0)).
"""
function build_C_corrections(K_arr::Vector, M_arr::Vector,
	α_damp::Float64, β_damp::Float64, basis::ThetaBasis)
	corr = MultilinearMap[]
	for (αidx, α) in enumerate(basis.mset.exponents)
		all(iszero, α) && continue
		Ck = β_damp != 0 ? β_damp * K_arr[αidx] : nothing
		if α_damp != 0
			Ck = Ck === nothing ? α_damp * M_arr[αidx] : Ck + α_damp * M_arr[αidx]
		end
		Ck === nothing && continue
		nnz(Ck) > 0 || continue
		m = sum(α); comp = _expand_multiindex(α)
		push!(corr, MultilinearMap(Base.invokelatest(_ps_linC, Val(m), comp, Ck), (0, 1, 0), m))
	end
	return corr
end

"""
	build_M_corrections(M_arr, basis) -> Vector{MultilinearMap}

`−θ^α · M_α · ü` for every α ≠ 0 (modal arity (0,0,1); requires ORD = 3).
"""
function build_M_corrections(M_arr::Vector, basis::ThetaBasis)
	corr = MultilinearMap[]
	for (αidx, α) in enumerate(basis.mset.exponents)
		all(iszero, α) && continue
		Mk = M_arr[αidx]
		nnz(Mk) > 0 || continue
		m = sum(α); comp = _expand_multiindex(α)
		push!(corr, MultilinearMap(Base.invokelatest(_ps_linM, Val(m), comp, Mk), (0, 0, 1), m))
	end
	return corr
end
