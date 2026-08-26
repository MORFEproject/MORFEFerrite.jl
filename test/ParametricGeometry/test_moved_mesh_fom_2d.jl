# The 2D counterpart of test_moved_mesh_fom.jl — the gate that makes
# "dimension-general" a MEASUREMENT rather than a claim.
#
# Every production physics in this package is 3D, so without this file nothing
# would ever drive dim = 2 through jacobian_series → determinant_series →
# adjugate_series → PullbackCache → QPContext → sweep_all!, and the module's
# d-generality would rest on inspection alone.
#
# The kernel below is TEST-ONLY plane-strain elasticity. Its physics is not under
# test; the COORDINATE TRANSFORM is. The references are chosen accordingly:
#
#   linear forms     `_plain_KM_2d`, an ordinary assembler that never touches
#                    ParametricGeometry — fully independent.
#   quadratic form   the SAME kernel driven with an IDENTITY transform on the
#                    MOVED mesh. There adj ≡ I and det ≡ 1, so none of the series
#                    algebra runs on the reference side and a bug in
#                    determinant_series, adjugate_series, reciprocal_series or
#                    the convolutions still shows up as a disagreement.
#
# In 2D an affine map has deg adj J ≤ (d−1)·deg J = 1, so the degree staircase
# has the same shape as in 3D: K (adj²) exact from MAXT 2, quadratic (adj³) from 3.

using Test
using MORFE, MORFEFerrite
using Ferrite, Tensors, StaticArrays
using LinearAlgebra, SparseArrays

const PG2 = MORFEFerrite.ParametricGeometry
import MORFEFerrite.ParametricGeometry: det_weight_power, qp_prepare, qp_integrand!,
	linear_qp_series!

const T2 = Tensor{2, 2, Float64, 4}
const _λ2, _μ2, _ρ2 = 1.2e3, 8.0e2, 2.7e-3

const _∇ψ2_SHEAR = T2((i, j) -> (i == 2 && j == 1) ? 1.0 : 0.0)    # det J ≡ 1
const _∇ψ2_STRETCH = T2((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)  # det J = 1+θ

# ── test-only plane-strain kernel ────────────────────────────────────────────
struct Elastic2D{DEG} <: PG2.AbstractPullbackKernel{DEG}
	λ::Float64
	μ::Float64
	ρ::Float64
end

@inline _σ2(E, λ, μ) = λ * tr(E) * one(E) + 2μ * E

det_weight_power(::Elastic2D{2}) = 2

function linear_qp_series!(k_ser, m_ser, k::Elastic2D{0}, ctx, i::Int, j::Int)
	b = ctx.basis
	cv, q = ctx.cv, ctx.q
	∇Ni, ∇Nj = shape_gradient(cv, q, i), shape_gradient(cv, q, j)
	εi = [symmetric(∇Ni ⋅ a) for a in ctx.adj]
	σj = [_σ2(symmetric(∇Nj ⋅ a), k.λ, k.μ) for a in ctx.adj]
	cw = PG2.poly_mul(PG2.poly_contract(εi, σj, b), ctx.inv_det, b)
	NiNj = shape_value(cv, q, i) ⋅ shape_value(cv, q, j)
	@inbounds for m in eachindex(k_ser)
		k_ser[m] = cw[m]
		m_ser[m] = k.ρ * NiNj * ctx.det[m]
	end
	return nothing
end

# Green-Lagrange quadratic form, the 2D image of StructuralSVK's DEG = 2 kernel.
function qp_prepare(k::Elastic2D{2}, ctx, ∇u_adj::NTuple{2, <:Vector})
	b = ctx.basis
	∇u1a, ∇u2a = ∇u_adj
	t1 = [transpose(g) for g in ∇u1a]
	t2 = [transpose(g) for g in ∇u2a]
	σ1 = [_σ2(symmetric(g), k.λ, k.μ) for g in ∇u1a]
	σ2 = [_σ2(symmetric(g), k.λ, k.μ) for g in ∇u2a]
	AB = PG2.poly_dot(t1, ∇u2a, b)
	BA = PG2.poly_dot(t2, ∇u1a, b)
	σE = [_σ2(symmetric(0.25 * (AB[m] + BA[m])), k.λ, k.μ) for m in eachindex(AB)]
	return (; t1, t2, σ1, σ2, σE)
end

function qp_integrand!(integ, k::Elastic2D{2}, ctx, st, ∇N_adj::Vector)
	b = ctx.basis
	c1 = PG2.poly_contract([symmetric(g) for g in ∇N_adj], st.σE, b)
	c2 = PG2.poly_contract([symmetric(g) for g in PG2.poly_dot(st.t1, ∇N_adj, b)], st.σ2, b)
	c3 = PG2.poly_contract([symmetric(g) for g in PG2.poly_dot(st.t2, ∇N_adj, b)], st.σ1, b)
	@inbounds for m in eachindex(integ)
		integ[m] = c1[m] + 0.5 * (c2[m] + c3[m])
	end
	return integ
end

# ── the independent linear reference: no ParametricGeometry anywhere ─────────
function _plain_KM_2d(dh, cv, λ, μ, ρ)
	K, M = allocate_matrix(dh), allocate_matrix(dh)
	aK, aM = start_assemble(K), start_assemble(M)
	nbf = getnbasefunctions(cv)
	ke, me = zeros(nbf, nbf), zeros(nbf, nbf)
	for cell in CellIterator(dh)
		reinit!(cv, cell)
		fill!(ke, 0.0)
		fill!(me, 0.0)
		for q in 1:getnquadpoints(cv)
			dΩ = getdetJdV(cv, q)
			for i in 1:nbf, j in 1:nbf
				εi = symmetric(shape_gradient(cv, q, i))
				εj = symmetric(shape_gradient(cv, q, j))
				ke[i, j] += (εi ⊡ _σ2(εj, λ, μ)) * dΩ
				me[i, j] += ρ * (shape_value(cv, q, i) ⋅ shape_value(cv, q, j)) * dΩ
			end
		end
		assemble!(aK, celldofs(cell), ke)
		assemble!(aM, celldofs(cell), me)
	end
	return K, M
end

# ── mesh, spaces, node motion ────────────────────────────────────────────────
_base_grid_2d() = generate_grid(Quadrilateral, (2, 2), Vec(0.0, 0.0), Vec(2.0, 1.0))

function _mm_space_2d(grid)
	ip = Lagrange{RefQuadrilateral, 2}()^2
	# generate_grid makes LINEAR quadrilaterals, so the geometric map is bilinear
	# and a linear ψ is reproduced by node-moving exactly.
	cv = CellValues(QuadratureRule{RefQuadrilateral}(3), ip, Lagrange{RefQuadrilateral, 1}())
	dh = DofHandler(grid)
	add!(dh, :u, ip)
	close!(dh)
	return dh, cv
end

# Same cells, displaced nodes ⇒ same topology ⇒ same DOF numbering.
function _moved_2d(grid, ∇ψs, θ)
	nodes = [Node(n.x + sum(θ[i] * (∇ψs[i] ⋅ n.x) for i in eachindex(θ)))
			 for n in grid.nodes]
	return Grid(collect(grid.cells), nodes)
end

_mono2(θ, α) = prod(θ[i]^α[i] for i in eachindex(α))
_at2(arr, basis, θ) = sum(_mono2(θ, α) * arr[m] for (m, α) in enumerate(basis.mset.exponents))
_cols_at2(A, basis, θ) =
	sum(_mono2(θ, α) * view(A, :, m) for (m, α) in enumerate(basis.mset.exponents))
_relerr2(a, b) = norm(a - b) / max(norm(b), eps())

function _mm2_case(∇ψs, maxt, θ, u)
	grid = _base_grid_2d()
	dh, cv = _mm_space_2d(grid)
	basis = PG2.GeometryParameterBasis(fill(maxt, length(∇ψs)))
	cache = PG2.PullbackCache(dh, cv, x -> (one(T2), ∇ψs...), basis; det_powers = [2])
	f2l = PG2.free_dof_map(ndofs(dh), 1:ndofs(dh))
	pd = PG2.ParametricDiscretisation(dh, cv, f2l, ndofs(dh), cache)

	L = PG2.nterms(basis)
	K = [allocate_matrix(dh) for _ in 1:L]
	M = [allocate_matrix(dh) for _ in 1:L]
	PG2.assemble_linear_series!(K, M, pd, Elastic2D{0}(_λ2, _μ2, _ρ2))

	pm = PG2.ParametricMap(pd, Elastic2D{2}(_λ2, _μ2, _ρ2))
	A = PG2.sweep_all!(zeros(ComplexF64, ndofs(dh), L), pm, (u, u))

	# ── references on the MOVED mesh ──
	dh_m, cv_m = _mm_space_2d(_moved_2d(grid, ∇ψs, θ))
	Kr, Mr = _plain_KM_2d(dh_m, cv_m, _λ2, _μ2, _ρ2)

	# Identity transform ⇒ adj ≡ I, det ≡ 1: the series algebra is trivial here,
	# so column 1 is plain assembly on the moved mesh.
	b0 = PG2.GeometryParameterBasis([1])
	c0 = PG2.PullbackCache(dh_m, cv_m, x -> (one(T2), zero(T2)), b0; det_powers = [2])
	pd0 = PG2.ParametricDiscretisation(dh_m, cv_m, PG2.free_dof_map(ndofs(dh_m), 1:ndofs(dh_m)),
		ndofs(dh_m), c0)
	pm0 = PG2.ParametricMap(pd0, Elastic2D{2}(_λ2, _μ2, _ρ2))
	A0 = PG2.sweep_all!(zeros(ComplexF64, ndofs(dh_m), PG2.nterms(b0)), pm0, (u, u))

	return (; eK = _relerr2(_at2(K, basis, θ), Kr), eM = _relerr2(_at2(M, basis, θ), Mr),
		e2 = _relerr2(_cols_at2(A, basis, θ), view(A0, :, 1)), cache)
end

_probe_u2(n) = ComplexF64[cis(0.53k) * (1 + 0.013k) for k in 1:n]

@testset "2D parametric transform ≡ FOM on the moved mesh" begin
	u = _probe_u2(ndofs(_mm_space_2d(_base_grid_2d())[1]))

	@testset "θ = 0 is the same assembly on the same mesh" begin
		r = _mm2_case((_∇ψ2_STRETCH,), 3, (0.0,), u)
		@test r.eK < 1e-13
		@test r.eM < 1e-13
		@test r.e2 < 1e-13
	end

	# det J ≡ 1 and deg(adj J) = 1, so each form is an EXACT polynomial of degree
	# equal to its gradient-factor count and must go machine-exact right there.
	@testset "shear: each form exact at its own degree (det J ≡ 1)" begin
		for θ in ((0.2,), (0.5,))
			r2 = _mm2_case((_∇ψ2_SHEAR,), 2, θ, u)
			r3 = _mm2_case((_∇ψ2_SHEAR,), 3, θ, u)

			@test r2.eK < 1e-12          # K uses adj² → exact from MAXT = 2
			@test r2.eM < 1e-12          # mass uses det J ≡ 1 → always exact
			@test r3.e2 < 1e-12          # quadratic uses adj³ → exact from 3

			# …and genuinely truncated one order below, or the staircase is vacuous.
			@test r2.e2 > 1e-9
		end
	end

	# det J = 1+θ ⇒ the reciprocal is a geometric series; error ~ θ^(MAXT+1).
	@testset "stretch: reciprocal series converges at the theoretical rate" begin
		θ = (0.1,)
		lo = _mm2_case((_∇ψ2_STRETCH,), 2, θ, u)
		hi = _mm2_case((_∇ψ2_STRETCH,), 4, θ, u)
		@test hi.eK < lo.eK
		@test hi.e2 < lo.e2
		@test 1e-4 < hi.eK / lo.eK < 1e-1
		@test hi.eM < 1e-12          # det J = 1+θ is inside the box ⇒ mass exact
	end

	@testset "two parameters: cross terms converge" begin
		θ = (0.1, 0.1)
		lo = _mm2_case((_∇ψ2_SHEAR, _∇ψ2_STRETCH), 2, θ, u)
		hi = _mm2_case((_∇ψ2_SHEAR, _∇ψ2_STRETCH), 4, θ, u)
		@test hi.eK < lo.eK
		@test hi.e2 < lo.e2
	end

	# The validity machinery must work in 2D too — det J = 1+θ ⇒ radius 1.
	@testset "validity radius is measured in 2D as well" begin
		stretch = _mm2_case((_∇ψ2_STRETCH,), 4, (0.1,), u).cache
		r = PG2.geometry_validity_report(stretch)
		@test !r.unconditional
		@test r.radius_pos[1] ≈ 1.0 rtol=1e-2
		@test PG2.geometry_validity_at(stretch, (1.0,)).convergent == false

		shear = _mm2_case((_∇ψ2_SHEAR,), 3, (0.1,), u).cache
		@test PG2.geometry_validity_report(shear).unconditional
	end
end
