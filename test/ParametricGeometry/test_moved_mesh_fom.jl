# The parametric transform against a FOM whose GEOMETRY ACTUALLY CHANGED.
#
# Every other gate in this directory compares the module against itself or
# against a frozen literal. This one compares it against physics: freeze θ, MOVE
# THE MESH NODES to x = x₀ + Σᵢ θᵢψᵢ(x₀), and reassemble with the ordinary
# non-parametric StructuralSVK code, which never touches ParametricGeometry.
# Topology and interpolation are unchanged, so the DOF numbering is identical and
# the operators compare entry-by-entry — no gauge, no permutation, no
# eigenvector normalisation to reconcile.
#
# The shape fields are LINEAR (ψ = ∇ψ⋅x with constant ∇ψ), hence exactly
# representable in the trilinear geometric map, so moving the nodes realises
# x(θ,x₀) exactly and the gate is sharp rather than approximate.
#
# Two regimes are deliberately separated:
#
#   SHEAR   ∇ψ = e₂⊗e₁ is NILPOTENT ⟹ det J ≡ 1 and adj J = I − θe₂⊗e₁ has
#           degree 1. Every integrand is then an exact polynomial of degree =
#           its number of gradient factors (K: adj², quadratic: adj³, cubic:
#           adj⁴), so each form must become MACHINE-EXACT precisely at its own
#           degree and stay there. That staircase is the sharp test — it pins
#           the adjugate contraction, the per-form (1/det J)^p powers and the
#           strain measures simultaneously.
#
#   STRETCH ∇ψ = e₁⊗e₁ ⟹ det J = 1+θ, so 1/det J is a genuine geometric series
#           and the error must fall like θ^(MAXT+1) — App. A.3's rate, measured.
#
# A failure in the SHEAR staircase is a pullback bug. A failure in the STRETCH
# rate is a reciprocal-recurrence bug. Do not widen these tolerances.

using Test
using MORFE, MORFEFerrite
using Ferrite, Tensors, StaticArrays
using LinearAlgebra, SparseArrays

const PGm = MORFEFerrite.ParametricGeometry
const SVKm = MORFEFerrite.StructuralSVK
const Tens3m = Tensor{2, 3, Float64, 9}

const _E_M, _ν_M, _ρ_M = 160e3, 0.22, 2.32e-3
const _λ_M = (_E_M * _ν_M) / ((1 + _ν_M) * (1 - 2_ν_M))
const _μ_M = _E_M / (2(1 + _ν_M))
const _STRESS_M = SVKm.IsotropicStress(_λ_M, _μ_M)

const _∇ψ_SHEAR = Tens3m((i, j) -> (i == 2 && j == 1) ? 1.0 : 0.0)   # det J ≡ 1
const _∇ψ_STRETCH = Tens3m((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0) # det J = 1+θ

_base_grid() = generate_grid(Hexahedron, (2, 1, 1), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))

function _mm_space(grid)
    ip = Lagrange{RefHexahedron, 2}()^3
    # generate_grid makes LINEAR hexahedra, so the geometric map is trilinear and
    # a linear ψ is reproduced by node-moving exactly.
    cv = CellValues(QuadratureRule{RefHexahedron}(3), ip, Lagrange{RefHexahedron, 1}())
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    return dh, cv
end

# Same cells, displaced nodes ⇒ same topology ⇒ same DOF numbering.
function _moved(grid, ∇ψs, θ)
    nodes = [Node(n.x + sum(θ[i] * (∇ψs[i] ⋅ n.x) for i in eachindex(θ)))
             for n in grid.nodes]
    return Grid(collect(grid.cells), nodes)
end

_mono(θ, α) = prod(θ[i]^α[i] for i in eachindex(α))
_at(arr, basis, θ) = sum(_mono(θ, α) * arr[m] for (m, α) in enumerate(basis.mset.exponents))
_cols_at(A, basis, θ) =
    sum(_mono(θ, α) * view(A, :, m) for (m, α) in enumerate(basis.mset.exponents))
_relerr(a, b) = norm(a - b) / max(norm(b), eps())

# Parametric operators and both nonlinear forms, plus the FOM reference at θ.
function _mm_case(∇ψs, maxt, θ, u)
    grid = _base_grid()
    dh, cv = _mm_space(grid)
    basis = PGm.GeometryParameterBasis(fill(maxt, length(∇ψs)))
    cache = PGm.PullbackCache(dh, cv, x -> (one(Tens3m), ∇ψs...), basis;
        det_powers = [2, 3])
    f2l = PGm.free_dof_map(ndofs(dh), 1:ndofs(dh))
    pd = PGm.ParametricDiscretisation(dh, cv, f2l, ndofs(dh), cache)

    L = PGm.nterms(basis)
    K = [allocate_matrix(dh) for _ in 1:L]
    M = [allocate_matrix(dh) for _ in 1:L]
    PGm.assemble_linear_series!(K, M, pd, SVKm.SVKPullbackKernel{0}(_STRESS_M, _ρ_M))

    A = Dict{Int, Matrix{ComplexF64}}()
    for DEG in (2, 3)
        pm = PGm.ParametricMap(pd, SVKm.SVKPullbackKernel{DEG}(_STRESS_M, _ρ_M))
        A[DEG] = PGm.sweep_all!(zeros(ComplexF64, ndofs(dh), L), pm, ntuple(_ -> u, DEG))
    end

    # ── the independent reference: ordinary SVK on the moved mesh ──
    dh_m, cv_m = _mm_space(_moved(grid, ∇ψs, θ))
    Kr = allocate_matrix(dh_m)
    Mr = allocate_matrix(dh_m)
    SVKm.assemble_KM!(Kr, Mr, dh_m, cv_m, _STRESS_M, _ρ_M)
    f2l_m = Dict(d => d for d in 1:ndofs(dh_m))
    nl = Dict{Int, Vector{ComplexF64}}()
    for DEG in (2, 3)
        term = SVKm.svk_nonlinearity(DEG, dh_m, cv_m, f2l_m, ndofs(dh_m), _λ_M, _μ_M;
            max_unique_cols = DEG)
        r = zeros(ComplexF64, ndofs(dh_m))
        MORFE.evaluate_term!(r, term, (u, u), nothing)
        nl[DEG] = r
    end

    return (; eK = _relerr(_at(K, basis, θ), Kr), eM = _relerr(_at(M, basis, θ), Mr),
        e2 = _relerr(_cols_at(A[2], basis, θ), -nl[2]),
        e3 = _relerr(_cols_at(A[3], basis, θ), -nl[3]), cache)
end

_probe_u(n) = ComplexF64[cis(0.7k) * (1 + 0.01k) for k in 1:n]

@testset "parametric transform ≡ FOM on the moved mesh" begin
    u = _probe_u(ndofs(_mm_space(_base_grid())[1]))

    @testset "θ = 0 is the same assembly on the same mesh" begin
        r = _mm_case((_∇ψ_STRETCH,), 3, (0.0,), u)
        # Not merely close: it is literally the reference configuration.
        @test r.eK < 1e-14
        @test r.eM < 1e-14
        @test r.e2 < 1e-14
        @test r.e3 < 1e-14
    end

    # det J ≡ 1 and deg(adj J) = 1, so each form is an EXACT polynomial of degree
    # equal to its gradient-factor count and must go machine-exact right there.
    @testset "shear: each form exact at its own degree (det J ≡ 1)" begin
        for θ in ((0.2,), (0.5,))
            r2 = _mm_case((_∇ψ_SHEAR,), 2, θ, u)
            r3 = _mm_case((_∇ψ_SHEAR,), 3, θ, u)
            r4 = _mm_case((_∇ψ_SHEAR,), 4, θ, u)

            @test r2.eK < 1e-13            # K uses adj² → exact from MAXT = 2
            @test r2.eM < 1e-13            # mass uses det J ≡ 1 → always exact
            @test r3.eK < 1e-13
            @test r3.e2 < 1e-13            # quadratic uses adj³ → exact from 3
            @test r4.e3 < 1e-13            # cubic uses adj⁴ → exact from 4

            # …and genuinely truncated one order below, or the staircase above
            # would be vacuous.
            @test r2.e2 > 1e-9
            @test r3.e3 > 1e-9
        end
    end

    # det J = 1+θ ⇒ the reciprocal is a geometric series; error ~ θ^(MAXT+1).
    @testset "stretch: reciprocal series converges at the theoretical rate" begin
        θ = (0.1,)
        lo = _mm_case((_∇ψ_STRETCH,), 2, θ, u)
        hi = _mm_case((_∇ψ_STRETCH,), 4, θ, u)
        @test hi.eK < lo.eK
        @test hi.e2 < lo.e2
        @test hi.e3 < lo.e3
        # Two extra orders at θ = 0.1 must buy about θ² = 1e-2; allow a decade
        # of slack either way, but a stagnating series would fail this outright.
        @test 1e-4 < hi.eK / lo.eK < 1e-1
        # det J = 1+θ is a degree-1 polynomial, inside the box ⇒ mass is exact.
        @test hi.eM < 1e-13
    end

    @testset "two parameters: cross terms converge" begin
        θ = (0.1, 0.1)
        lo = _mm_case((_∇ψ_SHEAR, _∇ψ_STRETCH), 2, θ, u)
        hi = _mm_case((_∇ψ_SHEAR, _∇ψ_STRETCH), 4, θ, u)
        @test hi.eK < lo.eK
        @test hi.e3 < lo.e3
    end

    # The radius the FOM sweep exposed, now asserted: outside it NO truncation
    # order converges, so this is the one diagnostic that must never regress.
    @testset "validity radius matches the analytic one" begin
        stretch = _mm_case((_∇ψ_STRETCH,), 4, (0.1,), u).cache
        r = PGm.geometry_validity_report(stretch)
        @test !r.unconditional
        @test r.radius_pos[1] ≈ 1.0 rtol=1e-2      # det J = 1+θ ⇒ |θ| < 1
        @test r.radius_neg[1] ≈ 1.0 rtol=1e-2
        # The truncated series measured against its own defining identity.
        @test PGm.geometry_validity_at(stretch, (0.1,)).reciprocal_residual < 1e-4
        @test PGm.geometry_validity_at(stretch, (1.0,)).convergent == false

        shear = _mm_case((_∇ψ_SHEAR,), 4, (0.1,), u).cache
        rs = PGm.geometry_validity_report(shear)
        @test rs.unconditional                      # det J ≡ 1: exact, no radius
        @test PGm.geometry_validity_at(shear, (0.9,)).reciprocal_residual == 0.0
    end
end
