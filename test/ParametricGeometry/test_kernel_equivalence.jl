# The parametric kernels against a path that never touches ParametricGeometry.
#
# At the identity coordinate transform (J ≡ I) the θ⁰ coefficient of every
# parametric form must equal StructuralSVK's own non-parametric assembly —
# `assemble_KM!` for the linear operators, `FerriteGeometricNonlinearity` for the
# quadratic and cubic forms. That makes this an INDEPENDENT check: an error would
# have to exist identically in two separately written implementations to survive.
#
# The curved-J₀ and per-form-basis regimes, which this cannot reach, are pinned by
# test_kernel_golden.jl. Both together replace the old new-vs-ParametricStructural
# equivalence suite, which was deleted with that module.

using Test
using MORFE, MORFEFerrite
using Ferrite, Tensors, StaticArrays
using LinearAlgebra, SparseArrays

const PG = MORFEFerrite.ParametricGeometry
const SVK = MORFEFerrite.StructuralSVK
const Tens3 = Tensor{2, 3, Float64, 9}

# ── A small clamped block: 2×1×1 hexahedra, quadratic Lagrange ───────────────
function _param_test_setup(; maxt = 2)
    grid = generate_grid(Hexahedron, (2, 1, 1), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
    ip = Lagrange{RefHexahedron, 2}()^3
    # generate_grid makes LINEAR hexahedra (8 nodes), so the geometric mapping is
    # order 1 while the field interpolation stays quadratic — sub-parametric.
    geo_ip = Lagrange{RefHexahedron, 1}()
    qr = QuadratureRule{RefHexahedron}(3)
    cv = CellValues(qr, ip, geo_ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch)
    update!(ch, 0.0)
    free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    free_to_local = Dict(d => i for (i, d) in enumerate(free))
    return (; grid, dh, cv, free, free_to_local, n_free = length(free))
end

const E_TEST, ν_TEST, ρ_TEST = 160e3, 0.22, 2.32e-3
const λ_TEST = (E_TEST * ν_TEST) / ((1 + ν_TEST) * (1 - 2ν_TEST))
const μ_TEST = E_TEST / (2(1 + ν_TEST))

# One geometry parameter whose shape field is identically zero, so J(θ) ≡ I for
# every θ. The θ ≠ 0 coefficients must then all vanish, which is itself a check.
_geom_identity(x₀) = (one(Tens3), zero(Tens3))

@testset "SVKPullbackKernel at J = I ≡ non-parametric StructuralSVK" begin
    s = _param_test_setup()
    b = PG.GeometryParameterBasis([1])                     # one parameter, ∇ψ = 0 ⇒ J(θ) ≡ I
    stress = SVK.IsotropicStress(λ_TEST, μ_TEST)
    cache = PG.PullbackCache(s.dh, s.cv, _geom_identity, b; det_powers = [2, 3])
    pd = PG.ParametricDiscretisation(s.dh, s.cv, s.free_to_local, s.n_free, cache)
    α0 = findfirst(α -> all(iszero, α), b.mset.exponents)

    @testset "linear operators ≡ svk_assemble_KM!" begin
        K_ref = allocate_matrix(s.dh)
        M_ref = allocate_matrix(s.dh)
        SVK.assemble_KM!(K_ref, M_ref, s.dh, s.cv, stress, ρ_TEST)

        L = PG.nterms(b)
        K_par = [allocate_matrix(s.dh) for _ in 1:L]
        M_par = [allocate_matrix(s.dh) for _ in 1:L]
        PG.assemble_linear_series!(K_par, M_par, pd, SVK.SVKPullbackKernel{0}(stress, ρ_TEST))

        @test Matrix(K_par[α0]) ≈ Matrix(K_ref) rtol=1e-12
        @test Matrix(M_par[α0]) ≈ Matrix(M_ref) rtol=1e-12
        # Every θ ≠ 0 coefficient must vanish: the transform does not depend on θ.
        for m in 1:L
            m == α0 && continue
            @test nnz(K_par[m]) == 0 || maximum(abs, K_par[m]) < 1e-9
            @test nnz(M_par[m]) == 0 || maximum(abs, M_par[m]) < 1e-9
        end
    end

    @testset "quadratic/cubic forms ≡ FerriteGeometricNonlinearity" begin
        u = ComplexF64[cis(0.7k) * (1 + 0.01k) for k in 1:s.n_free]

        # `evaluate_term!` indexes `xs` by DERIVATIVE-ORDER slot and repeats it
        # `multiindex[slot]` times, so it evaluates F(u,…,u) — hence one state here.
        for DEG in (2, 3)
            term = SVK.svk_nonlinearity(DEG, s.dh, s.cv, s.free_to_local, s.n_free,
                λ_TEST, μ_TEST; max_unique_cols = DEG)
            ref = zeros(ComplexF64, s.n_free)
            MORFE.evaluate_term!(ref, term, (u, u), nothing)

            pm = PG.ParametricMap(pd, SVK.SVKPullbackKernel{DEG}(stress, ρ_TEST))
            A = PG.sweep_all!(zeros(ComplexF64, s.n_free, PG.nterms(b)), pm,
                ntuple(_ -> u, DEG))
            # Both the SVK `accumulate_qp!` and the ParametricGeometry per-α closures
            # negate (internal-force convention); the raw sweep does not, so the raw
            # sweep is +∫ against `evaluate_term!`'s −∫.
            @test A[:, α0] ≈ -ref rtol=1e-10
        end
    end
end
