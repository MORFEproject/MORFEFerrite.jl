# Golden-value regression tests for the ParametricGeometry kernels.
#
# These pin the two regimes the `J = I ≡ non-parametric StructuralSVK` test in
# test_kernel_equivalence.jl cannot reach:
#
#   · a CURVED reference configuration — J₀ ≠ I and det J ≠ 1, varying with x₀,
#     so the adjugate and reciprocal series are genuinely non-trivial;
#   · PER-FORM θ-bases — the quadratic and cubic forms truncated at different
#     degrees (the arch case, examples/07).
#
# The values were generated from the implementation at the point it was verified
# **bit-identical (0.0 deviation)** to the previous hardcoded-SVK
# ParametricStructural module, on both the unit cases and example 07's full R
# coefficients. That module has since been deleted; these literals are what
# survives of it, so a change here means the kernels moved and needs explaining,
# not re-blessing.
#
# To re-derive after a DELIBERATE change: print `A[probe, αidx]` from the same
# setup below and paste. Do not widen the tolerance instead.

using Test
using MORFE, MORFEFerrite
using Ferrite, Tensors, StaticArrays
using LinearAlgebra, SparseArrays

const PGg = MORFEFerrite.ParametricGeometry
const SVKg = MORFEFerrite.StructuralSVK
const Tens3g = Tensor{2, 3, Float64, 9}

function _golden_setup()
    grid = generate_grid(Hexahedron, (2, 1, 1), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
    ip = Lagrange{RefHexahedron, 2}()^3
    qr = QuadratureRule{RefHexahedron}(3)
    # generate_grid makes LINEAR hexahedra, so the geometric mapping is order 1
    # while the field interpolation stays quadratic — sub-parametric.
    cv = CellValues(qr, ip, Lagrange{RefHexahedron, 1}())
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch); update!(ch, 0.0)
    free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    return (; dh, cv, free_to_local = Dict(d => i for (i, d) in enumerate(free)),
        n_free = length(free))
end

const _E_G, _ν_G, _ρ_G = 160e3, 0.22, 2.32e-3
const _λ_G = (_E_G * _ν_G) / ((1 + _ν_G) * (1 - 2_ν_G))
const _μ_G = _E_G / (2(1 + _ν_G))

# J₀ = I + 0.4 sin(x₁) e₂⊗e₁ + 0.15 x₂ e₃⊗e₃ ;  ∇ψ = 0.6 e₁⊗e₁ + 0.25 cos(x₁) e₂⊗e₂
function _golden_geom(x₀)
    J₀ = one(Tens3g) + Tens3g((i, j) -> (i == 2 && j == 1) ? 0.4 * sin(x₀[1]) :
                                        (i == 3 && j == 3) ? 0.15 * x₀[2] : 0.0)
    ∇ψ = Tens3g((i, j) -> (i == 1 && j == 1) ? 0.6 :
                          (i == 2 && j == 2) ? 0.25 * cos(x₀[1]) : 0.0)
    return (J₀, ∇ψ)
end

# Fixed DOF probes, spread across the free-DOF vector.
const _PROBES = [1, 7, 23, 61]

# θ^α coefficients of the quadratic form at inputs (u, v), θ-basis box [3].
const _GOLDEN_QUAD = [
    [9.72320666850908165e+04 + 1.28095234291978559e+04im,
     3.51062387309375772e+04 + 8.55358015932596463e+04im,
     -8.12123103098121501e+04 + 2.99431703395643737e+05im,
     2.26603709035501597e+04 + 1.17222608371730166e+05im],
    [-1.04333394765268065e+05 - 4.09357898430083806e+04im,
     -4.34352312379238283e+04 - 6.20836595117710167e+04im,
     -6.85677508210715605e+04 + 7.28479088670872152e+04im,
     -4.97244824910835887e+04 - 9.18199059750216547e+04im],
    [8.45928092562853417e+04 + 2.29352201666294495e+04im,
     2.36561059387936912e+04 + 8.19706805811998784e+04im,
     -6.90474194427040311e+03 - 3.78530192417116632e+04im,
     5.83362321360090427e+04 + 1.01652601541874144e+05im],
    [-6.58408303154367604e+04 - 1.83194356916218894e+04im,
     -1.64251354936084972e+04 - 7.22871356590723881e+04im,
     -2.06308249947286276e+03 + 1.55437941181939768e+04im,
     -4.48123095874047140e+04 - 8.11654172016775992e+04im],
]

# θ^α coefficients of the cubic form at inputs (u, v, w), θ-basis box [4].
const _GOLDEN_CUBIC = [
    [4.83157227791216719e+04 - 5.14564421943602283e+04im,
     -4.01390849518981631e+04 - 1.90628272516540237e+04im,
     2.54641601138632395e+04 - 2.73793732149901261e+04im,
     -2.34198895440395972e+04 + 1.06988439276637819e+04im],
    [-1.43011133222691078e+04 + 7.06504993320010981e+04im,
     4.33244983093812043e+04 - 1.07502856103734284e+04im,
     1.56251392905189368e+04 + 2.65744493765320331e+04im,
     8.24178593120714140e+03 - 3.03923741074194113e+04im],
    [1.30079579116485183e+04 - 7.63841545953962486e+04im,
     -3.87898549007623587e+04 + 1.35787405973952009e+04im,
     9.39194982291690849e+02 - 3.12473498097986303e+04im,
     -1.01659674939980723e+04 + 2.51934304364289601e+04im],
    [-8.74832487165031307e+03 + 7.13864490563136060e+04im,
     3.27667629906431539e+04 - 1.38226552845310798e+04im,
     -5.66765384887299115e+03 + 2.88049698428657248e+04im,
     7.59261002753943740e+03 - 2.60110827189193660e+04im],
    [5.97697953989370762e+03 - 6.20810491627410229e+04im,
     -2.69394645875115020e+04 + 1.28546706719347276e+04im,
     7.68820683186366568e+03 - 2.49015025395010816e+04im,
     -5.94350167642920132e+03 + 2.28232029660980043e+04im],
]

# Frobenius norms of the θ^α stiffness/mass coefficient matrices, θ-basis box [2].
const _GOLDEN_LINEAR = [
    (1.93169863146724645e+06, 1.24272513152882437e-03),
    (7.96666932072314317e+05, 8.99519887256407442e-04),
    (3.07458718123767118e+05, 1.15087175340239619e-04),
]

@testset "ParametricGeometry kernel goldens (curved J₀, per-form bases)" begin
    s = _golden_setup()
    stress = SVKg.IsotropicStress(_λ_G, _μ_G)
    u = ComplexF64[cis(0.7k) * (1 + 0.01k) for k in 1:s.n_free]
    v = ComplexF64[cis(-1.3k) * (1 - 0.005k) for k in 1:s.n_free]
    w = ComplexF64[cis(2.1k) * (0.5 + 0.002k) for k in 1:s.n_free]

    @testset "quadratic form, θ-basis [3]" begin
        b = PGg.GeometryParameterBasis([3])
        cache = PGg.PullbackCache(s.dh, s.cv, _golden_geom, b; det_powers = [2])
        pd = PGg.ParametricDiscretisation(s.dh, s.cv, s.free_to_local, s.n_free, cache)
        A = PGg.sweep_all!(zeros(ComplexF64, s.n_free, PGg.nterms(b)),
            PGg.ParametricMap(pd, SVKg.SVKPullbackKernel{2}(stress, _ρ_G)), (u, v))
        @test PGg.nterms(b) == length(_GOLDEN_QUAD)
        for (αidx, expected) in enumerate(_GOLDEN_QUAD)
            @test A[_PROBES, αidx] ≈ expected rtol=1e-12
        end
    end

    @testset "cubic form, θ-basis [4]" begin
        b = PGg.GeometryParameterBasis([4])
        cache = PGg.PullbackCache(s.dh, s.cv, _golden_geom, b; det_powers = [3])
        pd = PGg.ParametricDiscretisation(s.dh, s.cv, s.free_to_local, s.n_free, cache)
        A = PGg.sweep_all!(zeros(ComplexF64, s.n_free, PGg.nterms(b)),
            PGg.ParametricMap(pd, SVKg.SVKPullbackKernel{3}(stress, _ρ_G)), (u, v, w))
        @test PGg.nterms(b) == length(_GOLDEN_CUBIC)
        for (αidx, expected) in enumerate(_GOLDEN_CUBIC)
            @test A[_PROBES, αidx] ≈ expected rtol=1e-12
        end
    end

    @testset "linear K/M series, θ-basis [2]" begin
        b = PGg.GeometryParameterBasis([2])
        cache = PGg.PullbackCache(s.dh, s.cv, _golden_geom, b)
        pd = PGg.ParametricDiscretisation(s.dh, s.cv, s.free_to_local, s.n_free, cache)
        L = PGg.nterms(b)
        K = [allocate_matrix(s.dh) for _ in 1:L]
        M = [allocate_matrix(s.dh) for _ in 1:L]
        PGg.assemble_linear_series!(K, M, pd, SVKg.SVKPullbackKernel{0}(stress, _ρ_G))
        @test L == length(_GOLDEN_LINEAR)
        for (m, (nK, nM)) in enumerate(_GOLDEN_LINEAR)
            @test norm(K[m]) ≈ nK rtol=1e-12
            @test norm(M[m]) ≈ nM rtol=1e-12
        end
    end

    # A nested box must reproduce the smaller box's coefficients exactly: the
    # multiindex sets are graded, so widening the truncation only APPENDS terms.
    @testset "box [2] ⊂ box [3] on shared multiindices" begin
        out = map(([2], [3])) do bnd
            b = PGg.GeometryParameterBasis(bnd)
            cache = PGg.PullbackCache(s.dh, s.cv, _golden_geom, b; det_powers = [2])
            pd = PGg.ParametricDiscretisation(s.dh, s.cv, s.free_to_local, s.n_free, cache)
            PGg.sweep_all!(zeros(ComplexF64, s.n_free, PGg.nterms(b)),
                PGg.ParametricMap(pd, SVKg.SVKPullbackKernel{2}(stress, _ρ_G)), (u, v))
        end
        small, large = out
        @test size(large, 2) > size(small, 2)
        @test large[:, 1:size(small, 2)] ≈ small rtol=1e-12
    end
end
