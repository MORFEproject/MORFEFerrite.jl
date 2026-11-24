# The default quadrature order of `mechanical_model` integrates the element mass matrix
# exactly on every reference shape, and an order that leaves it singular is reported.
#
# Ferrite reads `QuadratureRule{RefShape}(n)` as Gauss points per direction on hexahedra
# but as a total polynomial degree on tetrahedra and prisms. The former default,
# `fe_order + 1`, was therefore exact on hexahedra only (and on linear tetrahedra). On the
# 18-node prism it gave 8 points for 18 shape functions, an element mass matrix of rank 7
# and zero-energy modes in K. On a turbine-blade mesh K and M then shared a null vector,
# so s²M + sC + K was singular for every s and the reduced model was meaningless; the
# 2 × 2 × 2 prism grid below reproduces that.

using MORFEFerrite
using MORFEFerrite.StructuralSVK: default_quad_order
using Ferrite
using LinearAlgebra
using Test

const _QO_SVK = MORFEFerrite.StructuralSVK
const _QO_WARN = Base.CoreLogging.Warn

# Element mass matrix of `Lagrange{R, p}` on the reference element.
function _qo_reference_mass(R, p, order)
    interpolation = Lagrange{R, p}()
    rule = QuadratureRule{R}(order)
    n = getnbasefunctions(interpolation)
    mass = zeros(n, n)
    for (ξ, w) in zip(Ferrite.getpoints(rule), Ferrite.getweights(rule))
        N = [Ferrite.reference_shape_value(interpolation, ξ, i) for i in 1:n]
        mass .+= w .* (N * N')
    end
    return mass
end

# An order exact for the mass integrand of every `p ≤ 2`: degree 4 on tetrahedra (5 is
# Ferrite's highest there), 8 on prisms, 2p per direction on hexahedra.
const _QO_EXACT = Dict(RefHexahedron => 10, RefTetrahedron => 5, RefPrism => 10)
const _QO_SHAPE = Dict(Hexahedron => RefHexahedron, Tetrahedron => RefTetrahedron,
    Wedge => RefPrism)

function _qo_case(Cell; kwargs...)
    return _QO_SVK.mechanical_model(generate_grid(Cell, (2, 2, 2));
        material = _QO_SVK.SVKMaterial(E = 1.0, ν = 0.3, ρ = 1.0),
        damping = _QO_SVK.RayleighDamping(α = 0.0, β = 0.0),
        dirichlet = "left", fe_order = 2, kwargs...)
end

# Smallest over largest eigenvalue: ≤ 0 for a singular or indefinite matrix.
_qo_definiteness(A) = (λ = eigvals(Symmetric(Matrix(A))); λ[1] / λ[end])

@testset "quadrature order" begin
    @testset "default_quad_order" begin
        for p in 1:2
            @test default_quad_order(RefHexahedron, p) == p + 1   # unchanged
            @test default_quad_order(RefTetrahedron, p) == 2p
            @test default_quad_order(RefPrism, p) == 4p
            @test default_quad_order(RefPyramid, p) == 2p + 2
        end
        @test default_quad_order(generate_grid(Wedge, (1, 1, 1)), 2) == 8
    end

    @testset "the default is the lowest exact order: $R, fe_order = $p" for R in (
            RefHexahedron, RefTetrahedron, RefPrism),
        p in 1:2

        q = default_quad_order(R, p)
        exact = _qo_reference_mass(R, p, _QO_EXACT[R])
        @test _qo_reference_mass(R, p, q)≈exact rtol=1e-12
        @test !isapprox(_qo_reference_mass(R, p, q - 1), exact; rtol = 1e-6)
    end

    # Pyramid shape functions are rational, so no rule is exact; the default must at
    # least give a full-rank element mass matrix.
    @testset "pyramid element mass is full rank, fe_order = $p" for p in 1:2
        M = _qo_reference_mass(RefPyramid, p, default_quad_order(RefPyramid, p))
        @test isposdef(Symmetric(M))
    end

    @testset "mechanical_model default on $Cell" for Cell in (
        Hexahedron, Tetrahedron, Wedge)
        case = @test_logs min_level=_QO_WARN _qo_case(Cell)
        @test case.info.quad_order == default_quad_order(_QO_SHAPE[Cell], 2)
        @test _qo_definiteness(case.M) > 1e-4
    end

    @testset "the former default fe_order + 1 = 3" begin
        # Quadratic tetrahedra: the Keast rule of order 3 has a negative weight and 5
        # points for 10 shape functions, so the assembled M is indefinite.
        tet = @test_logs (:warn, r"quad_order = 3 is too low.*indefinite") _qo_case(
            Tetrahedron; quad_order = 3)
        @test _qo_definiteness(tet.M) < -1e-3

        # Quadratic prisms: M is singular and shares a null vector with K, so the pencil
        # s²M + sC + K is singular for every s.
        prism = @test_logs (:warn, r"quad_order = 3 is too low.*singular") _qo_case(
            Wedge; quad_order = 3)
        @test _qo_definiteness(prism.M) < 1e-12
        n = size(prism.M, 1)
        @test rank(Matrix([prism.K; prism.M]); rtol = 1e-10) < n
    end

    # The warning is about rank, not exactness: order 6 does not integrate the 18-node
    # prism exactly but gives it a full-rank mass matrix.
    @testset "an inexact full-rank order is accepted silently" begin
        case = @test_logs min_level=_QO_WARN _qo_case(Wedge; quad_order = 6)
        @test _qo_definiteness(case.M) > 1e-4
    end
end
