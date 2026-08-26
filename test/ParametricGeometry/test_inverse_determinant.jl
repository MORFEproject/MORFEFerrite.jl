# Method 3 (auxiliary field s ≈ 1/det J) against Method 4 (power series).
#
# The two strategies are asked for the same object and differ only in how it is
# obtained, so they can be checked against each other AND against the identity
# they both approximate, `det J · (1/det J) = 1`.
#
# What pins Method 3 specifically:
#   · s₀ ≡ 1 to round-off whenever det J₀ ≡ 1 — `1 ∈ V_h` for any Lagrange space,
#     so the γ = 0 equation is the projection of the constant 1 and must return
#     it exactly. This checks the weighted mass matrix, its quadrature and the
#     right-hand side in one shot.
#   · J ≡ I ⟹ the two methods must agree EXACTLY, not approximately.
#   · raising the order of V_h must drive Method 3 towards Method 4, which is the
#     pointwise-exact-in-truncation limit. That convergence IS the theory's
#     "accuracy depends on the spatial discretisation of s".

using Test
using MORFE, MORFEFerrite
using Ferrite, Tensors, StaticArrays
using LinearAlgebra

const PGi = MORFEFerrite.ParametricGeometry
const Tens3i = Tensor{2, 3, Float64, 9}

function _inv_det_setup(; order = 2)
    grid = generate_grid(Hexahedron, (2, 2, 1), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
    ip = Lagrange{RefHexahedron, order}()^3
    qr = QuadratureRule{RefHexahedron}(3)
    cv = CellValues(qr, ip, Lagrange{RefHexahedron, 1}())
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    return (; dh, cv, qr)
end

_scalar_ip(order) = Lagrange{RefHexahedron, order}()

# Largest |1/det J − s| over all quadrature points, comparing two caches' series
# evaluated at the same θ.
function _max_series_gap(a::PGi.PullbackCache, b::PGi.PullbackCache, θ)
    g = 0.0
    for ci in eachindex(a.inv_det), q in eachindex(a.inv_det[ci])
        va = PGi.series_value(a.inv_det[ci][q], a.basis, θ)
        vb = PGi.series_value(b.inv_det[ci][q], b.basis, θ)
        g = max(g, abs(va - vb))
    end
    return g
end

@testset "inverse determinant: Method 3 vs Method 4" begin
    s = _inv_det_setup()
    basis = PGi.GeometryParameterBasis([4])

    # ψ = x₁e₁ : det J = 1 + θ, a genuinely non-constant determinant, so the two
    # methods have something to disagree about.
    stretch = x -> (one(Tens3i), Tens3i((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0))
    # ψ ≡ 0 : J ≡ I, det J ≡ 1 — the exactness case.
    identity_geom = x -> (one(Tens3i), zero(Tens3i))

    @testset "J ≡ I: the two methods agree exactly" begin
        m4 = PGi.PullbackCache(s.dh, s.cv, identity_geom, basis)
        m3 = PGi.PullbackCache(s.dh, s.cv, identity_geom, basis;
            inverse_determinant = PGi.AuxiliaryFieldInverseDet(_scalar_ip(1); qr = s.qr))
        for θ in ((0.0,), (0.3,), (0.9,))
            @test _max_series_gap(m3, m4, θ) < 1e-12
        end
        # 1/det J ≡ 1: the constant term is 1 and every other coefficient zero.
        for ci in eachindex(m3.inv_det), q in eachindex(m3.inv_det[ci])
            ser = m3.inv_det[ci][q]
            @test abs(ser[1] - 1) < 1e-10
            @test maximum(abs, ser[2:end]) < 1e-10
        end
    end

    @testset "s₀ ≡ 1 when det J₀ ≡ 1" begin
        # The γ = 0 equation is the L² projection of the constant 1. Any Lagrange
        # space contains it, so this is exact to round-off at every order.
        for order in (1, 2)
            m3 = PGi.PullbackCache(s.dh, s.cv, stretch, basis;
                inverse_determinant = PGi.AuxiliaryFieldInverseDet(_scalar_ip(order);
                    qr = s.qr))
            for ci in eachindex(m3.inv_det), q in eachindex(m3.inv_det[ci])
                @test abs(m3.inv_det[ci][q][1] - 1) < 1e-10
            end
        end
    end

    @testset "both satisfy their defining identity det J · (1/det J) = 1" begin
        m4 = PGi.PullbackCache(s.dh, s.cv, stretch, basis)
        m3 = PGi.PullbackCache(s.dh, s.cv, stretch, basis;
            inverse_determinant = PGi.AuxiliaryFieldInverseDet(_scalar_ip(2); qr = s.qr))
        # Well inside the radius |θ| < 1, both must invert det J to the accuracy
        # their own truncation allows.
        @test PGi.geometry_validity_at(m4, (0.1,)).reciprocal_residual < 1e-4
        @test PGi.geometry_validity_at(m3, (0.1,)).reciprocal_residual < 1e-3
    end

    @testset "uniform stretch: 1/det J is CONSTANT in x₀, so Method 3 is exact" begin
        # det J = 1 + θ does not depend on x₀, hence neither does 1/det J, and
        # every Lagrange space contains the constants exactly. Method 3 then has
        # no discretisation error to make — it reproduces Method 4 at order 1.
        m4 = PGi.PullbackCache(s.dh, s.cv, stretch, basis)
        m3 = PGi.PullbackCache(s.dh, s.cv, stretch, basis;
            inverse_determinant = PGi.AuxiliaryFieldInverseDet(_scalar_ip(1); qr = s.qr))
        @test _max_series_gap(m3, m4, (0.2,)) < 1e-13
    end

    @testset "Method 3 → Method 4 as V_h is enriched" begin
        # A SPATIALLY VARYING determinant is needed to see the discretisation
        # error at all: ∇ψ = sin(x₁) e₁⊗e₁ (i.e. ψ = −cos(x₁)e₁, a legitimate
        # gradient field) gives det J = 1 + θ sin x₁, so 1/det J is a genuinely
        # non-polynomial function of x₀ that V_h has to resolve. THIS is the
        # regime the theory means by "accuracy depends on the discretisation
        # of s"; the uniform case above cannot exhibit it.
        wavy = x -> (one(Tens3i),
            Tens3i((i, j) -> (i == 1 && j == 1) ? sin(x[1]) : 0.0))
        m4 = PGi.PullbackCache(s.dh, s.cv, wavy, basis)
        gaps = [_max_series_gap(
                    PGi.PullbackCache(s.dh, s.cv, wavy, basis;
                        inverse_determinant = PGi.AuxiliaryFieldInverseDet(
                            _scalar_ip(order); qr = s.qr)),
                    m4, (0.2,))
                for order in (1, 2)]
        # Method 4 is the pointwise-exact-in-truncation limit, so enriching V_h
        # must close the gap towards it.
        @test gaps[1] > 1e-8            # order 1 genuinely disagrees …
        @test gaps[2] < gaps[1] / 5     # … and order 2 is materially closer
    end

    @testset "mass lumping stays in the right ballpark" begin
        m4 = PGi.PullbackCache(s.dh, s.cv, stretch, basis)
        lumped = PGi.PullbackCache(s.dh, s.cv, stretch, basis;
            inverse_determinant = PGi.AuxiliaryFieldInverseDet(_scalar_ip(1);
                qr = s.qr, lump = true))
        # Coarser than the consistent projection, but must still be a 1/det J.
        @test _max_series_gap(lumped, m4, (0.2,)) < 1e-1
    end

    # The reportable form of the comparison above — this is what produces the
    # thesis numbers (scripts/inverse_determinant_study.jl), so it is gated too.
    @testset "inverse_determinant_comparison measures the trade-off" begin
        m4 = PGi.PullbackCache(s.dh, s.cv, identity_geom, basis)
        m3 = PGi.PullbackCache(s.dh, s.cv, identity_geom, basis;
            inverse_determinant = PGi.AuxiliaryFieldInverseDet(_scalar_ip(1); qr = s.qr))
        c = PGi.inverse_determinant_comparison(m4, m3, (0.3,))
        # J ≡ I ⟹ both methods return the single term 1, exactly.
        @test c.max_abs < 1e-14
        @test c.rms < 1e-14
        @test c.residual_a < 1e-12          # each against det J · s = 1 …
        @test c.residual_b < 1e-12          # … its own defining identity

        # A spatially varying det J: the gap is Method 3's interpolation error,
        # and enriching V_h must close it. Same premise as the gap test above —
        # a uniform stretch would show nothing, because its det J is constant.
        wavy = x -> (one(Tens3i),
            Tens3i((i, j) -> (i == 1 && j == 1) ? sin(x[1]) : 0.0))
        w4 = PGi.PullbackCache(s.dh, s.cv, wavy, basis)
        cs = [PGi.inverse_determinant_comparison(w4,
                  PGi.PullbackCache(s.dh, s.cv, wavy, basis;
                      inverse_determinant = PGi.AuxiliaryFieldInverseDet(
                          _scalar_ip(order); qr = s.qr)), (0.2,))
              for order in (1, 2)]
        @test cs[1].max_abs > 1e-8
        @test cs[2].max_abs < cs[1].max_abs / 5
        @test cs[2].rms < cs[1].rms
        @test 1 <= cs[1].worst_cell <= length(w4.det)

        # Two caches over different meshes cannot be lined up point-by-point.
        coarse = generate_grid(Hexahedron, (1, 1, 1), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
        dh_c = DofHandler(coarse)
        add!(dh_c, :u, Lagrange{RefHexahedron, 1}()^3)
        close!(dh_c)
        cv_c = CellValues(s.qr, Lagrange{RefHexahedron, 1}()^3, Lagrange{RefHexahedron, 1}())
        small = PGi.PullbackCache(dh_c, cv_c, identity_geom, basis)
        @test_throws ArgumentError PGi.inverse_determinant_comparison(m4, small, (0.1,))
    end

    @testset "quadrature mismatch is rejected, not silently tolerated" begin
        # det J is read at the points the cache already holds, so an auxiliary
        # rule with a different point count cannot be lined up with it.
        @test_throws ArgumentError PGi.PullbackCache(s.dh, s.cv, stretch, basis;
            inverse_determinant = PGi.AuxiliaryFieldInverseDet(_scalar_ip(1);
                qr = QuadratureRule{RefHexahedron}(2)))
    end
end
