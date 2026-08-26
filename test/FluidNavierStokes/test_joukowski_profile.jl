using LinearAlgebra
using SparseArrays
using Test

include(joinpath(@__DIR__, "..", "..", "examples", "11_parametric_karman_profile", "geometry.jl"))
using .ParametricKarmanGeometry
module KarmanModeTrackingTestHelpers
include(joinpath(@__DIR__, "..", "..", "examples",
    "11_parametric_karman_profile", "mode_tracking.jl"))
end
const KMT = KarmanModeTrackingTestHelpers

@testset "deterministic Joukowski profile geometry" begin
    spec = default_profile_spec()
    metrics = profile_metrics(spec, 0.5)

    @test metrics.length ≈ 0.1 atol = 1e-10 rtol = 0
    @test metrics.midpoint ≈ 0.2 + 0.2im atol = 1e-10 rtol = 0
    @test real(spec.circle_centre) < 0
    @test metrics.thickness_ratio ≈ 0.3709638117013695 atol = 1e-4 rtol = 0
    @test metrics.camber_ratio ≈ 0.06 atol = 1e-4 rtol = 0
    @test metrics.angle ≈ deg2rad(-0.193088744653491) atol = deg2rad(2e-3)
    @test real(metrics.leading) < real(metrics.trailing)

    # ζ=+c lies on the generating circle. At t=1 it maps to the downstream
    # cusp, never to the upstream leading edge.
    cusp_φ = angle(spec.c - spec.circle_centre)
    formal = profile_chord(spec, 1.0)
    @test profile_point(spec, 1.0, cusp_φ) ≈ formal.trailing atol = 1e-10
    @test abs(profile_tangent(spec, 1.0, cusp_φ)) < 1e-12

    @test signed_area(spec, 0.5) > 0
    @test boundary_is_simple(spec, 0.5; samples = 1024)
    @test boundary_is_simple(spec, 0.0; samples = 1024)
    @test boundary_is_simple(spec, 1.0; samples = 1024)
    @test profile_point(spec, 0.5, 0.0) ≈ profile_point(spec, 0.5, 2π) atol = 1e-14

    # The family is exactly affine in t. This identity is the boundary datum
    # used by x_μ = X + μψ on the stored midpoint mesh.
    for φ in range(0, 2π; length = 19)
        ψ = boundary_displacement(spec, φ)
        @test profile_point(spec, 0.6, φ) - profile_point(spec, 0.4, φ) ≈ 0.2ψ atol = 2e-14
        @test profile_tangent(spec, 0.5, φ) ≈
              (profile_point(spec, 0.5, φ + 1e-6) -
               profile_point(spec, 0.5, φ - 1e-6)) / 2e-6 rtol = 5e-9
    end
end

@testset "mass-overlap mode tracking" begin
    B = spdiagm(0 => [2.0, 1.0, 3.0, 1.0])
    phi = ComplexF64[1+im, 2-im, -0.5im, 0.25]
    phi ./= sqrt(real(dot(phi, B*phi)))
    other = ComplexF64[0, 0, 1, im]
    values = ComplexF64[-0.2-4im, -0.1+7im, -0.2+4im]
    modes = hcat(conj(phi).*cis(0.3), other, phi.*cis(-1.1))
    tracked = KMT.select_tracked_mode(phi, B, values, modes, B)
    @test tracked.selected == 3
    @test tracked.overlap ≈ 1.0 atol=1e-14
    pairing = dot(phi, B*tracked.mode)
    @test imag(pairing) ≈ 0.0 atol=1e-14
    @test real(pairing) > 0
    @test KMT.spectral_separation(values, tracked.selected) > 0

    # A strongly damped exact-frequency match must not beat the physical mode
    # that lies much closer to i*omega in the complex plane.
    seeded = ComplexF64[-20+24.48im, 0.8+19.9im, -1-24.48im]
    @test KMT.select_initial_mode(seeded, 24.48) == 2
end
