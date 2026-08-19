# The conjugate-symmetry invariant the fluid reduction rests on.
#
# `conjugate_permutation` asserts `modes[:, σ(r)] = conj(modes[:, r])`. Eigenvalue
# conjugacy does NOT imply it, and when it is violated the solve exploits a symmetry
# that is not there — silently. The failure signature is nasty: λ and every term linear
# in the master coordinate stay exact to machine precision, while the NONLINEAR
# coefficients keep their modulus and rotate in phase. That is easy to mistake for a
# gauge convention and hard to spot without a reference.
#
# These tests exercise the two pure helpers that establish the invariant. They need no
# mesh and no eigensolve, so they are cheap enough to keep in the default suite — which
# matters, because the run that would catch this end-to-end is order 9 on a 58k-DOF mesh.

using Test
using MORFEFerrite
using LinearAlgebra

const FNS = MORFEFerrite.FluidNavierStokes

@testset "_master_conjugate_pairing" begin
    @testset "an adjacent conjugate pair" begin
        λ = ComplexF64[0.004 + 16.8im, 0.004 - 16.8im]
        @test FNS._master_conjugate_pairing(λ) == [2, 1]
    end

    @testset "a real eigenvalue is self-paired" begin
        # THE case that makes a mode on the real axis usable at all: it is its own
        # conjugate and carries one real coordinate, not half of a complex pair.
        λ = ComplexF64[-2.113783 + 0.0im]
        @test FNS._master_conjugate_pairing(λ) == [1]
    end

    @testset "mixed: a pair plus two reals" begin
        λ = ComplexF64[0.004 + 16.8im, 0.004 - 16.8im, -2.11 + 0.0im, -5.12 + 0.0im]
        σ = FNS._master_conjugate_pairing(λ)
        @test σ == [2, 1, 3, 4]
        @test all(r -> σ[σ[r]] == r, eachindex(σ))       # an involution
    end

    @testset "the real test is RELATIVE, not absolute" begin
        # ARPACK's numerical zero is ~1e-7 of the magnitude, not machine epsilon. An
        # absolute 1e-8 threshold calls this mode complex and then demands a conjugate
        # that does not exist — which is how a real mode acquired a duplicate coordinate.
        λ = ComplexF64[-11.472351 - 1.4e-5im]             # |Im|/|λ| ≈ 1.2e-6
        @test FNS._master_conjugate_pairing(λ; atol = 1e-4) == [1]
        @test_throws ArgumentError FNS._master_conjugate_pairing(λ; atol = 1e-12)
    end

    @testset "half a conjugate pair is rejected, naming the mode" begin
        # A manifold spanned by one half of a pair is not conjugation-invariant and the
        # ROM has no real realisation, so this is an error rather than a silent fallback.
        λ = ComplexF64[0.004 + 16.8im, -2.11 + 0.0im]
        err = try
            FNS._master_conjugate_pairing(λ)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("conjugate is not in the master", err.msg)
    end
end

@testset "_realify_self_conjugate" begin
    v = ComplexF64[1.0, -2.5, 0.25, 4.0]                  # a real mode shape

    @testset "an arbitrary phase is removed" begin
        for θ in (0.0, 0.3, 1.9, π - 1e-3, -2.2)
            φ, ψ = FNS._realify_self_conjugate(v .* cis(θ), v .* cis(-θ), 1, -2.11 + 0.0im)
            # Real to round-off, hence equal to its own conjugate — which is exactly what
            # σ[k] = k asserts.
            @test norm(imag.(φ)) <= 1e-12 * norm(real.(φ))
            @test φ ≈ conj.(φ)
            @test ψ ≈ conj.(ψ)
            # The mode is unchanged up to sign: a phase rotation, not a rescaling.
            @test abs(dot(normalize(real.(φ)), normalize(v))) ≈ 1 atol=1e-12
        end
    end

    @testset "a genuinely complex vector is rejected" begin
        # No phase makes this real, so the eigenvalue is not truly real or the mode is
        # not simple. Reporting beats absorbing: a defective master coordinate corrupts
        # the whole reduction without any other symptom.
        w = ComplexF64[1.0, 2.0im, 0.5, -1.0]
        @test_throws ArgumentError FNS._realify_self_conjugate(w, w, 3, -6.67 + 0.0im)
    end
end
