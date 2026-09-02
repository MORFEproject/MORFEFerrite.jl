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

@testset "close_under_conjugation" begin
    # The eigenvalues below are the first entries of a real Kármán run at Re₀ = 49.03,
    # complex shift σ = 3 + 8i. That shift is why the set needs closing at all: ARPACK
    # returns only the modes near σ, so λ_Hopf arrives and its conjugate near σ̄ does not.
    λ = ComplexF64[0.004029 + 16.859170im,        # Hopf — conjugate NOT returned
        -2.113876 + 0.0im,                        # real — its own conjugate
        -2.911776 + 0.011415im,                   # a pair, both halves returned
        -2.911776 - 0.011415im,
        -11.022533 + 7.464033im]                  # conjugate NOT returned
    Φ = ComplexF64[i + 10im * j for i in 1:3, j in 1:5]
    eig = (; eigenvalues = λ, right_modes = Φ, hopf_index = 1, conjugate_index = 5)

    s = FNS.close_under_conjugation(eig)

    @testset "only the missing halves are appended" begin
        # Two added, for modes 1 and 5. The real mode gets NO partner — synthesising one
        # would duplicate the coordinate — and the pair already present is left alone.
        @test length(s.eigenvalues) == 7
        @test size(s.right_modes) == (3, 7)
        @test s.eigenvalues[1:5] == λ                       # existing entries untouched
        @test s.right_modes[:, 1:5] == Φ
        @test count(≈(ComplexF64(-2.113876)), s.eigenvalues) == 1
    end

    @testset "the appended halves are exact conjugates" begin
        # Exact, not approximate: B₀ and B₁ are real, so (λ̄, φ̄) is an eigenpair whenever
        # (λ, φ) is. Conjugating is also the only way to get the phase that
        # `conjugate_permutation` asserts — an independent solve pins it only up to a scalar.
        @test s.eigenvalues[6] == conj(λ[1])
        @test s.right_modes[:, 6] == conj.(Φ[:, 1])
        @test s.eigenvalues[7] == conj(λ[5])
        @test s.right_modes[:, 7] == conj.(Φ[:, 5])
    end

    @testset "conjugate_index is corrected, hopf_index carried through" begin
        # THE bug this function exists for. The eigensolve's own `conjugate_index` is an
        # argmin over what ARPACK returned, so it names the nearest available mode — here
        # entry 5, λ = -11.02 + 7.46i, nothing to do with the Hopf pair. Passing that as
        # `master` makes `build_model` throw.
        @test eig.conjugate_index == 5
        @test s.hopf_index == 1
        @test s.conjugate_index == 6
        @test s.eigenvalues[s.conjugate_index] == conj(s.eigenvalues[s.hopf_index])
        # The pairing the reduction consumes now succeeds on this master set.
        @test FNS._master_conjugate_pairing(s.eigenvalues[[s.hopf_index,
            s.conjugate_index]]) == [2, 1]
    end

    @testset "an already-closed spectrum is unchanged" begin
        closed = ComplexF64[0.004 + 16.8im, 0.004 - 16.8im, -2.11 + 0.0im]
        M = ComplexF64[1 2 3; 4 5 6]
        t = FNS.close_under_conjugation((; eigenvalues = closed, right_modes = M,
            hopf_index = 1, conjugate_index = 2))
        @test t.eigenvalues == closed
        @test t.right_modes == M
        @test t.conjugate_index == 2
    end

    @testset "the real test is RELATIVE, not absolute" begin
        # Same reason as `_master_conjugate_pairing` above: at an absolute threshold this
        # near-real mode reads as complex and acquires a spurious conjugate — a duplicate
        # coordinate, and one that would then fail the pairing involution.
        near_real = ComplexF64[-11.472351 - 1.4e-5im]       # |Im|/|λ| ≈ 1.2e-6
        N = ComplexF64[1.0; 2.0;;]
        e = (; eigenvalues = near_real, right_modes = N, hopf_index = 1,
            conjugate_index = 1)
        @test length(FNS.close_under_conjugation(e; rtol = 1e-4).eigenvalues) == 1
        @test length(FNS.close_under_conjugation(e; rtol = 1e-12).eigenvalues) == 2
    end
end
