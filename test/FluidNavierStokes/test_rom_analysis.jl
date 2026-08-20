# Run with:  julia --project=. test_invariants.jl
include(joinpath(@__DIR__, "invariants.jl"))
using Printf, Test

# Analytic check of the whole premise. Two ROMs describing IDENTICAL physics:
#
#   un-promoted :  ż = λz + c·z²z̄
#   promoted    :  ż = λz + c'·z²z̄ + g·z·y ,   ẏ = −μ·y + h·z·z̄
#
# Slaving the promoted coordinate gives y = (h/μ)zz̄, hence an effective cubic
# coefficient c' + gh/μ. Choosing c' = c − gh/μ makes the two ROMs the same system in
# different coordinates — so rom_invariants MUST return the same c₂₁₀ for both. This is
# exactly the mean-flow-distortion loop: y is driven by z z̄ and feeds back through z·y.
λ = 0.004028440003468248 + 16.859169021043915im
c = -0.11101107793 + 0.0706507599091im
g, h, μ = 0.7 - 0.3im, 1.9 + 0.4im, 2.113875        # μ from the real mode at λ ≈ −2.11
c′ = c - g * h / μ

# un-promoted: z₁, z̄₁, η
up = ROMPoly([[1,0,0], [2,1,0]], reshape(ComplexF64[λ, c], 2, 1), 3)

# promoted: z₁, z̄₁, y, η — component 1 is ż, component 3 is ẏ
exps = [[1,0,0,0], [2,1,0,0], [1,0,1,0], [0,0,1,0], [1,1,0,0]]
coef = zeros(ComplexF64, 5, 3)
coef[1,1] = λ; coef[2,1] = c′; coef[3,1] = g      # ż
coef[4,3] = -μ; coef[5,3] = h                     # ẏ
pr = ROMPoly(exps, coef, 4)

iu, ip = rom_invariants(up), rom_invariants(pr)
@printf("%-14s %-30s %-30s\n", "", "un-promoted (NVAR=3)", "promoted (NVAR=4)")
@printf("%-14s %+.12f %+.12fi %+.12f %+.12fi\n", "lambda", iu.σ, iu.ω, ip.σ, ip.ω)
@printf("%-14s %+.12f %+.12fi %+.12f %+.12fi\n", "c210_eff", iu.c210_re, iu.c210_im, ip.c210_re, ip.c210_im)
@printf("%-14s %+.12f                  %+.12f\n", "c210_ratio", iu.c210_ratio, ip.c210_ratio)
@printf("%-14s %d                                  %d\n", "n_promoted", iu.n_promoted, ip.n_promoted)

# What the naive reduction would report, for contrast: zeroing y instead of slaving it.
@printf("\nif y were ZEROED instead of slaved, c210 would read %+.6f %+.6fi (i.e. c'),\n",
        real(c′), imag(c′))
@printf("  which is off by g·h/mu = %+.6f %+.6fi — the mean-flow contribution.\n",
        real(g*h/μ), imag(g*h/μ))

@testset "promotion invariance" begin
    @test iu.σ ≈ ip.σ            rtol=1e-9
    @test iu.ω ≈ ip.ω            rtol=1e-9
    @test iu.c210_re ≈ ip.c210_re rtol=1e-7
    @test iu.c210_im ≈ ip.c210_im rtol=1e-7
    @test iu.c210_ratio ≈ ip.c210_ratio rtol=1e-7
end
