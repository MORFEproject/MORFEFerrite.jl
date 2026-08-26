# Resummation: summing the DPIM series past its radius of convergence.
#
# Mesh-free and fast — every case is analytic, so this belongs in the package suite rather
# than behind an example run.

using Test
using MORFEFerrite.FluidNavierStokes
using MORFEFerrite.FluidNavierStokes: _real_roots

const FNS = MORFEFerrite.FluidNavierStokes

@testset "resummation" begin
	@testset "pade reproduces a rational function exactly" begin
		# f(u) = (1 + 2u) / (1 − 3u + u²). Its Taylor coefficients determine it uniquely, so
		# [1/2] Padé must return f itself, not an approximation.
		P, Q = [1.0, 2.0], [1.0, -3.0, 1.0]
		c = Float64[]
		for k in 0:8                       # long division
			s = k + 1 <= length(P) ? P[k + 1] : 0.0
			for j in 1:min(k, length(Q) - 1)
				s -= Q[j + 1] * c[k - j + 1]
			end
			push!(c, s / Q[1])
		end
		f = pade(c, 1, 2)
		@test f !== nothing
		for u in (0.05, 0.1, 0.2, -0.3)
			@test f(u) ≈ evalpoly(u, P) / evalpoly(u, Q) rtol = 1e-10
		end
	end

	@testset "pade sums a divergent series outside its disc" begin
		# (1 − u)^(−1/2) has a branch point at u = 1 — the same character `domb_sykes`
		# measures for the manifold (γ ≈ 0.42). Beyond u = 1 the Taylor sum is meaningless;
		# Padé still tracks the function. This is the whole reason the module exists.
		n = 9
		c = Vector{Float64}(undef, n)
		c[1] = 1.0
		for k in 1:(n - 1)
			c[k + 1] = c[k] * (2k - 1) / (2k)
		end
		exact(u) = 1 / sqrt(1 - u)
		for u in (0.5, 0.8, 0.95)
			@test abs(resum(c)(u) - exact(u)) < abs(resum(c; method = :taylor)(u) - exact(u))
		end
		@test isapprox(resum(c)(0.8), exact(0.8); rtol = 5e-3)
		# At u = 0.95 — within 5 % of the branch point — Padé still holds to ~3 %, while
		# Taylor is off by a third. That gap IS the result this module exists for.
		@test abs(resum(c)(0.95) - exact(0.95)) / exact(0.95) < 0.05
		@test abs(resum(c; method = :taylor)(0.95) - exact(0.95)) / exact(0.95) > 0.3
	end

	@testset "resum takes the largest DIAGONAL shape" begin
		# An even coefficient count must DISCARD its spare rather than build an off-diagonal
		# approximant. Measured on the Kármán run, off-diagonal shapes are worse than the
		# smaller diagonal one: [1/2] from four coefficients scored −24 % against DNS where
		# [1/1] from three scored +10 %, and [2/3] from six collapsed the branch to Re 51
		# while [2/2] from five reached Re 69.9.
		c = [1.0, 0.5, 0.375, 0.3125, 0.2734375, 0.24609375]     # (1−u)^(−1/2), 6 terms
		@test resum(c)(0.7) ≈ resum(c[1:5])(0.7) rtol = 1e-12
		@test resum(c[1:2])(0.3) ≈ evalpoly(0.3, c[1:2]) rtol = 1e-12   # too few → Taylor
	end

	@testset "_real_roots" begin
		# (x−2)(x+3)(x²+1): two real roots, one complex pair that must NOT be returned.
		# (x−2)(x+3)(x²+1) = x⁴ + x³ − 5x² + x − 6, ascending.
		p = [-6.0, 1.0, -5.0, 1.0, 1.0]
		r = _real_roots(p)
		@test length(r) == 2
		@test r ≈ [-3.0, 2.0] rtol = 1e-8
		@test isempty(_real_roots([1.0]))                   # constant: no roots
		@test isempty(_real_roots([0.0, 0.0]))              # identically zero
	end

	@testset "amplitude_series recovers a Stuart–Landau branch" begin
		# ż = (σ₀ + aη)z + c z²z̄  ⇒  G(ρ,η) = σ₀ + aη + cρ², whose zero is exact:
		# ρ² = −(σ₀ + aη)/c. Checks the coefficient EXTRACTION, independent of resummation.
		σ₀, a, c = 4.0e-3, -2.1e2, -0.111
		p = ROMPoly([[1, 0, 0], [1, 0, 1], [2, 1, 0]],
			reshape(ComplexF64[σ₀ + 16.86im, a, c], 3, 1), 3)
		for η in (0.0, -1.0e-3, -6.11e-3)
			g = amplitude_series(p, η, 9)
			@test length(g) == 2
			@test g[1] ≈ σ₀ + a * η rtol = 1e-12
			@test g[2] ≈ c rtol = 1e-12
			@test only(_real_roots(g)) ≈ -(σ₀ + a * η) / c rtol = 1e-10
		end
	end

	@testset "tke_series consistent flag" begin
		# Two monomials at harmonics +1 and −1 (z₁ and z̄₁), core degree 1 each. Their
		# product survives the period average and lands at k = 1; nothing else does.
		A = [1 0 0; 0 1 0]
		G = ComplexF64[0.0 2.0; 2.0 0.0]
		t = tke_series(G, A, 0.0, 9)
		@test length(t) == 2
		@test t[1] == 0.0                       # no constant term in a FLUCTUATION energy
		@test t[2] ≈ 2.0 rtol = 1e-12           # ½·(G₁₂ + G₂₁) = 2
		# core(1) + core(2) = 2 ≤ N, so the consistent truncation keeps it too.
		@test tke_series(G, A, 0.0, 9; consistent = true) ≈ t
		# ...but not at N = 1, where the pair exceeds the total-degree budget.
		@test all(iszero, tke_series(G, A, 0.0, 1; consistent = true))
	end
end
