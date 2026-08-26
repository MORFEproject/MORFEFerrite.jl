# The θ-series algebra against its DEFINING IDENTITIES, in 2D and in 3D.
#
# `determinant_series` and `adjugate_series` implement theory App. A.1 and A.2 in
# a dimension-general way — no cofactor formula, no 3×3 anywhere. This file is
# what makes that a measurement rather than a claim, and it is mesh-free so it
# runs in milliseconds.
#
# The reference is never another series implementation. It is the DENSE tensor
# operation: evaluate the series at a concrete θ and compare against `det(J(θ))`
# and `det(J(θ)) · inv(J(θ))` computed by Tensors.jl. When the box is wide enough
# to hold the full polynomial degree the agreement must be to round-off, because
# both sides are then the same polynomial rather than two truncations of it.

using Test
using MORFEFerrite
using Ferrite, Tensors, StaticArrays
using LinearAlgebra: norm

const PGa = MORFEFerrite.ParametricGeometry

# Evaluate a θ-series (of scalars or tensors) at a concrete θ.
_ev(ser, basis, θ) = sum(
	prod(θ[i]^α[i] for i in eachindex(α)) * ser[m]
	for (m, α) in enumerate(basis.mset.exponents))

# A well-conditioned random affine Jacobian: J₀ near the identity so it stays
# invertible, J₁ arbitrary. Deliberately NOT J₀ = I — App. A.2's recurrence is
# derived assuming J₀ = I, and this module cannot make that assumption (example
# 07's arch has a curved reference configuration).
function _rand_affine(::Val{dim}, rng_seed) where {dim}
	m = dim * dim
	v0 = [0.3 * sin(3.1 * (rng_seed + k)) for k in 1:m]
	v1 = [0.7 * cos(2.3 * (rng_seed + k)) for k in 1:m]
	J₀ = one(Tensor{2, dim, Float64}) + Tensor{2, dim}(reshape(v0, dim, dim))
	J₁ = Tensor{2, dim}(reshape(v1, dim, dim))
	return J₀, J₁
end

@testset "series algebra is dimension-general (App. A.1 / A.2)" begin
	for dim in (2, 3)
		@testset "d = $dim" begin
			# deg det J ≤ d·deg J = d and deg adj J ≤ (d−1)·deg J = d−1, so a box
			# of bound d holds BOTH exactly and nothing below is a truncation.
			basis = PGa.GeometryParameterBasis([dim])
			J₀, J₁ = _rand_affine(Val(dim), 1.0)
			J = PGa.jacobian_series((J₀, J₁), basis)
			det_ser, adj_ser = PGa.det_adj_series(J, basis)

			@testset "det and adj match the dense tensor operations" begin
				for θ in ((0.0,), (0.13,), (-0.37,), (0.9,))
					Jθ = J₀ + θ[1] * J₁
					@test _ev(det_ser, basis, θ) ≈ det(Jθ) rtol=1e-12
					# adj A = det(A) · A⁻¹ — the definition, not another series.
					@test norm(_ev(adj_ser, basis, θ) - det(Jθ) * inv(Jθ)) <
						  1e-11 * max(1.0, norm(det(Jθ) * inv(Jθ)))
				end
			end

			@testset "J · adj J = det J · I as a truncated series identity" begin
				prodser = PGa.poly_dot(J, adj_ser, basis)
				Id = one(Tensor{2, dim, Float64})
				for m in eachindex(prodser)
					# Terms of degree > d are outside the box and legitimately
					# truncated; inside it the identity must hold exactly.
					sum(basis.mset.exponents[m]) > dim && continue
					@test norm(prodser[m] - det_ser[m] * Id) < 1e-11
				end
			end

			@testset "the proven degree bounds actually hold" begin
				# A wide box, so anything nonzero above the bound is a real
				# violation rather than something the box hid.
				wide = PGa.GeometryParameterBasis([dim + 3])
				Jw = PGa.jacobian_series((J₀, J₁), wide)
				dw, aw = PGa.det_adj_series(Jw, wide)
				scale = maximum(abs, dw)
				for (m, α) in enumerate(wide.mset.exponents)
					if sum(α) > dim                     # deg det J ≤ d·deg J
						@test abs(dw[m]) < 1e-11 * scale
					end
					if sum(α) > dim - 1                 # deg adj J ≤ (d−1)·deg J
						@test norm(aw[m]) < 1e-10
					end
				end
			end

			@testset "reciprocal series satisfies det J · (1/det J) = 1" begin
				inv_ser = PGa.reciprocal_series(det_ser, basis)
				one_ser = PGa.poly_mul(det_ser, inv_ser, basis)
				@test one_ser[1] ≈ 1.0 rtol=1e-14
				# Within the box the product must collapse to the constant 1.
				@test maximum(abs, one_ser[2:end]) < 1e-10
			end
		end
	end

	@testset "2D adjugate against the analytic [d −b; −c a]" begin
		basis = PGa.GeometryParameterBasis([2])
		J₀, J₁ = _rand_affine(Val(2), 4.0)
		_, adj_ser = PGa.det_adj_series(PGa.jacobian_series((J₀, J₁), basis), basis)
		for θ in ((0.0,), (0.21,), (-0.4,))
			A = J₀ + θ[1] * J₁
			analytic = Tensor{2, 2}((A[2, 2], -A[2, 1], -A[1, 2], A[1, 1]))
			@test norm(_ev(adj_ser, basis, θ) - analytic) < 1e-12
		end
	end

	# The recurrence is derived from J₀ A_σ + Σ J_α A_{σ−α} = c_σ I. With J₀ = I
	# it reduces to the appendix's published form; this pins the J₀ ≠ I branch
	# that the appendix does not cover but example 07 needs.
	@testset "J₀ = I reduces to the appendix's published recurrence" begin
		for dim in (2, 3)
			basis = PGa.GeometryParameterBasis([dim])
			_, J₁ = _rand_affine(Val(dim), 7.0)
			Id = one(Tensor{2, dim, Float64})
			J = PGa.jacobian_series((Id, J₁), basis)
			c, A = PGa.det_adj_series(J, basis)
			@test A[1] ≈ Id                       # A₀ = adj(I) = I
			# A_σ = c_σ I − Σ_{β<σ} J_{σ−β} A_β, evaluated at σ = e₁ (position 2).
			@test norm(A[2] - (c[2] * Id - J₁ ⋅ A[1])) < 1e-12
		end
	end

	@testset "two parameters, mixed degrees" begin
		dim = 3
		basis = PGa.GeometryParameterBasis([3, 3])
		J₀, J₁ = _rand_affine(Val(dim), 2.0)
		_, J₂ = _rand_affine(Val(dim), 5.0)
		J = PGa.jacobian_series((J₀, J₁, J₂), basis)
		det_ser, adj_ser = PGa.det_adj_series(J, basis)
		for θ in ((0.0, 0.0), (0.2, -0.15), (0.35, 0.4))
			Jθ = J₀ + θ[1] * J₁ + θ[2] * J₂
			@test _ev(det_ser, basis, θ) ≈ det(Jθ) rtol=1e-11
			@test norm(_ev(adj_ser, basis, θ) - det(Jθ) * inv(Jθ)) < 1e-10
		end
	end
end

@testset "general polynomial map x(x_ref, θ) = Σ_α x_α θ^α" begin
	basis = PGa.GeometryParameterBasis([3, 2])
	J₀, J₁ = _rand_affine(Val(3), 3.0)
	_, J₂ = _rand_affine(Val(3), 6.0)

	@testset "the pair form reproduces the affine constructor exactly" begin
		affine = PGa.jacobian_series((J₀, J₁, J₂), basis)
		general = PGa.jacobian_series(
			[SVector(0, 0) => J₀, SVector(1, 0) => J₁, SVector(0, 1) => J₂], basis)
		@test affine == general
	end

	@testset "genuinely non-affine coefficients are placed and used" begin
		# A θ₁² term — impossible to express through the affine constructor.
		Jq = 0.4 * J₂
		coeffs = [(0, 0) => J₀, (1, 0) => J₁, (2, 0) => Jq]
		@test PGa.jacobian_series(coeffs, basis)[PGa.position_of(basis, SVector(2, 0))] ≈ Jq

		# deg_θ₁ J = 2, so App. A.1 gives deg_θ₁ det J ≤ d·2 = 6 and App. A.2 gives
		# deg_θ₁ adj J ≤ (d−1)·2 = 4. A box bounded at 6 holds both EXACTLY; that
		# is what makes the comparison below a test of the algebra rather than of
		# how much the box happened to keep.
		wide = PGa.GeometryParameterBasis([6, 2])
		Jw = PGa.jacobian_series(coeffs, wide)
		dw, aw = PGa.det_adj_series(Jw, wide)
		for θ in ((0.1, 0.0), (0.3, 0.0), (-0.25, 0.0))
			Jθ = J₀ + θ[1] * J₁ + θ[1]^2 * Jq
			@test _ev(dw, wide, θ) ≈ det(Jθ) rtol=1e-11
			@test norm(_ev(aw, wide, θ) - det(Jθ) * inv(Jθ)) < 1e-10
		end
		# …and the bound is TIGHT-ish: nothing survives above it.
		for (m, α) in enumerate(wide.mset.exponents)
			α[1] > 6 && @test abs(dw[m]) < 1e-11
			α[1] > 4 && @test norm(aw[m]) < 1e-10
		end

		# The other half of the same statement: a box that CANNOT hold the exact
		# degree truncates, and silently. `basis` bounds θ₁ at 3 while det J needs
		# 6, so the θ₁⁴… terms are dropped and the series stops being exact. This
		# is the module's documented behaviour, and the reason per-parameter bounds
		# must be read off the geometry rather than chosen by symmetry.
		dn, _ = PGa.det_adj_series(PGa.jacobian_series(coeffs, basis), basis)
		θ = (0.3, 0.0)
		@test !isapprox(_ev(dn, basis, θ), det(J₀ + θ[1] * J₁ + θ[1]^2 * Jq); rtol = 1e-10)
	end

	@testset "repeated multiindices accumulate, as Σ_α implies" begin
		a = PGa.jacobian_series([(0, 0) => J₀, (1, 0) => J₁, (1, 0) => J₁], basis)
		b = PGa.jacobian_series([(0, 0) => J₀, (1, 0) => 2 * J₁], basis)
		@test a == b
	end

	# Silently truncating the MAP would change which geometry is modelled without
	# saying so — the precise failure mode that made example 04 wrong.
	@testset "a coefficient outside the box is an error, not a silent drop" begin
		@test_throws ArgumentError PGa.jacobian_series(
			[(0, 0) => J₀, (0, 3) => J₁], basis)          # bound on θ₂ is 2
	end

	# The map has to survive the PROVIDER contract too, not just the constructor:
	# a provider may return pairs instead of the affine tuple, and `PullbackCache`
	# must probe its dimension and build the same cache either way.
	@testset "a pair-returning provider drives PullbackCache end to end" begin
		grid = generate_grid(Hexahedron, (2, 1, 1), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
		ip = Lagrange{RefHexahedron, 1}()^3
		cv = CellValues(QuadratureRule{RefHexahedron}(2), ip, Lagrange{RefHexahedron, 1}())
		dh = DofHandler(grid)
		add!(dh, :u, ip)
		close!(dh)

		b1 = PGa.GeometryParameterBasis([3])
		∇ψ = Tensor{2, 3}((i, j) -> (i == 2 && j == 1) ? 0.4 : 0.0)
		affine = PGa.PullbackCache(dh, cv, x -> (one(Tensor{2, 3, Float64}), ∇ψ), b1)
		pairs = PGa.PullbackCache(dh, cv,
			x -> [SVector(0) => one(Tensor{2, 3, Float64}), SVector(1) => ∇ψ], b1)

		@test PGa.adj_tensor_type(pairs) === PGa.adj_tensor_type(affine)
		for ci in eachindex(affine.det), q in eachindex(affine.det[ci])
			@test affine.det[ci][q] == pairs.det[ci][q]
			@test affine.adj[ci][q] == pairs.adj[ci][q]
			@test affine.inv_det[ci][q] == pairs.inv_det[ci][q]
		end
	end
end
