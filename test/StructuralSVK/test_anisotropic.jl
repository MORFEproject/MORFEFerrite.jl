# Anisotropic St. Venant-Kirchhoff support.
#
# The load-bearing test is the ISOTROPIC LIMIT: a cubic crystal with
# c11 = λ+2μ, c12 = λ, c44 = μ is the same material as SVKMaterial(λ, μ), so
# every assembled operator and every ROM coefficient must agree to machine
# precision. That pins the anisotropic path against the long-validated
# isotropic one without needing any external reference.

using MORFE
using MORFEFerrite
using MORFEFerrite.StructuralSVK: svk_assemble_KM!, svk_nonlinearity,
	stress_model, IsotropicStress, VoigtStress, _σ
using Ferrite, Arpack, LinearMaps
using SparseArrays, LinearAlgebra, StaticArrays
using Test

const SVK = MORFEFerrite.StructuralSVK

const _E = 160e3
const _ν = 0.22
const _ρ = 2.32e-3
const _λ = (_E * _ν) / ((1 + _ν) * (1 - 2_ν))
const _μ = _E / (2(1 + _ν))

# Cubic constants that make the crystal isotropic.
iso_crystal(; rotation = nothing) = SVK.CubicCrystal(
	c11 = _λ + 2_μ, c12 = _λ, c44 = _μ, ρ = _ρ, rotation = rotation)

function _test_grid()
	# Rectangular section: a square one makes the bending pairs degenerate (see
	# the note on `_test_grid` in test_structural_svk.jl).
	grid = generate_grid(Hexahedron, (3, 1, 1), Vec(0.0, 0.0, 0.0), Vec(100.0, 5.0, 3.0))
	addfacetset!(grid, "Dirichlet", x -> abs(x[1]) < 1e-8 || abs(x[1] - 100.0) < 1e-8)
	return grid
end

@testset "Voigt stiffness and rotation" begin
	iso = SVK.voigt_stiffness(SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ))
	cub = iso_crystal().D
	@test iso ≈ cub                                   # the isotropic identity itself

	# Q = I is a no-op.
	@test SVK.rotate_voigt(cub, Matrix(I, 3, 3)) ≈ cub
	# An isotropic tensor is invariant under ANY rotation.
	θ = 0.37
	Qz = [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]
	Qx = [1.0 0.0 0.0; 0.0 cos(θ) -sin(θ); 0.0 sin(θ) cos(θ)]
	@test SVK.rotate_voigt(cub, Qz) ≈ cub
	@test SVK.rotate_voigt(cub, Qx * Qz) ≈ cub

	# A genuinely cubic (non-isotropic) crystal: silicon.
	si = SVK.CubicCrystal(c11 = 165.7e3, c12 = 63.9e3, c44 = 79.6e3, ρ = 2.33e-3).D
	@test !(si ≈ SVK.voigt_stiffness(SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ)))
	# 90° about z maps the cubic axes onto themselves → invariant.
	Q90 = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
	@test SVK.rotate_voigt(si, Q90) ≈ si
	# A generic rotation does change it, but preserves symmetry and trace-type invariants.
	sir = SVK.rotate_voigt(si, Qz)
	@test !(sir ≈ si)
	@test sir ≈ sir'                                   # stays symmetric
	# Rotating back recovers the original.
	@test SVK.rotate_voigt(sir, Qz') ≈ si
	# The two genuine quadratic invariants of the stiffness tensor:
	#   C_iijj = Σ_{i,j≤3} D[i,j]   and   C_ijij = Σ_i D[i,i] + 2Σ_{a≥4} D[a,a]
	# (note Σ_{i≤3} D[i,i] alone is NOT invariant).
	ciijj(D) = sum(D[i, j] for i in 1:3, j in 1:3)
	cijij(D) = sum(D[i, i] for i in 1:3) + 2 * sum(D[a, a] for a in 4:6)
	@test ciijj(sir) ≈ ciijj(si)
	@test cijij(sir) ≈ cijij(si)
end

@testset "stress model: Voigt ≡ Lamé in the isotropic limit" begin
	iso = stress_model(SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ))
	ani = stress_model(iso_crystal())
	@test iso isa IsotropicStress
	@test ani isa VoigtStress
	for _ in 1:20
		g = rand(Tensor{2, 3, Float64})
		ε = symmetric(g)
		@test _σ(ε, iso) ≈ _σ(ε, ani)
	end
end

@testset "assembled operators: anisotropic ≡ isotropic in the limit" begin
	grid = _test_grid()
	ip = Lagrange{RefHexahedron, 2}()^3
	qr = QuadratureRule{RefHexahedron}(3)
	cv = CellValues(qr, ip)
	dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)

	Ki = allocate_matrix(dh); Mi = allocate_matrix(dh)
	Ka = allocate_matrix(dh); Ma = allocate_matrix(dh)
	svk_assemble_KM!(Ki, Mi, dh, cv, SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ))
	svk_assemble_KM!(Ka, Ma, dh, cv, iso_crystal())
	@test maximum(abs.(Ki - Ka)) / maximum(abs.(Ki)) < 1e-14
	@test maximum(abs.(Mi - Ma)) == 0.0        # mass does not see the stiffness law

	# Nonlinear terms: same residual for the same input.
	n = ndofs(dh); f2l = Dict(d => d for d in 1:n)
	for deg in (2, 3)
		ti = svk_nonlinearity(deg, dh, cv, f2l, n,
			SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ); max_unique_cols = deg)
		ta = svk_nonlinearity(deg, dh, cv, f2l, n, iso_crystal(); max_unique_cols = deg)
		us = [randn(ComplexF64, n) for _ in 1:deg]
		ri = zeros(ComplexF64, n); ra = zeros(ComplexF64, n)
		MORFE.evaluate_term!(ri, ti, us, 1.0)
		MORFE.evaluate_term!(ra, ta, us, 1.0)
		@test maximum(abs.(ri - ra)) / max(maximum(abs.(ri)), 1e-30) < 1e-13
	end
end

@testset "ROM: anisotropic path reproduces the isotropic ROM" begin
	miso = SVK.mechanical_model(_test_grid();
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = 1e-3, β = 1e-4),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
	mani = SVK.mechanical_model(_test_grid();
		material = iso_crystal(),
		damping = SVK.RayleighDamping(α = 1e-3, β = 1e-4),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
	@test mani.info.n_dofs == miso.info.n_dofs

	ri = SVK.parametrise(miso; master = [1], order = 3)
	ra = SVK.parametrise(mani; master = [1], order = 3)
	a = ri.R.poly.coefficients
	b = ra.R.poly.coefficients
	@test size(a) == size(b)
	dev = maximum(abs.(a .- b) ./ max.(abs.(a), 1e-12))
	@info "anisotropic-vs-isotropic ROM max rel dev = $dev"
	@test dev < 1e-9
end

@testset "frame indifference: rotating crystal and mesh together" begin
	# A cubic crystal rotated by 90° about z, on a mesh whose x/y are swapped
	# accordingly, must give the same spectrum as the unrotated pair.
	si = SVK.CubicCrystal(c11 = 165.7e3, c12 = 63.9e3, c44 = 79.6e3, ρ = 2.33e-3)
	Q90 = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0]
	si90 = SVK.CubicCrystal(c11 = 165.7e3, c12 = 63.9e3, c44 = 79.6e3, ρ = 2.33e-3,
		rotation = Q90)
	# 90° about a cubic axis is a symmetry of the crystal → identical material.
	@test si.D ≈ si90.D
end
