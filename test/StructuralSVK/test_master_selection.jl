# Gate C: `master` may name arbitrary (non-leading, non-contiguous) mode pairs.
# The high-level call must reproduce a hand-built solve that slices the
# eigenpairs at 2p-1, 2p for the requested pairs p — the pattern the IFX-style
# studies use (master = [2, 5], reducing onto a 2:1 internally resonant pair).

using MORFE
using MORFEFerrite
using MORFEFerrite.StructuralSVK: svk_assemble_KM!, svk_nonlinearity
using Ferrite, Arpack, LinearMaps
using SparseArrays, LinearAlgebra, StaticArrays
using Test

const SVKm = MORFEFerrite.StructuralSVK

# Rectangular (not square) cross-section on purpose: a square section makes the
# y- and z-bending modes degenerate, and inside a degenerate subspace Arpack is
# free to return any basis — so two separate solves would not be comparable
# mode-by-mode. 5 × 3 separates every mode used below.
function _sel_grid()
	grid = generate_grid(Hexahedron, (4, 1, 1), Vec(0.0, 0.0, 0.0), Vec(100.0, 5.0, 3.0))
	addfacetset!(grid, "Dirichlet", x -> abs(x[1]) < 1e-8 || abs(x[1] - 100.0) < 1e-8)
	return grid
end

const _sE = 160e3
const _sν = 0.22
const _sρ = 2.32e-3
const _sα = 1e-3
const _sβ = 1e-4

_sel_model() = SVKm.mechanical_model(_sel_grid();
	material = SVKm.SVKMaterial(E = _sE, ν = _sν, ρ = _sρ),
	damping = SVKm.RayleighDamping(α = _sα, β = _sβ),
	dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)

# Hand-built reference for an arbitrary set of physical mode pairs.
function _reference_rom(master::Vector{Int}, order::Int, nev::Int)
	grid = _sel_grid()
	ip = Lagrange{RefHexahedron, 2}()^3
	qr = QuadratureRule{RefHexahedron}(3)
	cv = CellValues(qr, ip)
	dh = DofHandler(grid)
	add!(dh, :u, ip)
	close!(dh)
	ch = ConstraintHandler(dh)
	add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
	close!(ch)
	update!(ch, 0.0)

	λm = (_sE * _sν) / ((1 + _sν) * (1 - 2_sν))
	μm = _sE / (2(1 + _sν))
	K_full = allocate_matrix(dh)
	M_full = allocate_matrix(dh)
	svk_assemble_KM!(K_full, M_full, dh, cv, λm, μm, _sρ)
	free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
	free_to_local = Dict(d => i for (i, d) in enumerate(free))
	n_free = length(free)
	K = K_full[free, free]
	M = M_full[free, free]
	C = _sα * M + _sβ * K

	ROM = 2 * length(master)
	mset = all_multiindices_up_to(ROM, order; min_degree = 1)
	terms = (
		svk_nonlinearity(2, dh, cv, free_to_local, n_free, λm, μm;
			max_unique_cols = length(mset)),
		svk_nonlinearity(3, dh, cv, free_to_local, n_free, λm, μm;
			max_unique_cols = length(mset)),
	)
	model = NthOrderModel((K, C, M), terms)

	eigenproblem = spectrum(model,
		solver = SVKm.RayleighEigensolver(nothing, nothing, nev, _sα, _sβ),
		sorter! = (args...) -> nothing)
	eigenvalues, Y, X = eigenproblem.eigenvalues, eigenproblem.eigenmodes, eigenproblem.left_eigenmodes

	idx = reduce(vcat, [[2p - 1, 2p] for p in master])
	master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[idx])
	master_modes = Y[:, 1, idx]
	left_eigenmodes = X[:, idx]
	master_modes_derivatives = zeros(ComplexF64, n_free, 1, ROM)
	for (r, i) in enumerate(idx)
		master_modes_derivatives[:, 1, r] .= Y[:, 2, i]
	end
	left_modes_derivatives = eigenproblem.left_eigenmodes_orders[:, 1:1, idx]

	tol = [[0.05 * abs(master_eigenvalues[j]) for j in 1:ROM] for _ in 1:length(mset)]
	resonance_set = resonance_set_from_complex_normal_form_style(
		mset, Vector{ComplexF64}(master_eigenvalues), tol)
	perm = reduce(vcat, [[2p, 2p - 1] for p in 1:length(master)])
	spectral = SpectralData(; eigenvalues = master_eigenvalues,
		right_modes = master_modes, right_derivatives = master_modes_derivatives,
		left_modes = left_eigenmodes, left_blocks = Array(left_modes_derivatives),
		conjugate_permutation = perm)
	W, R = solve_cohomological_problem(model, mset, spectral, resonance_set)
	return R, eigenvalues
end

@testset "non-leading master pairs (Gate C)" begin
	beam = _sel_model()

	for master in ([2], [1, 3])
		rom = SVKm.parametrise(beam; master = master, order = 3,
			nev = 10, resonance_tol_rel = 0.05)
		R_ref, eigs_ref = _reference_rom(master, 3, 10)

		@test rom.eigenvalues[1:(2*maximum(master))] ≈
			  eigs_ref[1:(2*maximum(master))]
		a = rom.R.poly.coefficients
		b = R_ref.poly.coefficients
		@test size(a) == size(b)
		dev = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-12))
		@info "Gate C master = $master max rel dev = $dev"
		@test dev < 1e-10
	end
end

@testset "relative vs absolute resonance tolerance" begin
	beam = _sel_model()

	# For one master pair both targets share |λ|, so the relative threshold
	# 0.05·|λ₁| is exactly the absolute threshold below — the two flavours must
	# then produce the same ROM.
	λ1 = SVKm.eigenfrequencies(beam; nev = 10)[1]
	rom_abs = SVKm.parametrise(beam; master = [1], order = 3, nev = 10,
		resonance_tol = 0.05 * abs(λ1))
	rom_rel = SVKm.parametrise(beam; master = [1], order = 3, nev = 10,
		resonance_tol_rel = 0.05)
	dev = maximum(abs.(rom_abs.R.poly.coefficients .- rom_rel.R.poly.coefficients) ./
				  max.(abs.(rom_abs.R.poly.coefficients), 1e-12))
	@info "relative-vs-absolute tolerance max rel dev = $dev"
	@test dev < 1e-12
end

@testset "reusing a precomputed spectrum" begin
	beam = _sel_model()

	ep = SVKm.spectrum(beam; nev = 10)
	rom_a = SVKm.parametrise(beam; master = [1], order = 3, eigenproblem = ep)
	rom_b = SVKm.parametrise(beam; master = [1], order = 3, eigenproblem = ep)
	# Same spectrum in ⇒ bit-identical ROM out.
	@test rom_a.R.poly.coefficients == rom_b.R.poly.coefficients
	@test rom_a.info.eig_time_s < 1e-3   # no eigenproblem was solved

	# …and it agrees with letting parametrise solve its own.
	rom_own = SVKm.parametrise(beam; master = [1], order = 3, nev = 10)
	dev = maximum(abs.(rom_a.R.poly.coefficients .- rom_own.R.poly.coefficients) ./
				  max.(abs.(rom_own.R.poly.coefficients), 1e-12))
	@info "precomputed-vs-internal spectrum max rel dev = $dev"
	@test dev < 1e-10

	# An eigenproblem too small for the requested masters is rejected (this one
	# holds 4 physical modes; master = [8] needs 8).
	small = SVKm.spectrum(beam; nev = 4)
	@test_throws AssertionError SVKm.parametrise(beam; master = [8], order = 2,
		eigenproblem = small)
end

@testset "master argument validation" begin
	beam = _sel_model()

	@test_throws AssertionError SVKm.parametrise(beam; master = [2, 2], order = 2)
	@test_throws AssertionError SVKm.parametrise(beam; master = [3, 1], order = 2)
	@test_throws AssertionError SVKm.parametrise(beam; master = [0], order = 2)
	@test_throws AssertionError SVKm.parametrise(beam; master = [8], order = 2, nev = 4)
end

@testset "node → DOF lookups" begin
	beam = _sel_model()
	dh = beam.info.dh
	ftl = beam.info.free_to_local

	# An interior node's three DOFs are distinct, free, and consistent between
	# the global and the free-DOF views.
	interior = findfirst(n -> 40.0 < n.x[1] < 60.0, dh.grid.nodes)
	@test interior !== nothing
	g = [MORFEFerrite.node_dof(dh, interior, d) for d in 1:3]
	@test length(unique(g)) == 3
	f = MORFEFerrite.free_dofs_at_nodes(dh, ftl, fill(interior, 3), [1, 2, 3])
	@test f == [ftl[gi] for gi in g]
	@test all(1 .<= f .<= beam.info.n_dofs)

	@test_throws ArgumentError MORFEFerrite.node_dof(dh, interior, 4)
	@test_throws ArgumentError MORFEFerrite.free_dofs_at_nodes(dh, ftl, [interior], [1, 2])
	# A clamped node (x = 0) has no free-DOF row.
	clamped = findfirst(n -> abs(n.x[1]) < 1e-8, dh.grid.nodes)
	@test_throws ArgumentError MORFEFerrite.free_dofs_at_nodes(dh, ftl, [clamped], [1])
end
