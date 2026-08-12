# Tests for MORFEFerrite.StructuralSVK (high-level SVK + Ferrite UI).
# Gate A (small): high-level parametrise ≡ hand-built ex01-style pipeline.
# Gate B (small): forced(amplitude = 0) ≡ autonomous on shared monomials.

using MORFE
using MORFEFerrite
using MORFEFerrite.StructuralSVK: svk_assemble_KM!, svk_nonlinearity
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using SparseArrays, LinearAlgebra, StaticArrays
using Test
include(joinpath(@__DIR__, "rom_pipeline.jl"))

const SVK = MORFEFerrite.StructuralSVK
@test SVK !== nothing

# ── Tiny in-memory beam (subparametric: Hex8 geometry, quadratic field) ──────
function _test_grid()
	# Rectangular 5×3 cross-section, matching test_master_selection.jl. A square
	# section makes the two bending pairs DEGENERATE, so reducing onto master =
	# [1] leaves its twin off-manifold at the same frequency — an ill-posed
	# fixture that the outer-resonance check correctly complains about.
	grid = generate_grid(Hexahedron, (4, 1, 1), Vec(0.0, 0.0, 0.0), Vec(100.0, 5.0, 3.0))
	addfacetset!(grid, "Dirichlet",
		x -> abs(x[1]) < 1e-8 || abs(x[1] - 100.0) < 1e-8)
	return grid
end

const _E = 160e3
const _ν = 0.22
const _ρ = 2.32e-3
const _α = 1e-3
const _β = 1e-4
const _order = 3

@testset "high-level vs hand-built (Gate A, small)" begin
	grid = _test_grid()
	beam = SVK.mechanical_model(grid;
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = _α, β = _β),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
	rom = svk_build_rom(beam; master = [1], order = _order)

	# Hand-built reference: ex01 §§1–8 inline on an identical grid.
	grid2 = _test_grid()
	ip = Lagrange{RefHexahedron, 2}()^3
	qr = QuadratureRule{RefHexahedron}(3)
	cv = CellValues(qr, ip)
	dh = DofHandler(grid2)
	add!(dh, :u, ip)
	close!(dh)
	ch = ConstraintHandler(dh)
	add!(ch, Dirichlet(:u, getfacetset(grid2, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
	close!(ch)
	update!(ch, 0.0)

	λm = (_E * _ν) / ((1 + _ν) * (1 - 2_ν))
	μm = _E / (2(1 + _ν))
	K_full = allocate_matrix(dh)
	M_full = allocate_matrix(dh)
	svk_assemble_KM!(K_full, M_full, dh, cv, λm, μm, _ρ)
	free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
	free_to_local = Dict(d => i for (i, d) in enumerate(free))
	n_free = length(free)
	K = K_full[free, free]
	M = M_full[free, free]
	C = _α * M + _β * K

	ROM = 2
	mset = all_multiindices_up_to(ROM, _order; min_degree = 1)
	term_quad = svk_nonlinearity(
		2, dh, cv, free_to_local, n_free, λm, μm; max_unique_cols = length(mset))
	term_cubic = svk_nonlinearity(
		3, dh, cv, free_to_local, n_free, λm, μm; max_unique_cols = length(mset))
	model = NthOrderModel((K, C, M), (term_quad, term_cubic))

	eigenproblem = spectrum(model,
		solver = SVK.RayleighEigensolver(10, SVK.RayleighDamping(α = _α, β = _β)),
		sorter! = (args...) -> nothing)
	eigenvalues, Y, X = eigenproblem.eigenvalues, eigenproblem.eigenmodes, eigenproblem.left_eigenmodes
	master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
	master_modes = Y[:, 1, 1:ROM]
	left_eigenmodes = X[:, 1:ROM]
	master_modes_derivatives = zeros(ComplexF64, n_free, 1, ROM)
	for r in 1:ROM
		master_modes_derivatives[:, 1, r] .= Y[:, 2, r]
	end
	left_modes_derivatives = eigenproblem.left_eigenmodes_orders[:, 1:1, 1:ROM]
	resonance_set = resonance_set_from_complex_normal_form_style(
		mset, Vector{ComplexF64}(master_eigenvalues), 0.05)
	spectral = SpectralData(; eigenvalues = master_eigenvalues,
		right_modes = master_modes, right_derivatives = master_modes_derivatives,
		left_modes = left_eigenmodes, left_blocks = Array(left_modes_derivatives),
		conjugate_permutation = [2, 1])
	W_ref, R_ref = solve_cohomological_problem(model, mset, spectral, resonance_set)

	@test beam.info.n_dofs == n_free
	a = rom.R.poly.coefficients
	b = R_ref.poly.coefficients
	@test size(a) == size(b)
	dev = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-12))
	@info "Gate A (small) max rel dev = $dev"
	@test dev < 1e-10
end

@testset "zero-amplitude forcing consistency (Gate B, small)" begin
	grid = _test_grid()
	beam = SVK.mechanical_model(grid;
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = _α, β = _β),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)

	rom0 = svk_build_rom(beam; master = [1], order = _order)
	romf = svk_build_rom(beam; master = [1], order = _order,
		forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.0))

	@test romf.info.N_EXT == 2
	@test length(romf.forcing) == 1
	@test romf.info.Ω[1] ≈ abs(rom0.eigenvalues[1])

	exps0 = rom0.R.poly.multiindex_set.exponents
	expsf = romf.R.poly.multiindex_set.exponents
	lookup = Dict(e => i for (i, e) in enumerate(expsf))
	maxdev = 0.0
	for (i, e) in enumerate(exps0)
		target = SVector{4, Int}(e[1], e[2], 0, 0)
		j = get(lookup, target, nothing)
		@test j !== nothing
		c0 = rom0.R.poly.coefficients[:, i]
		cf = romf.R.poly.coefficients[1:2, j]
		maxdev = max(maxdev, maximum(abs.(c0 .- cf) ./ max.(abs.(c0), 1e-12)))
	end
	@info "Gate B (small) max rel dev = $maxdev"
	@test maxdev < 1e-10
end

@testset "forced ROM is non-trivial at finite amplitude" begin
	grid = _test_grid()
	beam = SVK.mechanical_model(grid;
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = _α, β = _β),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
	romf = svk_build_rom(beam; master = [1], order = _order,
		forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.1))
	expsf = romf.R.poly.multiindex_set.exponents
	ext_cols = [i for (i, e) in enumerate(expsf) if e[3] + e[4] > 0]
	@test !isempty(ext_cols)
	# The forcing must inject SOMETHING into the external-monomial coefficients.
	@test maximum(abs.(romf.R.poly.coefficients[:, ext_cols])) > 0
end

@testset "multiple forcings" begin
	grid = _test_grid()
	beam = SVK.mechanical_model(grid;
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = _α, β = _β),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)

	rom0 = svk_build_rom(beam; master = [1], order = _order)
	ω1 = abs(rom0.eigenvalues[1])

	# Two zero-amplitude forcings must collapse onto the autonomous ROM
	# (multi-forcing Gate B), with one external pair per forcing.
	rom2 = svk_build_rom(beam; master = [1], order = _order,
		forcing = [SVK.HarmonicForcing(mode = 1, amplitude = 0.0),
			SVK.HarmonicForcing(mode = 1, amplitude = 0.0, Ω = 3.0 * ω1)])

	@test rom2.info.N_EXT == 4
	@test length(rom2.forcing) == 2
	@test rom2.info.Ω ≈ [ω1, 3.0 * ω1]

	exps0 = rom0.R.poly.multiindex_set.exponents
	exps2 = rom2.R.poly.multiindex_set.exponents
	lookup = Dict(e => i for (i, e) in enumerate(exps2))
	maxdev = 0.0
	for (i, e) in enumerate(exps0)
		j = get(lookup, SVector{6, Int}(e[1], e[2], 0, 0, 0, 0), nothing)
		@test j !== nothing
		c0 = rom0.R.poly.coefficients[:, i]
		c2 = rom2.R.poly.coefficients[1:2, j]
		maxdev = max(maxdev, maximum(abs.(c0 .- c2) ./ max.(abs.(c0), 1e-12)))
	end
	@info "Gate B (small, 2 forcings) max rel dev = $maxdev"
	@test maxdev < 1e-10

	# Per-forcing selectivity: forcing 1 excites its own external pair (slots
	# 3,4) and forcing 2, at zero amplitude, must excite nothing on slots 5,6.
	# A closure using `sum(r)` instead of indexing `r` would drive both pairs
	# with forcing 1's load vector and fail this.
	romsel = svk_build_rom(beam; master = [1], order = _order,
		forcing = [SVK.HarmonicForcing(mode = 1, amplitude = 0.1),
			SVK.HarmonicForcing(mode = 1, amplitude = 0.0, Ω = 3.0 * ω1)])
	expsel = romsel.R.poly.multiindex_set.exponents
	only1 = [i for (i, e) in enumerate(expsel) if e[3] + e[4] > 0 && e[5] + e[6] == 0]
	only2 = [i for (i, e) in enumerate(expsel) if e[5] + e[6] > 0 && e[3] + e[4] == 0]
	@test !isempty(only1) && !isempty(only2)
	# Master rows only: rows 3:6 carry the *prescribed* external dynamics
	# ṙ = diag(±iΩₖ)·r, which is non-zero regardless of amplitude.
	@test maximum(abs.(romsel.R.poly.coefficients[1:2, only1])) > 0
	@test maximum(abs.(romsel.R.poly.coefficients[1:2, only2])) < 1e-14
end

@testset "outer-resonance diagnostic" begin
	# The warning is on FREQUENCY alone: the cohomological operator is built from
	# s = ⟨λ, α⟩ and its conditioning is independent of the right-hand side, so
	# shaping the load away from the offending mode is no protection.
	grid = _test_grid()
	_beam(α, β) = SVK.mechanical_model(grid;
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = α, β = β),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)

	# `collect_test_logs` returns (logs, value).
	function _run(f)
		logs, val = Test.collect_test_logs(f)
		return filter(r -> r.level >= Base.CoreLogging.Warn, logs), val
	end

	undamped = _beam(0.0, 0.0)

	# 0. AUTONOMOUS outer resonance. A square cross-section makes bending pairs 1
	#    and 2 degenerate, so reducing onto master = [1] alone leaves its twin
	#    off-manifold at the same frequency — flagged through the LINEAR monomials
	#    z₁, z̄₁ with no forcing anywhere in sight. This is the case the old
	#    "only monomials involving a forcing state" filter hid entirely.
	sq = generate_grid(Hexahedron, (4, 1, 1),
		Vec(0.0, 0.0, 0.0), Vec(100.0, 5.0, 5.0))
	addfacetset!(sq, "Dirichlet",
		x -> abs(x[1]) < 1e-8 || abs(x[1] - 100.0) < 1e-8)
	square_beam = SVK.mechanical_model(sq;
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = 0.0, β = 0.0),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
	warns0, rom0 = _run() do
		svk_build_rom(square_beam; master = [1], order = 3)
	end
	@test isempty(rom0.forcing)              # genuinely autonomous
	@test length(warns0) == 1                # one per PAIR, not per conjugate
	@test occursin("mode pair 2", warns0[1].message)
	@test occursin("(1, 0)", warns0[1].message)   # z₁  — superharmonic λ₁
	@test occursin("(0, 1)", warns0[1].message)   # z̄₁ — superharmonic conj(λ₁)

	# 1. Undamped, forcing mode 2 off the manifold at its own frequency: detuning
	#    is ~0, so the direction solve is near-singular and must be flagged.
	warns, rom = _run() do
		svk_build_rom(undamped; master = [1], order = 3,
			forcing = SVK.HarmonicForcing(mode = 2, amplitude = 0.1))
	end
	@test length(warns) == 1
	@test occursin("mode pair 2", warns[1].message)
	@test rom.info.N_EXT == 2
	@test rom.forcing[1].mode == 2

	# 1b. HIGHER-ORDER superharmonic. Shape the load like the MASTER mode (so the
	#     shape is beyond reproach) but put Ω at |λ₂|/3, so the cubic monomials
	#     r³ carry s = ±3iΩ = ±i|λ₂| onto non-master pair 2. Nothing at degree 1
	#     is resonant here — only the resonance-set scan over the whole monomial
	#     set can see this, which is exactly what a ±iΩ-only check would miss.
	λ = SVK.eigenfrequencies(undamped; nev = 10)
	warns1b, _ = _run() do
		svk_build_rom(undamped; master = [1], order = 3,
			forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.1,
				Ω = abs(λ[3]) / 3))
	end
	@test length(warns1b) == 1
	@test occursin("mode pair 2", warns1b[1].message)
	@test occursin("(0, 0, 3, 0)", warns1b[1].message)   # r₃³ ⇒ s = +3iΩ
	@test occursin("(0, 0, 0, 3)", warns1b[1].message)   # r₄³ ⇒ s = -3iΩ

	# 2. Same undamped beam and same non-master shape mode, but Ω detuned well
	#    away from every computed eigenvalue ⇒ well-conditioned ⇒ silent. This is
	#    the discriminating case: the criterion measures conditioning, not merely
	#    "shape mode ∉ master".
	ωs = sort(abs.(λ))
	i = argmax(diff(ωs))
	Ω_gap = 0.5 * (ωs[i] + ωs[i+1])
	# Silence must be earned at every order the scan looks at, not just degree 1:
	# state independently that Ω, 2Ω and 3Ω all miss the whole spectrum, so this
	# case cannot start passing for the wrong reason if the mesh ever changes.
	for n in 1:3
		@test minimum(abs.(im * n * Ω_gap .- λ)) > 0.05
	end
	warns2, _ = _run() do
		svk_build_rom(undamped; master = [1], order = 3,
			forcing = SVK.HarmonicForcing(mode = 2, amplitude = 0.1, Ω = Ω_gap))
	end
	@test isempty(warns2)

	# 3. Damping regularises the same resonant solve ⇒ silent. Rayleigh damping
	#    puts the detuning at |σ₂| = (α + β ω₂²)/2, so pick β to clear the 0.05
	#    rad/s default tolerance by a comfortable margin rather than hardcoding.
	ω2 = abs(λ[3])
	warns3, _ = _run() do
		svk_build_rom(_beam(0.0, 0.6 / ω2^2); master = [1], order = 3,
			forcing = SVK.HarmonicForcing(mode = 2, amplitude = 0.1))
	end
	@test isempty(warns3)

	# 4. Pair 2 on the manifold: not an outer direction at all ⇒ silent.
	warns4, _ = _run() do
		svk_build_rom(undamped; master = [1, 2], order = 3,
			forcing = SVK.HarmonicForcing(mode = 2, amplitude = 0.1))
	end
	@test isempty(warns4)
end

@testset "error paths" begin
	grid = _test_grid()
	beam = SVK.mechanical_model(grid;
		material = SVK.SVKMaterial(E = _E, ν = _ν, ρ = _ρ),
		damping = SVK.RayleighDamping(α = _α, β = _β),
		dirichlet = "Dirichlet", fe_order = 2, quad_order = 3)
	# Non-leading master pairs are supported (see test_master_selection.jl); a
	# forcing outside the master set is now a warning, not an error, so the
	# rejected cases are malformed master lists and malformed forcing modes.
	@test_throws AssertionError svk_build_rom(beam; master = [2, 1], order = 3)
	@test_throws AssertionError svk_build_rom(beam; master = [1], order = 3,
		forcing = SVK.HarmonicForcing(mode = 0, amplitude = 0.1))
end
