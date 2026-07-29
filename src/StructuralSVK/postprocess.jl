"""
	spectrum(m::AssembledMechanicalModel; nev = 10, eigensolver = nothing)
		-> Eigenproblem

Solve the model's eigenproblem. Pass the result to `parametrise` as
`eigenproblem = …` to inspect the spectrum first without paying for a second
solve — and so that inspecting it cannot perturb the ROM.
"""
function spectrum(m::AssembledMechanicalModel; nev::Int = 10, eigensolver = nothing)
	solver = eigensolver === nothing ?
			 RayleighEigenSolver(nothing, nothing, nev,
		Float64(m.damping.α), Float64(m.damping.β)) : eigensolver
	return solver isa StructureModalDampingEigensolver ?
		   solve_eigenproblem(m.K, m.M, solver; sorter! = (args...) -> nothing) :
		   solve_eigenproblem(
		NDOrderModel((m.K, m.C, m.M),
			Tuple(m.term_factory(d, 1) for d in m.nonlinear_degrees));
		solver = solver, sorter! = (args...) -> nothing)
end

"""
	eigenfrequencies(m::AssembledMechanicalModel; nev = 10, eigensolver = nothing)
		-> Vector{ComplexF64}

Damped eigenvalues of the assembled model, ordered in conjugate pairs: physical
mode `p` occupies entries `2p-1, 2p`, so `abs(λ[2p-1]) / 2π` is its frequency in
Hz. Use it to inspect the spectrum — and pick `master` — before committing to a
parametrisation.
"""
eigenfrequencies(m::AssembledMechanicalModel; kwargs...) =
	collect(get_eigenpairs(spectrum(m; kwargs...))[1])

"""
	print_mode_table(eigenvalues; master = Int[], io = stdout)

Tabulate the physical modes behind `eigenvalues` (one row per conjugate pair:
decay rate, frequency in Hz and rad/s), marking the pairs listed in `master`.
"""
function print_mode_table(eigenvalues::AbstractVector; master::Vector{Int} = Int[],
	io::IO = stdout)
	n = length(eigenvalues) ÷ 2
	rule = "  " * "-"^68
	println(io, "  Physical mode table ($n pairs computed):")
	println(io, rule)
	println(io, "  Mode  EV idx     σ (decay)          f (Hz)          ω (rad/s)")
	println(io, rule)
	for p in 1:n
		λ = eigenvalues[2p-1]
		mark = p in master ? "  ← MASTER ★" : ""
		@printf(io, "  %3d   %2d, %2d   %+.6e   %12.4f   %14.4f%s\n",
			p, 2p - 1, 2p, real(λ), imag(λ) / (2π), imag(λ), mark)
	end
	println(io, rule)
	return nothing
end

"""
	real_dynamics(rom) -> DensePolynomial

Realified master equation ż₁ in the real pair z₁ = x₁ + i y₁:
Re(c) is the ẋ₁-equation, Im(c) the ẏ₁-equation (the imaginary parts carry
the frequency content; do not discard them). External (forcing) states also
come in conjugate ±iΩ pairs, so the same pairwise conjugation map applies.
"""
function real_dynamics(rom::InvariantManifoldROM)
	NVAR = 2 * length(rom.master) + rom.info.N_EXT
	conj_map = [isodd(i) ? i + 1 : i - 1 for i in 1:NVAR]
	return realify(extract_component(rom.R.poly, 1), conj_map)
end

"""
	print_equations(rom; tol = 1e-12, io = stdout)

Print the realified ż₁ monomial table. In the forced case the trailing two
exponents belong to the external states (e^{+iΩt}, e^{-iΩt}).
"""
function print_equations(rom::InvariantManifoldROM; tol = 1e-12, io = stdout)
	Rr = real_dynamics(rom)
	header = rom.info.N_EXT == 0 ?
			 "(x₁,y₁) exponents : ẋ₁-coeff, ẏ₁-coeff" :
			 "(x₁,y₁,ext₊,ext₋) exponents : ẋ₁-coeff, ẏ₁-coeff"
	println(io, "Reduced dynamics ż₁ = ẋ₁ + i·ẏ₁ in real variables:")
	println(io, "  " * header)
	for (m, mi) in enumerate(Rr.multiindex_set.exponents)
		c = Rr.coefficients[m]
		abs(c) > tol && println(io,
			"  $(Tuple(mi)) : " *
			"$(round(real(c); sigdigits = 6)), $(round(imag(c); sigdigits = 6))")
	end
end

"""
	save_rom(rom, dir)

Write `dir/data/{W.jls, R.jls, R_coefficients.csv}` and `dir/summary.txt`.
Extends `MORFE.save_rom` for the SVK `InvariantManifoldROM`, delegating the
file layout and CSV schema to `MORFE.RomIO` (the equivalence gates diff those
CSVs, so the schema is shared library-wide).
"""
function save_rom(rom::InvariantManifoldROM, dir::AbstractString)
	metadata = Pair{String, Any}[
		"model" => "SVK + Ferrite (MORFEFerrite.StructuralSVK)",
		"n_dofs" => rom.info.n_dofs,
		"master_pairs" => rom.master,
		"master_eigenvalues" => rom.eigenvalues[reduce(
			vcat, [[2p - 1, 2p] for p in rom.master])],
		"parametrisation_order" => rom.order,
		"n_monomials" => rom.info.n_monomials,
		"forcing" => rom.forcing === nothing ? "none" :
					 "mode=$(rom.forcing.mode) amplitude=$(rom.forcing.amplitude) Omega=$(rom.info.Ω)",
		"eigenproblem_time_s" => rom.info.eig_time_s,
		"cohomological_solve_time_s" => rom.info.solve_time_s,
	]
	MORFE.save_rom(dir, rom.W, rom.R; metadata = metadata)
	return nothing
end
