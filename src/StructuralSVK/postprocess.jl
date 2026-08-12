"""
	spectrum(m::AssembledMechanicalModel; nev = 10, eigensolver = nothing)
		-> Spectrum

Solve the model's eigenproblem. Pass the result to `parametrise` as
`eigenproblem = …` to inspect the spectrum first without paying for a second
solve — and so that inspecting it cannot perturb the ROM.
"""
function spectrum(m::AssembledMechanicalModel; nev::Int = 10, eigensolver = nothing)
	solver = eigensolver === nothing ? RayleighEigensolver(nev, m.damping) : eigensolver
	return solver isa StructureModalDampingEigensolver ?
		   spectrum(m.K, m.M, solver; sorter! = (args...) -> nothing) :
		   spectrum(
		NthOrderModel((m.K, m.C, m.M),
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
	collect((spectrum(m; kwargs...)).eigenvalues)

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
the frequency content; do not discard them). Every forcing's external states
also come in a conjugate ±iΩ pair, so the same pairwise conjugation map applies
regardless of how many forcings there are.

The adjacent-pair map below is valid exactly because `parametrise` builds `N_EXT =
2 · length(forcing)` external states as ±iΩ pairs from a diagonal `ExternalSystem`, so no
change of external coordinates ever takes place. The assertion pins that invariant: a ROM
with an odd `N_EXT` (a self-conjugate real external variable) or a re-based external system
needs its permutation derived with `MORFE.full_conjugate_permutation` instead, which this
signature has no access to since `InvariantManifoldROM` does not retain the model.
"""
function real_dynamics(rom::InvariantManifoldROM)
	N_EXT = rom.info.N_EXT
	@assert iseven(N_EXT) """
	real_dynamics assumes external states come in adjacent conjugate ±iΩ pairs, but this ROM
	has N_EXT = $(N_EXT), which is odd. Derive the conjugation map from the model's
	ExternalSystem with `MORFE.full_conjugate_permutation(master_block, external_system)`.
	"""
	NVAR = 2 * length(rom.master) + N_EXT
	conj_map = [isodd(i) ? i + 1 : i - 1 for i in 1:NVAR]
	return realify(extract_component(rom.R.poly, 1), conj_map)
end

_sub(n::Integer) = join('₀' + d for d in reverse(digits(n)))

"""
	print_equations(rom; tol = 1e-12, io = stdout)

Print the realified ż₁ monomial table. `realify` groups the variables as
`(x₁…xₙ, y₁…yₙ)` — all real parts, then all imaginary parts — over
`n = length(master) + N_EXT÷2` conjugate pairs. Pairs `1 … length(master)` are
the master coordinates; the rest are the forcing pairs, in the order the
forcings were passed to `parametrise`.
"""
function print_equations(rom::InvariantManifoldROM; tol = 1e-12, io = stdout)
	Rr = real_dynamics(rom)
	n_pairs = length(rom.master) + rom.info.N_EXT ÷ 2
	header = "(" * join(["x" * _sub(i) for i in 1:n_pairs], ",") * "," *
			 join(["y" * _sub(i) for i in 1:n_pairs], ",") *
			 ") exponents : ẋ₁-coeff, ẏ₁-coeff"
	println(io, "Reduced dynamics ż₁ = ẋ₁ + i·ẏ₁ in real variables:")
	println(io, "  " * header)
	if rom.info.N_EXT > 0
		println(io,
			"  (pairs $(length(rom.master) + 1)…$n_pairs are the forcing states)")
	end
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
		"n_forcings" => length(rom.forcing),
		"forcing" => isempty(rom.forcing) ? "none" :
					 join(["mode=$(f.mode) amplitude=$(f.amplitude) Omega=$Ω"
						   for (f, Ω) in zip(rom.forcing, rom.info.Ω)], "; "),
		"eigenproblem_time_s" => rom.info.eig_time_s,
		"cohomological_solve_time_s" => rom.info.solve_time_s,
	]
	MORFE.save_rom(dir, rom.W, rom.R; metadata = metadata)
	return nothing
end
