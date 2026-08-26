"""
	spectrum(m::AssembledMechanicalModel; nev = 10, eigensolver = nothing)
		-> Spectrum

Solve the model's eigenproblem. Pass the result to `build_model` as
`spectrum = …` to inspect the spectrum first without paying for a second
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
function eigenfrequencies(m::AssembledMechanicalModel; kwargs...)
    collect((spectrum(m; kwargs...)).eigenvalues)
end

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
        λ = eigenvalues[2p - 1]
        mark = p in master ? "  ← MASTER ★" : ""
        @printf(io, "  %3d   %2d, %2d   %+.6e   %12.4f   %14.4f%s\n",
            p, 2p - 1, 2p, real(λ), imag(λ) / (2π), imag(λ), mark)
    end
    println(io, rule)
    return nothing
end
