"""
	spectrum(m::AssembledMechanicalModel; nev = 10, eigensolver = nothing)
		-> Spectrum

Solve the assembled model's eigenproblem. `nev` requests physical modes; the
default `RayleighEigensolver` returns `2nev` eigenvalues and eigenmodes in adjacent
conjugate pairs. A supplied `eigensolver` is used as-is. In particular, callers
supplying a different modal-damping solver are responsible for making its damping
consistent with `m.C`.

Pass the returned `MORFE.Spectrum` to `build_model` as `spectrum = ...` to inspect
or report it without paying for a second solve—and without letting a repeated
iterative eigensolve choose a different basis in a clustered eigenspace.
"""
function spectrum(m::AssembledMechanicalModel; nev::Int = 10, eigensolver = nothing)
    solver = eigensolver === nothing ? RayleighEigensolver(nev, m.damping) : eigensolver
    return solver isa StructureModalDampingEigensolver ?
           spectrum(m.K, m.M, solver; sorter! = (args...) -> nothing) :
           spectrum(
        NthOrderModel(m.B,
            Tuple(m.term_factory(d, 1) for d in m.nonlinear_degrees));
        solver = solver, sorter! = (args...) -> nothing)
end

"""
	eigenfrequencies(m::AssembledMechanicalModel; nev = 10, eigensolver = nothing)
		-> Vector{ComplexF64}

Return the complex damped eigenvalues from [`spectrum`](@ref). Physical mode `p`
occupies adjacent conjugate entries `2p-1, 2p`. For an underdamped pair,
`abs(imag(λ[2p-1]))/(2π)` is its damped oscillation frequency in Hz; the
undamped natural frequency `ω` used by the Rayleigh construction is distinct.
For an underdamped or critically damped mode `abs(λ) = ω`; this identity does
not hold for each individual eigenvalue of an overdamped pair.

Use the result to inspect the spectrum and choose `master` before parametrisation.
"""
function eigenfrequencies(m::AssembledMechanicalModel; kwargs...)
    collect((spectrum(m; kwargs...)).eigenvalues)
end

"""
	print_mode_table(eigenvalues; master = Int[], io = stdout)

Tabulate adjacent conjugate pairs in `eigenvalues`, ignoring an unmatched final
entry. Each row reports `real(λ)` as the decay rate and `imag(λ)` as the damped
angular frequency, together with `imag(λ)/(2π)` in Hz. Pairs listed in `master`
are marked. Output is written to `io` and the function returns `nothing`.
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
