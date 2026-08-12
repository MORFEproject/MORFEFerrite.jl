"""
	resonances(eigenvalues, master, order; external_eigenvalues = ComplexF64[],
			   outer_eigenvalues = ComplexF64[], resonance_tol = 0.05,
			   resonance_tol_rel = nothing, mset = nothing)
		-> (mset, resonance_set, master_eigenvalues)

Build the monomial set and the complex-normal-form `ResonanceSet` that a
`parametrise` call with the same arguments would use. Call it on the output of
[`eigenfrequencies`](@ref) to preview — before paying for the cohomological
solve — which monomials will be kept in the reduced dynamics.

`resonance_tol` is the absolute detuning threshold (rad/s); `resonance_tol_rel`,
when given, replaces it by the per-target relative threshold
`resonance_tol_rel * |λⱼ|`.

`outer_eigenvalues` adds off-manifold resonance targets, populating the
`outer_resonances` block of the returned set (query it with
`resonant_multiindices(rset, ROM + j)`). Those flags are diagnostic: the
cohomological solve reads only the inner block. It cannot be combined with
`resonance_tol_rel` — see the assertion below.
"""
function resonances(eigenvalues::AbstractVector, master::Vector{Int}, order::Int;
	external_eigenvalues::Vector{ComplexF64} = ComplexF64[],
	outer_eigenvalues::Vector{ComplexF64} = ComplexF64[],
	resonance_tol::Real = 0.05,
	resonance_tol_rel::Union{Nothing, Real} = nothing,
	mset = nothing)
	# MORFE hands the SAME `tol` to the inner and outer conditions, and each
	# indexes it by its own local target number — so `tol[k][j]` would have to
	# serve master target j and outer target j at once. The per-target vectors
	# below are sized ROM, so combining them with outer targets is a bounds
	# error waiting to happen; require a scalar tolerance in that case.
	@assert isempty(outer_eigenvalues)||resonance_tol_rel === nothing "resonance_tol_rel cannot be combined with outer_eigenvalues: MORFE indexes one per-target tolerance vector by both the inner and the outer local target number. Pass an absolute resonance_tol instead."
	master_indices = reduce(vcat, [[2p - 1, 2p] for p in master])
	λm = ComplexF64[eigenvalues[i] for i in master_indices]
	ROM = length(λm)
	NVAR = ROM + length(external_eigenvalues)
	ms = mset === nothing ? all_multiindices_up_to(NVAR, order; min_degree = 1) : mset
	# Per-monomial, per-target threshold when a relative tolerance is requested:
	# tol[k][j] = resonance_tol_rel * |λⱼ|, so detuning is measured on each
	# target's own frequency scale.
	tol = resonance_tol_rel === nothing ? Float64(resonance_tol) :
		  [[Float64(resonance_tol_rel) * abs(λm[j]) for j in 1:ROM] for _ in 1:length(ms)]
	rset = resonance_set_from_complex_normal_form_style(ms, λm, tol;
		external_eigenvalues = external_eigenvalues,
		outer_eigenvalues = outer_eigenvalues)
	return ms, rset, λm
end

"""
	print_resonances(mset, resonance_set, master_eigenvalues; io = stdout)

List, per reduced-dynamics target, the monomials flagged as resonant — the terms
that survive in the reduced dynamics `R`.
"""
function print_resonances(mset, resonance_set, master_eigenvalues::Vector{ComplexF64};
	io::IO = stdout)
	n_int = n_internal(resonance_set)
	n_out = resonance_set.outer_resonances === nothing ? 0 :
			size(resonance_set.outer_resonances, 1)
	n_flag = count(resonance_set.inner_resonances) +
			 (n_out > 0 ? count(resonance_set.outer_resonances) : 0)
	@printf(io, "  Monomials: %d,  targets: %d,  resonant flags: %d / %d\n",
		length(mset), n_int + n_out, n_flag, (n_int + n_out) * length(mset))
	for t in 1:(n_int+n_out)
		cols = resonant_multiindices(resonance_set, t)
		λt = t ≤ n_int ? master_eigenvalues[t] : ComplexF64(0)
		@printf(io, "\n    Target %d (λ = %+.6e %+.6e im): %d resonant monomials\n",
			t, real(λt), imag(λt), length(cols))
		println(io, "      ", join(["$(Tuple(mset.exponents[k]))" for k in cols], "  "))
	end
	return nothing
end
