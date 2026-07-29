"""
	resonances(eigenvalues, master, order; external_eigenvalues = ComplexF64[],
			   resonance_tol = 0.05, resonance_tol_rel = nothing, mset = nothing)
		-> (mset, resonance_set, master_eigenvalues)

Build the monomial set and the complex-normal-form `ResonanceSet` that a
`parametrise` call with the same arguments would use. Call it on the output of
[`eigenfrequencies`](@ref) to preview — before paying for the cohomological
solve — which monomials will be kept in the reduced dynamics.

`resonance_tol` is the absolute detuning threshold (rad/s); `resonance_tol_rel`,
when given, replaces it by the per-target relative threshold
`resonance_tol_rel * |λⱼ|`.
"""
function resonances(eigenvalues::AbstractVector, master::Vector{Int}, order::Int;
	external_eigenvalues::Vector{ComplexF64} = ComplexF64[],
	resonance_tol::Real = 0.05,
	resonance_tol_rel::Union{Nothing, Real} = nothing,
	mset = nothing)
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
		external_eigenvalues = external_eigenvalues)
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
	for t in 1:(n_int + n_out)
		cols = resonant_multiindices(resonance_set, t)
		λt = t ≤ n_int ? master_eigenvalues[t] : ComplexF64(0)
		@printf(io, "\n    Target %d (λ = %+.6e %+.6e im): %d resonant monomials\n",
			t, real(λt), imag(λt), length(cols))
		println(io, "      ", join(["$(Tuple(mset.exponents[k]))" for k in cols], "  "))
	end
	return nothing
end

"""
	parametrise(m::AssembledMechanicalModel; master = [1], order,
				forcing = nothing, resonance_tol = 0.05, resonance_tol_rel = nothing,
				eigenproblem = nothing, nev = ..., eigensolver = nothing)

Compute the DPIM invariant-manifold ROM. `master` lists physical mode PAIRS —
`[1]` is the first conjugate pair (ROM = 2), and the pairs need not be leading
or contiguous (`master = [2, 5]` reduces onto the 2nd and 5th physical modes).
With `forcing::HarmonicForcing`, two external states with eigenvalues ±iΩ are
appended (N_EXT = 2).

Resonances are detected with `|λⱼ - ⟨λ, α⟩| < tol`. `resonance_tol` sets that
threshold in absolute units (rad/s); `resonance_tol_rel`, when given, replaces
it by the per-target relative threshold `resonance_tol_rel * |λⱼ|`, which is the
physically meaningful criterion when the master modes span decades in frequency
(e.g. a 2:1 internally resonant pair).

Pass `eigenproblem` (from [`spectrum`](@ref)) to reuse a spectrum you already
solved — inspecting the modes then costs nothing and cannot perturb the ROM.
"""
function parametrise(m::AssembledMechanicalModel;
	master::Vector{Int} = [1],
	order::Int,
	forcing::Union{Nothing, HarmonicForcing} = nothing,
	resonance_tol::Real = 0.05,
	resonance_tol_rel::Union{Nothing, Real} = nothing,
	eigenproblem = nothing,
	nev::Int = max(10,
		2 * max(maximum(master), forcing === nothing ? 0 : forcing.mode) + 4),
	eigensolver = nothing)
	@assert all(>(0), master)&&allunique(master) "master must list distinct positive mode pairs; got master = $master"
	@assert issorted(master) "master must be sorted (the ROM coordinate order follows it); got master = $master"
	@assert eigenproblem !== nothing||nev >= maximum(master) "nev = $nev is too small for master = $master: at least $(maximum(master)) physical modes must be computed"
	if forcing !== nothing
		@assert forcing.mode in master "forcing.mode = $(forcing.mode) must be a master mode pair (master = $master): the near-resonant reduction requires the forced mode on the manifold"
	end

	ROM = 2 * length(master)
	N_EXT = forcing === nothing ? 0 : 2
	NVAR = ROM + N_EXT
	mset = all_multiindices_up_to(NVAR, order; min_degree = 1)

	terms = Tuple(m.term_factory(d, length(mset)) for d in m.nonlinear_degrees)

	# ── Eigenproblem (autonomous operator; the forcing does not enter) ──────
	eig_model = NDOrderModel((m.K, m.C, m.M), terms)
	t_eig = @elapsed ep = eigenproblem === nothing ?
							  spectrum(m; nev = nev, eigensolver = eigensolver) :
							  eigenproblem
	(eigenvalues, Y, X) = get_eigenpairs(ep)
	@assert length(eigenvalues)>=2 * maximum(master) "the eigenproblem holds $(length(eigenvalues) ÷ 2) physical modes, too few for master = $master"

	# Eigenvalues come in conjugate pairs: physical mode p occupies columns 2p-1, 2p.
	master_indices = reduce(vcat, [[2p - 1, 2p] for p in master])
	select_master_modes_by_hand(ep,
		[i in master_indices for i in 1:length(eigenvalues)])
	master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[master_indices])
	master_modes = Y[:, 1, master_indices]
	left_eigenmodes = X[:, master_indices]

	ORD_model = length(eig_model.linear_terms) - 1   # = 2
	FOM = m.info.n_dofs
	master_modes_derivatives = zeros(ComplexF64, FOM, ORD_model - 1, ROM)
	for (r, idx) in enumerate(master_indices), k in 1:(ORD_model-1)
		master_modes_derivatives[:, k, r] .= Y[:, k+1, idx]
	end
	# Lower-order left eigenvector blocks (must be scale-consistent with the
	# physical slice above — both come from the same Eigenproblem storage).
	left_modes_derivatives = ep.left_eigenmodes_orders[
		:, 1:(ORD_model-1), master_indices]

	# ── Model with forcing (shape mode == frequency mode) ───────────────────
	Ω = nothing
	if forcing === nothing
		model = eig_model
		ext_eigs = ComplexF64[]
	else
		Ω = forcing.Ω === nothing ? abs(eigenvalues[2*forcing.mode-1]) : forcing.Ω
		fv = real((forcing.amplitude / 2) .* (m.M * Y[:, 1, 2*forcing.mode-1]))
		# Degree-1 term: multiindex (0, 0) in the state derivatives, multiplicity 1
		# in the external state ⇒ f! signature is exactly (res, r).
		force_term = MultilinearMap((res, r) -> (res .+= fv * sum(r)), (0, 0), 1)
		ext_eigs = ComplexF64[im*Ω, -im*Ω]
		model = NDOrderModel((m.K, m.C, m.M), (terms..., force_term),
			ExternalSystem(Tuple(ext_eigs)))
	end

	_, resonance_set, _ = resonances(eigenvalues, master, order;
		external_eigenvalues = ext_eigs, resonance_tol = resonance_tol,
		resonance_tol_rel = resonance_tol_rel, mset = mset)

	conjugate_permutation = reduce(vcat, [[2p, 2p - 1] for p in 1:length(master)])
	for k in 1:(N_EXT ÷ 2)
		append!(conjugate_permutation, [ROM + 2k, ROM + 2k - 1])
	end

	t_solve = @elapsed W,
	R = solve_cohomological_problem(
		model, mset, master_eigenvalues, master_modes, left_eigenmodes, resonance_set;
		master_modes_derivatives = master_modes_derivatives,
		left_modes_derivatives = left_modes_derivatives,
		conjugate_permutation = conjugate_permutation)

	return InvariantManifoldROM(W, R, collect(eigenvalues), master, order, forcing,
		(; m.info..., n_monomials = length(mset), N_EXT = N_EXT, Ω = Ω,
			eig_time_s = t_eig, solve_time_s = t_solve))
end
