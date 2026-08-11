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

"""
	_forcings(forcing) -> Vector{<:HarmonicForcing}

Normalise the `forcing` keyword (`nothing`, a single `HarmonicForcing`, or a
vector of them) to a vector — empty when the problem is autonomous.
"""
_forcings(::Nothing) = HarmonicForcing[]
_forcings(f::HarmonicForcing) = [f]
_forcings(fs::AbstractVector{<:HarmonicForcing}) = collect(fs)

# Largest physical mode pair a forcing draws its shape from (0 when unforced).
# Used before the eigensolve to size `nev`, so it must accept the raw keyword.
function _max_forcing_mode(forcing)
	fs = _forcings(forcing)
	return isempty(fs) ? 0 : maximum(f -> f.mode, fs)
end

"""
	_warn_outer_resonances(eigenvalues, master, master_indices, mset,
						   external_eigenvalues, order, resonance_tol)

Warn when a monomial is near-resonant with a physical mode that is *not* on the
manifold, which makes that direction's solve ill-conditioned.

The check is delegated to MORFE: [`resonances`](@ref) is re-run with the
non-master eigenvalues supplied as `outer_eigenvalues`, so the resulting
`ResonanceSet` carries a populated `outer_resonances` block — the same
`|λ_s - ⟨λ, α⟩| < tol` test the reduced dynamics already uses, applied to the
off-manifold targets. Reading it back with `resonant_multiindices` covers
**every** monomial at every order.

`ALL` outer resonances are reported, forced and autonomous alike. A monomial
built purely from master coordinates that lands on an off-manifold eigenvalue
makes the solve exactly as near-singular as a forced one; that its cause is the
chosen `master` set rather than the forcing makes it more worth surfacing, not
less. Consequently this runs for autonomous ROMs too.

The test is on frequency alone, and deliberately so. `solve_single_monomial!`
forms `s = ⟨λ, α⟩` and builds its operator from `s` only, so the conditioning is
**independent of the right-hand side** — shaping the load away from the
offending mode is no protection. Rounding in the factorisation injects a
component along the near-null direction which `1/(λ_s - s)` then amplifies, and
at higher orders the right-hand side is built from lower-order `W`'s and so is
not orthogonal to that mode even when the raw load is.

This resonance set is diagnostic only. `outer_resonances` is never read by the
cohomological solve — `_resonance_vector` builds an `SVector{ROM,Bool}` from the
inner block alone — so it cannot perturb `W` or `R`.
"""
function _warn_outer_resonances(eigenvalues::AbstractVector,
	master::Vector{Int}, master_indices::Vector{Int}, mset,
	external_eigenvalues::Vector{ComplexF64}, order::Int, resonance_tol::Real)
	outer_idx = [s for s in 1:length(eigenvalues) if !(s in master_indices)]
	isempty(outer_idx) && return nothing
	ROM = length(master_indices)
	λ_outer = ComplexF64[eigenvalues[s] for s in outer_idx]

	_, rset, _ = resonances(eigenvalues, master, order;
		external_eigenvalues = external_eigenvalues,
		outer_eigenvalues = λ_outer,
		resonance_tol = resonance_tol, mset = mset)

	# Both conjugates of a physical pair are separate outer targets; collapse them
	# so each offending pair warns once instead of twice.
	flagged = Dict{Int, Vector{Int}}()
	pair_eigenvalue = Dict{Int, ComplexF64}()
	for (j, s) in enumerate(outer_idx)
		cols = resonant_multiindices(rset, ROM + j)
		isempty(cols) && continue
		p = (s + 1) ÷ 2
		append!(get!(flagged, p, Int[]), cols)
		get!(pair_eigenvalue, p, λ_outer[j])
	end

	N_EXT = length(external_eigenvalues)
	legend = N_EXT == 0 ? "reduced coordinates 1:$ROM are the master states" :
			 "reduced coordinates 1:$ROM are master, $(ROM+1):$(ROM + N_EXT) the forcing states"
	remedy = N_EXT == 0 ? "Add pair %s to `master`, or add damping." :
			 "Add pair %s to `master`, detune Ω, or add damping."
	for p in sort!(collect(keys(flagged)))
		monomials = join([string(Tuple(mset.exponents[c]))
						  for c in sort!(unique(flagged[p]))], ", ")
		@warn """
		Monomials are near-resonant with non-master physical mode pair $p \
		(λ = $(pair_eigenvalue[p])). That mode is not on the manifold, so its \
		direction is solved through a near-singular operator and the ROM will \
		lose accuracy there regardless of how the load is shaped. Offending \
		monomial exponents ($legend): $monomials. \
		$(replace(remedy, "%s" => string(p)))"""
	end
	return nothing
end

"""
	build_model(m::AssembledMechanicalModel; master = [1], forcing = nothing,
				nev = ..., spectrum = nothing) -> (NthOrderModel, SpectralData)

Assemble the full-order model and its spectral data — the pair
`MORFE.parametrise(model, sd, expansion_order)` consumes.

This is StructuralSVK's implementation of the [`build_model`](@ref) contract, and it is
where every SVK-specific convention lives:

- **`master` lists physical mode PAIRS.** Eigenvalues come in conjugate pairs, so physical
  mode `p` occupies spectrum entries `2p-1, 2p`; `master = [1]` means `ROM = 2`. The pairs
  need not be leading or contiguous.
- **`forcing`** takes one [`HarmonicForcing`](@ref) or a vector of them, giving
  `f(t) = Σₖ aₖ · M·ϕ_{pₖ} · cos(Ωₖ t)`. Each appends its own conjugate pair of external
  states with eigenvalues `±iΩₖ`, so `N_EXT = 2·length(forcing)`. A forcing whose shape
  mode is not a master pair is allowed — it only supplies the load shape.
- **`spectrum`** reuses an already-solved [`spectrum`](@ref) instead of solving again;
  `nev` sizes the solve when one is needed.

The conjugate permutation is derived rather than written out: the master block pairs
adjacent modes, and the external block comes from the external system itself, so it stays
correct for an odd `N_EXT` or a change of external coordinates.
"""
function build_model(m::AssembledMechanicalModel;
	master::Vector{Int} = [1],
	forcing::Union{Nothing, HarmonicForcing, AbstractVector{<:HarmonicForcing}} = nothing,
	nev::Int = max(10, 2 * max(maximum(master), _max_forcing_mode(forcing)) + 4),
	spectrum = nothing,
	n_monomials::Int = 1)
	forcings = _forcings(forcing)
	@assert all(>(0), master)&&allunique(master) "master must list distinct positive mode pairs; got master = $master"
	@assert issorted(master) "master must be sorted (the ROM coordinate order follows it); got master = $master"
	@assert all(f -> f.mode > 0, forcings) "every forcing.mode must be a positive mode pair; got modes = $([f.mode for f in forcings])"
	n_modes_needed = max(maximum(master), _max_forcing_mode(forcing))
	@assert spectrum !== nothing||nev >= n_modes_needed "nev = $nev is too small for master = $master and forcing modes $([f.mode for f in forcings]): at least $n_modes_needed physical modes must be computed"

	ROM = 2 * length(master)
	N_EXT = 2 * length(forcings)
	terms = Tuple(m.term_factory(d, n_monomials) for d in m.nonlinear_degrees)

	# ── Spectrum of the autonomous operator (the forcing does not enter) ────
	eig_model = NthOrderModel((m.K, m.C, m.M), terms)
	t_eig = @elapsed sp = spectrum === nothing ?
						  MORFE.spectrum(m; nev = nev) : spectrum
	eigenvalues = sp.eigenvalues
	@assert length(eigenvalues)>=2 * n_modes_needed "the spectrum holds $(length(eigenvalues) ÷ 2) physical modes, too few for master = $master and forcing modes $([f.mode for f in forcings])"

	master_indices = reduce(vcat, [[2p - 1, 2p] for p in master])
	FOM = m.info.n_dofs

	# ── Model, with forcing if any ──────────────────────────────────────────
	# Forcing k contributes the conjugate pair of external states (+iΩₖ, −iΩₖ) at
	# reduced coordinates ROM+2k-1, ROM+2k. Both carry the same load vector, so
	# fvₖ·(r_{2k-1} + r_{2k}) reproduces aₖ · M·ϕ · cos(Ωₖ t).
	Ωs = Float64[]
	if isempty(forcings)
		model = eig_model
	else
		ext_eigs = ComplexF64[]
		force_matrix = zeros(Float64, FOM, N_EXT)   # column j ↔ external state j
		for (k, f) in enumerate(forcings)
			Ωk = f.Ω === nothing ? abs(eigenvalues[2*f.mode-1]) : Float64(f.Ω)
			fv = real((f.amplitude / 2) .* (m.M * sp.eigenmodes[:, 1, 2*f.mode-1]))
			push!(Ωs, Ωk)
			force_matrix[:, 2k-1] .= fv
			force_matrix[:, 2k] .= fv
			append!(ext_eigs, ComplexF64[im*Ωk, -im*Ωk])
		end
		# Degree-1 term: multiindex (0, 0) in the state derivatives, multiplicity 1
		# in the external state ⇒ f! signature is exactly (res, r).
		#
		# `r` reaches this closure two ways: the cohomological solve passes a UNIT
		# vector eⱼ selecting one external state, while direct evaluation passes the
		# whole external state vector. Being linear in `r` makes it correct under
		# both — but it must INDEX `r`, not sum it: `sum(r)` would drive every
		# harmonic with every load vector. Skipping zero entries also makes the
		# solve-path cost independent of N_EXT.
		force_term = MultilinearMap((res, r) -> begin
				@inbounds for j in axes(force_matrix, 2)
					iszero(r[j]) || axpy!(r[j], view(force_matrix, :, j), res)
				end
				res
			end, (0, 0), 1)
		model = NthOrderModel((m.K, m.C, m.M), (terms..., force_term),
			ExternalSystem(Tuple(ext_eigs)))
	end

	# The master block pairs adjacent modes; SpectralData extends it over the external
	# variables from the external system at solve time, so no literal is written here.
	master_block = reduce(vcat, [[2p, 2p - 1] for p in 1:length(master)])
	sd = SpectralData(model, sp; master = master_indices,
		conjugate_permutation = master_block)

	return model, sd, (; forcings = forcings, Ω = Ωs, N_EXT = N_EXT,
		eig_time_s = t_eig, master_indices = master_indices, spectrum = sp,
		ext_eigs = model.external_system === nothing ? ComplexF64[] :
				   Vector(model.external_system.eigenvalues))
end

"""
	parametrise(m::AssembledMechanicalModel; master = [1], order, forcing = nothing, …)

Compute the DPIM invariant-manifold ROM for an assembled structural model.

Thin wrapper over the unified pipeline — it exists only to attach SVK's metadata:

```julia
model, sd = build_model(m; master, forcing)
W, R      = parametrise(model, sd, order)
```

See [`build_model`](@ref) for what `master` and `forcing` mean.
"""
function parametrise(m::AssembledMechanicalModel;
	master::Vector{Int} = [1],
	order::Int,
	forcing::Union{Nothing, HarmonicForcing, AbstractVector{<:HarmonicForcing}} = nothing,
	resonance_tol::Real = 0.05,
	resonance_tol_rel::Union{Nothing, Real} = nothing,
	eigenproblem = nothing,
	nev::Int = max(10, 2 * max(maximum(master), _max_forcing_mode(forcing)) + 4),
	eigensolver = nothing)
	ROM = 2 * length(master)
	N_EXT = 2 * length(_forcings(forcing))
	mset = all_multiindices_up_to(ROM + N_EXT, order; min_degree = 1)

	model, sd, meta = build_model(m; master = master, forcing = forcing, nev = nev,
		spectrum = eigenproblem, n_monomials = length(mset))

	# Off-manifold near-resonances are reported by SVK rather than by MORFE's generic
	# `warn_outer`. The detection is identical, but the REPORTING needs conventions only
	# this backend has: eigenvalues come in conjugate pairs, so warnings are collapsed to
	# one per physical mode PAIR and named by that pair's index. MORFE's spectrum has no
	# notion of pairs and would warn twice, once per conjugate.
	_warn_outer_resonances(meta.spectrum.eigenvalues, master, meta.master_indices, mset,
		meta.ext_eigs, order, resonance_tol)

	config = ResonanceConfig(style = :complex_normal_form,
		tol = Float64(resonance_tol), tol_relative = resonance_tol_rel,
		warn_outer = false)

	t_solve = @elapsed W, R = parametrise(model, sd, mset; resonance = config)

	return InvariantManifoldROM(W, R, collect(meta.spectrum.eigenvalues), master, order,
		meta.forcings,
		(; m.info..., n_monomials = length(mset), N_EXT = meta.N_EXT, Ω = meta.Ω,
			eig_time_s = meta.eig_time_s, solve_time_s = t_solve))
end

