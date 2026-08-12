# ── The `build_model` contract for StructuralSVK ────────────────────────────
#
# This file holds the ONE `build_model` method and the forcing helpers it needs.
# The reduction sugar that calls it lives in `parametrise.jl`.

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

# Number of multiindices in `nvar` variables of total degree 1…order — the size of
# `all_multiindices_up_to(nvar, order; min_degree = 1)` WITHOUT building it. Counting
# keeps `build_model` free of `MultiindexSet`, which the contract requires.
_n_multiindices(nvar::Int, order::Int) = binomial(nvar + order, order) - 1

"""
	build_model(m::AssembledMechanicalModel; master = [1], forcing = nothing,
				nev = ..., spectrum = nothing) -> (; model, spectral, meta)

Assemble the full-order model and its spectral data — what
`MORFE.parametrise(model, spectral, expansion_order)` consumes.

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
	expansion_order::Union{Nothing, Int} = nothing,
	n_monomials::Union{Nothing, Int} = nothing)
	forcings = _forcings(forcing)
	@assert all(>(0), master)&&allunique(master) "master must list distinct positive mode pairs; got master = $master"
	@assert issorted(master) "master must be sorted (the ROM coordinate order follows it); got master = $master"
	@assert all(f -> f.mode > 0, forcings) "every forcing.mode must be a positive mode pair; got modes = $([f.mode for f in forcings])"
	n_modes_needed = max(maximum(master), _max_forcing_mode(forcing))
	@assert spectrum !== nothing||nev >= n_modes_needed "nev = $nev is too small for master = $master and forcing modes $([f.mode for f in forcings]): at least $n_modes_needed physical modes must be computed"

	ROM = 2 * length(master)
	N_EXT = 2 * length(forcings)
	# Sizes the FEM terms' batched-column cache. `expansion_order` COUNTS the monomials
	# of a total-degree truncation rather than building the set — `build_model` must not
	# construct a `MultiindexSet`. A caller with a custom set passes `n_monomials` instead.
	n_cols = something(n_monomials,
		expansion_order === nothing ? 1 : _n_multiindices(ROM + N_EXT, expansion_order))
	terms = Tuple(m.term_factory(d, n_cols) for d in m.nonlinear_degrees)

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

	# The SPECTRUM-WIDE involution, not just the master block. The eigensolvers used here
	# return conjugate pairs in adjacent entries, so `2p ↔ 2p-1` is exactly true for every
	# entry — and stating it for the outer modes too is what lets MORFE number physical
	# modes rather than eigenvalues, so its outer-resonance warning names a mode PAIR and
	# fires once per pair. The master restriction MORFE derives from this is identical to
	# the ROM-length block, so the solve is unchanged.
	# SpectralData extends the pairing over the external variables from the external system
	# at solve time, so no literal is written here.
	n_pairs = length(sp.eigenvalues) ÷ 2
	σ = reduce(vcat, [[2p, 2p - 1] for p in 1:n_pairs])
	sd = SpectralData(model, sp; master = master_indices, conjugate_permutation = σ)

	# `case_info` and `external_system` are carried so a caller can package an
	# `InvariantManifoldROM` and call `MORFE.save_rom` without holding the case.
	meta = (; forcings = forcings, Ω = Ωs, N_EXT = N_EXT, n_monomials = n_cols,
		eig_time_s = t_eig, master_indices = master_indices, spectrum = sp,
		case_info = m.info, external_system = model.external_system,
		ext_eigs = model.external_system === nothing ? ComplexF64[] :
				   Vector(model.external_system.eigenvalues))
	return (; model = model, spectral = sd, meta = meta)
end
