# =====================================================================
# The `build_model` contract for ParametricGeometry — physics-blind.
# =====================================================================

using SparseArrays: spzeros
using MORFE: NthOrderModel, ExternalSystem, SpectralData, full_conjugate_permutation

"""
	build_model(m::AssembledParametricModel; master, spectrum, …)
		-> (; model, spectral, meta)

Assemble the augmented full-order model and its spectral data for a parametric
problem.

The θ parameters enter as **frozen external states**: `N_EXT = n_geometry_parameters` external
variables with eigenvalue `0` (`θ̇ = 0`), so a monomial `z^a θ^b` in the reduced
dynamics is a genuine parameter dependence rather than a transient.

What this method does, and what it deliberately leaves to the caller:

- Zero-pads the base physics' linear terms to [`model_order`](@ref), because a
  correction on the highest derivative (a parametric mass) only exists on a
  model one order higher.
- Collects every θ ≠ 0 correction of every operator, plus the nonlinear maps.
- Reconciles the spectral data against the augmented model's order with
  `SpectralData(model, spectrum; master)`. The base eigenproblem is second-order
  while the model is third — this is exactly the reconciliation MORFE owns, and
  hand-rolling it (multiplying the last block by `λ` versus forming a fresh
  `λ^{k-1}ψ`) is silently wrong.
- Derives the conjugate permutation with `full_conjugate_permutation`, never a
  literal — the θ states are real, hence self-conjugate, so an odd `n_geometry_parameters`
  works without a special case.
- Does **not** build a `MultiindexSet`. The θ-box truncation is a modelling
  choice; pass your own `mset` to `parametrise`.

`master` lists physical mode PAIRS, as elsewhere: pair `p` occupies spectrum
entries `2p-1, 2p`.
"""
function build_model(m::AssembledParametricModel;
	master::Vector{Int} = [1],
	spectrum,
	conjugate_permutation = nothing,
	diagnostics::Bool = true,
	diagnostics_rtol::Real = 1e-12)
	@assert all(>(0), master)&&allunique(master) "master must list distinct positive mode pairs; got master = $master"
	@assert issorted(master) "master must be sorted (the ROM coordinate order follows it); got master = $master"

	# Measured, not assumed: which θ-multiindices carry nothing. Silent when all do.
	diagnostics && report_zero_coefficients(m; rtol = diagnostics_rtol)

	b = basis(m.pd)
	sp = spectrum
	ORD = model_order(m)
	n_free = m.pd.n_free
	N_EXT = m.n_geometry_parameters

	# ── Linear terms: base coefficients, zero-padded to the augmented order ──
	base_terms = [op.arrays[1] for op in m.operators]
	@assert length(base_terms)<=ORD "got $(length(base_terms)) linear operators but ORD = $ORD"
	ZERO = spzeros(eltype(base_terms[1]), n_free, n_free)
	linear_terms = Tuple(vcat(base_terms, fill(ZERO, ORD + 1 - length(base_terms))))

	# ── Nonlinear terms: the θ-expanded forms, then every α ≠ 0 correction ──
	terms = Any[]
	for (pm, arity) in zip(m.maps, m.map_arities)
		append!(terms, multilinear_maps(pm; arity = arity))
	end
	for op in m.operators
		append!(terms, build_linear_corrections(op.arrays, b, op.arity))
	end

	# ── The frozen θ states ────────────────────────────────────────────────
	ext_sys = ExternalSystem(ntuple(_ -> complex(0.0, 0.0), N_EXT))
	model = NthOrderModel(linear_terms, Tuple(terms), ext_sys)

	# ── Spectral data, reconciled against the AUGMENTED order ──────────────
	master_indices = reduce(vcat, [[2p - 1, 2p] for p in master])
	n_pairs = length(sp.eigenvalues) ÷ 2
	σ = reduce(vcat, [[2p, 2p - 1] for p in 1:n_pairs])
	sd = SpectralData(model, sp; master = master_indices, conjugate_permutation = σ)

	# ── The NVAR-length permutation, derived. It spans the external states, so
	#    it goes to `parametrise`, which uses it verbatim — not to SpectralData,
	#    which validates a master-block or spectrum-wide vector. ─────────────
	master_block = reduce(vcat, [[2p, 2p - 1] for p in 1:length(master)])
	perm = conjugate_permutation === nothing ?
		   full_conjugate_permutation(master_block, ext_sys) :
		   collect(Int, conjugate_permutation)

	meta = (; conjugate_permutation = perm, master_indices = master_indices,
		N_EXT = N_EXT, n_geometry_parameters = m.n_geometry_parameters, ORD = ORD, spectrum = sp,
		n_terms = length(terms), geometry_parameter_terms = nterms(b))
	return (; model = model, spectral = sd, meta = meta)
end
