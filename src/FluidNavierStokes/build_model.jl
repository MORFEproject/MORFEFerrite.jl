# =====================================================================
# The `build_model` contract for FluidNavierStokes.
# =====================================================================

"""
	build_model(m::AssembledFluidModel; n_monomials = 1, nev = 40,
				sigma_re = 3.0, sigma_im = 8.0, target_freq = nothing,
				scale = 1e-2, normalisation = SymmetricBiorthogonal(),
				verbose = true) -> (; model, spectral, meta)

Turn an assembled fluid case into the first-order model and spectral data a
reduction consumes.

**`ORD = 1`.** `NthOrderModel((B₀, B₁), …)` has two linear terms, so there are no
companion derivative blocks and `SpectralData` takes plain `FOM × ROM` matrices.
Nothing needs reconciling against a higher model order — the clause in the
[`build_model`](@ref) contract about extending order-blocks is vacuous here.

The reduction is around the **Hopf pair** at `Re₀`, with the Reynolds
continuation variable `η′ = 1/Re − 1/Re₀` as a single frozen external state. That
makes `N_EXT = 1`, which is *odd*: `η′` has a real eigenvalue and is therefore its
own conjugate, a pairing the usual adjacent-pairs formula cannot express. The
permutation is built with `full_conjugate_permutation`, which handles it, and
comes out `[2, 1, 3]`.

## The gauge is explicit, because it changes what `W` and `R` mean

- `normalisation` — see [`AbstractModeNormalisation`](@ref). The historical
  default scales `ψ` and `φ` symmetrically by `1/√α`.
- `scale` — a further uniform factor on both sides, purely for conditioning.
  `1e-2` is the Kármán value, which makes `φᴴBψ = 1e-4` rather than `1`.
  `SpectralData` deliberately has no `scale` field so that such a tweak stays
  visible where it is made; this keyword is that visibility.

Raw `W`/`R` coefficients from two different gauges are **not** comparable —
compare gauge-invariant quantities instead.

`n_monomials` sizes `FluidConvection`'s batched-column cache. It is a cache
capacity, not a reduction concept: `build_model` builds no `MultiindexSet` and no
`ResonanceSet`.
"""
function build_model(m::AssembledFluidModel;
	n_monomials::Int = 1,
	nev::Int = 40,
	sigma_re::Float64 = 3.0,
	sigma_im::Float64 = 8.0,
	target_freq::Union{Nothing, Float64} = nothing,
	scale::Real = 1e-2,
	normalisation::AbstractModeNormalisation = SymmetricBiorthogonal(),
	verbose::Bool = true)

	# ── Hopf eigenproblem. A_lin = −B₀ is this module's convention, not the
	#    caller's; the mode scaling goes through the solver rather than being
	#    applied to the arrays afterwards. ─────────────────────────────────────
	t_eig = @elapsed eig = solve_hopf_eigenproblem(-m.B₀, m.B₁;
		nev = nev, sigma_re = sigma_re, sigma_im = sigma_im,
		target_freq = target_freq, normalisation = normalisation,
		scale = scale, verbose = verbose)

	# ── The model: convection, the Reynolds coupling, the base-flow forcing ──
	convection = FluidConvection(m.fom; max_unique_cols = n_monomials)
	g₁ = make_param_coupling(m.K_visc)
	h₀ = make_base_forcing(m.h₀_vec)
	ext_sys = ExternalSystem((0.0 + 0.0im,))       # η̇′ = 0 — a frozen parameter
	model = NthOrderModel((m.B₀, m.B₁), (convection, g₁, h₀), ext_sys)

	# ── Spectral data. ORD = 1 ⇒ physical slices only, no companion blocks. ──
	spectral = SpectralData(; eigenvalues = eig.eigenvalues,
		right_modes = eig.right_modes, left_modes = eig.left_modes)

	# ── Derived, never a literal: the master pair maps 1 ↔ 2, and the external
	#    block comes from the external system itself, so this stays correct if
	#    η′ ever gains a non-triangular linear part (which would re-base it). ──
	perm = full_conjugate_permutation([2, 1], ext_sys)

	# `external_system` is carried so a caller can hand it to `MORFE.save_rom`, which
	# records the external coordinates the ROM was written in.
	meta = (; conjugate_permutation = perm, spectrum = eig, external_system = ext_sys,
		N_EXT = 1, ROM = length(eig.eigenvalues), Re₀ = m.Re₀,
		scale = scale, normalisation = normalisation,
		eig_time_s = t_eig, m.info...)
	return (; model = model, spectral = spectral, meta = meta)
end
