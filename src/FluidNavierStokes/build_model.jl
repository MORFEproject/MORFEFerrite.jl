# =====================================================================
# The `build_model` contract for FluidNavierStokes.
# =====================================================================

"""
	build_model(m::AssembledFluidModel, eig; master, outer = Int[],
				n_monomials = 1, conjugate_permutation = nothing)
		-> (; model, spectral, meta)

Turn an assembled fluid case and a solved eigenproblem into the first-order model
and spectral data a reduction consumes.

**It makes no choice about modes.** `master` and `outer` are index vectors into
`eig.eigenvalues`, supplied by the caller — this method does not select, filter,
rank or detect anything. Which modes span the manifold and which are off-manifold
targets is a modelling decision, and it belongs in the driver where it can be read
and changed, not behind a default here.

`eig` is whatever [`solve_hopf_eigenproblem`](@ref) returned — eigenvalues and right
modes for the whole computed spectrum. Left eigenvectors are computed **here**, for
the master modes only, because each one costs its own adjoint factorisation (see
[`left_eigenvector`](@ref)) and outer modes need eigenvalues alone.

`scale` and `normalisation` set the mode gauge. They change what `W` and `R` mean —
coefficients from two gauges are not comparable — so they are keywords rather than
constants, and are recorded in `meta`.

**`ORD = 1`.** `NthOrderModel((B₀, B₁), …)` has two linear terms, so there are no
companion derivative blocks and `SpectralData` takes plain `FOM × ROM` matrices.
Nothing needs reconciling against a higher model order.

The Reynolds continuation variable `η′ = 1/Re − 1/Re₀` is a single frozen external
state, so `N_EXT = 1` — *odd*. `η′` is real and therefore its own conjugate, a
pairing the usual adjacent-pairs formula cannot express, so the permutation is
built with `full_conjugate_permutation`.

`conjugate_permutation` overrides the derived master-block pairing. Give it when
the master set is not a plain sequence of adjacent conjugate pairs — for instance
when real (self-conjugate) modes have been included alongside a Hopf pair.

`n_monomials` sizes `FluidConvection`'s batched-column cache. It is a cache
capacity, not a reduction concept: `build_model` builds no `MultiindexSet` and no
`ResonanceSet`.
"""
function build_model(m::AssembledFluidModel, eig;
	master::AbstractVector{Int},
	outer::AbstractVector{Int} = Int[],
	n_monomials::Int = 1,
	scale::Real = 1.0,
	normalisation::AbstractModeNormalisation = SymmetricBiorthogonal(),
	conjugate_permutation = nothing,
	conjugate_atol::Real = 1e-4)

	allunique(master) || throw(ArgumentError("master indices must be distinct, got $master"))
	isempty(intersect(master, outer)) || throw(ArgumentError(
		"a mode cannot be both master and outer; overlap = $(intersect(master, outer))"))
	nλ = length(eig.eigenvalues)
	all(i -> 1 <= i <= nλ, master) || throw(ArgumentError(
		"master indices $master out of range 1:$nλ"))
	all(i -> 1 <= i <= nλ, outer) || throw(ArgumentError(
		"outer indices $outer out of range 1:$nλ"))

	ROM = length(master)

	# ── The model: convection, the Reynolds coupling, the base-flow forcing ──
	convection = FluidConvection(m.fom; max_unique_cols = n_monomials)
	g₁ = make_param_coupling(m.K_visc)
	h₀ = make_base_forcing(m.h₀_vec)
	ext_sys = ExternalSystem((0.0 + 0.0im,))       # η̇′ = 0 — a frozen parameter
	model = NthOrderModel((m.B₀, m.B₁), (convection, g₁, h₀), ext_sys)

	# ── Spectral data. ORD = 1 ⇒ physical slices only, no companion blocks. ──
	# The outer eigenvalues are carried so off-manifold resonance can be detected
	# (`outer_targets = true` in the `ResonanceConfig`), not merely warned about.
	λ_master = collect(ComplexF64, eig.eigenvalues[master])
	λ_outer = collect(ComplexF64, eig.eigenvalues[outer])

	# The pairing is needed BEFORE the modes, because it decides which of them are
	# solved for and which are conjugated from a partner.
	σ_master = conjugate_permutation === nothing ?
			   _master_conjugate_pairing(λ_master; atol = conjugate_atol) :
			   collect(Int, conjugate_permutation)

	# ── Master modes, conjugate-symmetric BY CONSTRUCTION ────────────────────
	# `conjugate_permutation` asserts `modes[:, σ(r)] = conj(modes[:, r])`, and MORFE
	# states plainly that eigenvalue conjugacy does NOT imply it. So the partner of a
	# pair is CONJUGATED here, never solved for separately: an independent adjoint
	# solve pins its left vector only up to a scalar `c`, and after the `1/√α` gauge
	# that leaves `Ψ[:, σ(r)] = √c · conj(Ψ[:, r])`. ARPACK returns unit-norm vectors,
	# so `√c` is a pure PHASE — moduli survive, phases rotate, and the solve then
	# exploits a symmetry that is not there. It is silent, and it corrupts every
	# nonlinear coefficient while leaving λ untouched.
	#
	# Conjugating also halves the adjoint factorisations: one per conjugate pair.
	Φ = Matrix{ComplexF64}(undef, size(eig.right_modes, 1), ROM)
	Ψ = similar(Φ)
	pairings = fill(ComplexF64(NaN), ROM)
	for r in 1:ROM
		σ_master[r] < r && continue                 # already filled by its partner
		k = master[r]
		φ, ψ, α = left_eigenvector(-m.B₀, m.B₁, eig.eigenvalues[k], eig.right_modes[:, k];
			normalisation = normalisation, scale = scale)
		if σ_master[r] == r
			φ, ψ = _realify_self_conjugate(φ, ψ, r, eig.eigenvalues[k])
		end
		Φ[:, r] = φ
		Ψ[:, r] = ψ
		pairings[r] = α
		if σ_master[r] > r                          # the partner, by conjugation
			Φ[:, σ_master[r]] = conj.(φ)
			Ψ[:, σ_master[r]] = conj.(ψ)
			pairings[σ_master[r]] = conj(α)
		end
	end
	# The invariant the whole conjugate-symmetric solve rests on. Silent when violated,
	# which is exactly why it is checked rather than assumed.
	for r in 1:ROM
		s = σ_master[r]
		@assert Φ[:, s]≈conj.(Φ[:, r]) "master right modes violate conjugate symmetry at ($r, $s)"
		@assert Ψ[:, s]≈conj.(Ψ[:, r]) "master left modes violate conjugate symmetry at ($r, $s)"
	end

	# The outer block is left self-paired. `detect_conjugate_permutation` over the
	# whole selection is NOT used: a complex shift computes only the modes near σ, so
	# a complex mode's conjugate is generally absent from the outer set and detection
	# fails over the lot — which previously fell back to the identity, quietly claiming
	# the Hopf pair was two unrelated modes.
	σ_full = vcat(σ_master, collect((ROM + 1):(ROM + length(λ_outer))))

	spectral = SpectralData(; eigenvalues = λ_master,
		right_modes = Φ, left_modes = Ψ,
		outer_eigenvalues = λ_outer,
		conjugate_permutation = σ_full === nothing ? σ_master : σ_full)

	# ── Derived, never a literal: the master block from the pairing above, the
	#    external block from the external system itself. ─────────────────────
	master_block = σ_master === nothing ? collect(1:ROM) : σ_master
	perm = full_conjugate_permutation(master_block, ext_sys)

	meta = (; conjugate_permutation = perm, spectrum = eig, external_system = ext_sys,
		master = collect(Int, master), outer = collect(Int, outer),
		bilinear_pairings = pairings, scale = scale, normalisation = normalisation,
		N_EXT = 1, ROM = ROM, Re₀ = m.Re₀, m.info...)
	return (; model = model, spectral = spectral, meta = meta)
end

"""
	_master_conjugate_pairing(λ; atol = 1e-8) -> Vector{Int}

The involution `σ` over a master set: `σ[r] = s` where `λ[s] = conj(λ[r])`.

Two cases, and the first is the one that makes a mode on the real axis usable at
all:

- **`λ[r]` is real** ⇒ it is its own conjugate, so `σ[r] = r`. A non-oscillatory
  mode carries a single real coordinate rather than half of a complex pair.
- **`λ[r]` is complex** ⇒ its partner must also be in the master set. If it is not,
  that is an error rather than a fallback: a manifold spanned by one half of a
  conjugate pair is not invariant under conjugation, and the resulting ROM has no
  real realisation.

This is constructed, not detected. Detection over the selected modes fails here
because the shift-invert eigensolve uses a COMPLEX shift σ and therefore returns
only the modes near σ — a strongly oscillatory mode's conjugate sits near `conj(σ)`
and is simply not computed. Those conjugates are synthesised analytically by the
caller (`λ̄, φ̄`), and this function then pairs them.
"""
function _master_conjugate_pairing(λ::AbstractVector; atol::Real = 1e-8)
	n = length(λ)
	σ = zeros(Int, n)
	for r in 1:n
		# RELATIVE test. ARPACK's numerical zero is ~1e-7 of the magnitude, not
		# machine epsilon, so an absolute threshold mistakes a real mode for a complex
		# one and then demands a conjugate that does not exist.
		if abs(imag(λ[r])) <= atol * abs(λ[r])
			σ[r] = r                                   # real ⇒ unpaired, self-conjugate
			continue
		end
		s = argmin(abs.(λ .- conj(λ[r])))
		abs(λ[s] - conj(λ[r])) <= atol * abs(λ[r]) || throw(ArgumentError(
			"master mode $r has λ = $(λ[r]) but its conjugate is not in the master " *
			"set (nearest is mode $s, λ = $(λ[s])). A conjugate pair must be included " *
			"whole: include conj(λ) with the conjugated eigenvector, or drop mode $r."))
		σ[r] = s
	end
	all(r -> σ[σ[r]] == r, 1:n) || throw(ArgumentError(
		"the derived conjugate pairing $σ is not an involution"))
	return σ
end

"""
	_realify_self_conjugate(φ, ψ, r, λ; rtol = 1e-6) -> (φ_real, ψ_real)

Rotate a self-paired mode's eigenvectors onto the real axis.

A mode with `σ[r] = r` is its own conjugate, which asserts `modes[:, r] =
conj(modes[:, r])` — the eigenvector must be **real-valued**, not merely have a real
eigenvalue. An eigensolver returns `e^{iθ}·(real vector)` for an arbitrary `θ`, and a
biorthogonal gauge adds more phase, so the assertion generally fails on arrival even
though the mode is perfectly good.

The phase is removed by rotating against the largest component (largest, so the
reference entry is well away from a node of the mode). What remains must be real to
round-off; if it is not, the eigenvalue is not truly real or the mode is not simple,
and that is reported rather than absorbed — a defective mode used as a master
coordinate corrupts the whole reduction silently.
"""
function _realify_self_conjugate(φ::AbstractVector, ψ::AbstractVector, r::Int, λ;
	rtol::Real = 1e-6)
	out = map((φ, ψ)) do v
		p = v[argmax(abs.(v))]
		w = v .* conj(p / abs(p))                  # unit-modulus rotation
		nr = norm(real.(w))
		norm(imag.(w))<=rtol * max(nr, eps()) || throw(ArgumentError(
			"master mode $r is self-paired (λ = $λ, treated as real) but its eigenvector " *
			"is not real after phase removal: ‖Im‖/‖Re‖ = $(norm(imag.(w)) / max(nr, eps())). " *
			"Either the eigenvalue is not real (loosen nothing — check REAL_MODE_RTOL), " *
			"or the mode is degenerate and cannot carry a single real coordinate."))
		complex.(real.(w))
	end
	return out[1], out[2]
end
