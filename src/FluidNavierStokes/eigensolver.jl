# =====================================================================
# Shift-invert ARPACK eigensolver for the NSE descriptor system.
#
# The linearised system B₁ṡ = −B₀ s has a SINGULAR mass B₁ (pressure carries no
# time derivative), so this is a descriptor system:
#     B₁ = [M 0; 0 0]
#
# Solve A y = λ B y with A = −B₀, B = B₁ by shift-invert:
#     σ = sigma_re + im·sigma_im,  F = klu(A − σB),  T = F⁻¹B,
#     eigs(T; which = :LM) → μ,   λ = σ + 1/μ
#
# That part is standard. What is NOT standard is the GAUGE — see
# `AbstractModeNormalisation` below, which is why it is a named argument rather
# than something baked into this function.
# =====================================================================

using Arpack
using KLU
using LinearAlgebra
using LinearMaps
using Printf
using SparseArrays
using StaticArrays

"""
	AbstractModeNormalisation

How a computed eigenpair `(φ, ψ)` is scaled before it becomes `SpectralData`.

**This is a real modelling choice, not an implementation detail**, which is why
it is stated at the call site. Every option below satisfies the biorthogonality
the solve needs; they differ by a scalar gauge, and that gauge propagates into
`W` and `R`. Raw coefficients from two different gauges are not comparable —
compare gauge-invariant quantities (eigenvalues, `Im/Re` ratios) instead.
"""
abstract type AbstractModeNormalisation end

"""
	SymmetricBiorthogonal()

`α = ψᵀBφ`, then scale **both** `ψ` and `φ` by `1/√α`, giving `ψᵀBφ = 1` with
`‖ψ‖ ~ ‖φ‖`.

Splitting the scaling across both sides is deliberate: for this problem `ψ` can
be orders of magnitude larger than `φ`, and putting the whole factor on one side
leaves an ill-conditioned bordered system in the resonant solves.

This is the historical default and the gauge every archived Kármán result was
computed in — changing it re-bases the reference data.
"""
struct SymmetricBiorthogonal <: AbstractModeNormalisation end

"""
	LeftBiorthogonal()

`α = ψᵀBφ`, then scale `ψ` alone by `1/α`, leaving `φ` untouched. The
convention MORFE's own `normalise_biorthogonal!` uses.
"""
struct LeftBiorthogonal <: AbstractModeNormalisation end

"""
	NoNormalisation()

Return the eigenvectors as ARPACK produced them. `ψᵀBφ` is then whatever it is,
and the caller is responsible for the biorthogonality the solve assumes.
"""
struct NoNormalisation <: AbstractModeNormalisation end

function _normalise_pair(::SymmetricBiorthogonal, φ, ψ, α)
	s = sqrt(α)
	return φ ./ s, ψ ./ s
end
_normalise_pair(::LeftBiorthogonal, φ, ψ, α) = (φ, ψ ./ α)
_normalise_pair(::NoNormalisation, φ, ψ, α) = (φ, ψ)

"""
	solve_hopf_eigenproblem(A_lin, B_mass; nev, sigma_re, sigma_im,
							target_freq = nothing,
							normalisation = SymmetricBiorthogonal(),
							scale = 1.0, verbose = true)
		-> (; eigenvalues, right_modes, left_modes, all_eigenvalues, all_modes)

Compute `nev` eigenvalues of `A_lin y = λ B_mass y` by shift-invert ARPACK and
return the Hopf conjugate pair with its adjoint.

`sigma_re` offsets the shift from the imaginary axis; `sigma_im` targets a
frequency band. Neither affects which mode is selected — only the factorisation.

The Hopf mode is the eigenvalue with the smallest `|Re λ|` among those with
`Im λ > 0`. That heuristic is reliable **near** `Re_c`, where the shedding mode
IS the least damped; away from it another oscillatory mode can sit closer to the
imaginary axis and be picked silently, so pass `target_freq` (rad/s) to pin the
frequency instead.

## The two gauge choices, both explicit

- `normalisation` — see [`AbstractModeNormalisation`](@ref). Defaults to the
  historical [`SymmetricBiorthogonal`](@ref).
- `scale` — a further uniform factor applied to **both** sides, purely for
  conditioning. The Kármán case uses `1e-2`, which makes `φᴴBψ = 1e-4` rather
  than `1`. `SpectralData` deliberately has no `scale` field so that such a
  tweak stays visible where it is made; this keyword is that visibility.

Returns a `NamedTuple` so the fields are named at every call site: `eigenvalues`
is the `SVector{2}` Hopf pair (`Im λ₁ > 0`), `right_modes` and `left_modes` are
`n × 2`, and `all_*` carry the positive-imaginary half of the spectrum sorted by
`|Re λ|`.
"""
function solve_hopf_eigenproblem(
	A_lin::AbstractSparseMatrix,
	B_mass::AbstractSparseMatrix;
	nev::Int,
	sigma_re::Float64,
	sigma_im::Float64,
	target_freq::Union{Nothing, Float64} = nothing,
	normalisation::AbstractModeNormalisation = SymmetricBiorthogonal(),
	scale::Real = 1.0,
	verbose::Bool = true,
)
	n = size(A_lin, 1)
	sigma = complex(sigma_re, sigma_im)
	verbose && println("  Shift σ = $sigma,  nev = $nev,  n = $n")

	# ── Shift-invert factorisation ─────────────────────────────────────────────
	Ac = complex.(A_lin)
	Bc = complex.(B_mass)
	F = klu(Ac - sigma * Bc)
	LM = LinearMap{ComplexF64}(n, n; ismutating = false) do x
		F \ (Bc * x)
	end
	mu, vecs, = eigs(LM; nev = nev, which = :LM, maxiter = 3000,
		ncv = min(max(nev + 30, 120), n - 1))

	# Guard against zero mu (spurious pressure modes of the descriptor system).
	tiny = eps(Float64)
	mu_safe = map(m -> abs(m) < tiny ? complex(tiny) : m, mu)
	vals = sigma .+ inv.(mu_safe)

	if verbose
		println()
		@printf("  %3s   %-14s  %-14s  %-12s\n", "#", "Re(λ)", "Im(λ)", "|λ|")
		println("  " * "─"^52)
		for (i, λ) in enumerate(vals)
			@printf("  %3d   %+12.6f  %+12.6f  %12.6f\n", i, real(λ), imag(λ), abs(λ))
		end
		println()
	end

	# ── Select the Hopf mode ───────────────────────────────────────────────────
	# A_lin and B_mass are real, so eigenpairs come in exact conjugate pairs.
	hopf_tol = 0.1
	cand = findall(λ -> imag(λ) > hopf_tol, vals)
	isempty(cand) && error("No eigenvalue with Im(λ) > $hopf_tol found; increase nev.")
	i_best = target_freq === nothing ?
			 cand[argmin(abs(real(vals[i])) for i in cand)] :
			 cand[argmin(abs(imag(vals[i]) - target_freq) for i in cand)]
	λ₁ = vals[i_best]
	φ₁ = vecs[:, i_best]
	verbose && @printf("  Selected Hopf mode:  λ₁ = %+.6f %+.6f·i  (ω_c = %.4f rad/s)\n",
		real(λ₁), imag(λ₁), imag(λ₁))

	# ── Left eigenvector (adjoint shift-invert) ───────────────────────────────
	# Shift at λ₁, NOT conj(λ₁): AᵀΓ − λ₁Bᵀ is singular there, so ARPACK returns
	# ψ₁. Shifting at conj(λ₁) would find ψ₂ instead, making ψᵀBφ₁ = 0.
	F_adj = klu(Ac' - λ₁ * Bc')
	LM_adj = LinearMap{ComplexF64}(n, n; ismutating = false) do x
		F_adj \ (Bc' * x)
	end
	_, xl, = eigs(LM_adj; nev = 1, which = :LM, maxiter = 3000, ncv = min(60, n - 1))
	ψ₁ = xl[:, 1]

	# ── Gauge: the caller's normalisation, then the caller's scale ─────────────
	α = transpose(ψ₁) * (Bc * φ₁)
	φ₁, ψ₁ = _normalise_pair(normalisation, φ₁, ψ₁, α)
	if scale != 1
		φ₁ = φ₁ .* scale
		ψ₁ = ψ₁ .* scale
	end

	# ψ₁ is the BILINEAR left vector (ψ₁ᵀ(A − λ₁B) = 0). The orthogonality
	# equations use the sesquilinear convention φᴴ(λB − A) = 0, which for the real
	# matrices here means conj(ψ). Passing ψ un-conjugated would pair each border
	# row with the WRONG mode of the conjugate pair — cross-mode pairing ≈ 0, hence
	# near-singular resonant solves.
	eigenvalues = SVector{2, ComplexF64}(λ₁, conj(λ₁))
	right_modes = hcat(φ₁, conj.(φ₁))
	left_modes = hcat(conj.(ψ₁), ψ₁)

	# ── All modes (positive-imaginary half, most weakly damped first) ─────────
	all_idx = sort(findall(λ -> imag(λ) > hopf_tol, vals); by = i -> abs(real(vals[i])))
	if verbose
		println("  Returning $(length(all_idx)) modes (Im > $hopf_tol, sorted by |Re(λ)|):")
		for (k, i) in enumerate(all_idx)
			@printf("    mode %2d:  λ = %+.6f %+.6f·i\n", k, real(vals[i]), imag(vals[i]))
		end
	end

	return (; eigenvalues = eigenvalues, right_modes = right_modes,
		left_modes = left_modes, all_eigenvalues = vals[all_idx],
		all_modes = vecs[:, all_idx])
end
