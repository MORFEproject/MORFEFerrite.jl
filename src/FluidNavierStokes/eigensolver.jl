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
							scale = 1.0, tol = 0.0, maxiter = 3000,
							ncv = nothing, close_conjugates = true,
							conjugate_rtol = 1e-4, verbose = true)
		-> (; eigenvalues, right_modes, hopf_index, conjugate_index)

Compute eigenvalues of `A_lin y = λ B_mass y` by shift-invert ARPACK, close the result
under conjugation, and point at the Hopf pair.

`sigma_re` offsets the shift from the imaginary axis; `sigma_im` targets a
frequency band. Neither affects which mode is selected — only the factorisation.
`tol`, `maxiter`, and `ncv` are forwarded to ARPACK. Their defaults preserve
the historical solver configuration.

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

## The spectrum comes back closed under conjugation

The shift `σ` is **complex**, so ARPACK returns only the modes near `σ` — a strongly
oscillatory mode's conjugate sits near `σ̄` and is never computed. The result is
therefore passed through [`close_under_conjugation`](@ref) before it is returned, which
appends the missing halves analytically (exact, because `A_lin` and `B_mass` are real)
and makes `conjugate_index` name the true partner of `hopf_index`.

That is on by default because the raw `conjugate_index` is a footgun: it is an `argmin`
over what ARPACK happened to return, so it names the *nearest available* mode, and
handing that pair to `build_model` as `master` throws. Two consequences worth stating:

- **`nev` is a lower bound on `length(eigenvalues)`**, not the count. Closure appends.
- Appended conjugates go at the **end**, not next to their partners, so the result is
  not a sequence of adjacent pairs.

`close_conjugates = false` returns exactly what ARPACK produced. Pass it when a caller
must post-process the raw modes before closing — for instance one that phase-aligns a
tracked eigenvector and needs the synthesised partner to inherit that phase.
`conjugate_rtol` is forwarded as `close_under_conjugation`'s `rtol`; it is relative
because ARPACK's numerical zero is ~1e-7 of a mode's magnitude, not machine epsilon.

Returns a `NamedTuple` so the fields are named at every call site: `eigenvalues` is a
`Vector{ComplexF64}` in ARPACK order with any synthesised conjugates appended,
`right_modes` is the matching `n × length(eigenvalues)` matrix, and `hopf_index` /
`conjugate_index` locate the Hopf pair within them.
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
	tol::Real = 0.0,
	maxiter::Int = 3000,
	ncv::Union{Nothing, Int} = nothing,
	close_conjugates::Bool = true,
	conjugate_rtol::Real = 1e-4,
	verbose::Bool = true,
)
	n = size(A_lin, 1)
	0 <= tol < 1 || throw(ArgumentError("tol must satisfy 0 <= tol < 1"))
	maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
	nev > 0 || throw(ArgumentError("nev must be positive"))
	nev < n || throw(ArgumentError("nev must be smaller than the matrix dimension"))
	ncv_used = isnothing(ncv) ? min(max(nev + 30, 120), n - 1) : Int(ncv)
	nev + 1 <= ncv_used <= n - 1 || throw(ArgumentError(
		"ncv must satisfy nev + 1 <= ncv <= n - 1"))
	sigma = complex(sigma_re, sigma_im)
	verbose && println("  Shift σ = $sigma,  nev = $nev,  ncv = $ncv_used,  n = $n")

	# ── Shift-invert factorisation ─────────────────────────────────────────────
	Ac = complex.(A_lin)
	Bc = complex.(B_mass)
	F = klu(Ac - sigma * Bc)
	LM = LinearMap{ComplexF64}(n, n; ismutating = false) do x
		F \ (Bc * x)
	end
	mu, vecs, = eigs(LM; nev = nev, which = :LM, maxiter, ncv = ncv_used,
		tol = Float64(tol))

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

	# ── Report the Hopf mode, but do NOT select on the caller's behalf ────────
	# Which modes are master and which are outer is the caller's choice; this only
	# points at the least-damped oscillatory mode, the usual one to want. The
	# heuristic is reliable near Re_c; away from it another oscillatory mode can sit
	# closer to the imaginary axis, so `target_freq` pins the frequency instead.
	hopf_tol = 0.1
	cand = findall(λ -> imag(λ) > hopf_tol, vals)
	isempty(cand) && error("No eigenvalue with Im(λ) > $hopf_tol found; increase nev.")
	i_hopf = target_freq === nothing ?
			 cand[argmin(abs(real(vals[i])) for i in cand)] :
			 cand[argmin(abs(imag(vals[i]) - target_freq) for i in cand)]
	i_conj = argmin(abs.(vals .- conj(vals[i_hopf])))

	if verbose
		@printf("  Hopf mode: index %d,  λ = %+.6f %+.6f·i  (ω_c = %.4f rad/s); conjugate at %d\n",
			i_hopf, real(vals[i_hopf]), imag(vals[i_hopf]), imag(vals[i_hopf]), i_conj)
		n_real = count(λ -> abs(imag(λ)) <= hopf_tol, vals)
		@printf("  %d modes computed, %d on/near the real axis (|Im λ| ≤ %.2g)\n",
			length(vals), n_real, hopf_tol)
	end

	# NOTE: no left eigenvectors here. They cost one adjoint factorisation EACH (see
	# `left_eigenvector`), so they are computed only for the modes a caller actually
	# puts in the master set — never for the whole spectrum.
	raw = (; eigenvalues = vals, right_modes = vecs,
		hopf_index = i_hopf, conjugate_index = i_conj)
	close_conjugates || return raw

	# The complex shift returns half of each conjugate pair, so `i_conj` above is an argmin
	# over the wrong candidate set. Closing here rather than leaving it to the caller is the
	# difference between a usable master pair and one `build_model` rejects.
	closed = close_under_conjugation(raw; rtol = conjugate_rtol)
	if verbose
		n_added = length(closed.eigenvalues) - length(vals)
		@printf("  %d analytic conjugates appended; conjugate of the Hopf mode at %d\n",
			n_added, closed.conjugate_index)
	end
	return closed
end

"""
	close_under_conjugation(eig; rtol = 1e-4)
		-> (; eigenvalues, right_modes, hopf_index, conjugate_index)

Complete a computed spectrum with the conjugates the eigensolve could not return, and
report where the Hopf pair ended up.

[`solve_hopf_eigenproblem`](@ref) shifts at a **complex** `σ`, so ARPACK returns only the
modes near `σ`. A strongly oscillatory mode's conjugate sits near `σ̄` and is simply never
computed — at `Re₀ = 49.03` the Kármán mode `λ = 0.004 + 16.859i` comes back while
`λ̄ = 0.004 − 16.859i` does not. `conjugate_index` as returned by the eigensolve is then the
*nearest available* eigenvalue rather than the true partner, and passing that pair as
`master` makes `build_model` throw.

The missing halves are synthesised rather than solved for. `B₀` and `B₁` are real, so
`(λ̄, φ̄)` is an eigenpair exactly whenever `(λ, φ)` is — this is an identity, not an
approximation, and it costs nothing. It is also the only way to get the partner with the
phase `conjugate_permutation` asserts: an independent solve would pin it only up to a
scalar.

Three cases per computed mode:

- `|Im λ| ≤ rtol·|λ|` — the mode is **real**, hence its own conjugate. Nothing is added;
  giving it a partner would duplicate the coordinate.
- the conjugate is already in the set — nothing is added.
- otherwise `conj(λ)` and `conj.(φ)` are appended.

`rtol` is **relative**. ARPACK's numerical zero is ~1e-7 of a mode's magnitude, not machine
epsilon, so an absolute threshold reads a real mode as complex and then demands a conjugate
that does not exist.

`hopf_index` is carried through unchanged — the closure only appends — and
`conjugate_index` is recomputed afterwards, so it now names the true partner.
"""
function close_under_conjugation(eig; rtol::Real = 1e-4)
	λ = collect(ComplexF64, eig.eigenvalues)
	Φ = Matrix{ComplexF64}(eig.right_modes)
	for k in eachindex(eig.eigenvalues)
		abs(imag(λ[k])) <= rtol * abs(λ[k]) && continue
		any(l -> abs(l - conj(λ[k])) <= rtol * abs(λ[k]), λ) && continue
		push!(λ, conj(λ[k]))
		Φ = hcat(Φ, conj.(eig.right_modes[:, k]))
	end
	i_hopf = eig.hopf_index
	i_conj = argmin(abs.(λ .- conj(λ[i_hopf])))
	return (; eigenvalues = λ, right_modes = Φ,
		hopf_index = i_hopf, conjugate_index = i_conj)
end

"""
	left_eigenvector(A_lin, B_mass, λ, φ; normalisation, scale)
		-> (φ_gauged, ψ_gauged, α)

The left eigenvector for ONE mode, by adjoint shift-invert at that mode's own `λ`,
biorthonormalised against its right partner `φ`.

**One factorisation per mode, and there is no cheaper correct way.** A single
adjoint solve at the common shift σ returns its own subset of the spectrum in its
own order; matching those to the right modes by eigenvalue pairs most of them with
the wrong partner, which shows up as `ψᵀBφ ≈ 0` — the pairing is degenerate and the
mode is unusable as a master. Shifting at `λ` makes `Aᵀ − λBᵀ` singular exactly
there, so ARPACK returns the partner that belongs to `φ`.

For the same reason the shift is `λ` and **not** `conj(λ)`: the latter finds the
conjugate mode's left vector instead, and the pairing collapses again.

Returns the gauged pair and the raw bilinear pairing `α = ψᵀBφ` before scaling, so
a caller can see how well-conditioned the mode was.

!!! warning "Do not call this for both halves of a conjugate pair"
    `eigs` pins `ψ` only up to a scalar, so two independent calls give
    `ψ_partner = c · conj(ψ)` for an arbitrary `c` — and after the `1/√α` gauge the
    pair no longer satisfies `modes[:, σ(r)] = conj(modes[:, r])`, which is what
    `conjugate_permutation` asserts. Solve ONE half and conjugate the other, as
    `build_model` does.
"""
function left_eigenvector(A_lin::AbstractSparseMatrix, B_mass::AbstractSparseMatrix,
	λ::Number, φ::AbstractVector;
	normalisation::AbstractModeNormalisation = SymmetricBiorthogonal(),
	scale::Real = 1.0)
	n = size(A_lin, 1)
	Ac = complex.(A_lin)
	Bc = complex.(B_mass)
	F_adj = klu(Ac' - ComplexF64(λ) * Bc')
	LM_adj = LinearMap{ComplexF64}(n, n; ismutating = false) do x
		F_adj \ (Bc' * x)
	end
	_, xl, = eigs(LM_adj; nev = 1, which = :LM, maxiter = 3000, ncv = min(60, n - 1))
	ψ = xl[:, 1]

	α = transpose(ψ) * (Bc * φ)
	abs(α) > 1e-10 || @warn """
	Degenerate bilinear pairing ψᵀBφ = $α for λ = $λ. The left and right vectors do not \
	belong to the same mode, so this mode cannot carry a master coordinate — its \
	normalisation divides by ~0. Check that the eigenvalue is converged and simple."""

	φg, ψg = _normalise_pair(normalisation, φ, ψ, α)
	if scale != 1
		φg = φg .* scale
		ψg = ψg .* scale
	end
	# ψ is the BILINEAR left vector (ψᵀ(A − λB) = 0). The orthogonality equations use
	# the sesquilinear convention φᴴ(λB − A) = 0, which for the real matrices here
	# means conj(ψ). Passing ψ un-conjugated pairs each border row with the WRONG mode
	# of a conjugate pair — cross-mode pairing ≈ 0, hence near-singular resonant solves.
	return φg, conj.(ψg), α
end
