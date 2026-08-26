# rom_analysis.jl — everything that reads a finished ROM: slaving, limit-cycle
# continuation, promotion-invariant physical quantities, and the convergence
# diagnostics that say how far the manifold can be trusted.
#
# This file exists because three copies of the same algorithm had drifted apart:
# `solver/rom_palc.jl:_rom_R1`, `invariants.jl:slaved_R1` and (in Python)
# `compare_orders.py:slaved_states`. The first two are now ONE function here. The
# Python one remains a port and is commented as such where it lives.
#
# Coordinate layout throughout: z₁, z̄₁, then any PROMOTED modes y, then η′ LAST.

using Printf: @printf, @sprintf
using DelimitedFiles: readdlm

# ── The reduced dynamics, from either source ─────────────────────────────────
# Continuation runs on MORFE's in-memory `ReducedDynamics`; validation runs on an
# `R_coefficients.csv` written by an earlier session. Both answer `_nvar`/`_eval`,
# so the slaving algorithm below is written once.

"""
	ROMPoly

Reduced dynamics loaded from `R_coefficients.csv`. Same coordinate layout as the
in-memory `ReducedDynamics`; see `load_rom_poly`.
"""
struct ROMPoly
	exps::Vector{Vector{Int}}      # monomial exponents, one entry per row
	coef::Matrix{ComplexF64}       # nmono × ncomp
	nvar::Int
end

"""
	load_rom_poly(csv) → ROMPoly

Read an `R_coefficients.csv` (`exp_1…exp_NVAR, R1_re, R1_im, R2_re, …`). NVAR comes
from the header, so this reads a promoted and an un-promoted run alike.
"""
function load_rom_poly(csv::AbstractString)
	raw, hdr = readdlm(csv, ','; header = true)
	cols = vec(String.(hdr))
	nvar = count(startswith("exp_"), cols)
	ncomp = (length(cols) - nvar) ÷ 2
	exps = [Int[raw[i, j] for j in 1:nvar] for i in 1:size(raw, 1)]
	coef = Matrix{ComplexF64}(undef, size(raw, 1), ncomp)
	for k in 1:ncomp
		coef[:, k] = complex.(raw[:, nvar + 2k - 1], raw[:, nvar + 2k])
	end
	return ROMPoly(exps, coef, nvar)
end

_nvar(p::ROMPoly) = p.nvar
_nvar(R) = length(first(R.poly.multiindex_set.exponents))

function _eval(p::ROMPoly, z::AbstractVector{ComplexF64})
	out = zeros(ComplexF64, size(p.coef, 2))
	@inbounds for i in eachindex(p.exps)
		m = one(ComplexF64)
		e = p.exps[i]
		for j in 1:(p.nvar)
			ej = e[j]
			ej == 0 || (m *= z[j]^ej)
		end
		iszero(m) && continue
		for k in eachindex(out)
			out[k] += p.coef[i, k] * m
		end
	end
	return out
end
_eval(R, z::AbstractVector{ComplexF64}) = evaluate(R.poly, z)

n_promoted(p) = _nvar(p) - 3

"""
	slaved_R1(R, ρ, η) → ComplexF64

First component of the reduced dynamics at the canonical phase z₁ = z̄₁ = ρ, η′ = η,
with every promoted coordinate SLAVED to its quasi-steady value. Reduces to a plain
evaluation when nothing was promoted.

The promoted coordinates are slaved, never zeroed. They carry the mean-flow distortion
— ẏ_k is driven by z₁z̄₁ — and hand it back to the oscillator through z₁·y_k, which is
the dominant stabilising contribution to the Landau coefficient. Zeroing them removes it
and reports this supercritical Hopf as subcritical.

On the orbit they are quasi-steady, so they solve R_k(ρ, ρ, y, η) = 0. The monomial set
carries at most ONE promoted coordinate to the first power, so R is exactly AFFINE in y:
this is one small linear solve, not an iteration and not an approximation.
"""
_exps(p::ROMPoly) = p.exps
_exps(R) = R.poly.multiindex_set.exponents
_coefk(p::ROMPoly, k::Int, m::Int) = p.coef[m, k]
_coefk(R, k::Int, m::Int) = R.poly.coefficients[k, m]

"Frequency of the linear Hopf mode — the seed for the fixed point on Ω."
function _linear_omega(R)
	for (m, e) in enumerate(_exps(R))
		e[1] == 1 && e[2] == 0 && sum(e) == 1 && return imag(_coefk(R, 1, m))
	end
	return 1.0
end

"""
	_harmonic_R1(R, ρ, η, Ω) → ComplexF64

Fundamental (harmonic s = 1) component of `R₁` on the orbit `z₁ = ρe^{iΩt}`, with the
promoted coordinates closed by HARMONIC BALANCE.

Each monomial `z₁^a z̄₁^b η^c y_j` sits at harmonic `s = a − b`, so on the orbit the drive of
promoted row k splits as `b_k(t) = Σ_s b_{k,s} e^{isΩt}` and its response solves

	(i s Ω I − A₀) y_s = b_s

rather than `A y = −b`. The old quasi-steady form is the `s = 0` case of this and is exact
only for a drive that is constant on the orbit — true for the mean-flow modes (all of the
6-mode set), false for the −6.542148 ± 18.415390i pair, which is driven at `s = ±1`. There
the denominators differ by a factor 2.9 in magnitude plus a large phase error:
`|−λ| = 19.54` against `|iΩ − λ| = 6.72`. The small one IS the near-resonance that made the
mode resonant in the first place (detuning 6.73 < 8.43), so slaving discarded exactly the
amplification worth capturing.

APPROXIMATION, deliberate: only the `s = 0` part of `A = ∂R_k/∂y_j` is kept, so harmonics do
not couple through `A`. `A`'s nonzero harmonics come from monomials like `z₁²y_j`, which are
higher order in ρ than the `λ_k y_k` diagonal that dominates it. Lifting this would need a
block-coupled solve across harmonics.
"""
function _harmonic_R1(R, ρ::Float64, η::Float64, Ω::Float64)
	nv = _nvar(R)
	npro = nv - 3
	r1_core = zero(ComplexF64)                    # y-free monomials of R₁ at harmonic 1
	r1_y = Dict{Int, Vector{ComplexF64}}()        # core harmonic → coefficient on each y_j

	for (m, e) in enumerate(_exps(R))
		a, b, c = e[1], e[2], e[nv]
		s = a - b
		v = ρ^(a + b) * (c == 0 ? 1.0 : η^c)
		j = 0                                     # which promoted coordinate, 0 = none
		for t in 3:(nv - 1)
			e[t] != 0 && (j = t - 2; break)
		end
		c1 = _coefk(R, 1, m)
		iszero(c1) && continue
		if j == 0
			s == 1 && (r1_core += c1 * v)
		else
			d = get!(() -> zeros(ComplexF64, npro), r1_y, s)
			d[j] += c1 * v
		end
	end
	npro == 0 && return r1_core

	ys = _promoted_ys(R, ρ, η, Ω)
	# A term `c · z₁^a z̄₁^b η^c · y_j` lands at harmonic (a−b) + s_j, so the fundamental
	# picks up y_j at harmonic 1 − (a−b).
	r1 = r1_core
	for (s_core, cvec) in r1_y
		yv = get(ys, 1 - s_core, nothing)
		yv === nothing && continue
		for j in 1:npro
			r1 += cvec[j] * yv[j]
		end
	end
	return r1
end

"""
	_promoted_ys(R, ρ, η, Ω) → Dict{Int, Vector{ComplexF64}}

Response of the promoted coordinates on the orbit, per harmonic: `y_s` solving
`(isΩ·I − A₀) y_s = b_s`, with `b_s` the drive of the promoted rows at harmonic `s` and `A₀`
the `s = 0` part of `∂R_k/∂y_j`.

Split out of `_harmonic_R1` so the branch, the base-flow shift and the activity check all
read the SAME closure. Three copies of the slaving algorithm drifting apart is what moved
this into `src` in the first place; a fourth would undo that.
"""
function _promoted_ys(R, ρ::Float64, η::Float64, Ω::Float64)
	nv = _nvar(R)
	npro = nv - 3
	ys = Dict{Int, Vector{ComplexF64}}()
	npro == 0 && return ys
	bs = Dict{Int, Vector{ComplexF64}}()          # drive of promoted rows, per harmonic
	A0 = zeros(ComplexF64, npro, npro)
	for (m, e) in enumerate(_exps(R))
		a, b, c = e[1], e[2], e[nv]
		s = a - b
		v = ρ^(a + b) * (c == 0 ? 1.0 : η^c)
		j = 0
		for t in 3:(nv - 1)
			e[t] != 0 && (j = t - 2; break)
		end
		for k in 1:npro
			ck = _coefk(R, 2 + k, m)
			iszero(ck) && continue
			if j == 0
				d = get!(() -> zeros(ComplexF64, npro), bs, s)
				d[k] += ck * v
			elseif s == 0
				A0[k, j] += ck * v
			end
		end
	end
	for (s, bvec) in bs
		ys[s] = ((im * s * Ω) * I - A0) \ bvec
	end
	return ys
end

"""
	promoted_equilibrium(R, η) → Vector{ComplexF64}

Equilibrium of the promoted coordinates on the trivial (`z₁ = 0`) branch: `y*(η)` solving
`R_k(0, 0, y, η) = 0`, which on that slice is the linear system `A₀ y = −b₀`.

**`y*` is not zero, and requiring it to be is what broke two redesigns.** Once φ_k is a
master direction, the base flow at Re ≠ Re₀ has a component along it, and `y*(η)` is exactly
that component expressed in the new coordinate — the base-flow shift, not an error. The
`s = 0` harmonic of `_promoted_ys` already solves for it, so the branch and the observables
are taken about `y*(η)` and always were.

What is worth checking is its SIZE. A shift comparable to the orbit amplitude means the
promoted coordinate has stopped representing the base flow, and in practice means the
pure-η forcing has diverged — which `eta_series_report` measures directly.
"""
function promoted_equilibrium(R, η::Float64)
	npro = _nvar(R) - 3
	npro == 0 && return ComplexF64[]
	return get(_promoted_ys(R, 0.0, η, 0.0), 0, zeros(ComplexF64, npro))
end

"""
	promoted_amplitude(R, ρ, η) → Float64

Largest promoted-coordinate response on the orbit at `(ρ, η)`, `max_s max_k |y_{k,s}|`.

The activity check. A promoted run whose branch has this at 0 is a change of coordinates
that changed nothing: `{y = 0}` is invariant, `R₁` never sees a y-monomial, and the ROM is
the un-promoted one in disguise. That is exactly what a normal-form tolerance tight enough
to keep η out of `R_k` produces, and it went unnoticed for two runs because every physical
quantity agreed — of course it did, it was the same ROM.
"""
function promoted_amplitude(R, ρ::Float64, η::Float64)
	_nvar(R) == 3 && return 0.0
	(_, Ω, _) = rom_po_R1(R, ρ, η)
	ys = _promoted_ys(R, ρ, η, Ω)
	return isempty(ys) ? 0.0 : maximum(maximum(abs, y) for y in values(ys))
end

"""
	rom_po_R1(R, ρ, η; Ω0, tol, maxit) → (R₁, Ω, converged)

Fundamental of `R₁` and the orbit frequency, solved together. Ω enters the harmonic closure
and is itself read off `R₁`, so the two are found by fixed point — seeded from the linear
Hopf frequency, and reported as non-converged rather than silently accepted.
"""
function rom_po_R1(R, ρ::Float64, η::Float64;
		Ω0::Float64 = NaN, tol::Float64 = 1e-12, maxit::Int = 50)
	if _nvar(R) == 3
		r1 = _harmonic_R1(R, ρ, η, 0.0)          # Ω unused when nothing is promoted
		return r1, imag(r1) / ρ, true
	end
	Ω = isnan(Ω0) ? _linear_omega(R) : Ω0
	r1 = zero(ComplexF64)
	for _ in 1:maxit
		r1 = _harmonic_R1(R, ρ, η, Ω)
		Ωn = imag(r1) / ρ
		abs(Ωn - Ω) <= tol * max(abs(Ω), 1.0) && return r1, Ωn, true
		Ω = Ωn
	end
	return r1, Ω, false
end

"First component of the reduced dynamics on the orbit, promoted coordinates closed."
slaved_R1(R, ρ::Float64, η::Float64) = first(rom_po_R1(R, ρ, η))

"Periodic-orbit residual F(ρ, η′) = Re(R₁); vanishes on the limit-cycle branch."
rom_po_residual(ρ::Float64, η::Float64, R) = real(first(rom_po_R1(R, ρ, η)))

"Angular frequency Ω = Im(R₁)/ρ at the periodic orbit."
rom_po_frequency(ρ::Float64, η::Float64, R) = rom_po_R1(R, ρ, η)[2]

"""
	rom_hopf_eta(R; ε, η0, tol, max_iter) → Float64

η′ at which the linearised ROM growth rate vanishes — the true Hopf point. Re₀ is only
the FOM's expansion point, not necessarily the critical Reynolds number, so this root is
generally nonzero.
"""
function rom_hopf_eta(R; ε::Float64 = 1e-6, η0::Float64 = 0.0,
		tol::Float64 = 1e-12, max_iter::Int = 50)
	η = η0
	for _ in 1:max_iter
		F = rom_po_residual(ε, η, R)
		abs(F) < tol * ε && return η
		ε_η = 1e-6 * max(abs(η), 1e-6)
		dF = (rom_po_residual(ε, η + ε_η, R) - F) / ε_η
		abs(dF) < 1e-300 && break
		η -= F / dF
	end
	return η
end

"Unit tangent to the branch F(ρ, η′) = 0, oriented consistently with `τ_prev`."
function rom_palc_tangent(ρ::Float64, η::Float64, R, τ_prev::Vector{Float64})
	ε_ρ = 1e-7 * max(ρ, 1.0)
	ε_η = 1e-7 * max(abs(η), 1e-8)
	F0 = rom_po_residual(ρ, η, R)
	dF_dρ = (rom_po_residual(ρ + ε_ρ, η, R) - F0) / ε_ρ
	dF_dη = (rom_po_residual(ρ, η + ε_η, R) - F0) / ε_η
	τ = [-dF_dη, dF_dρ]                 # ⊥ gradient
	nrm = sqrt(τ[1]^2 + τ[2]^2)
	nrm < 1e-300 && return copy(τ_prev)
	τ ./= nrm
	(τ[1] * τ_prev[1] + τ[2] * τ_prev[2]) < 0.0 && (τ .*= -1.0)
	return τ
end

"""
	rom_palc_step(ρ, η, τ, Δs, R; tol, max_iter)
	→ (ρ, η, T, τ, n_iter, converged)

One pseudo-arclength step: predictor Δs along τ, then a 2×2 Newton corrector on
[F(ρ,η); τ·(p − last) − Δs].
"""
function rom_palc_step(ρ::Float64, η::Float64, τ::Vector{Float64}, Δs::Float64, R;
		tol::Float64 = 1e-10, max_iter::Int = 20)
	ρ_p = max(ρ + Δs * τ[1], 1e-12)
	η_p = η + Δs * τ[2]
	converged = false
	n_iter = 0
	for iter in 1:max_iter
		n_iter = iter
		F = rom_po_residual(ρ_p, η_p, R)
		N = τ[1] * (ρ_p - ρ) + τ[2] * (η_p - η) - Δs
		if abs(F) < tol && abs(N) < tol
			converged = true
			break
		end
		ε_ρ = 1e-7 * max(ρ_p, 1.0)
		ε_η = 1e-7 * max(abs(η_p), 1e-8)
		a = (rom_po_residual(ρ_p + ε_ρ, η_p, R) - F) / ε_ρ
		b = (rom_po_residual(ρ_p, η_p + ε_η, R) - F) / ε_η
		# 2×2 solve written out: avoids a StaticArrays dependency in src for two rows.
		det = a * τ[2] - b * τ[1]
		abs(det) < 1e-300 && break
		δρ = (-F * τ[2] + b * N) / det
		δη = (-a * N + τ[1] * F) / det
		ρ_p = max(ρ_p + δρ, 1e-12)
		η_p = η_p + δη
	end
	Ω = rom_po_frequency(ρ_p, η_p, R)
	τ_new = rom_palc_tangent(ρ_p, η_p, R, τ)
	return ρ_p, η_p, 2π / abs(Ω), τ_new, n_iter, converged
end

"""
	branch_amplitude_scale(R, η; lo, hi, n) → Float64

Characteristic limit-cycle amplitude at parameter `η`, found without assuming one.

**ρ has no natural size.** It is a master coordinate, so its scale is whatever the
eigenvector gauge gives it: under `SymmetricBiorthogonal` the Kármán orbit happens to sit at
ρ ~ O(1), under `LeftBiorthogonal` the SAME physical orbit sits some 400× further out,
because that gauge leaves φ at its natural length instead of dividing by √α. Anything
carrying units of ρ — a continuation step, a finite-difference increment, an initial guess —
is therefore meaningless as a bare number, and hardcoding one silently pins the code to a
gauge. That is not hypothetical: continuation constants tuned for ρ ~ O(1) traced 5001 steps
to ρ = 0.002 in the left gauge and reported it as a collapsed branch.

The fix is to measure the scale first and work in `ρ/ρ_ref`. Bracketing the first sign change
of `G(ρ) = Re(R₁)/ρ` on a LOG grid spanning `lo`…`hi` is scale-free by construction: below
the orbit `G ≈ σ > 0`, above it the saturating term wins. Returns `NaN` if no sign change is
bracketed, and callers should fall back to 1.0 rather than propagate it.
"""
function branch_amplitude_scale(R, η::Float64; lo::Float64 = 1e-8, hi::Float64 = 1e8,
		n::Int = 321)
	g(ρ) = (v = try rom_po_residual(ρ, η, R) / ρ catch; NaN end;
		isfinite(v) ? v : NaN)
	prev_ρ, prev_g = NaN, NaN
	for ρ in exp10.(range(log10(lo), log10(hi); length = n))
		v = g(ρ)
		if isnan(v)
			prev_ρ, prev_g = NaN, NaN
			continue
		end
		if !isnan(prev_g) && sign(v) != sign(prev_g)
			# Bisect in log ρ — the bracket spans decades, so the geometric midpoint is the
			# one that halves the remaining interval.
			a, b, ga = prev_ρ, ρ, prev_g
			for _ in 1:60
				m = sqrt(a * b)
				gm = g(m)
				isnan(gm) && break
				sign(gm) == sign(ga) ? (a = m) : (b = m)
			end
			return sqrt(a * b)
		end
		prev_ρ, prev_g = ρ, v
	end
	return NaN
end

"""
	truncate_dynamics(R, N) → ReducedDynamics

Zero every monomial whose CORE degree — (z₁, z̄₁, η′), excluding promoted coordinates —
exceeds N. Because the cohomological solve is graded this IS the order-N reduced
dynamics, bit-exact.

Truncating on the TOTAL degree instead would drop the mean-flow coupling z₁·y_k at low N
and silently return a ROM with no saturation mechanism — the Landau coefficient would
flip sign. Promoted coordinates are first order by construction and are not part of the
order hierarchy.
"""
function truncate_dynamics(R, N::Int)
	Rt = deepcopy(R)
	exps = Rt.poly.multiindex_set.exponents
	nv = length(first(exps))
	for (m, e) in enumerate(exps)
		(e[1] + e[2] + e[nv]) > N && (Rt.poly.coefficients[:, m] .= 0)
	end
	return Rt
end

"""
	roots_at_re(R, re; re0, rho_lo, rho_hi, n) → Vector{Float64}

Every ρ > 0 with F(ρ, η(re)) = 0, by sign change on a log grid then bisection. Used to
continue a branch where the arclength corrector cannot, and to cross-check a traced one.
"""
function roots_at_re(R, re::Float64; re0::Float64, rho_lo::Float64 = 1e-4,
		rho_hi::Float64 = 15.0, n::Int = 2000)
	η = 1 / re - 1 / re0
	rs = 10 .^ range(log10(rho_lo), log10(rho_hi), length = n)
	f(ρ) = (v = try rom_po_residual(ρ, η, R) / ρ catch; NaN end; v)
	Fs = map(f, rs)
	out = Float64[]
	for i in 1:(n - 1)
		(isfinite(Fs[i]) && isfinite(Fs[i + 1])) || continue
		sign(Fs[i]) == sign(Fs[i + 1]) && continue
		a, b = rs[i], rs[i + 1]
		fa = Fs[i]
		for _ in 1:60                     # bisection: no derivative, cannot diverge
			m = 0.5 * (a + b)
			fm = f(m)
			isfinite(fm) || break
			sign(fm) == sign(fa) ? (a = m; fa = fm) : (b = m)
		end
		push!(out, 0.5 * (a + b))
	end
	return out
end

"""
	sweep_branch_in_re(R, ρ_start, re_start; re0, re_max, dre, rho_max)
	→ Vector{NTuple{6,Float64}}

Continue a branch by stepping **Re** and solving F(ρ, η) = 0 for ρ at each step, following
the root nearest the previous one.

This exists because a turning point in ρ is NOT a turning point in Re. Pseudo-arclength
continuation parametrises by arclength in (ρ, η) and has to negotiate the fold; where its
corrector cannot — the 6-mode order-7 branch oscillated across one turning point 73 times
without resolving it — the branch is still perfectly single-valued as ρ(Re) and a plain
sweep walks straight through. A root scan confirms the solutions are there: the 6-mode
order-9 branch runs ρ = 1.53 at Re 52 to 9.04 at Re 69.5 while PALC gave up at Re 52.6.

A jump limit rejects hops onto a disconnected root: this ROM's truncated polynomial also
carries spurious high-amplitude roots (ρ ≈ 9–17), which are not the physical branch.
"""
function sweep_branch_in_re(R, ρ_start::Float64, re_start::Float64; re0::Float64,
		re_max::Float64 = 70.0, dre::Float64 = 0.25, rho_max::Float64 = 15.0)
	rows = Vector{NTuple{6, Float64}}()
	ρ = ρ_start
	slope = NaN                       # dρ/dRe, from the previous accepted step
	re = re_start + dre
	while re <= re_max + 1e-9
		cands = roots_at_re(R, re; re0 = re0, rho_hi = rho_max)
		isempty(cands) && break
		# SECANT prediction, not "nearest to the last ρ". Nearest-value following is what
		# let this hop onto a spurious low-amplitude root: at Re 70 it reported ρ = 0.66
		# for an order-7 branch whose order-3 counterpart sits at 5.92, and produced a
		# branch with ρ DECREASING in Re. The prediction carries the slope the branch was
		# actually travelling at, so a root going the wrong way is no longer "nearest".
		ρ_pred = isnan(slope) ? ρ : ρ + slope * dre
		ρn = cands[argmin(abs.(cands .- ρ_pred))]
		# Tolerance about the PREDICTION, scaled by amplitude so it is meaningful at both
		# ρ ≈ 0.5 and ρ ≈ 5.
		tol = max(0.15, 0.30 * max(ρ_pred, 1.0), 2 * abs(isnan(slope) ? 0.0 : slope) * dre)
		abs(ρn - ρ_pred) > tol && break
		slope = (ρn - ρ) / dre
		ρ = ρn
		η = 1 / re - 1 / re0
		Ω = rom_po_frequency(ρ, η, R)
		push!(rows, (η, re, ρ, Ω, 2π / abs(Ω), 0.0))
		re += dre
	end
	return rows
end

"""
	trace_limit_cycle_branch(R; re0, re_max, re_min, rho_max, ds0, max_steps)
	→ Vector{NTuple{6,Float64}}   # (η, Re, ρ, Ω, T, fold)

PALC continuation of F(ρ, η′) = 0 from the Hopf point. Returns one row per branch point,
with `fold` counting how many times the branch has turned in Re — downstream code
segments the sheets on that counter instead of re-deriving it from Re decreasing, which
mis-segments whenever consecutive rows repeat.

Three defects fixed relative to the original `solve_rom.jl:trace_branch`:

  · it stopped at `re < re_c - 0.5`, abandoning any branch that folds and would come
    back up — folds are exactly what PALC exists to traverse;
  · a stalled step was still pushed as a row, so a collapsing Δs produced 82 identical
    rows at one point (order-7 of the 6-mode run) before the arclength floor hit;
  · nothing bounded ρ, so past a fold it would chase a disconnected root out to ρ ≈ 17.
"""
function trace_limit_cycle_branch(R; re0::Float64,
		re_max::Float64 = 70.0, re_min::Float64 = 40.0, rho_max::Float64 = 15.0,
		ds0::Float64 = 1e-4, max_steps::Int = 2000, max_folds::Int = 4)
	η_c = rom_hopf_eta(R)
	ρ = 1e-4
	η = η_c
	τ = rom_palc_tangent(ρ, η, R, [1.0, 0.0])
	τ[1] < 0 && (τ .*= -1.0)          # orient: ρ increasing

	rows = Vector{NTuple{6, Float64}}()
	push!(rows, (η, 1 / (η + 1 / re0), ρ, rom_po_frequency(ρ, η, R),
		2π / abs(rom_po_frequency(ρ, η, R)), 0.0))

	Δs = ds0
	folds = 0
	stalls = 0
	re_prev = rows[end][2]
	dir_prev = 0
	for _ in 1:max_steps
		local ρn, ηn, Tn, τn, n_iter, ok
		try
			(ρn, ηn, Tn, τn, n_iter, ok) = rom_palc_step(ρ, η, τ, Δs, R)
			isfinite(ρn) && isfinite(ηn) || (ok = false)
		catch err
			err isa LinearAlgebra.SingularException || err isa DomainError || rethrow()
			ok = false
		end
		if !ok
			Δs /= 2
			Δs < 1e-12 && break
			continue
		end
		# No-progress guard. A converged step that does not move is not a branch point:
		# pushing it wastes rows and starves the arclength. Grow Δs and retry instead.
		if hypot(ρn - ρ, ηn - η) < 1e-14 * max(1.0, ρ)
			stalls += 1
			stalls >= 5 && break
			Δs *= 4
			continue
		end
		stalls = 0
		ρ, η, τ = ρn, ηn, τn
		re = 1 / (η + 1 / re0)
		dir = re > re_prev ? 1 : -1
		dir_prev != 0 && dir != dir_prev && (folds += 1)
		dir_prev = dir
		re_prev = re
		push!(rows, (η, re, ρ, rom_po_frequency(ρ, η, R), Tn, Float64(folds)))
		re > re_max && break
		re < re_min && break
		ρ > rho_max && break
		# Fold THRASHING. A genuine branch turns a handful of times; the order-7 branch of
		# the 6-mode run recorded 73 reversals inside a span of 1e-9 in Re, oscillating
		# across a turning point the corrector could not resolve. The no-progress guard
		# misses it because each step does move — just alternately forwards and back.
		folds > max_folds && break
		ρ < 1e-8 && break              # folded back onto the trivial branch
		n_iter <= 4 && (Δs = min(Δs * 1.5, 100 * ds0))
		n_iter >= 10 && (Δs /= 2)
	end

	# ── Continue to re_max where the corrector gave up ───────────────────────
	# PALC stops for three reasons that are NOT "the branch ended": it thrashed at a
	# turning point, it folded and ran back down to re_min, or ρ collapsed to the trivial
	# branch. In all three the stable sheet may still extend past the largest Re reached —
	# and a sweep in Re, where the branch is single-valued, walks straight through.
	# Without this, orders 7 and 9 of the 6-mode run stopped at Re 56.6 and 52.6 although
	# roots exist all the way to 70.
	stable = filter(r -> r[6] == 0.0, rows)
	if !isempty(stable)
		i_top = argmax(r[2] for r in stable)
		(re_top, ρ_top) = (stable[i_top][2], stable[i_top][3])
		if re_top < re_max - 1e-6
			extra = sweep_branch_in_re(R, ρ_top, re_top; re0 = re0, re_max = re_max,
				rho_max = rho_max)
			if !isempty(extra)
				# Keep only the stable sheet up to the handover, then the swept tail: the
				# post-fold rows describe a different sheet at Re values the sweep now
				# covers, and interleaving the two would make ρ(Re) multi-valued.
				rows = vcat(stable[1:i_top], extra)
			end
		end
	end
	return rows
end

# ── Promotion-invariant physical quantities ──────────────────────────────────
# Raw R coefficients are not comparable between runs: the Arpack eigenvector gauge
# differs run to run (z → e^{iφ}z), and promoting outer modes is a CHANGE OF
# COORDINATES — the ROM gains columns and shares no monomials with an un-promoted one.
# The quantities below survive both transformations.

# R₁(ρ, ρ, η)/ρ is even in ρ (phase symmetry z → e^{iφ}z), so it expands as
# λ(η) + c₂₁₀(η)ρ² + O(ρ⁴). One Richardson step in ρ² removes the ρ⁴ term exactly.
function _lambda_and_c210(p, η::Float64; ρ::Float64 = 1e-3)
	coarse = slaved_R1(p, ρ, η) / ρ
	fine = slaved_R1(p, ρ / 2, η) / (ρ / 2)
	# NOTE the explicit `*`. `4fine` would be fine, but `4f2` — the name this once
	# carried — parses as the Float32 literal 4e2 = 400.0, because `f` is Julia's
	# Float32 exponent marker. It silently returned λ = 133.33 instead of 16.86i.
	λ = (4 * fine - coarse) / 3
	c210 = (coarse - λ) / ρ^2
	return λ, c210
end

"""
	rom_invariants(p; η_step) → NamedTuple

  · `σ`, `ω`      — the Hopf eigenvalue λ = σ + iω at η′ = 0. Fully invariant.
  · `c101`        — ∂λ/∂η′. Fully invariant. Computed by a central difference, because
                    in a PROMOTED run the mean-flow coordinates respond to η′ directly
                    and feed back through z₁·y_k, so the `[1,0,1]` coefficient alone
                    would miss part of it.
  · `c210_eff`    — the effective Landau coefficient after slaving. Scales as |c|² under
                    z → cz, so comparable only at a fixed `MODE_SCALE`.
  · `c210_ratio`  — Im/Re of c₂₁₀. The |c|² cancels, so this is the gauge-free
                    fingerprint — and the quantity whose SIGN the conjugate-pairing bug
                    flipped.
  · `criticality` — sign(Re c₂₁₀).
"""
function rom_invariants(p; η_step::Float64 = 1e-4)
	λ, c210 = _lambda_and_c210(p, 0.0)
	# λ(η) is a polynomial in η, so a central difference leaves an O(η²) error that the
	# reference's large c₁₀₂ ≈ 2.9e3 makes visible (3e-6 relative at η = 1e-4). One
	# Richardson step in η² takes c₁₀₁ to ~1e-10 for four extra evaluations.
	d_coarse = (first(_lambda_and_c210(p, η_step)) -
				first(_lambda_and_c210(p, -η_step))) / (2η_step)
	d_fine = (first(_lambda_and_c210(p, η_step / 2)) -
			  first(_lambda_and_c210(p, -η_step / 2))) / η_step
	c101 = (4 * d_fine - d_coarse) / 3
	return (; σ = real(λ), ω = imag(λ),
		c101_re = real(c101), c101_im = imag(c101),
		c210_re = real(c210), c210_im = imag(c210),
		c210_ratio = imag(c210) / real(c210),
		criticality = real(c210) < 0 ? "supercritical" : "subcritical",
		nvar = _nvar(p), n_promoted = n_promoted(p))
end

# Flat `key = value` text: diffable, greppable, no TOML dependency.
const INVARIANT_KEYS = ("σ", "ω", "c101_re", "c101_im", "c210_re", "c210_im", "c210_ratio")

function write_invariants(path::AbstractString, inv::NamedTuple; note::AbstractString = "")
	open(path, "w") do io
		println(io, "# Kármán vortex street — promotion-invariant ROM reference.")
		println(io, "# Valid for a run WITH or WITHOUT promoted outer modes: every quantity")
		println(io, "# here is computed after slaving the promoted coordinates.")
		isempty(note) || println(io, "# ", note)
		println(io)
		for k in INVARIANT_KEYS
			@printf(io, "%-12s = %.12g\n", k, getproperty(inv, Symbol(k)))
		end
		println(io)
		println(io, "# informational only — not compared")
		println(io, "criticality  = ", inv.criticality)
	end
	return path
end

function read_invariants(path::AbstractString)
	d = Dict{String, Float64}()
	for line in eachline(path)
		s = strip(first(split(line, '#')))
		isempty(s) && continue
		i = findfirst('=', s)
		i === nothing && continue
		k = strip(s[1:(i - 1)])
		k in INVARIANT_KEYS && (d[k] = parse(Float64, strip(s[(i + 1):end])))
	end
	return d
end

# ── Convergence diagnostics ──────────────────────────────────────────────────
# What limits how far the manifold can be pushed, measured rather than assumed.

"""
	homological_denominators(λ, master; max_ord, tol, pairings) → Vector{NamedTuple}

For every monomial `z₁^a z̄₁^b` up to `max_ord`, the eigenvalue combination
`μ = a·λ₁ + b·λ̄₁` is what the homological solve inverts against on the outer block:
`(L − μ B₁)`. A small `|μ − λ_k|` is a NEAR-RESONANCE — the outer mode k is nearly
excited by that monomial and its manifold component is amplified by 1/|μ − λ_k|.

Returns one row per outer mode: its smallest denominator over the whole monomial set,
the monomial achieving it, and (when `pairings` is supplied) the α ratio that says
whether the mode could be promoted out of the outer block at all.

A denominator is only a CANDIDATE ranking — it is the amplification factor, not the
amplitude. A large denominator with a large numerator beats a small one with a
negligible numerator. Use `modal_growth` to find which is actually excited.
"""
function homological_denominators(λ::AbstractVector{<:Complex}, master::AbstractVector{Int};
		max_ord::Int = 9, tol::Float64 = Inf,
		pairings::Union{Nothing, Dict{Int, Float64}} = nothing)
	λ₁ = λ[master[1]]
	outer = setdiff(eachindex(λ), master)
	out = NamedTuple[]
	for k in outer
		best = (Inf, 0, 0)
		for a in 0:max_ord, b in 0:max_ord
			(0 < a + b <= max_ord) || continue
			d = abs(a * λ₁ + b * conj(λ₁) - λ[k])
			d < best[1] && (best = (d, a, b))
		end
		push!(out, (; mode = k, λ = λ[k], denom = best[1], a = best[2], b = best[3],
			s = best[2] - best[3],
			α_ratio = pairings === nothing ? NaN : get(pairings, k, NaN),
			flagged = best[1] < tol))
	end
	sort!(out, by = r -> r.denom)
	return out
end

"""
	manifold_ratio_test(W) → NamedTuple

Ratio test on ‖W‖ per degree, along the amplitude backbone `z₁^{n+1}z̄₁^n` and along
`η′^n`. A ratio that SETTLES to a constant is the signature of a genuine analytic
singularity, and its reciprocal gives the radius; a ratio that keeps growing would
instead indicate round-off contamination.

⚠ Valid WITHIN one run only. Promoted coordinates each carry a 1/√α scaling, and α
spans orders of magnitude across modes, so ‖W‖ along the backbone means a different
thing in each coordinate system. Compare runs on observables (lift vs DNS, the branch),
never on this.
"""
function manifold_ratio_test(W)
	exps = W.poly.multiindex_set.exponents
	C = W.poly.coefficients
	nv = length(first(exps))
	col(m) = sqrt(sum(abs2, @view C[:, 1, m]))

	amp = NamedTuple[]
	prev = 0.0
	for n in 0:((length(exps) > 0 ? 4 : 0))
		m = findfirst(e -> e[1] == n + 1 && e[2] == n && sum(e) == 2n + 1, exps)
		m === nothing && continue
		nrm = col(m)
		r = prev > 0 ? nrm / prev : NaN
		push!(amp, (; deg = 2n + 1, norm = nrm, ratio = r,
			rho_conv = isnan(r) ? NaN : 1 / sqrt(r)))
		prev = nrm
	end

	eta = NamedTuple[]
	prev = 0.0
	for n in 1:9
		m = findfirst(e -> e[1] == 0 && e[2] == 0 && e[nv] == n && sum(e) == n, exps)
		m === nothing && continue
		nrm = col(m)
		r = prev > 0 ? nrm / prev : NaN
		push!(eta, (; deg = n, norm = nrm, ratio = r,
			eta_conv = isnan(r) ? NaN : 1 / r))
		prev = nrm
	end
	# Radius from the LAST ratio, where the sequence has settled.
	#
	# ONLY when nothing was promoted. With promoted coordinates the mean-flow content
	# moves out of the pure backbone z₁^{n+1}z̄₁^n and into the mixed monomials z₁^a z̄₁^b y_k,
	# so the backbone norms measure a RESIDUAL rather than the manifold's convergence.
	# Reporting them anyway gave ρ_conv = 0.67 for a 6-mode run whose branch is accurate
	# against DNS out to ρ ≈ 2.1 — an artefact, not a radius. Returning NaN makes callers
	# (and figures.py) fall back to marking nothing rather than mark it wrongly.
	# ρ_conv is suppressed for a promoted run: the mean-flow content moves out of the pure
	# backbone z₁^{n+1}z̄₁^n into the mixed monomials z₁^a z̄₁^b y_k, so those norms are a
	# RESIDUAL rather than a radius.
	#
	# η_conv is NOT suppressed. The pure-η column of W is the same object however many
	# coordinates were promoted, so it is directly comparable across runs — and it is the
	# number that exposed the tolerance bug (1.88e-2 un-promoted against 2.56e-4 promoted,
	# now back to ≈2.1e-2 once the normal-form tolerance stopped dumping the η-expansion
	# into R_k).
	promoted = nv > 3
	ρc = (promoted || isempty(amp)) ? NaN : last(amp).rho_conv
	ηc = isempty(eta) ? NaN : last(eta).eta_conv
	return (; amplitude = amp, eta = eta, rho_conv = ρc, eta_conv = ηc,
		valid = !promoted)
end

"""
	eta_series_report(R; re0, re_max, orders, gate_radius) → NamedTuple

Convergence of the PARAMETER expansion, read off `R` directly. Run this before believing any
plot.

`manifold_ratio_test` measures `W`, which is coordinate-dependent and therefore not comparable
between a promoted and an un-promoted run. `R`'s coefficient FAMILIES are comparable: for every
run of the same problem, `R₁`'s `z₁η^c` is the eigenvalue series and `z₁²z̄₁η^c` is the Landau
series, whatever else was promoted. Two numbers per family:

  · **radius** — the ratio test `|a_c| / |a_{c+1}|`, taken at the largest `c` where both are
    nonzero, i.e. where the sequence has settled.
  · **truncation ratio** — `|a_c η^c| / |a_0|` at `η(re_max)` with `c` the highest η power the
    order-N truncation retains. This is how much the LAST kept term still contributes at the top
    of the sweep, and it is the honest statement of where the Re range stops being trustworthy.
    It is not a promoted-run concern: un-promoted, `η(70) = 6.1e-3` against a radius of 1.05e-2
    is 58% of the way out, so the tail is not negligible there either.

This table is what distinguishes the three ways promotion has failed here, and it costs a
deserialise and a dictionary lookup:

  · **inert** — promoted rows carry no forcing at all, `R₁` identical to the un-promoted run.
  · **divergent** — the promoted rows' pure-η forcing has its own radius (1.5e-4 measured), far
    inside the Hopf block's, and it drags `R₁` down with it.
  · **contaminated** — radii look normal but a low-degree coefficient is wrong by orders of
    magnitude and every higher one inherits the offset, which is why `families[2].coeffs[1]`
    (the η-independent Landau coefficient) is reported separately: it is the one entry that must
    agree with the un-promoted run to full precision, and it did even in the run whose `R₁`
    reached 1e36.
"""
function eta_series_report(R; re0::Float64 = 49.03, re_max::Float64 = 70.0,
		orders = (3, 5, 7, 9), gate_radius::Float64 = 5e-3,
		gate_truncation::Float64 = 1.0)
	ex = _exps(R)
	nv = _nvar(R)
	npro = nv - 3
	idx = Dict{Vector{Int}, Int}()
	for (m, e) in enumerate(ex)
		idx[collect(Int, e)] = m
	end
	c_top = maximum(Int(e[nv]) for e in ex)
	η_max = 1 / re_max - 1 / re0

	function coeffs(row::Int, a::Int, b::Int)
		v = zeros(Float64, c_top + 1)
		for c in 0:c_top
			k = zeros(Int, nv)
			k[1] = a
			k[2] = b
			k[nv] = c
			m = get(idx, k, 0)
			m == 0 || (v[c + 1] = abs(_coefk(R, row, m)))
		end
		return v
	end
	# Last ratio with both terms nonzero. Early ratios have not settled, and a family that is
	# zero above some power (a promoted row with no η forcing) must report NaN, not a radius.
	function radius(v)
		r = NaN
		for i in 1:(length(v) - 1)
			(v[i] > 0 && v[i + 1] > 0) && (r = v[i] / v[i + 1])
		end
		return r
	end
	# `d0` is the family's degree in (z₁, z̄₁), so `truncate_dynamics(R, N)` keeps η powers up
	# to N − d0 — the CORE degree, matching how the branch is traced.
	function truncation(v, d0::Int)
		out = NamedTuple[]
		for N in orders
			c = N - d0
			(c < 1 || c + 1 > length(v) || v[1] <= 0) && continue
			push!(out, (; order = N, eta_power = c,
				ratio = v[c + 1] * abs(η_max)^c / v[1]))
		end
		return out
	end

	fams = NamedTuple[]
	for (name, row, a, b) in (("R1  z1*eta^c", 1, 1, 0), ("R1  z1^2 z1b*eta^c", 1, 2, 1))
		v = coeffs(row, a, b)
		push!(fams, (; name, row, core_degree = a + b, coeffs = v,
			radius = radius(v), truncation = truncation(v, a + b)))
	end
	for k in 1:npro
		v = coeffs(2 + k, 0, 0)
		push!(fams, (; name = "R$(2 + k)  eta^c (pure-parameter forcing)", row = 2 + k,
			core_degree = 0, coeffs = v, radius = radius(v),
			truncation = truncation(v, 0)))
	end

	lines = String[@sprintf("eta-series report   nvar=%d   eta(Re=%.1f) = %.4e",
		nv, re_max, η_max)]
	for f in fams
		push!(lines, @sprintf("  %-34s radius %.3e", f.name, f.radius))
		push!(lines, "      c:     " * join([@sprintf("%10.3e", x) for x in f.coeffs], ""))
		isempty(f.truncation) ||
			push!(lines, "      trunc: " * join([@sprintf("ord%d(eta^%d) %.2e  ",
					t.order, t.eta_power, t.ratio) for t in f.truncation], ""))
	end
	# The gate is on R₁ alone. A promoted row may legitimately carry a shorter η series — what
	# must not happen is that shortness propagating into the oscillator's own coefficients.
	#
	# BOTH tests are needed and neither subsumes the other. The radius is blind to a level
	# shift: the run whose Landau series went 1.32e-1, 3.40e5, 3.88e16 reports a perfectly
	# healthy radius of 1.24e-2, because from c = 2 on the ratios are normal and only the
	# offset is wrong. The truncation ratio catches that instantly (1.1e13 at order 5). Equally,
	# a run can have every term small at η(re_max) and still be diverging in a way the radius
	# exposes. Fail on either.
	r1 = [f for f in fams if f.row == 1]
	# `init = -Inf`, not NaN: `max(NaN, x)` is NaN in Julia, so a NaN seed poisons the whole
	# fold and every run reports NaN. Empty (no order reaches this family) then reads as -Inf
	# and is mapped back to NaN below.
	worst = maximum(Float64[t.ratio for f in r1 for t in f.truncation]; init = -Inf)
	isfinite(worst) || (worst = NaN)
	# A family needs two nonzero coefficients before a ratio means anything. At order 3 the
	# Landau family has only c = 0, so its radius is NaN — that is "not measurable at this
	# order", not "diverging", and failing on it made the FAST profile unable to pass its own
	# gate. Judge on the families that HAVE a radius, and require at least one.
	measurable = [f for f in r1 if !isnan(f.radius)]
	ok = !isempty(measurable) && all(f -> f.radius >= gate_radius, measurable) &&
		 !isnan(worst) && worst <= gate_truncation
	push!(lines, @sprintf("  gate: R1 radius >= %.1e AND max trunc <= %.1f ? %s%s",
		gate_radius, gate_truncation, ok ? "PASS" : "FAIL",
		length(measurable) == length(r1) ? "" :
		@sprintf("   (%d/%d families measurable at this order)",
			length(measurable), length(r1))))
	push!(lines, @sprintf("        max trunc = %.3e   landau(c=0) = %.8e",
		worst, fams[2].coeffs[1]))
	return (; families = fams, eta_max = η_max, pass = ok, max_truncation = worst,
		landau_c0 = fams[2].coeffs[1], lines = lines)
end

"""
	domb_sykes(W) → NamedTuple

Locate and classify the singularity that limits the amplitude expansion.

For a series `Σ aₙ ρⁿ` whose nearest singularity is at `ρ_c` and behaves like
`(1 − ρ/ρ_c)^{-γ}`, the coefficient ratios obey asymptotically

	rₙ = aₙ / aₙ₋₁ ≈ (1/ρ_c) · (1 + (γ − 1)/n)

so plotting `rₙ` against `1/n` gives a straight line whose **intercept is 1/ρ_c** and whose
**slope is (γ − 1)/ρ_c**. That separates the two cases that matter here:

  · `γ = 1` (slope ≈ 0) — a simple **POLE**. This is what "the radius is set by the nearest
    pole, which signals an outer mode" predicts, and it is the case where carrying that mode
    as a coordinate should push the singularity out.
  · `γ ∉ ℤ` (slope ≠ 0) — a **branch point**, which is what a quadratic convolution generically
    produces and which no single mode is responsible for.

Only the backbone `z₁^{n+1}z̄₁^n` is used, so this is the ρ direction. Five degrees (1…9) is few
for an extrapolation — the fit residual is returned so the reader can judge, and a two-point
Richardson estimate is given alongside the least-squares one.
"""
function domb_sykes(W)
	exps = W.poly.multiindex_set.exponents
	C = W.poly.coefficients
	norms = Float64[]
	degs = Int[]
	for n in 0:4
		m = findfirst(e -> e[1] == n + 1 && e[2] == n && sum(e) == 2n + 1, exps)
		m === nothing && continue
		push!(norms, sqrt(sum(abs2, @view C[:, 1, m])))
		push!(degs, 2n + 1)
	end
	length(norms) < 3 && return (; rho_c = NaN, gamma = NaN, resid = NaN,
		x = Float64[], r = Float64[])
	# Successive backbone degrees differ by 2, so the ratio per unit degree is √(aₙ/aₙ₋₁).
	r = [sqrt(norms[i] / norms[i - 1]) for i in 2:length(norms)]
	x = [1.0 / degs[i] for i in 2:length(degs)]
	# least squares r = A + B x
	n = length(r)
	x̄ = sum(x) / n
	r̄ = sum(r) / n
	Sxx = sum((xi - x̄)^2 for xi in x)
	B = Sxx > 0 ? sum((x[i] - x̄) * (r[i] - r̄) for i in 1:n) / Sxx : 0.0
	A = r̄ - B * x̄
	resid = sqrt(sum((r[i] - (A + B * x[i]))^2 for i in 1:n) / n)
	ρc = A > 0 ? 1 / A : NaN
	γ = A > 0 ? 1 + B * ρc : NaN
	return (; rho_c = ρc, gamma = γ, resid = resid, x = x, r = r,
		degrees = degs, norms = norms)
end

"""
	backbone_direction(W) → NamedTuple

Does the manifold's high-degree content settle onto ONE direction? Needs only `W` — no
eigenbasis, no `B₁`, no solve.

If the series `Σ Wₙ ρⁿ` is limited by a single singularity, its coefficient VECTORS align
with that singularity's direction as `n → ∞`, so `cos(Wₙ, W_top) → 1`. Promotion works by
making that direction a coordinate, and it can only work if the direction exists: a
`cos` that stalls well below 1 means the high-degree content is spread over several
directions and no single promoted mode captures it.

This is the measurement `modal_growth` cannot make. Ranking modes by their amplitude at the
top degree answers "which mode is largest there", which is not the same question and gave a
misleading answer here — it picked λ = −5.135420 on a degree-9 amplitude of 3.50e-8 against
2.23e-8 and 1.32e-8 for its neighbours, i.e. no dominance at all, and that mode's pairing
`α/α_Hopf = 8.41e-3` makes it a poor coordinate for unrelated reasons.

Measured on the un-promoted order-9 run, `cos(Wₙ, W_top)` climbs monotonically — 0.27, 0.46,
0.72, 0.87 along `z₁^{n+1}z̄₁^n` and 0.48, 0.78, 0.92 along the mean-flow `z₁^n z̄₁^n`. The
direction is settling, and the mean-flow family settles faster, which is consistent with the
fold at ρ_c ≈ 1.81 being a mean-flow-distortion effect. Five backbone terms is few, so read
the trend, not the last digit.
"""
function backbone_direction(W)
	exps = W.poly.multiindex_set.exponents
	C = W.poly.coefficients
	col(m) = @view C[:, 1, m]
	function cosang(a, b)
		na = sqrt(sum(abs2, a))
		nb = sqrt(sum(abs2, b))
		return (na == 0 || nb == 0) ? NaN : abs(dot(a, b)) / (na * nb)
	end
	function family(pick)
		ms = Int[]
		for n in 0:4
			m = findfirst(e -> pick(e, n), exps)
			m === nothing || push!(ms, m)
		end
		rows = NamedTuple[]
		for i in eachindex(ms)
			push!(rows, (; deg = sum(Int, exps[ms[i]]),
				norm = sqrt(sum(abs2, col(ms[i]))),
				cos_next = i < length(ms) ? cosang(col(ms[i]), col(ms[i + 1])) : NaN,
				cos_top = i < length(ms) ? cosang(col(ms[i]), col(ms[end])) : NaN))
		end
		return rows
	end
	amp = family((e, n) -> e[1] == n + 1 && e[2] == n && sum(e) == 2n + 1)
	mean = family((e, n) -> n >= 1 && e[1] == n && e[2] == n && sum(e) == 2n)
	# "Settling" = the alignment with the top degree is still climbing at the last step we
	# can measure. It is a trend statement, not a converged limit; with five terms it cannot
	# be anything more.
	#
	# `nothing` when no family has three degrees to compare — the FAST profile stops at
	# order 3 and has two. Reporting `false` there reads as "the direction has stalled",
	# which is a claim the data cannot support either way.
	trend(f) = length(f) < 3 || isnan(f[end - 1].cos_top) ? nothing :
			   f[end - 1].cos_top > f[max(end - 2, 1)].cos_top
	ta, tm = trend(amp), trend(mean)
	return (; amplitude = amp, mean_flow = mean,
		settling = (ta === nothing && tm === nothing) ? nothing :
				   (ta === true || tm === true))
end

"""
	fold_overlap(W, Φ, λ; master) → NamedTuple

WHICH outer modes span the direction the manifold is straining in. Use this to choose what to
promote.

`backbone_direction` establishes THAT the high-degree content settles onto a single
direction; this decomposes that direction, as the cosine
`|⟨φ_k, W_d⟩| / (‖φ_k‖ ‖W_d‖)` over the top backbone degrees `d`.

**Right eigenvectors only, and that is the point.** `modal_growth` projects with `ψ`, and in
this descriptor pencil the modes carrying the fastest-growing manifold content are precisely
the ones whose pairing `α = ψᵀB₁φ` is degenerate (~1e-13) — `left_eigenvector` warns on every
one of them that its left and right vectors are not the same mode. Ranking on that measures
the adjoint solve, not the manifold. No adjoint enters here, so those modes cannot corrupt
the answer, and it costs nothing extra because `Φ` is already in hand.

Measured on the un-promoted Kármán order-9 run, the degree-9 backbone is led by
λ = −6.542148 + 18.415390i at 0.184 and −11.812255 + 19.322862i at 0.115, with **every real
mode below 0.038** — including all three that had been promoted on `modal_growth`'s advice.
Promoting a mode nearly orthogonal to this direction is a change of coordinates that changes
nothing, which is exactly what was measured: ρ_conv 1.95 either way and the order-9 fold
unmoved at Re 55.1.
"""
function fold_overlap(W, Φ::AbstractMatrix, λ::AbstractVector{<:Complex};
		master::AbstractVector{Int} = Int[])
	exps = W.poly.multiindex_set.exponents
	C = W.poly.coefficients
	cols = Tuple{Int, Int}[]
	for n in 4:-1:0                      # highest backbone degree first
		m = findfirst(e -> e[1] == n + 1 && e[2] == n && sum(e) == 2n + 1, exps)
		m === nothing || push!(cols, (2n + 1, m))
	end
	isempty(cols) && return (; degrees = Int[], modes = NamedTuple[])
	out = NamedTuple[]
	for k in axes(Φ, 2)
		k in master && continue
		φ = @view Φ[:, k]
		nφ = sqrt(sum(abs2, φ))
		ov = [begin
			w = @view C[:, 1, m]
			nw = sqrt(sum(abs2, w))
			(nφ > 0 && nw > 0) ? abs(dot(φ, w)) / (nφ * nw) : NaN
		end for (_, m) in cols]
		push!(out, (; mode = k, λ = λ[k], overlaps = ov, top = first(ov)))
	end
	sort!(out; by = r -> isnan(r.top) ? -Inf : -r.top)
	return (; degrees = [d for (d, _) in cols], modes = out)
end

"""
	modal_growth(W, B₁, ψ, λ_outer) → Vector{NamedTuple}

Decompose the manifold's amplitude backbone into outer modes and rank them by how fast
each grows with degree. `ψ` holds the left eigenvectors of the outer modes (columns), so
the modal amplitude of mode k at backbone degree d is `c_k = ψ_kᵀ B₁ W[:, m_d]`.

⚠ **Do not select promotion candidates with this — use `fold_overlap`.** The ranking needs
`ψ`, and here the top eight rows are all modes whose pairing is degenerate at ~1e-13, where
`left_eigenvector` warns that the left and right vectors are not the same mode. Acting on it
sent three separate runs to real modes whose actual overlap with the fold direction is below
0.038, and all three changed nothing. Kept because the growth ratio is still the honest
answer to a different question — how fast a given mode's component grows with degree, given a
trustworthy `ψ`.
"""
function modal_growth(W, B₁, ψ::AbstractMatrix, λ_outer::AbstractVector{<:Complex})
	exps = W.poly.multiindex_set.exponents
	C = W.poly.coefficients
	degs = Int[]
	cols = Int[]
	for n in 0:4
		m = findfirst(e -> e[1] == n + 1 && e[2] == n && sum(e) == 2n + 1, exps)
		m === nothing && continue
		push!(degs, 2n + 1)
		push!(cols, m)
	end
	# Normalise by ‖ψ_k‖. `left_eigenvector` returns ψ in the SymmetricBiorthogonal gauge,
	# i.e. divided by √α — and α spans 1e-6 down to 1e-14 across this spectrum, so an
	# unnormalised projection is amplified by up to 1e7 for exactly the modes whose pairing
	# is degenerate. Ranking on that measures the gauge, not the manifold: it put
	# λ = −10.25+4.47i (α ≈ 9e-14) on top purely because its ψ had been blown up.
	amp = Matrix{Float64}(undef, size(ψ, 2), length(cols))
	for (j, m) in enumerate(cols)
		Bw = B₁ * @view C[:, 1, m]
		for k in axes(ψ, 2)
			nψ = sqrt(sum(abs2, @view ψ[:, k]))
			amp[k, j] = nψ > 0 ? abs(dot(conj(@view ψ[:, k]), Bw)) / nψ : NaN
		end
	end
	out = NamedTuple[]
	for k in axes(ψ, 2)
		seq = @view amp[k, :]
		g = (length(seq) >= 2 && seq[end - 1] > 0) ? seq[end] / seq[end - 1] : NaN
		push!(out, (; mode = k, λ = λ_outer[k], amplitudes = collect(seq), growth = g))
	end
	sort!(out, by = r -> isnan(r.growth) ? -Inf : -r.growth)
	return (; degrees = degs, modes = out)
end
