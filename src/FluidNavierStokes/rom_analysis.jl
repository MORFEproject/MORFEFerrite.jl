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
	ex = _exps(R)

	bs = Dict{Int, Vector{ComplexF64}}()          # drive of promoted rows, per harmonic
	A0 = zeros(ComplexF64, max(npro, 1), max(npro, 1))
	r1_core = zero(ComplexF64)                    # y-free monomials of R₁ at harmonic 1
	r1_y = Dict{Int, Vector{ComplexF64}}()        # core harmonic → coefficient on each y_j

	for (m, e) in enumerate(ex)
		a, b, c = e[1], e[2], e[nv]
		s = a - b
		v = ρ^(a + b) * (c == 0 ? 1.0 : η^c)
		j = 0                                     # which promoted coordinate, 0 = none
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

	ys = Dict{Int, Vector{ComplexF64}}()
	for (s, bvec) in bs
		ys[s] = ((im * s * Ω) * I - A0) \ bvec
	end
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
	promoted = nv > 3
	ρc = (promoted || isempty(amp)) ? NaN : last(amp).rho_conv
	ηc = (promoted || isempty(eta)) ? NaN : last(eta).eta_conv
	return (; amplitude = amp, eta = eta, rho_conv = ρc, eta_conv = ηc,
		valid = !promoted)
end

"""
	modal_growth(W, B₁, ψ, λ_outer) → Vector{NamedTuple}

Decompose the manifold's amplitude backbone into outer modes and rank them by how fast
each grows with degree. `ψ` holds the left eigenvectors of the outer modes (columns), so
the modal amplitude of mode k at backbone degree d is `c_k = ψ_kᵀ B₁ W[:, m_d]`.

This is the measurement `homological_denominators` cannot make: it reports which outer
mode is ACTUALLY excited as the order grows, numerator included, rather than which one
merely has the smallest denominator. The mode with the largest growth ratio is the one
limiting the expansion.
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
