"""
	resummation.jl — evaluating the DPIM series past its radius of convergence.

The Kármán manifold has a singularity at `ρ_c ≈ 1.95` and `domb_sykes` classifies it as
`γ ≈ 0.42` — a square-root **branch point**, not a pole. The limit cycle reaches ρ ≈ 2.2 at
Re 54 and ρ ≈ 2.7 at Re 55, so every observable above Re ≈ 53 is a Taylor series being summed
OUTSIDE its disc. That is why more terms make things worse and why the orders alternate
(3 fine, 5 folds, 7 fine, 9 folds): the partial sums of a divergent series oscillate.

Measured on the un-promoted order-9 run, period-averaged TKE against DNS at the DNS amplitude:

| Re    | ρ    | Taylor-3 | Taylor-9   | Padé[2/2] |
|-------|------|----------|------------|-----------|
| 50.50 | 1.13 |  −5.4 %  |   +0.2 %   |  −0.1 %   |
| 52.00 | 1.64 |  −7.6 %  |   +9.2 %   |  +0.5 %   |
| 54.00 | 2.17 | +31.0 %  | +1116.4 %  |  −0.0 %   |

Same coefficients, different summation. Nothing here touches `R` or `W` — resummation is
post-processing, so `rom_invariants` and every reference comparison are unaffected.

**Padé** `[L/M]` — a rational function fitted to the series — is the whole toolkit. It needs no
knowledge of the singularity, and a rational function represents a branch CUT by stacking poles
along it, so the spare order goes in the denominator: with four coefficients `[2/1]` gives
−50 % against DNS at Re 54 while `[1/2]` gives −24 %.

An Euler/conformal map `w = 1 − (1 − u/u_c)^γ` was tried and removed. Two reasons, both
measured. Substituting `u(w)` back into the truncated `u`-polynomial and evaluating at the
matching `w` returns the original value minus whatever the `w`-truncation drops, so it is a
no-op that can only lose accuracy — it scored −104 % at Re 52 where Padé scores +0.1 %. And
`domb_sykes` puts `ρ_c` at 1.81 while the DNS shows the TKE perfectly smooth at ρ = 2.17, so the
singularity is NOT on the positive real axis and a real `u_c` is the wrong map for it.

Everything is a series in `u = ρ²`, never in ρ: `G(ρ) = Re(R₁)/ρ` and the period-averaged TKE
are both even in ρ, so working in `u` halves the degree and doubles the effective order.
"""

"""
	pade(c, L, M) → Function

`[L/M]` Padé approximant of `Σ c[k+1] u^k`, returned as a callable in `u`.

Requires `L + M + 1` coefficients. Returns `nothing` when the Toeplitz system is singular —
that happens for a genuinely degenerate series and must not be papered over, because a
silently-wrong approximant is far worse than a missing one.
"""
function pade(c::AbstractVector{<:Real}, L::Int, M::Int)
	length(c) >= L + M + 1 || return nothing
	# `c` is 1-based, so `c[k + 1]` is the coefficient of `u^k`. The condition is
	# `Σ_{j=1..M} Q_j c_{L+i-j} = −c_{L+i}` for `i = 1…M`; writing `A[i,j] = c_{L+i-j}` with a
	# 1-based `c` means index `L + i - j + 1`, NOT `L + 1 + i - j`. That off-by-one put a
	# pole–zero pair at `u ≈ 1` and turned +0.5 % into −42 %.
	A = zeros(Float64, M, M)
	b = zeros(Float64, M)
	for i in 1:M
		b[i] = -c[L + i + 1]
		for j in 1:M
			k = L + i - j
			A[i, j] = (0 <= k <= length(c) - 1) ? c[k + 1] : 0.0
		end
	end
	q = M == 0 ? Float64[] : begin
		F = lu(A; check = false)
		issuccess(F) || return nothing
		F \ b
	end
	Q = vcat(1.0, q)
	P = [sum(c[k - j + 1] * Q[j + 1] for j in 0:min(k, M)) for k in 0:L]
	return function (u)
		den = evalpoly(u, Q)
		den == 0 && return NaN
		return evalpoly(u, P) / den
	end
end

"""
	resum(c; method) → Function

Callable summation of `Σ c[k+1] u^k`. `method`:

  · `:taylor` — plain partial sum. Reproduces the pre-resummation behaviour exactly, so it is
    the control every comparison should carry.
  · `:pade` — diagonal (or near-diagonal) Padé from all available coefficients.

Falls back to `:taylor` rather than returning `nothing`, so a caller never silently loses a
curve; the fallback is visible because the resummed and Taylor curves then coincide. Fewer than
three coefficients is such a case — there is nothing to fit.
"""
function resum(c::AbstractVector{<:Real}; method::Symbol = :pade)
	taylor = u -> evalpoly(u, c)
	method === :taylor && return taylor
	n = length(c)
	n >= 3 || return taylor
	# ALWAYS the largest DIAGONAL [M/M], discarding the last coefficient when the count is
	# even. Diagonal Padé is the standard choice for a function with a branch cut, and the
	# measurements here are unambiguous: every off-diagonal shape has been worse than the
	# smaller diagonal one built from FEWER coefficients.
	#
	#   3 coeffs → [1/1]  TKE +10 % at Re 54            5 coeffs → [2/2]  TKE −2.8 %
	#   4 coeffs → [1/2]  TKE −24 %  (worse than [1/1]) 6 coeffs → [2/3]  branch folds at Re 51
	#
	# So an even count buys nothing and can cost a great deal — order 11 gives six
	# coefficients and its branch collapsed to Re 50.97 while order 9's reached Re 69.88.
	# Useful order therefore advances two at a time in the ODD counts: order 9 → [2/2],
	# order 13 → [3/3], order 17 → [4/4].
	M = (n - 1) ÷ 2
	f = pade(c, M, M)
	return f === nothing ? taylor : f
end

"""
	amplitude_series(R, η, N) → Vector{Float64}

Coefficients of `G(ρ) = Re(R₁(ρ, ρ, η)) / ρ = Σ gₙ uⁿ`, `u = ρ²`, for the order-`N` truncation.

The ρ-direction sibling of `eta_series_report`: `G`'s zero IS the limit cycle, so these are the
coefficients whose divergence caps the branch. Closed form off `R₁`'s `z₁^{n+1} z̄₁^n η^c`
family — those are the only monomials that survive on the orbit at harmonic 1, since
`z₁^a z̄₁^b` sits at harmonic `a − b`.

Promoted coordinates are NOT included: this is the y-free part of `R₁`. Use `rom_po_residual`
when the y feedback matters; use this when the object of interest is the series itself.
"""
function amplitude_series(R, η::Float64, N::Int)
	ex = _exps(R)
	nv = _nvar(R)
	g = Dict{Int, Float64}()
	for (m, e) in enumerate(ex)
		a, b, c = Int(e[1]), Int(e[2]), Int(e[nv])
		a == b + 1 || continue
		a + b + c <= N || continue
		any(t -> e[t] != 0, 3:(nv - 1)) && continue          # y-free part
		g[b] = get(g, b, 0.0) + real(_coefk(R, 1, m) * (c == 0 ? 1.0 : η^c))
	end
	isempty(g) && return Float64[]
	return [get(g, k, 0.0) for k in 0:maximum(keys(g))]
end

"""
	_real_roots(p; rtol) → Vector{Float64}

Real roots of `Σ p[k+1] x^k`, by companion-matrix eigenvalues. No external dependency and no
bracketing, so a root cannot be missed between grid points the way a sign-change scan misses
double roots.
"""
function _real_roots(p::AbstractVector{Float64}; rtol::Float64 = 1e-8)
	q = collect(Float64, p)
	scale = maximum(abs, q; init = 0.0)
	scale > 0 || return Float64[]
	while length(q) > 1 && abs(q[end]) <= rtol * scale
		pop!(q)
	end
	length(q) < 2 && return Float64[]
	n = length(q) - 1
	c = q ./ q[end]
	Cm = zeros(Float64, n, n)
	for i in 1:(n - 1)
		Cm[i + 1, i] = 1.0
	end
	for i in 1:n
		Cm[i, n] = -c[i]
	end
	λ = eigvals(Cm)
	mag = maximum(abs, λ; init = 1.0)
	return sort([real(z) for z in λ if abs(imag(z)) <= rtol * max(abs(real(z)), mag)])
end

"""
	branch_by_amplitude(R, N; re0, rho_max, n_rho, re_min, re_max, method) → rows

The limit-cycle branch, parametrised by AMPLITUDE instead of by Reynolds number.

`G(ρ, η) = 0` is one equation in two unknowns; which one you solve for is a free choice, and
the conventional choice is the bad one here. Continuing in Re makes ρ the unknown, so the
solver works in the direction whose series is DIVERGENT over most of the range (ρ ≈ 2.2 at
Re 54 and ≈ 3.5 at Re 70, against ρ_conv ≈ 1.95). Fixing ρ instead makes η the unknown, and at
fixed ρ, `G` is exactly a POLYNOMIAL in η — solved by companion matrix, no iteration — in the
direction that converges comfortably (η(70) is 58 % of its radius, last term 1.9 % at order 9).

Everything the arclength continuation needed disappears with it: no `ds`/`dsmax` tuning, no
`ρ_ref` scaling, no Newton tolerance to match a finite-difference Jacobian, no runaway (order 7
once reached ρ = 234), and no fold-chasing — a fold in `ρ(Re)` is just a monotone `Re(ρ)`.
Measured at order 9 against the DNS points: ρ = 0.65 → Re 49.50, 1.13 → 50.50, 1.64 → 52.00,
2.17 → 53.83, against DNS 49.5 / 50.5 / 52.0 / 54.0.

The ρ direction is still resummed — each η-coefficient `p_c(ρ) = Σ_n Re(c_{n+1,n,c}) u^n` is a
series in `u = ρ²` and gets `resum`'d before the η-polynomial is assembled.

Branch selection marches in ρ and takes the η root nearest the previous one, seeded from the
Hopf. That is continuation, but in the variable that cannot fold.
"""
function branch_by_amplitude(R, N::Int; re0::Float64 = 49.03, rho_max::Float64 = 8.0,
		n_rho::Int = 600, re_min::Float64 = 40.0, re_max::Float64 = 70.0,
		method::Symbol = :pade)
	Rt = truncate_dynamics(R, N)
	ex = _exps(Rt)
	nv = _nvar(Rt)
	tab = Dict{Int, Vector{Float64}}()          # η power → series in u
	for (m, e) in enumerate(ex)
		a, b, c = Int(e[1]), Int(e[2]), Int(e[nv])
		(a == b + 1 && a + b + c <= N) || continue
		any(t -> e[t] != 0, 3:(nv - 1)) && continue      # y-free part
		v = get!(() -> Float64[], tab, c)
		while length(v) <= b
			push!(v, 0.0)
		end
		v[b + 1] += real(_coefk(Rt, 1, m))
	end
	isempty(tab) && return NTuple{6, Float64}[]
	cmax = maximum(keys(tab))
	sums = Dict(c => resum(v; method = method) for (c, v) in tab)

	rows = NTuple{6, Float64}[]
	η_prev = NaN
	re_prev = NaN
	folds = 0
	dir_prev = 0
	for ρ in range(rho_max / n_rho, rho_max; length = n_rho)
		u = ρ^2
		p = [haskey(sums, c) ? sums[c](u) : 0.0 for c in 0:cmax]
		all(isfinite, p) || continue
		cand = [η for η in _real_roots(p) if η + 1 / re0 > 0]
		isempty(cand) && continue
		# Nearest to the previous point; at the first point take the root closest to η = 0,
		# which is the Hopf the branch is born from.
		η = isnan(η_prev) ? cand[argmin(abs.(cand))] : cand[argmin(abs.(cand .- η_prev))]
		re = 1 / (η + 1 / re0)
		(isfinite(re) && re_min <= re <= re_max) || continue
		# Count TURNING POINTS, not decreasing steps: a fold is a reversal of direction in Re
		# along increasing ρ. Same convention as the CSV's `fold` column downstream.
		if !isnan(re_prev)
			dir = re > re_prev ? 1 : -1
			dir_prev != 0 && dir != dir_prev && (folds += 1)
			dir_prev = dir
		end
		Ω = rom_po_frequency(ρ, η, Rt)
		push!(rows, (η, re, ρ, Ω, 2π / abs(Ω), Float64(folds)))
		η_prev = η
		re_prev = re
	end
	return rows
end

"""
	tke_series(G, A, η, N; consistent = false) → Vector{Float64}

Coefficients of the period-averaged fluctuation TKE, `T(ρ) = Σ t_k u^k`, `u = ρ²`.

Closed form — no orbit sampling. On `z₁ = ρe^{iθ}` the monomial `z₁^a z̄₁^b η^c` sits at
harmonic `s = a − b`, so:

  · `a = b` is constant on the orbit and is removed by the period-mean subtraction (it IS the
    mean-flow distortion — correctly excluded from a FLUCTUATION energy);
  · a product `ζ_m ζ_n` survives the period average only when `s_m + s_n = 0`.

so `t_k = ½ Σ_{s_m + s_n = 0} Re(G_mn η^{c_m + c_n})` collected at `k = (a_m+b_m+a_n+b_n)/2`.
Equivalent to `figures.py::tke_from_state` with `NS → ∞`, and agrees with it to round-off;
having it as a series is what allows the sum to be resummed instead of merely evaluated.

`G` and `A` are the Gram and exponent table written by `write_energy_gram`. `N` truncates on
the CORE degree, matching `truncate_dynamics`.

**`consistent` decides which coefficients you get, and the right choice depends on what you do
with them.**

  · `false` (default) — every product of two kept monomials, so the series runs to `ρ^{2N}`.
    This is what `figures.py` evaluates and it is the more ACCURATE Taylor sum: `‖u'_N‖²` has
    error `O(ρ^{N+2})` against `O(ρ^{N+1})` for a degree-`N` truncation. But its coefficients
    above total degree `N` are INCOMPLETE — they are products of known terms missing the
    contributions of the unknown ones.
  · `true` — only pairs with `core(m) + core(n) ≤ N`, every coefficient exact.

**Resummation must use `consistent = true`.** Padé fits the coefficient SEQUENCE, so feeding
it the incomplete tail makes it model an artefact: at Re 54 the full sequence gives −106 %
against DNS while the exact one gives −0.0 %. A Taylor sum merely adds the tail on and is
barely harmed; a rational fit is led by it.
"""
function tke_series(G::AbstractMatrix{<:Complex}, A::AbstractMatrix{<:Integer},
		η::Float64, N::Int; consistent::Bool = false)
	L = size(A, 1)
	nv = size(A, 2)
	core(m) = Int(A[m, 1]) + Int(A[m, 2]) + Int(A[m, nv])
	keep = [m for m in 1:L if core(m) <= N && A[m, 1] != A[m, 2]]
	t = Dict{Int, Float64}()
	for m in keep
		am, bm, cm = Int(A[m, 1]), Int(A[m, 2]), Int(A[m, nv])
		for n in keep
			an, bn, cn = Int(A[n, 1]), Int(A[n, 2]), Int(A[n, nv])
			(am - bm) + (an - bn) == 0 || continue
			consistent && (core(m) + core(n) > N) && continue
			k = (am + bm + an + bn) ÷ 2
			ηp = (cm + cn) == 0 ? 1.0 : η^(cm + cn)
			t[k] = get(t, k, 0.0) + 0.5 * real(G[m, n] * ηp)
		end
	end
	isempty(t) && return Float64[]
	return [get(t, k, 0.0) for k in 0:maximum(keys(t))]
end
