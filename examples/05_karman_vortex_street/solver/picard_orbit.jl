"""
	picard_orbit.jl — ROM-seeded Picard iteration for FOM periodic orbits.

Starting from a ROM-predicted initial condition (s_guess, T_ROM), drives the
trajectory onto the attracting FOM limit cycle:

  1. n_spinup UNDAMPED orbits: the cycle is the attractor and the ROM seed
	 starts near it, so the natural dynamics are the correct relaxation.
  2. A 3-orbit probe estimating the FOM period as the gated normalised-
	 autocorrelation peak of the mean-subtracted lift signal (lags in
	 [0.5, 2]·T_ROM, peak correlation > 0.5 required). Zero-crossing counting
	 is not viable: the pressure-borne lift carries fast components that are
	 invisible to the velocity mass norm but dominate the crossings.
  3. Picard orbits s ← Φ(s, T) until the mass-norm periodicity residual
	 ‖Φ(s) − s‖_{B₁} / X_max drops below `tol`.  Near the cycle the period is
	 refined each orbit by projecting the residual onto the trajectory tangent
	 (δT = ⟨ṡ, E⟩_{B₁}/⟨ṡ, ṡ⟩_{B₁}) — a fixed-T map floors at the phase drift
	 |T − T_exact|·‖ṡ‖, which this removes.  δT is evaluated BEFORE the
	 convergence check and the final correction is applied to the returned
	 period, so T_fom is always FOM-measured (never just the ROM seed period).
	 All period updates are clamped to [0.5, 2]·T_ROM.

At convergence, s_po = s(0) satisfies  Φ(s_po, T_fom) ≈ s_po  to the requested
tolerance and is the FOM periodic-orbit representative for that period.

(Picard-only revival of the pre-restructure solver/shooting.jl; the stepping
lives in solver/time_integration.jl.)
"""

"""
	_estimate_period_autocorr(signal, dt, τ_min, τ_max) -> (T_est, c_peak)

Estimate the dominant period of `signal` (assumed mean-subtracted) as the lag
of the highest NORMALISED autocorrelation inside `[τ_min, τ_max]`, refined to
sub-sample accuracy by a parabola through the peak and its neighbours.

Returns `(NaN, 0.0)` when the peak sits on the window edge (no interior
maximum). `c_peak ∈ [−1, 1]` is the normalised correlation at the peak — ≈ 1
for a genuinely periodic signal, ≈ 0 for noise; gate on it.

Zero-crossing counting is NOT robust here: the lift is a pressure functional,
and pressure — the algebraic variable of the descriptor system — carries fast
components invisible to the velocity mass norm that pollute the crossings.
Fast components only correlate at their own short lags, outside the gated
window, so the autocorrelation peak stays on the shedding period.
"""
function _estimate_period_autocorr(signal::Vector{Float64}, dt::Float64,
		τ_min::Float64, τ_max::Float64)
	n = length(signal)
	k_min = max(1, floor(Int, τ_min / dt))
	k_max = min(n - 2, ceil(Int, τ_max / dt))
	k_max <= k_min + 1 && return NaN, 0.0

	c = fill(-Inf, k_max)
	for k in k_min:k_max
		num = 0.0
		d1 = 0.0
		d2 = 0.0
		@inbounds for i in 1:(n - k)
			num += signal[i] * signal[i+k]
			d1 += signal[i]^2
			d2 += signal[i+k]^2
		end
		den = sqrt(d1 * d2)
		c[k] = den > 0.0 ? num / den : -Inf
	end

	k_star = k_min
	for k in (k_min+1):k_max
		c[k] > c[k_star] && (k_star = k)
	end
	(k_star == k_min || k_star == k_max) && return NaN, 0.0

	cm, c0, cp = c[k_star-1], c[k_star], c[k_star+1]
	curv = cm - 2c0 + cp
	δ = curv < 0.0 ? 0.5 * (cm - cp) / curv : 0.0
	return (k_star + δ) * dt, c0
end

"""
	find_periodic_orbit(s_guess, η_prime, T, fom, B₀, B₁, K_visc, h₀_vec; kwargs...)
	→ (s_po, T_fom, E_norm, X_max, n_orbits, converged)

Find the FOM periodic orbit near the ROM prediction `(s_guess, T)` by Picard
iteration (integrate-and-map) with an undamped spin-up and a lift-based period
probe.

## Keyword arguments
- `Δt`          : time-step size (default `5e-4`)
- `θ`           : implicit weight (0.5 = Crank-Nicolson, 1 = backward Euler)
- `n_spinup`    : undamped spin-up orbits before the probe (default 3)
- `γ_spinup`    : OPTIONAL artificial damping `−γ B₁ s` during spin-up, for
				  seeds that overshoot badly (default 0.0).  WARNING: any
				  γ ≫ σ (the Hopf growth rate, ~0.1 s⁻¹ near onset) kills the
				  shedding oscillation itself within a couple of orbits and
				  collapses the trajectory onto the damped system's forced
				  equilibrium — leave at 0 unless you know the seed is bad.
- `max_picard`  : maximum undamped Picard orbits before giving up (default 400)
- `tol`         : convergence tolerance on `‖E‖_M / X_max` (default 1e-4)
- `lift_weights`: if provided (`Vector{Float64}` of length n_free_dpim), enables
				  the autocorrelation period probe on
				  `F_L(t) = dot(lift_weights, s)`; keeps the ROM period T when
				  the gated estimate is rejected.
- `verbose`     : print residual every orbit if `true`

## Returns
- `s_po`      : periodic-orbit initial condition
- `T_fom`     : period at convergence (probe estimate + tangent refinement)
- `E_norm`    : `‖Φ(s_po,T)−s_po‖_{B₁}` at convergence
- `X_max`     : `max_t ‖s(t)‖_{B₁}` over the last orbit
- `μ̂`         : dominant transversal Floquet multiplier estimated from the
				ratio of consecutive Picard residuals at fixed T (NaN when no
				valid pair occurred, e.g. first-orbit convergence). The anchor
				residual under-reports the distance to the cycle by 1/(1−μ̂):
				one orbit maps a transversal deviation d to μ̂·d, so the
				measured closure is only (1−μ̂)·d.
- `n_orbits`  : total number of periods integrated (spin-up + probe + Picard)
- `converged` : `true` if tolerance was met before `max_picard` was reached
"""
function find_periodic_orbit(
	s_guess::Vector{Float64},
	η_prime::Float64,
	T::Float64,
	fom,
	B₀, B₁,
	K_visc,
	h₀_vec::Vector{Float64};
	Δt::Float64 = 5e-4,
	θ::Float64 = 0.5,
	n_spinup::Int = 3,
	γ_spinup::Float64 = 0.0,
	max_picard::Int = 400,
	tol::Float64 = 1e-4,
	lift_weights::Union{Vector{Float64}, Nothing} = nothing,
	verbose::Bool = false,
)
	T_seed = T   # ROM period — all period updates are gated to [0.5, 2]·T_seed
	s = copy(s_guess)
	f2 = similar(s)
	rhs = similar(s)
	tmp = similar(s)
	total_orbits = 0

	# ── Phase 1: undamped spin-up (γ_spinup > 0 only for pathological seeds) ─
	if n_spinup > 0
		L_su, RHS_su, _, n_steps = build_imex_operators(
			B₀, B₁, K_visc, η_prime, T, Δt; θ, γ = γ_spinup)
		for orbit in 1:n_spinup
			ok = integrate_orbit!(s, n_steps, η_prime, fom, L_su, RHS_su,
				h₀_vec, f2, rhs)
			if !ok
				@warn "find_periodic_orbit: blow-up during spin-up orbit " *
					  "$orbit/$n_spinup — decrease Δt (or set γ_spinup ~ σ)"
				return s, T, NaN, NaN, NaN, total_orbits + orbit, false
			end
			verbose && @printf("  spin-up %3d/%3d  ‖s‖ = %.3e\n",
				orbit, n_spinup, sqrt(max(dot(s, s), 0.0)))
		end
		total_orbits += n_spinup
	end

	L_klu, RHS_M, Δt_exact, n_steps = build_imex_operators(
		B₀, B₁, K_visc, η_prime, T, Δt; θ)

	# ── Phase 2: period-estimation probe (3×T_ROM, mean-subtracted lift) ─────
	if lift_weights !== nothing
		n_probe = 3 * n_steps
		F_L_probe = Vector{Float64}(undef, n_probe)
		record = (k, sv) -> (F_L_probe[k] = dot(lift_weights, sv))
		ok = integrate_orbit!(s, n_probe, η_prime, fom, L_klu, RHS_M,
			h₀_vec, f2, rhs; on_step = record)
		if ok
			total_orbits += 3
			F_L_probe .-= sum(F_L_probe) / n_probe   # correlate the oscillation, not the offset
			T_probe, c_peak = _estimate_period_autocorr(
				F_L_probe, Δt_exact, 0.5 * T_seed, 2.0 * T_seed)
			if isfinite(T_probe) && c_peak > 0.5
				verbose && @printf("  period probe: T_ROM=%.6f → T_fom=%.6f  (peak corr %.3f)\n",
					T, T_probe, c_peak)
				T = T_probe
				L_klu, RHS_M, Δt_exact, n_steps = build_imex_operators(
					B₀, B₁, K_visc, η_prime, T, Δt; θ)
			else
				@warn "find_periodic_orbit: period probe rejected " *
					  "(T_probe = $T_probe, peak corr = $c_peak) — keeping T_ROM = $T_seed"
			end
		else
			@warn "find_periodic_orbit: blow-up during period-estimation probe"
			return s, T, NaN, NaN, NaN, total_orbits, false
		end
	end

	# ── Phase 3: undamped Picard with tangent-projection period refinement ──
	E_norm = NaN
	X_max = NaN
	μ̂ = NaN            # transversal Floquet multiplier from residual ratios
	E_norm_prev = NaN
	T_rebuilt = true   # residual ratios are only μ when T was NOT rebuilt between the orbits
	s_prev = similar(s)   # state one step before the orbit end → ṡ estimate
	E = similar(s)

	for orbit in 1:max_picard
		s_start = copy(s)
		X_max = 0.0
		track = function (k, sv)
			mul!(tmp, B₁, sv)
			X_max = max(X_max, sqrt(max(dot(sv, tmp), 0.0)))
			k == n_steps - 1 && (s_prev .= sv)
			return nothing
		end

		ok = integrate_orbit!(s, n_steps, η_prime, fom, L_klu, RHS_M,
			h₀_vec, f2, rhs; on_step = track)
		if !ok
			@warn "find_periodic_orbit: blow-up during Picard orbit $orbit"
			return s_start, T, NaN, NaN, NaN, total_orbits + orbit, false
		end

		E .= s .- s_start
		mul!(tmp, B₁, E)
		E_norm = sqrt(max(dot(E, tmp), 0.0))
		rel = X_max > 0.0 ? E_norm / X_max : NaN

		if !T_rebuilt && isfinite(E_norm_prev) && E_norm_prev > 0.0
			ratio = E_norm / E_norm_prev
			0.0 < ratio < 1.0 && (μ̂ = ratio)
		end
		E_norm_prev = E_norm
		T_rebuilt = false

		# Near the cycle the residual is dominated by phase drift (T off by δ):
		# Φ_T(s*) ≈ s* + (T − T_exact)·ṡ  ⇒  T_exact ≈ T − ⟨ṡ,E⟩_M/⟨ṡ,ṡ⟩_M.
		# Compute δT BEFORE the convergence check so the returned period is
		# always FOM-measured — even when the seed period was already accurate
		# enough for the residual to pass tol on the first orbit.
		δT = 0.0
		if isfinite(rel) && rel < 5e-2 && n_steps > 1
			@. s_prev = (s - s_prev) / Δt_exact   # s_prev ← ṡ(T)
			mul!(rhs, B₁, E)
			num = dot(s_prev, rhs)
			mul!(rhs, B₁, s_prev)
			den = dot(s_prev, rhs)
			den > 0.0 && (δT = num / den)
		end

		verbose && @printf("  Picard %3d  ‖E‖_M = %.3e  X_max = %.3e  rel = %.3e  T = %.6f  δT = %+.2e\n",
			total_orbits + orbit, E_norm, X_max, rel, T, δT)

		if isfinite(rel) && rel < tol
			T = clamp(T - δT, 0.5 * T_seed, 2.0 * T_seed)   # final phase-drift removal
			verbose && @printf("  converged: T_fom = %.8f  (final δT = %+.3e s)\n", T, δT)
			return s_start, T, E_norm, X_max, μ̂, total_orbits + orbit, true
		end

		if δT != 0.0
			T_new = clamp(T - δT, 0.5 * T_seed, 2.0 * T_seed)
			if abs(T_new - T) > 1e-10 * T
				T = T_new
				L_klu, RHS_M, Δt_exact, n_steps = build_imex_operators(
					B₀, B₁, K_visc, η_prime, T, Δt; θ)
				T_rebuilt = true
			end
		end
	end

	@warn "find_periodic_orbit: did not converge in $(total_orbits + max_picard) orbits " *
		  "(rel = $(X_max > 0 ? E_norm/X_max : NaN))"
	return s, T, E_norm, X_max, μ̂, total_orbits + max_picard, false
end
