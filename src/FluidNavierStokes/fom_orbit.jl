# fom_orbit.jl — full-order time integration and periodic-orbit solution for the
# perturbation Navier-Stokes system, used to produce the DNS reference the ROM is
# judged against.
#
# Integrates the descriptor system
#
#     B₁ ṡ = −B₀ s + f₂(s,s) + η′ K_visc s + η′ h₀_vec
#
# on the free_dpim DOF subspace with an IMEX θ-method (linear part implicit at weight
# θ, quadratic convection explicit), then drives the result onto a periodic orbit by
# ROM-seeded Picard iteration.
#
# Moved here from examples/05_karman_vortex_street/solver/{time_integration,picard_orbit}.jl
# verbatim: both were already free of example configuration, taking every parameter as an
# argument. The example now supplies those from its own constants.

"""
    time_integration.jl — IMEX θ-method FOM integrator for the perturbation NSE.

Integrates

    B₁ ṡ = −B₀ s + f₂(s,s) + η′ K_visc s + η′ h₀_vec

in the free_dpim DOF subspace with an implicit-explicit θ-method:
  • linear part (−B₀ s + η′ K_visc s) : implicit with weight θ
  • quadratic convection f₂(s,s)      : explicit (evaluated at the current step)

The LHS matrix B₁/Δt + θ(B₀ − η′ K_visc) is constant per (η′, T, Δt) and is
factorised once with KLU.  `K_visc` and `h₀_vec` follow main.jl's convention:
both already scaled by −_CYL_D, and h₀ built from the rectangular free×ALL
block of K_raw times the FULL base flow (the prescribed inlet DOFs carry the
Poiseuille profile — see stage 5 of main.jl).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Element-level assembly of f₂(s, s) = −∫ φ · (u·∇u) dΩ
# ─────────────────────────────────────────────────────────────────────────────

"""
    eval_perturbation_convection!(accum, s_free, fom)

Assemble the quadratic perturbation convection into `accum` (length n_free_dpim):

    accum[k] += −∫_Ω  φᵢ · (∇u · u)  dΩ     (velocity DOFs k only)

where u is the velocity part of `s_free` extracted element-by-element via
`fom.free_to_local_dpim`.  `accum` is zeroed on entry.
"""
function eval_perturbation_convection!(
    accum::Vector{Float64},
    s_free::Vector{Float64},
    fom,
)
    fill!(accum, 0.0)

    n_dpc = ndofs_per_cell(fom.dh)
    n_vel = fom.n_vel_dofs_per_cell

    Fe = zeros(Float64, n_dpc)
    u_e = zeros(Float64, n_vel)

    for element in CellIterator(fom.dh)
        reinit!(fom.cv_vel, element)
        dofs = celldofs(element)
        vel_dofs = dofs[fom.dof_range_u]   # global velocity DOF indices for this cell

        # Extract velocity values from the free-DOF state vector
        fill!(u_e, 0.0)
        for (i, d) in enumerate(vel_dofs)
            li = get(fom.free_to_local_dpim, d, 0)
            li != 0 && (u_e[i] = s_free[li])
        end

        fill!(Fe, 0.0)
        for q in 1:getnquadpoints(fom.cv_vel)
            dΩ = getdetJdV(fom.cv_vel, q)
            u_q = function_value(fom.cv_vel, q, u_e)   # Vec{2,Float64}
            ∇u_q = function_gradient(fom.cv_vel, q, u_e)   # Tensor{2,2,Float64}

            # f₂(s,s) = −(u·∇)u;  in Ferrite: (∇u_q ⋅ u_q)[i] = Σⱼ ∂_j uᵢ · uⱼ = (u·∇u)ᵢ
            conv = -(∇u_q ⋅ u_q)   # Vec{2,Float64}

            for i in 1:n_vel
                ri = fom.dof_range_u[i]
                φᵢ = shape_value(fom.cv_vel, q, i)   # Vec{2,Float64}
                Fe[ri] += (φᵢ ⋅ conv) * dΩ
            end
        end

        # Scatter element residual into free-DOF accumulator
        for (r, d) in enumerate(dofs)
            li = get(fom.free_to_local_dpim, d, 0)
            li != 0 && (accum[li] += Fe[r])
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# IMEX θ-method operators and orbit stepping
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_imex_operators(B₀, B₁, K_visc, η_prime, T, Δt; θ = 0.5, γ = 0.0)
    -> (L_klu, RHS_M, Δt_exact, n_steps)

Assemble and KLU-factorise the θ-method operators for one orbit of period `T`,
with the step count rounded so that `n_steps · Δt_exact == T` exactly.
`γ > 0` adds artificial damping `γ B₁` to the implicit operator, shifting every
eigenvalue left by γ — used for transient suppression during spin-up.
"""
function build_imex_operators(B₀, B₁, K_visc, η_prime, T, Δt; θ = 0.5, γ = 0.0)
    n_steps = max(1, round(Int, T / Δt))
    Δt_exact = T / n_steps
    inv_dt = 1.0 / Δt_exact

    A_imp = B₀ .- η_prime .* K_visc
    γ > 0.0 && (A_imp = A_imp .+ γ .* B₁)

    LHS = inv_dt .* B₁ .+ θ .* A_imp
    RHS_M = inv_dt .* B₁ .- (1.0 - θ) .* A_imp

    return klu(LHS), RHS_M, Δt_exact, n_steps
end

"""
    integrate_orbit!(s, n_steps, η_prime, fom, L_klu, RHS_M, h₀_vec, f2, rhs;
                     on_step = nothing) -> Bool

Advance `s` in place by `n_steps` IMEX steps:

    (B₁/Δt + θ A) s⁺ = (B₁/Δt − (1−θ) A) s + f₂(s,s) + η′ h₀_vec

`f2` and `rhs` are pre-allocated work vectors (length of `s`).  If given,
`on_step(step, s)` is called after every accepted step (lift recording, state
storage, norm tracking).  Returns `false` on blow-up (non-finite state).
"""
function integrate_orbit!(
    s::Vector{Float64},
    n_steps::Int,
    η_prime::Float64,
    fom,
    L_klu,
    RHS_M::AbstractMatrix,
    h₀_vec::Vector{Float64},
    f2::Vector{Float64},
    rhs::Vector{Float64};
    on_step = nothing,
)
    for step in 1:n_steps
        eval_perturbation_convection!(f2, s, fom)
        mul!(rhs, RHS_M, s)
        axpy!(1.0, f2, rhs)
        axpy!(η_prime, h₀_vec, rhs)
        ldiv!(s, L_klu, rhs)
        isfinite(dot(s, s)) || return false
        on_step === nothing || on_step(step, s)
    end
    return true
end
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

# ─────────────────────────────────────────────────────────────────────────────
# Periodic-orbit measurement — seeding from a ROM branch, and the observables of a
# converged orbit. Moved from examples/05_karman_vortex_street/fom_reference.jl.
#
# ONE adaptation on the way: `measure_orbit` read the example's global `FOM_REF_DT`.
# It is now the keyword `dt`, so the function carries no configuration of its own.
# ─────────────────────────────────────────────────────────────────────────────

"""
	seed_from_branch(branch, re_target) -> (ρ, T) or nothing

Linear interpolation of the branch amplitude ρ and period T at `re_target`
(columns eta,Re,rho,omega,T; rows ordered along the branch).
"""
function seed_from_branch(branch::Matrix{Float64}, re_target::Float64)
	for i in 1:(size(branch, 1) - 1)
		lo, hi = branch[i, 2], branch[i+1, 2]
		if (lo - re_target) * (hi - re_target) <= 0.0 && lo != hi
			w = (re_target - lo) / (hi - lo)
			ρ = (1 - w) * branch[i, 3] + w * branch[i+1, 3]
			T = (1 - w) * branch[i, 5] + w * branch[i+1, 5]
			return ρ, T
		end
	end
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Measurement orbit: lift statistics + period-averaged fluctuation TKE
# ─────────────────────────────────────────────────────────────────────────────

"""
	_periodic_lift(FL, K) -> FL_per

Project the one-period lift signal onto its first `K` temporal harmonics (the
measurement orbit spans EXACTLY one period at uniform steps, so the DFT bins
are exact Fourier coefficients of the periodic signal).

The converged orbit is T-periodic by construction, so any non-harmonic content
in the measured signal is numerical: under Crank–Nicolson (θ = ½) the algebraic
pressure mode of the descriptor system has amplification factor −1, and seed
inconsistencies ring at the Nyquist frequency forever — invisible to the
velocity mass norm that governs Picard convergence, but added on top of every
raw pressure-lift extremum (measured ≈ +9% at Re 49.5, growing with seed
error). The physical lift harmonics decay fast (j ≥ 2 are ~5 orders below the
fundamental at Re 49.5), so K = 10 is generous.
"""
function _periodic_lift(FL::Vector{Float64}, K::Int)
	n = length(FL)
	FL_per = zeros(Float64, n)
	for j in 0:min(K, div(n - 1, 2))
		c = sum(FL[k+1] * cis(-2π * j * k / n) for k in 0:(n-1)) / n
		w = j == 0 ? 1.0 : 2.0
		for k in 0:(n-1)
			FL_per[k+1] += w * real(c * cis(2π * j * k / n))
		end
	end
	return FL_per
end

"""
	measure_orbit(s_po, η_prime, T, fom, ops, l_free, M_vel, vel_rows, area)
	-> (; lift_max, lift_min, lift_mean, lift_ringing, FL, FL_per, Δt_meas,
		  avg_tke, periodicity_max, lift_amp_drift, μ_meas, ok)

Integrate TWO periods from the converged periodic-orbit point `s_po` and
evaluate the observables plus a definition-level periodicity verification.

Period 1 stores the velocity history and the lift signal; observables come
from it: lift statistics from the harmonic projection `FL_per` (see
[`_periodic_lift`](@ref); `lift_ringing` is the peak raw-minus-periodic
residual — the numerical pollution amplitude), and the TKE subtracts the
PERIOD MEAN (fluctuation TKE — matches tke_from_gram's ζ̄ subtraction;
velocity is constraint-consistent, so it needs no filtering).

Period 2 verifies periodicity BY ITS DEFINITION — the state at t must equal
the state at t+T for ALL t, not just at the Picard anchor (with a finite
anchor residual the deviation propagates and can transiently grow within a
period; the flow is non-normal):

- `periodicity_max = max_t ‖v(t+T) − v(t)‖_{M_vel} / max_t ‖v(t)‖_{M_vel}` —
  the uniform-in-time periodicity error (velocity mass norm; pressure is
  excluded — its θ = ½ Nyquist ringing is handled by the harmonic projection).
- `lift_amp_drift` — relative change of the harmonic-projected lift amplitude
  between the two periods (observable stationarity).
- `μ_meas = ‖v(2T)−v(T)‖ / ‖v(T)−v(0)‖` — the dominant transversal Floquet
  multiplier measured from the two anchor mismatches (deviation δ contracts to
  μδ per period, so consecutive closures scale by μ).
"""
function measure_orbit(s_po, η_prime, T, fom, ops, l_free, M_vel, vel_rows, area;
		dt::Float64 = 5e-4)
	L_klu, RHS_M, Δt_meas, n_steps = build_imex_operators(
		ops.B₀, ops.B₁, ops.K_visc, η_prime, T, dt; θ = 0.5)
	n_steps < 100 && @warn "measure_orbit: only $n_steps steps per period — " *
		  "T collapsed towards Δt, upstream period gate failed?"

	V = Matrix{Float64}(undef, length(vel_rows), n_steps)
	FL = Vector{Float64}(undef, n_steps)
	record = function (k, sv)
		@views V[:, k] .= sv[vel_rows]
		FL[k] = dot(l_free, sv)
		return nothing
	end

	s = copy(s_po)
	f2 = similar(s)
	rhs = similar(s)
	fail = (; lift_max = NaN, lift_min = NaN, lift_mean = NaN,
		lift_ringing = NaN, FL = FL, FL_per = FL, Δt_meas = Δt_meas,
		avg_tke = NaN, periodicity_max = NaN, lift_amp_drift = NaN,
		μ_meas = NaN, ok = false)

	ok = integrate_orbit!(s, n_steps, η_prime, fom, L_klu, RHS_M,
		ops.h₀_vec, f2, rhs; on_step = record)
	ok || return fail

	# ── Period 2: pointwise periodicity check  d(t) = ‖v(t+T) − v(t)‖_{M_vel} ─
	FL2 = Vector{Float64}(undef, n_steps)
	w2 = zeros(Float64, length(vel_rows))
	Mw2 = similar(w2)
	d_max = 0.0
	d_end = 0.0
	check = function (k, sv)
		@views w2 .= sv[vel_rows] .- V[:, k]
		mul!(Mw2, M_vel, w2)
		d = sqrt(max(dot(w2, Mw2), 0.0))
		d_max = max(d_max, d)
		k == n_steps && (d_end = d)   # ‖v(2T) − v(T)‖
		FL2[k] = dot(l_free, sv)
		return nothing
	end
	ok = integrate_orbit!(s, n_steps, η_prime, fom, L_klu, RHS_M,
		ops.h₀_vec, f2, rhs; on_step = check)
	ok || return fail

	v̄ = vec(sum(V, dims = 2)) ./ n_steps
	w = similar(v̄)
	Mw = similar(v̄)
	tke_sum = 0.0
	X_max_vel = 0.0
	for k in 1:n_steps
		@views w .= V[:, k]
		mul!(Mw, M_vel, w)
		X_max_vel = max(X_max_vel, sqrt(max(dot(w, Mw), 0.0)))
		w .-= v̄
		mul!(Mw, M_vel, w)
		tke_sum += 0.5 * dot(w, Mw)
	end

	# anchor mismatch of period 1: ‖v(T) − v(0)‖ → Floquet ratio with d_end
	@views w .= V[:, n_steps] .- s_po[vel_rows]
	mul!(Mw, M_vel, w)
	d_anchor = sqrt(max(dot(w, Mw), 0.0))
	μ_meas = (d_anchor > 0.0 && d_end > 0.0) ? d_end / d_anchor : NaN

	FL_per = _periodic_lift(FL, 10)
	FL2_per = _periodic_lift(FL2, 10)
	amp1 = (maximum(FL_per) - minimum(FL_per)) / 2
	amp2 = (maximum(FL2_per) - minimum(FL2_per)) / 2

	return (; lift_max = maximum(FL_per), lift_min = minimum(FL_per),
		lift_mean = sum(FL_per) / n_steps,
		lift_ringing = maximum(abs.(FL .- FL_per)),
		FL = FL, FL_per = FL_per, Δt_meas = Δt_meas,
		avg_tke = tke_sum / n_steps / area,
		periodicity_max = X_max_vel > 0.0 ? d_max / X_max_vel : NaN,
		lift_amp_drift = amp1 > 0.0 ? abs(amp2 - amp1) / amp1 : NaN,
		μ_meas = μ_meas, ok = true)
end
