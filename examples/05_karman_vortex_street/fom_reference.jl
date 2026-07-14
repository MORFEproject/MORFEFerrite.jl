"""
	fom_reference.jl — full-order reference for the Hopf tube (STEP 4).

For each Reynolds number in `FOM_REF_RE` (config.jl) this script converges the
ACTUAL full-order periodic orbit by time integration — no DPIM involved beyond
the initial guess — and records the physical observables the order-comparison
plots use:

  1. Seed:  s(0) = Re(W(ρ, ρ, η′)) at the ROM branch point nearest the target Re
	 (rom_branch_ord<N>.csv of the highest available order). Seeding from the ROM
	 does not bias the result — Picard converges to the FOM's own attracting
	 cycle; the seed only shortens the transient.
  2. Converge the orbit:  damped spin-up → lift-zero-crossing period probe →
	 Picard s ← Φ(s, T) with tangent-projection period refinement
	 (solver/picard_orbit.jl, solver/time_integration.jl).
  3. Measure one orbit:  shedding period/frequency, lift statistics, and the
	 period-averaged fluctuation TKE  ⟨½‖u′−ū‖²⟩/|Ω|  (the FOM counterpart of
	 tke_from_gram's ζ̄-subtracted evaluation).

Lift comparability: compare_orders.py drops the (0,0,c) monomials — the steady
base-flow correction in η′ — so the FOM counterpart of its max|lift| subtracts
the lift of the Newton STEADY solution at the target Re (the unstable fixed
point above Re_c, warm-started along the target list).

Writes  data/fom_reference.csv  per run directory; compare_orders.py overlays
the points automatically when the file exists.

Usage:
	julia --project=. fom_reference.jl                  # all results/Re*_ord*
	julia --project=. fom_reference.jl results/Re49.03_ord9 [more dirs...]

Linear operators are cached in data/linear_ops.jls (versioned — pre-h₀-fix
caches from the old pipeline are detected and recomputed).
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MORFE
using MORFE.Polynomials: evaluate
using Ferrite
using FerriteGmsh
using Gmsh
using LinearAlgebra
using SparseArrays
using KLU
using Printf
using Serialization
using DelimitedFiles

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fem", "mesh.jl"))
using MORFEFerrite.FluidNavierStokes
include(joinpath(@__DIR__, "solver", "time_integration.jl"))
include(joinpath(@__DIR__, "solver", "picard_orbit.jl"))

const OPS_CACHE_VERSION = 2   # v2 = post-h₀-fix (rectangular free×ALL block)

# ─────────────────────────────────────────────────────────────────────────────
# Operators: versioned cache per run directory
# ─────────────────────────────────────────────────────────────────────────────

"""
	load_or_build_operators(data_dir, fom, re0) -> (; version, B₀, B₁, K_visc, h₀_vec, s₀_full)

Load `data_dir/linear_ops.jls` if it carries the current cache version; otherwise
recompute (Newton base flow at `re0` + linear assembly, exactly as main.jl stages
3–5) and overwrite the cache. Old caches from the pre-restructure pipeline lack
the version field AND carry the buggy inlet-columns h₀ — they are never reused.
"""
function load_or_build_operators(data_dir::AbstractString, fom, re0::Float64)
	path = joinpath(data_dir, "linear_ops.jls")
	if isfile(path)
		ops = deserialize(path)
		if ops isa NamedTuple && haskey(ops, :version) && ops.version == OPS_CACHE_VERSION
			@info "Loaded cached operators from $path"
			return ops
		end
		@info "Stale linear_ops.jls (pre-h₀-fix cache) — recomputing"
	end

	@info "Computing operators (Newton base flow at Re₀ = $re0 + linear assembly) ..."
	(_, _, s₀_full) = solve_steady_state(fom; Re0 = re0)
	B₀, B₁ = assemble_linear_operators(s₀_full, fom; Re0 = re0)
	(K_visc, _K_visc_rect) = assemble_K_visc(fom)
	K_visc .*= -_CYL_D
	h₀_vec = -_CYL_D .* (_K_visc_rect * s₀_full)   # rectangular free×ALL block × FULL base flow

	ops = (; version = OPS_CACHE_VERSION, B₀, B₁, K_visc, h₀_vec, s₀_full)
	serialize(path, ops)
	@info "Cached operators to $path"
	return ops
end

# ─────────────────────────────────────────────────────────────────────────────
# ROM branch seeding
# ─────────────────────────────────────────────────────────────────────────────

"""
	load_seed_branch(data_dir, preferred_ord) -> (branch::Matrix{Float64}, order::Int)

Read rom_branch_ord<preferred_ord>.csv (columns eta,Re,rho,omega,T) for
seeding, falling back to the highest available order when that file is
missing. The seed only sets the transient length — never the converged FOM
orbit — so a lower-order branch is fine, and preferable when a higher order
folds back before covering the FOM_REF_RE targets (at Re₀ = 49.03 the order-9
branch folds before Re 56; order 7 continues past it).
"""
function load_seed_branch(data_dir::AbstractString, preferred_ord::Int)
	orders = Int[]
	for f in readdir(data_dir)
		m = match(r"^rom_branch_ord(\d+)\.csv$", f)
		m === nothing || push!(orders, parse(Int, m.captures[1]))
	end
	isempty(orders) && error("no rom_branch_ord*.csv in $data_dir — run solve_rom.jl first")
	N = preferred_ord in orders ? preferred_ord : maximum(orders)
	N == preferred_ord ||
		@warn "rom_branch_ord$(preferred_ord).csv not found — seeding from order $N instead"
	raw, _ = readdlm(joinpath(data_dir, "rom_branch_ord$(N).csv"), ',', Float64, '\n';
		header = true)
	return raw, N
end

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
function measure_orbit(s_po, η_prime, T, fom, ops, l_free, M_vel, vel_rows, area)
	L_klu, RHS_M, Δt_meas, n_steps = build_imex_operators(
		ops.B₀, ops.B₁, ops.K_visc, η_prime, T, FOM_REF_DT; θ = 0.5)
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

# ─────────────────────────────────────────────────────────────────────────────
# Per-run-directory driver
# ─────────────────────────────────────────────────────────────────────────────

function process_dir(run_dir::AbstractString, fom, l_free, M_vel, vel_rows, area)
	data_dir = joinpath(run_dir, "data")
	m = match(r"^Re([\d.]+)_ord(\d+)$", basename(run_dir))
	m === nothing && (println("  skip (unrecognised dir name): $run_dir"); return)
	re0 = parse(Float64, m.captures[1])

	w_path = joinpath(data_dir, "W.jls")
	isfile(w_path) || (println("  skip (no data/W.jls): $run_dir"); return)
	println("── $run_dir  (Re₀ = $re0)")

	W = deserialize(w_path)
	branch, branch_ord = load_seed_branch(data_dir, FOM_REF_SEED_ORD)
	@printf("  seeding from rom_branch_ord%d.csv  (%d points, Re %.3f → %.3f)\n",
		branch_ord, size(branch, 1), branch[1, 2], branch[end, 2])

	ops = load_or_build_operators(data_dir, fom, re0)
	L0_lift = dot(l_free, ops.s₀_full[fom.free_dpim])

	s_steady_warm = ops.s₀_full   # continuation chain for the steady branch
	rows = NamedTuple[]

	for re_target in sort(FOM_REF_RE)
		η′ = 1.0 / re_target - 1.0 / re0
		seed = seed_from_branch(branch, re_target)
		if seed === nothing
			@warn "Re = $re_target outside the ROM branch — skipped"
			continue
		end
		(ρ, T_rom) = seed
		@printf("\n  ● Re = %.3f  (η′ = %+.5e,  seed ρ = %.4e,  T_ROM = %.6f)\n",
			re_target, η′, ρ, T_rom)

		z = ComplexF64[ρ, ρ, η′]   # phase θ = 0; ρ already in the run's scaled coords
		s_init = vec(real.(evaluate(W.poly, z)))

		t_start = time()
		(s_po, T_fom, E_norm, X_max, μ̂_picard, n_orb, conv) = find_periodic_orbit(
			s_init, η′, T_rom, fom, ops.B₀, ops.B₁, ops.K_visc, ops.h₀_vec;
			Δt = FOM_REF_DT, θ = 0.5,
			n_spinup = FOM_REF_N_SPINUP, γ_spinup = FOM_REF_GAMMA,
			max_picard = FOM_REF_MAX_PICARD, tol = FOM_REF_TOL,
			lift_weights = l_free, verbose = true)
		conv || @warn "Re = $re_target: Picard did not converge (rel = $(X_max > 0 ? E_norm/X_max : NaN))"

		# Steady lift shift: compare_orders.py drops the (0,0,c) monomials, whose
		# FOM counterpart is the steady-branch lift change from Re₀ to Re.
		(_, _, s_steady) = solve_steady_state(fom; Re0 = re_target,
			s_init = s_steady_warm, tol = 1e-10)
		s_steady_warm = s_steady
		ΔL_steady = dot(l_free, s_steady[fom.free_dpim]) - L0_lift

		obs = measure_orbit(s_po, η′, T_fom, fom, ops, l_free, M_vel, vel_rows, area)
		obs.ok || (@warn "Re = $re_target: measurement orbit blew up — skipped"; continue)

		max_abs_lift = maximum(abs, obs.FL_per .- ΔL_steady)
		lift_amp = (obs.lift_max - obs.lift_min) / 2
		E_M_rel = X_max > 0.0 ? E_norm / X_max : NaN

		# Distance-to-cycle bound: anchor closure measures (1−μ)·distance, so
		# distance ≈ rel/(1−μ). Prefer the Picard-iterate μ̂; fall back to the
		# measurement-orbit anchor ratio (always available).
		μ = isfinite(μ̂_picard) ? μ̂_picard : obs.μ_meas
		dist_rel_est = (isfinite(μ) && 0.0 < μ < 1.0) ? E_M_rel / (1.0 - μ) : NaN

		# per-point lift trace: raw vs harmonic projection (pollution visible)
		lift_csv = joinpath(data_dir, @sprintf("fom_lift_Re%.2f.csv", re_target))
		open(lift_csv, "w") do io
			println(io, "t,FL_raw,FL_periodic")
			for k in eachindex(obs.FL)
				@printf(io, "%.8e,%.10e,%.10e\n", k * obs.Δt_meas, obs.FL[k], obs.FL_per[k])
			end
		end

		@printf("    T_fom = %.6f s  (ω = %.5f rad/s),  %d orbits,  rel = %.2e,  %.0f s wall\n",
			T_fom, 2π / T_fom, n_orb, E_M_rel, time() - t_start)
		@printf("    max|lift| = %.6e,  lift amp = %.6e,  ringing = %.2e,  ⟨TKE⟩ = %.6e,  ΔL_steady = %+.3e\n",
			max_abs_lift, lift_amp, obs.lift_ringing, obs.avg_tke, ΔL_steady)
		@printf("    periodicity: max_t‖s(t+T)−s(t)‖/X = %.2e,  μ = %.4f,  dist ≈ %.2e,  amp drift = %.2e\n",
			obs.periodicity_max, μ, dist_rel_est, obs.lift_amp_drift)

		push!(rows, (; Re = re_target, eta = η′, T_rom, T_fom,
			omega_fom = 2π / T_fom, max_abs_lift, lift_amp,
			lift_max = obs.lift_max, lift_min = obs.lift_min,
			lift_mean = obs.lift_mean, lift_ringing = obs.lift_ringing,
			avg_TKE = obs.avg_tke, E_M_rel,
			periodicity_max = obs.periodicity_max,
			lift_amp_drift = obs.lift_amp_drift,
			floquet_mu = μ, dist_rel_est,
			n_orbits = n_orb, converged = conv, dt = FOM_REF_DT))
	end

	isempty(rows) && (println("  no FOM reference points computed"); return)
	out_csv = joinpath(data_dir, "fom_reference.csv")
	open(out_csv, "w") do io
		println(io, "Re,eta,T_rom,T_fom,omega_fom,max_abs_lift,lift_amp," *
					"lift_max,lift_min,lift_mean,lift_ringing,avg_TKE,E_M_rel," *
					"periodicity_max,lift_amp_drift,floquet_mu,dist_rel_est,n_orbits,converged,dt")
		for r in rows
			@printf(io,
				"%.8f,%.12e,%.10f,%.10f,%.10f,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e,%.3e,%.3e,%.3e,%.5f,%.3e,%d,%s,%.1e\n",
				r.Re, r.eta, r.T_rom, r.T_fom, r.omega_fom, r.max_abs_lift,
				r.lift_amp, r.lift_max, r.lift_min, r.lift_mean, r.lift_ringing,
				r.avg_TKE, r.E_M_rel, r.periodicity_max, r.lift_amp_drift,
				r.floquet_mu, r.dist_rel_est, r.n_orbits, r.converged, r.dt)
		end
	end
	@printf("\n  %d FOM reference points → %s\n", length(rows), out_csv)
end

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

run_dirs = if isempty(ARGS)
	base = joinpath(@__DIR__, "results")
	sort(filter(d -> occursin(r"^Re[\d.]+_ord\d+$", basename(d)) && isdir(d),
		readdir(base; join = true)))
else
	[abspath(a) for a in ARGS]
end
isempty(run_dirs) && error("no run directories found — run main.jl first")

meshfile = joinpath(@__DIR__, "fem", "cylinder_flow.msh")
if !isfile(meshfile)
	@info "Mesh file missing — regenerating with config.jl parameters"
	generate_mesh(; h_cyl = MESH_H_CYL, h_wake = MESH_H_WAKE, h_bulk = MESH_H_BULK)
end

@info "Setting up FEM ..."
fom = setup_fem(meshfile)
l_free = compute_pressure_lift_weights(fom)[fom.free_dpim]
(M_vel, vel_rows, area) = prepare_energy_gram(fom)

foreach(d -> process_dir(d, fom, l_free, M_vel, vel_rows, area), run_dirs)
