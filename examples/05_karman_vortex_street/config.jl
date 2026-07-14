# config.jl — all tunable parameters for the Kármán vortex street DPIM demo.
# Edit this file to reproduce different figures from arXiv:2510.26542v1.

# ── Physics ───────────────────────────────────────────────────────────────────
const Re₀ = 49.03   # expansion Re; paper uses 20, Re_c≈49.03, 70, 80
const MAX_ORD = 9   # DPIM expansion order (single run — lower orders are truncations)
# The cohomological solve is graded: coefficients of degree ≤ N never depend on higher
# degrees, so the order-9 W/R contain the order-N ROMs EXACTLY (verified bit-exact).
const TRUNC_ORDERS = [3, 5, 7, 9]   # truncation orders for the convergence comparison
const ROM = 2        # number of Hopf master modes
const N_EXT = 1        # external parameter dimensions (η′ = 1/Re − 1/Re₀)
const NVAR = ROM + N_EXT   # = 3

# ── Mesh ──────────────────────────────────────────────────────────────────────
const MESH_H_CYL = 0.005   # element size on cylinder surface
const MESH_H_WAKE = 0.015   # element size in the cylinder wake
const MESH_H_BULK = 0.04    # element size in the bulk channel

# ── Eigensolver ───────────────────────────────────────────────────────────────
const EIG_NEV = 40      # number of eigenvalues to compute and display
const EIG_SIGMA_RE = 3.0   # real part of ARPACK shift; offset from the imaginary
# axis avoids near-singularity when Re(λ_Hopf) ≈ 0
const EIG_SIGMA_IM = 8.0   # imag part ≈ ω₀/2 (Hopf freq ω₀ ≈ 16.86 rad/s); any
# O(ω₀) value works — the shift only conditions the factorisation, Hopf-mode
# selection is shift-independent
const EIG_TARGET_FREQ = nothing   # rad/s: pin the Hopf mode by Im(λ) ≈ this frequency;
# nothing → smallest |Re(λ)| among Im(λ) > 0 (only reliable near Re_c ≈ 49)

# ── ROM limit-cycle branch (solve_rom.jl) ─────────────────────────────────────
const BRANCH_RE_MAX = 70.0     # stop continuation when Re exceeds this
const BRANCH_DS0 = 1e-4        # initial PALC arclength step (scaled-ρ/η units)
const BRANCH_MAX_STEPS = 2000  # hard cap on PALC steps

# ── FOM reference for the Hopf tube (fom_reference.jl) ────────────────────────
const FOM_REF_RE = [49.5, 50.5, 52.0, 54.0, 56.0]   # Re sample points along the branch
const FOM_REF_SEED_ORD = 7   # branch order used for seeding: order 9 folds back
# before Re 56 (cannot bracket the last target), order 7 continues past it; the
# seed only sets the transient length, never the converged FOM orbit
const FOM_REF_DT = 5e-4      # IMEX time step; θ = 0.5 (Crank-Nicolson) on the linear part
const FOM_REF_TOL = 1e-4     # rel. mass-norm periodicity tolerance ‖E‖_M / X_max
const FOM_REF_N_SPINUP = 3   # undamped spin-up orbits before the period probe
const FOM_REF_GAMMA = 0.0    # optional spin-up damping; leave 0 — any γ ≫ σ (Hopf
# growth rate ~0.1 s⁻¹ near onset) kills the shedding oscillation itself and the
# probe then reads a spurious period off the quasi-steady signal
const FOM_REF_MAX_PICARD = 400   # undamped Picard orbit cap per Re point

# ── Optional MATLAB/COCO export (validation/generate_matlab.py) ───────────────
const EXPORT_MATLAB = false
