# config.jl — all tunable parameters for the Kármán vortex street DPIM demo.
# Edit this file to reproduce different figures from arXiv:2510.26542v1.
#
# MORFE_FAST=1 selects a smoke profile (coarse mesh, order 3, fewer eigenvalues,
# ≈ minutes). NOTE: unlike the structural examples, FAST here changes the MESH,
# so its numbers are not comparable to the full-profile reference — it verifies
# the pipeline end-to-end, not the physics.
FAST = get(ENV, "MORFE_FAST", "0") == "1"

# ── Physics ───────────────────────────────────────────────────────────────────
const Re₀ = 49.03   # expansion Re; paper uses 20, Re_c≈49.03, 70, 80
const MAX_ORD = FAST ? 3 : 9   # DPIM expansion order (single run — lower orders are truncations)
# The cohomological solve is graded: coefficients of degree ≤ N never depend on higher
# degrees, so the order-9 W/R contain the order-N ROMs EXACTLY (verified bit-exact).
const TRUNC_ORDERS = FAST ? [3] : [3, 5, 7, 9]   # truncation orders for the convergence comparison
const ROM = 2        # number of Hopf master modes
const N_EXT = 1        # external parameter dimensions (η′ = 1/Re − 1/Re₀)
const NVAR = ROM + N_EXT   # = 3

# ── Mesh ──────────────────────────────────────────────────────────────────────
const MESH_H_CYL = FAST ? 0.010 : 0.005   # element size on cylinder surface
const MESH_H_WAKE = FAST ? 0.030 : 0.015   # element size in the cylinder wake
const MESH_H_BULK = FAST ? 0.080 : 0.04    # element size in the bulk channel

# ── Eigensolver ───────────────────────────────────────────────────────────────
const EIG_NEV = FAST ? 20 : 40      # number of eigenvalues to compute and display
const EIG_SIGMA_RE = 3.0   # real part of ARPACK shift; offset from the imaginary
# axis avoids near-singularity when Re(λ_Hopf) ≈ 0
const EIG_SIGMA_IM = 8.0   # imag part ≈ ω₀/2 (Hopf freq ω₀ ≈ 16.86 rad/s); any
# O(ω₀) value works — the shift only conditions the factorisation, Hopf-mode
# selection is shift-independent
const EIG_TARGET_FREQ = nothing   # rad/s: pin the Hopf mode by Im(λ) ≈ this frequency;
# nothing → smallest |Re(λ)| among Im(λ) > 0 (only reliable near Re_c ≈ 49)

# ── Mode gauge and resonance policy ───────────────────────────────────────────
const MODE_SCALE = 1e-2   # uniform factor on BOTH eigenvector sides, for conditioning.
# It makes φᴴBψ = 1e-4 rather than 1. `SpectralData` has no `scale` field precisely so
# that this stays visible; raw W/R from two different gauges are NOT comparable.

# ── Which analysis: Hopf pair only, or resonant modes promoted? ───────────────
# MORFE_PROMOTE=1 selects the SECOND analysis, in which the resonant outer modes join
# the master set at first order. Promotion is a CHANGE OF COORDINATES — the mean-flow
# distortion becomes an explicit coordinate y_k instead of being folded into W — so both
# analyses must yield the same λ, c₁₀₁ and effective Landau coefficient. Comparing them
# is the point; see compare_runs.jl.
const PROMOTE_RESONANT = get(ENV, "MORFE_PROMOTE", "0") == "1"

# Reject a candidate whose bilinear pairing α = ψᵀB₁φ is too weak to carry a coordinate.
# α is the denominator of the SymmetricBiorthogonal gauge (BOTH sides ÷ √α), and B₁ here
# is SINGULAR — 6636 of its 57860 rows are the P1 pressure DOFs — so this is a descriptor
# pencil whose null-space vectors are algebraic incompressibility modes with no dynamics
# and α = 0. A near-zero α means the mode is defective, unconverged, or algebraic; in
# every case ż = λz cannot be obtained by projecting with ψᵀB₁, and 1/√α amplifies the
# eigenvector enormously (1.7e5 at the measured α = 3.3e-11).
#
# RELATIVE to the Hopf pair's own α, measured in the same run, because the absolute value
# depends on the mesh and on Arpack's vector scaling. Measured at the FULL mesh:
# λ = −2.11 → 1.09e-6 (keep), −5.14 → 1.31e-8 (drop, ~1e4 weaker), −6.69 → 3.3e-11 (drop).
const PROMOTE_ALPHA_RTOL = 1e-3

const RESONANCE_TOL_FACTOR = PROMOTE_RESONANT ? 0.5 : 0.1   # detuning threshold, as a fraction of the MASTER
# frequency: the tolerance handed to `ResonanceConfig` is the ABSOLUTE value
# `RESONANCE_TOL_FACTOR · |λ_Hopf|`, so a mode is flagged when its distance to a
# superharmonic is less than that. With |λ_Hopf| ≈ 16.86 that is 1.69 un-promoted
# (nothing is caught — the Hopf pair alone) and 8.43 promoted, i.e. "catch everything
# down to λ ≈ −8". For a mode on the real axis the nearest superharmonic is s = 0, at
# distance exactly |λ|, so the promoted set is essentially {|λ| < 8.43}.
#
# Absolute, NOT `tol_relative`. MORFE scales a relative tolerance by each TARGET's own
# eigenvalue, `tol_relative · |λⱼ|`. For an outer mode on the real axis the nearest
# superharmonic is s = 0 (any pure-η′ monomial), at distance exactly |λⱼ| — so the ratio
# is 1/tol_relative for EVERY real mode regardless of magnitude, and no real mode can be
# flagged unless tol_relative > 1. Scaling by the master frequency instead is what makes
# "catch everything down to λ ≈ −8" mean what it says.

const PROMOTED_CORE_ORD = 2   # highest (z₁,z̄₁,η′) degree a promoted coordinate may
# multiply. The promoted block is first order either way; this bounds how far it MIXES.
#
# 2 is what closes the mean-flow loop and no more: ẏ_k is driven by z₁z̄₁ (core degree 2)
# and hands back through z₁·y_k (core degree 1). Higher mixing contributes little and
# costs a great deal — the mixed block is core(≤ this) × n_promoted, so at order 9 an
# uncapped version is ~165 monomials PER promoted mode. With ~20 promoted that is ~3500
# monomials and a 3.3 GB W, which the OS kills. At 2 it is 9 per mode.

const REAL_MODE_RTOL = 1e-4   # a mode counts as REAL when |Im λ| ≤ REAL_MODE_RTOL·|λ|.
# Relative, not absolute: ARPACK's numerical zero is ~1e-7, not ~1e-16, so an absolute
# 1e-8 test calls a real mode complex and then invents a conjugate for it — a duplicate
# coordinate. Measured on this spectrum the two families are three orders apart
# (real ≤ 1.2e-6, smallest genuinely complex 3.6e-3), so 1e-4 sits well clear of both.

# ── ROM limit-cycle branch (solve_rom.jl) ─────────────────────────────────────
const BRANCH_RE_MAX = 70.0     # stop continuation when Re exceeds this
const BRANCH_DS0 = 1e-4        # initial PALC arclength step (scaled-ρ/η units)
const BRANCH_MAX_STEPS = 2000  # hard cap on PALC steps

# ── FOM reference for the Hopf tube (fom_reference.jl) ────────────────────────
const FOM_REF_RE = [49.5, 50.5, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0]   # Re sample points along the branch
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

# ── Where a run writes, and where validate.jl reads ───────────────────────────
# ONE directory per (Re₀, order) — the same config overwrites its own previous run.
# Deliberately NOT tagged by mode selection: promoting outer modes into the master
# set is a CHANGE OF COORDINATES, so a promoted and an un-promoted run describe the
# same physics and are validated against the same reference (see reference_data/).
# Tagging them apart produced four near-duplicate directories and four "references",
# which is exactly the confusion this avoids.
#
# It lives HERE rather than in main.jl because validate.jl has to read exactly what
# main.jl wrote; when the two built the path independently they drifted, and
# validate.jl silently validated a days-old directory from a different mode set.
# The `_promoted` suffix marks the COORDINATE SYSTEM, so the two analyses coexist for
# comparison instead of overwriting each other. It comes from the switch (known upfront —
# the log file opens before the resonance probe runs) rather than from the outcome, and
# the un-promoted name stays bare so the validated run keeps its path.
using Printf
const RUN_SUFFIX = PROMOTE_RESONANT ? "_promoted" : ""
const RESULTS_DIR = joinpath(@__DIR__, "results",
	@sprintf("Re%.2f_ord%d%s", Re₀, MAX_ORD, RUN_SUFFIX))
