# ── CASE — the sinusoidal-arch problem definition. Edit ONLY here. ──────────
# Consumed by main.jl (the generic parametric pipeline) and validate.jl.
#
# Single shape-field parameter θ scaling the arch rise:
#   x(θ,x₀) = x₀ + (1+θ)·w(x₀),  w = h₀ sin(πx₁/L)·e₂
#   J(θ,x₀) = J₀(x₀) + θ·J₁(x₀)          (J₀ = I + ∇w, J₁ = ∇w, per QP, analytic)
# θ = −1 → straight beam · θ = 0 → base arch (h₀) · θ = +1 → doubled arch.

FAST = get(ENV, "MORFE_FAST", "0") == "1"

# Geometry / mesh / material -------------------------------------------------
const h0_L_ratio = 0.005            # base arch rise / span (θ = 0 configuration)
MESH = joinpath(@__DIR__, "beam_h27_10x2x2.msh")
const L_SPAN = 1000.0               # beam span (mm)
const H0 = h0_L_ratio * L_SPAN      # arch rise (mm)
E = 160e3; ν = 0.22; RHO = 2.32e-3
LAMBDA_LAME = (E * ν) / ((1 + ν) * (1 - 2ν))
MU_LAME = E / (2(1 + ν))
ALPHA = 0.0; BETA = 0.0             # Rayleigh damping (conservative arch)
NEV = 10

# Reduction -------------------------------------------------------------------
ROM = 2                             # master conjugate pair
N_EXT = 1                           # one frozen external state: θ
NVAR = ROM + N_EXT
PERMUTATION = [2, 1, 3]             # z₁ ↔ z₂; θ real → self-conjugate
MAXZ = FAST ? 5 : 11                # expansion order in (z₁, z₂)
MAXT = FAST ? 3 : 7                 # expansion order in θ
RUN_SANITY_CHECKS = false

# θ-series bases (exact polynomial degrees of the arch forms:
# K(θ) ≤ 2, quadratic form ≤ 3, cubic form ≤ 4 — J_arch² = 0, det J ≡ 1).
BASIS_K = PS.ThetaBasis([2])
BASIS_QUAD = PS.ThetaBasis([3])
BASIS_CUBIC = PS.ThetaBasis([4])

# Analytic shape-field geometry: provider (J₀, ∇ψ₁) per quadrature point.
include(joinpath(@__DIR__, "fem", "arch_geometry.jl"))
GEOM(x₀) = arch_jacobian_pair(x₀, H0, L_SPAN)
GEOM_BUILDER(dh, free, master_modes, ndofs_total) = GEOM   # analytic: no eigen data needed

# Base operators for the eigenproblem come from the θ⁰ coefficient of the
# parametric assembly (the base ARCH, not the straight beam) — computed once,
# cached, and reused by PARAM_KM.
const _KM_CACHE = Ref{Any}(nothing)
function BASE_KM(dh, cv)
	K_full = [allocate_matrix(dh) for _ in 1:PS.nterms(BASIS_K)]
	M_full = [allocate_matrix(dh) for _ in 1:PS.nterms(BASIS_K)]
	PS.assemble_parametric_K_M!(K_full, M_full, dh, cv,
		LAMBDA_LAME, MU_LAME, RHO, GEOM, BASIS_K)
	_KM_CACHE[] = (K_full, M_full)
	return K_full[1], M_full[1]
end
function PARAM_KM(dh, cv, provider, free)
	K_full, M_full = _KM_CACHE[]
	return [Kf[free, free] for Kf in K_full], [Mf[free, free] for Mf in M_full]
end

SANITY(K_arr, M_arr, K_ref, master_eigenvalues, master_modes) = nothing

# Anisotropic multiindex set: total degree ≤ MAXZ in (z₁,z₂), θ-degree ≤ MAXT,
# joint total capped at MAXZ.
BUILD_MSET() = MultiindexSet([
	SVector{NVAR, Int}(a, b, c)
	for a in 0:MAXZ for b in 0:MAXZ for c in 0:MAXT
	if a + b ≤ MAXZ && c ≤ MAXT && 1 ≤ a + b + c ≤ MAXZ])

META() = ["example" => "07_parametric_arch",
	"h0_L_ratio" => h0_L_ratio, "L_mm" => L_SPAN,
	"max_degree_z" => MAXZ, "max_degree_theta" => MAXT,
	"fast" => FAST]
