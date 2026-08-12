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
ALPHA = 0.0; BETA = 0.0             # Rayleigh damping (conservative arch)
NEV = 10

# Reduction -------------------------------------------------------------------
# No PERMUTATION literal: `build_model` derives it with
# `full_conjugate_permutation`, which is the only way to express an ODD N_EXT —
# θ is real, hence its own conjugate, so the adjacent-pairs formula cannot.
# It comes out as [2, 1, 3] here, and main.jl prints it.
ROM = 2                             # master conjugate pair
N_EXT = 1                           # one frozen external state: θ
NVAR = ROM + N_EXT
MAXZ = FAST ? 5 : 11                # expansion order in (z₁, z₂)
MAXT = FAST ? 3 : 7                 # expansion order in θ
RUN_SANITY_CHECKS = false

# θ-series bases (exact polynomial degrees of the arch forms:
# K(θ) ≤ 2, quadratic form ≤ 3, cubic form ≤ 4 — J_arch² = 0, det J ≡ 1).
# Three DISTINCT bases, so `parametric_model` builds three geometry caches —
# truncating them all at the largest would be wasted work.
BASIS_K = PG.GeometryParameterBasis([2])
BASIS_QUAD = PG.GeometryParameterBasis([3])
BASIS_CUBIC = PG.GeometryParameterBasis([4])
GEOMETRY_PARAMETER_BASES = (; linear = BASIS_K, quadratic = BASIS_QUAD, cubic = BASIS_CUBIC)

MATERIAL = SVKMaterial(E = E, ν = ν, ρ = RHO)
DAMPING = RayleighDamping(α = ALPHA, β = BETA)

# Analytic shape-field geometry: provider (J₀, ∇ψ₁) per quadrature point.
include(joinpath(@__DIR__, "fem", "arch_geometry.jl"))
GEOM(x₀) = arch_jacobian_pair(x₀, H0, L_SPAN)
GEOM_BUILDER(dh, free, master_modes, ndofs_total) = GEOM   # analytic: no eigen data needed

# The geometry is analytic, so the parametric assembly needs no eigen data and
# can run FIRST; the eigenproblem is then solved on its θ⁰ coefficient — the
# base ARCH, not the straight beam. That ordering is what dissolves the old
# `_KM_CACHE` global: assembly happens once, inside `parametric_model`.
function BUILD_CASE(dh, cv, free)
	pcase = SVK.parametric_model(dh, cv, GEOM;
		geometry_parameter_basis = GEOMETRY_PARAMETER_BASES, material = MATERIAL, damping = DAMPING, free = free)
	K0, M0 = SVK.base_operators(pcase)
	eigenproblem = spectrum(K0, M0,
		StructureModalDampingEigensolver(NEV, ALPHA, BETA);
		sorter! = (args...) -> nothing)
	return pcase, eigenproblem
end

SANITY(pcase, eigenproblem) = nothing

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
