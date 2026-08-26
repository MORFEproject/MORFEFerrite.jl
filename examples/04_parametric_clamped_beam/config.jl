# ── CASE — the two-parameter clamped-clamped beam problem definition. ────────
# Edit ONLY here. Consumed by main.jl (the generic parametric pipeline),
# validate.jl and frequency_accuracy_map.jl.
#
# Two independent, ANALYTIC shape fields (general additive formulation,
# per-parameter box truncation):
#
#   x(θ₁,θ₂,x₀) = x₀ + θ₁ ψ₁(x₀) + θ₂ ψ₂(x₀)
#
#   ψ₁ = x₁ e₁                    ⟹  ∇ψ₁ = e₁⊗e₁          the SPAN
#   ψ₂ = h₀ sin(πx₁/L) e₂         ⟹  ∇ψ₂ = (πh₀/L)cos(πx₁/L)·e₂⊗e₁   the ARCH
#
# ψ₂ is the same isochoric sine bend example 07 uses (there as its *reference*
# configuration, here as an additive shape field). This example is therefore
# "example 07's arch, plus span control".
#
# θ₁ = 0 is the nominal span L, θ₁ = 0.2 a 20 % longer beam.
# θ₂ = 0 is the straight beam, θ₂ = 20 a rise of 20·h₀ = 10 % of the span.

FAST = get(ENV, "MORFE_FAST", "0") == "1"

# Geometry / mesh / material -------------------------------------------------
MESH = joinpath(@__DIR__, "beam_h27_10x2x2.msh")
const L_SPAN = 1000.0               # beam span (mm) — matches the mesh extent
const h0_L_ratio = 0.005            # rise / span at θ₂ = 1 (example 07's value)
const H0 = h0_L_ratio * L_SPAN      # 5 mm, so θ₂ = 20 ⇒ 100 mm = 10 % of span
E = 160e3; ν = 0.22; RHO = 2.32e-3
ALPHA = 0.0; BETA = 0.0
NEV = 10

# Reduction -------------------------------------------------------------------
# No PERMUTATION literal: `build_model` derives it with
# `full_conjugate_permutation`, so it stays correct if the external system
# changes. It came out as [2, 1, 3, 4] here — z₁ ↔ z₂, θ₁/θ₂ real hence
# self-conjugate — and main.jl prints it.
ROM = 2                             # master conjugate pair
N_EXT = 2                           # two frozen external states: θ₁, θ₂
NVAR = ROM + N_EXT
MAXZ = FAST ? 3 : 9                 # expansion order in (z₁, z₂)
MAXT = FAST ? 2 : 4                 # θ degree kept in the REDUCED DYNAMICS (see BUILD_MSET)
RUN_SANITY_CHECKS = true

# ── VALIDITY AND TRUNCATION OF THE GEOMETRY EXPANSION ────────────────────────
# Intended parameter range: |θ₁| ≤ 0.2, 0 ≤ θ₂ ≤ 20.
#
# Both shape-field gradients are constant-in-θ, so the Jacobian is lower
# triangular and everything below is EXACT, not estimated:
#
#     J     = [[1+θ₁, 0, 0], [θ₂c, 1, 0], [0, 0, 1]],   c = (πh₀/L)cos(πx₁/L)
#     det J = 1 + θ₁                       ← independent of θ₂
#     adj J = [[1,0,0], [−θ₂c, 1+θ₁, 0], [0,0,1+θ₁]]
#
# Two consequences, and they pull in opposite directions on the two axes:
#
#   θ₂ (the ARCH) is EXACTLY isochoric — ∇ψ₂ is nilpotent, so it never touches
#      det J. `adj J` is degree 1 in θ₂ with no cross term, so a form with n
#      gradient factors is an exact polynomial of θ₂-degree n: 2 for the
#      stiffness, 3 for the quadratic form, 4 for the cubic. Truncating at
#      exactly those bounds is LOSSLESS at any θ₂, however large.
#
#   θ₁ (the SPAN) is the hard one. ∇ψ₁ is not nilpotent, so det J = 1 + θ₁ and
#      1/det J is a genuine geometric series about det J = 1. Its radius is
#      |det J − 1| < 1, i.e. |θ₁| < 1: at θ₁ = 1 it DIVERGES at every truncation
#      order, and near it convergence is only ~θ₁ per order. ALL of this
#      example's geometry truncation error lives on this axis.
#
# So the box is deliberately asymmetric, and the θ₂ bound is per-FORM rather
# than shared — truncating the stiffness at 4 would be wasted work, and
# truncating the cubic form at 2 would be wrong.
#
# `frequency_accuracy_map.jl` measures the consequence: relative error in the
# first bending frequency over the (θ₁,θ₂) box, against an exact frozen-θ
# assembly. Expect the θ₁ = 0 column to be at machine precision.
#
# NOTE: ‖ΔK‖/‖K‖ badly understates modal error — 1.3e-05 operator error has
# corresponded to a 267 % frequency error, because the Frobenius norm is
# dominated by stiff axial directions while ω₁ is the softest bending mode.
# Judge accuracy on MODAL quantities, never on operator norms.
#
# `build_model` prints the measured validity range; `PG.geometry_validity_at(
# cache, θ).reciprocal_residual` gives the truncation error at a specific θ.
GEOM_T1 = FAST ? 4 : 8              # θ₁ box bound — the span, needs the terms

# θ₂ bounds are the EXACT polynomial degrees derived above. Three DISTINCT
# bases, so `parametric_model` builds three geometry caches — sharing one object
# is how a caller says "same truncation", and here they genuinely differ.
BASIS_K = PG.GeometryParameterBasis([GEOM_T1, 2])
BASIS_QUAD = PG.GeometryParameterBasis([GEOM_T1, 3])
BASIS_CUBIC = PG.GeometryParameterBasis([GEOM_T1, 4])
GEOMETRY_PARAMETER_BASES = (; linear = BASIS_K, quadratic = BASIS_QUAD, cubic = BASIS_CUBIC)

MATERIAL = SVKMaterial(E = E, ν = ν, ρ = RHO)
DAMPING = RayleighDamping(α = ALPHA, β = BETA)

# ── Shape-field geometry: fully analytic, one argument per quadrature point ──
include(joinpath(@__DIR__, "fem", "sine_bend.jl"))

const ∇ψ₁ = Tens3((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)

# (J₀, ∇ψ₁, ∇ψ₂) with J₀ = I: the reference configuration is the STRAIGHT beam,
# unlike example 07 where the arch itself is the reference (J₀ = I + ∇w).
GEOM(x₀) = (one(Tens3), ∇ψ₁, sine_bend_jacobian(x₀, H0, L_SPAN))

# The exact Jacobian at a FROZEN θ — the untruncated reference used by
# frequency_accuracy_map.jl. Same map, evaluated rather than expanded.
J_EXACT(x₀, θ) = one(Tens3) + θ[1] * ∇ψ₁ + θ[2] * sine_bend_jacobian(x₀, H0, L_SPAN)

# The geometry is analytic, so the parametric assembly needs no eigen data and
# runs FIRST; the eigenproblem is then solved on its θ⁰ coefficient, which here
# is the straight beam. (This is why the old bending-eigenmode shape field
# needed the reverse ordering — it had to eigensolve before it could build the
# geometry at all.)
function BUILD_CASE(dh, cv, free)
	pcase = SVK.parametric_model(dh, cv, GEOM;
		geometry_parameter_basis = GEOMETRY_PARAMETER_BASES, material = MATERIAL,
		damping = DAMPING, free = free)
	K0, M0 = SVK.base_operators(pcase)
	eigenproblem = spectrum(K0, M0,
		StructureModalDampingEigensolver(NEV, ALPHA, BETA);
		sorter! = (args...) -> nothing)
	return pcase, eigenproblem
end

# Pre-solve sanity, both analytic:
#   ∂ω/∂θ₁ = −2ω₀ — a clamped-clamped beam has ω ∝ 1/L², and θ₁ scales L.
#   ∂ω/∂θ₂ ≈ 0    — arch stiffening is SECOND order in the rise, so the first
#                    derivative at the straight configuration vanishes.
function SANITY(pcase, eigenproblem)
	K_arr = pcase.operators[1].arrays
	M_arr = pcase.operators[3].arrays
	master_eigenvalues = eigenproblem.eigenvalues[1:ROM]
	master_modes = eigenproblem.eigenmodes[:, 1, 1:ROM]
	M = M_arr[1]
	i10 = BASIS_K.index[SVector(1, 0)]
	i01 = BASIS_K.index[SVector(0, 1)]
	φ = real(master_modes[:, 1]); φ ./= maximum(abs, φ)
	ω₀ = abs(master_eigenvalues[1])
	φᵀMφ = dot(φ, M * φ)
	dω_dθ1 = (dot(φ, K_arr[i10] * φ) - ω₀^2 * dot(φ, M_arr[i10] * φ)) / (2ω₀ * φᵀMφ)
	dω_dθ2 = (dot(φ, K_arr[i01] * φ) - ω₀^2 * dot(φ, M_arr[i01] * φ)) / (2ω₀ * φᵀMφ)
	@printf "sanity: ∂ω/∂θ₁ (FEM) = %+.6f   expected ≈ %+.6f (−2ω₀)\n" dω_dθ1 (-2ω₀)
	@printf "sanity: ∂ω/∂θ₂ (FEM) = %+.6f   expected ≈ 0\n" dω_dθ2
	return nothing
end

# Corrected truncation: group-total ≤ MAXZ in (z₁,z₂), per-parameter BOX in θ.
#
# MAXT here is the θ degree kept in the REDUCED DYNAMICS, which is a different
# truncation from the geometry box (GEOM_T1 and the θ₂ degrees above). The
# geometry box controls how accurately the operators represent the actual
# configuration; this one controls how many θ powers the ROM carries. They are
# independent modelling choices and need not agree.
BUILD_MSET() = MultiindexSet([
	SVector{NVAR, Int}(a, b, c, d)
	for a in 0:MAXZ for b in 0:MAXZ
	for c in 0:MAXT for d in 0:MAXT
	if a + b ≤ MAXZ && 1 ≤ a + b + c + d])

META() = ["example" => "04_parametric_clamped_beam",
	"formulation" => "additive analytic shape fields (span + isochoric sine bend), " *
					 "per-parameter box truncation",
	"max_degree_z" => MAXZ, "max_degree_theta" => MAXT,
	"geometry_box_theta1" => GEOM_T1,
	"geometry_degrees_theta2" => "2 (K) / 3 (quadratic) / 4 (cubic) — exact",
	"h0_over_L" => h0_L_ratio,
	"validity_range" => "|θ₁| ≤ 0.2, 0 ≤ θ₂ ≤ 20",
	"fast" => FAST]
