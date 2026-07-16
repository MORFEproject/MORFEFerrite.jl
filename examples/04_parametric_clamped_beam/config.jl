# ── CASE — the two-parameter clamped-beam problem definition. Edit ONLY here. ─
# Consumed by main.jl (the generic parametric pipeline) and validate.jl.
#
# Two independent shape fields (general additive formulation, per-parameter
# box truncation — this CORRECTS the old example 04, whose total-degree
# θ-truncation was superseded; the old reference is not comparable):
#   x(θ₁,θ₂,x₀) = x₀ + θ₁ψ₁(x₀) + θ₂ψ₂(x₀)
#   ∇ψ₁ = e₁⊗e₁              uniform axial stretch (constant)
#   ∇ψ₂ = ∇φ₁(x₀)            first bending eigenmode gradient (FE field, per QP)

FAST = get(ENV, "MORFE_FAST", "0") == "1"

# Geometry / mesh / material -------------------------------------------------
MESH = joinpath(@__DIR__, "beam_h27_10x2x2.msh")
E = 160e3; ν = 0.22; RHO = 2.32e-3
LAMBDA_LAME = (E * ν) / ((1 + ν) * (1 - 2ν))
MU_LAME = E / (2(1 + ν))
ALPHA = 0.0; BETA = 0.0
NEV = 10

# Reduction -------------------------------------------------------------------
ROM = 2                             # master conjugate pair
N_EXT = 2                           # two frozen external states: θ₁, θ₂
NVAR = ROM + N_EXT
PERMUTATION = [2, 1, 3, 4]          # z₁ ↔ z₂; θ₁, θ₂ real → self-conjugate
MAXZ = FAST ? 3 : 9                 # expansion order in (z₁, z₂)
MAXT = FAST ? 2 : 4                 # per-parameter θ box bound
RUN_SANITY_CHECKS = true

# θ-series bases: one per-parameter box [MAXT, MAXT] for all forms.
BASIS_K = PS.ThetaBasis([MAXT, MAXT])
BASIS_QUAD = BASIS_K
BASIS_CUBIC = BASIS_K

# Shape-field geometry: ψ₁ analytic (constant gradient), ψ₂ built from the
# first bending eigenmode — the provider needs element/QP context (4-arg form).
const ∇ψ₁ = Tens3((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)
struct BeamParametricGeom
	J₁::Tens3
	φ_full::Vector{Float64}
	φe::Vector{Float64}
end
function (g::BeamParametricGeom)(x₀, cell, cv, q)
	dofs = celldofs(cell)
	@inbounds for (i, d) in pairs(dofs)
		g.φe[i] = g.φ_full[d]
	end
	J₂ = Tens3(function_gradient(cv, q, g.φe))
	return (one(Tens3), g.J₁, J₂)
end
function GEOM_BUILDER(dh, free, master_modes, ndofs_total)
	mode = real(master_modes[:, 1])
	mode ./= maximum(abs, mode)                 # max-normalised bending mode
	φ_full = zeros(Float64, ndofs_total)
	φ_full[free] .= mode
	return BeamParametricGeom(∇ψ₁, φ_full, zeros(Float64, ndofs_per_cell(dh)))
end

# Base operators for the eigenproblem: at θ = 0 the reference is the straight
# beam (J = I), so K₀/M₀ are the standard isotropic SVK assembly.
function BASE_KM(dh, cv)
	K0 = allocate_matrix(dh)
	M0 = allocate_matrix(dh)
	SVK.svk_assemble_KM!(K0, M0, dh, cv, LAMBDA_LAME, MU_LAME, RHO)
	return K0, M0
end
function PARAM_KM(dh, cv, provider, free)
	K_full = [allocate_matrix(dh) for _ in 1:PS.nterms(BASIS_K)]
	M_full = [allocate_matrix(dh) for _ in 1:PS.nterms(BASIS_K)]
	PS.assemble_parametric_K_M!(K_full, M_full, dh, cv,
		LAMBDA_LAME, MU_LAME, RHO, provider, BASIS_K)
	return [Kf[free, free] for Kf in K_full], [Mf[free, free] for Mf in M_full]
end

# Pre-solve sanity: θ⁰ coefficient ≡ SVK assembly (exact) and the analytic
# frequency sensitivities ∂ω/∂θ₁ ≈ −2ω₀ (stretch), ∂ω/∂θ₂ ≈ 0 (bending arch).
function SANITY(K_arr, M_arr, K_ref, master_eigenvalues, master_modes)
	M = M_arr[1]
	dev = maximum(abs.(K_arr[1] - K_ref))
	@printf "sanity: ‖K_θ⁰ − K_svk‖_max = %.3e  (expect ~1e-12)\n" dev
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
BUILD_MSET() = MultiindexSet([
	SVector{NVAR, Int}(a, b, c, d)
	for a in 0:MAXZ for b in 0:MAXZ
	for c in 0:MAXT for d in 0:MAXT
	if a + b ≤ MAXZ && 1 ≤ a + b + c + d])

META() = ["example" => "04_parametric_clamped_beam",
	"formulation" => "additive shape fields, per-parameter box truncation",
	"max_degree_z" => MAXZ, "max_degree_theta" => MAXT,
	"fast" => FAST]
