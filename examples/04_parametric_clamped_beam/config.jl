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
MAXT = FAST ? 2 : 4                 # per-parameter θ box bound
RUN_SANITY_CHECKS = true

# θ-series bases: one per-parameter box [MAXT, MAXT] for all forms. Sharing ONE
# object is how a caller says "same truncation" — `parametric_model` then builds
# a single geometry cache instead of three.
BASIS_K = PG.GeometryParameterBasis([MAXT, MAXT])
BASIS_QUAD = BASIS_K
BASIS_CUBIC = BASIS_K
GEOMETRY_PARAMETER_BASES = (; linear = BASIS_K, quadratic = BASIS_QUAD, cubic = BASIS_CUBIC)

MATERIAL = SVKMaterial(E = E, ν = ν, ρ = RHO)
DAMPING = RayleighDamping(α = ALPHA, β = BETA)

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

# The shape field ψ₂ is built FROM the first bending mode, so the eigenproblem
# comes first: assemble the straight beam (θ = 0 reference, J = I) with the
# standard SVK kernels, solve, build the provider, then expand parametrically.
function BUILD_CASE(dh, cv, free)
	K0 = allocate_matrix(dh)
	M0 = allocate_matrix(dh)
	SVK.svk_assemble_KM!(K0, M0, dh, cv, MATERIAL)
	eigenproblem = spectrum(K0[free, free], M0[free, free],
		StructureModalDampingEigensolver(NEV, ALPHA, BETA);
		sorter! = (args...) -> nothing)
	provider = GEOM_BUILDER(dh, free, eigenproblem.eigenmodes[:, 1, 1:ROM], ndofs(dh))
	pcase = SVK.parametric_model(dh, cv, provider;
		geometry_parameter_basis = GEOMETRY_PARAMETER_BASES, material = MATERIAL, damping = DAMPING, free = free)
	return pcase, eigenproblem
end

# Pre-solve sanity: θ⁰ coefficient ≡ SVK assembly (exact) and the analytic
# frequency sensitivities ∂ω/∂θ₁ ≈ −2ω₀ (stretch), ∂ω/∂θ₂ ≈ 0 (bending arch).
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
BUILD_MSET() = MultiindexSet([
	SVector{NVAR, Int}(a, b, c, d)
	for a in 0:MAXZ for b in 0:MAXZ
	for c in 0:MAXT for d in 0:MAXT
	if a + b ≤ MAXZ && 1 ≤ a + b + c + d])

META() = ["example" => "04_parametric_clamped_beam",
	"formulation" => "additive shape fields, per-parameter box truncation",
	"max_degree_z" => MAXZ, "max_degree_theta" => MAXT,
	"fast" => FAST]
