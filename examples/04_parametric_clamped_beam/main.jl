"""
	main.jl — two-parameter parametric ROM for a 3D clamped-clamped beam,
	using the GENERAL parametric formulation in `MORFEFerrite.ParametricStructural`.

Reference map (additive, each θᵢ scaling an independent shape field):

	x(θ₁,θ₂,x₀) = x₀ + θ₁ ψ₁(x₀) + θ₂ ψ₂(x₀)
	J(θ₁,θ₂,x₀) = I + θ₁ ∇ψ₁ + θ₂ ∇ψ₂(x₀)

	∇ψ₁ = e₁⊗e₁           uniform axial stretch (constant)
	∇ψ₂ = ∇φ₁(x₀)         first bending eigenmode gradient (FE field, per QP)

CORRECTED FORMULATION — this replaces the old example 04, whose bivariate
machinery truncated the θ-series by TOTAL degree (k₁+k₂ ≤ N). The general
engine truncates PER PARAMETER (multiindex box: k₁ ≤ N and k₂ ≤ N), both in
the assembly θ-series (`ThetaBasis`) and in the DPIM multiindex set. Its
outputs are therefore the corrected reference; the old committed reference is
NOT comparable.

Orders overridable via `MORFE_MAXZ` / `MORFE_MAXT` (defaults 9 / 4).
NOTE: the θ-series kernel still allocates in its inner loop (see ex07 note), so
run reduced orders (e.g. `MORFE_MAXZ=3 MORFE_MAXT=2`) until the allocation pass
lands. DPIM is graded — lower orders are exact truncations.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	# MORFE.jl is expected as a sibling checkout (folder MORFE.jl or MORFE_jl),
	# either next to this repository or one directory above it; override with
	# ENV["MORFE_PATH"] if it lives elsewhere.
	morfe = get(ENV, "MORFE_PATH", "")
	if isempty(morfe)
		cands = [joinpath(@__DIR__, "..", "..", "..", n) for n in ("MORFE.jl", "MORFE_jl")]
		append!(cands, [joinpath(@__DIR__, "..", "..", "..", "..", n) for n in ("MORFE.jl", "MORFE_jl")])
		morfe = first(filter(isdir, cands))
	end
	Pkg.develop([
		Pkg.PackageSpec(path = morfe),
		Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..")),
	])
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays", "Tensors"])
end
Pkg.instantiate()

using MORFE, MORFEFerrite
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using SparseArrays, LinearAlgebra, Printf, Tensors, StaticArrays
const PS = MORFEFerrite.ParametricStructural
const SVK = MORFEFerrite.StructuralSVK
const Tens3 = Tensor{2, 3, Float64, 9}

const max_degree_z = parse(Int, get(ENV, "MORFE_MAXZ", "9"))
const max_degree_θ = parse(Int, get(ENV, "MORFE_MAXT", "4"))
const N_θ = max_degree_θ          # per-parameter assembly-series bound
const ROM = 2
const N_EXT = 2                   # two frozen external states θ₁, θ₂
const NVAR = ROM + N_EXT

# ── §1 mesh + FE ───────────────────────────────────────────────────────────
grid = togrid(joinpath(@__DIR__, "beam_h27_10x2x2.msh"))
ip = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr = QuadratureRule{RefHexahedron}(3)
cv = CellValues(qr, ip, geo_ip)
dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch); update!(ch, 0.0)
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free = length(free); FOM = n_free
println("Total DOFs: ", ndofs(dh), "   free: ", n_free)

# ── §2 material ────────────────────────────────────────────────────────────
E = 160e3; ν = 0.22; ρ = 2.32e-3
λ_lame = (E * ν) / ((1 + ν) * (1 - 2ν))
μ_lame = E / (2(1 + ν))
α = 0.0; β = 0.0

# ── §3 eigenproblem at the reference configuration (J = I) ─────────────────
# At θ = 0 the parametric K-integrand reduces to the standard isotropic SVK
# assembly, so K₀/M₀ come straight from StructuralSVK (cross-module reuse).
K0_full = allocate_matrix(dh)
M0_full = allocate_matrix(dh)
SVK.svk_assemble_KM!(K0_full, M0_full, dh, cv, λ_lame, μ_lame, ρ)
K_ref = K0_full[free, free]; M_ref = M0_full[free, free]

solver_eig = StructureModalDampingEigensolver(10, α, β)
eigenproblem = solve_eigenproblem(K_ref, M_ref, solver_eig; sorter! = (args...) -> nothing)
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)
select_master_modes_by_sorting(eigenproblem, ROM)
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[:, 1, 1:ROM]
left_eigenmodes = X[:, 1:ROM]
master_modes_derivatives = zeros(ComplexF64, FOM, 2, ROM)
for r in 1:ROM
	master_modes_derivatives[:, 1, r] .= Y[:, 2, r]
	master_modes_derivatives[:, 2, r] .= master_eigenvalues[r] .* Y[:, 2, r]
end
println("master eigenvalues: ", collect(master_eigenvalues))

# ── §4 shape fields ─────────────────────────────────────────────────────────
# ψ₁: uniform axial stretch, ∇ψ₁ = e₁⊗e₁ (constant).
# ψ₂: first bending eigenmode, max-normalised; ∇ψ₂ evaluated per QP from the
#     FE field (full-DOF vector, constrained entries zero).
const ∇ψ₁ = Tens3((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0)
arch_mode_free = real(master_modes[:, 1])
arch_mode_free ./= maximum(abs, arch_mode_free)
φ_full = zeros(Float64, ndofs(dh))
φ_full[free] .= arch_mode_free

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
geom = BeamParametricGeom(∇ψ₁, φ_full, zeros(Float64, ndofs_per_cell(dh)))

# ── §5 parametric K/M coefficient matrices (per-parameter box) ──────────────
basis = PS.ThetaBasis([N_θ, N_θ])
println("θ-basis: box [$N_θ, $N_θ] → ", PS.nterms(basis), " terms")
K_full = [allocate_matrix(dh) for _ in 1:PS.nterms(basis)]
M_full = [allocate_matrix(dh) for _ in 1:PS.nterms(basis)]
@time PS.assemble_parametric_K_M!(K_full, M_full, dh, cv, λ_lame, μ_lame, ρ, geom, basis)
K_arr = [K_full[m][free, free] for m in 1:PS.nterms(basis)]
M_arr = [M_full[m][free, free] for m in 1:PS.nterms(basis)]
K = K_arr[1]; M = M_arr[1]; C = α * M + β * K

# Sanity 1: parametric base coefficient ≡ standard SVK assembly (machine ε).
let dev = maximum(abs.(K - K_ref))
	@printf "sanity: ‖K_θ⁰ − K_svk‖_max = %.3e  (expect ~1e-12)\n" dev
end

# Sanity 2: frequency sensitivities from the θ¹ coefficient matrices.
#   ∂ω/∂θ₁ ≈ −2ω₀ (axial stretch);   ∂ω/∂θ₂ ≈ 0 (bending-mode arch).
i10 = basis.index[SVector(1, 0)]
i01 = basis.index[SVector(0, 1)]
let φ = arch_mode_free, ω₀ = abs(master_eigenvalues[1])
	φᵀMφ = dot(φ, M * φ)
	dω_dθ1 = (dot(φ, K_arr[i10] * φ) - ω₀^2 * dot(φ, M_arr[i10] * φ)) / (2ω₀ * φᵀMφ)
	dω_dθ2 = (dot(φ, K_arr[i01] * φ) - ω₀^2 * dot(φ, M_arr[i01] * φ)) / (2ω₀ * φᵀMφ)
	@printf "sanity: ∂ω/∂θ₁ (FEM) = %+.6f   expected ≈ %+.6f (−2ω₀)\n" dω_dθ1 (-2ω₀)
	@printf "sanity: ∂ω/∂θ₂ (FEM) = %+.6f   expected ≈ 0\n" dω_dθ2
end

# ── §6 nonlinear maps + corrections (GENERAL library) ───────────────────────
pgn_quad = PS.ParametricGeometricNonlinearity{2}(dh, cv, λ_lame, μ_lame, geom,
	free_to_local, n_free, basis)
pgn_cube = PS.ParametricGeometricNonlinearity{3}(dh, cv, λ_lame, μ_lame, geom,
	free_to_local, n_free, basis)
quad_maps = PS.multilinear_maps(pgn_quad)
cube_maps = PS.multilinear_maps(pgn_cube)
K_corrections = PS.build_K_corrections(K_arr, basis)
C_corrections = PS.build_C_corrections(K_arr, M_arr, α, β, basis)
M_corrections = PS.build_M_corrections(M_arr, basis)
println("maps: quad=$(length(quad_maps)) cube=$(length(cube_maps)) ",
	"Kcorr=$(length(K_corrections)) Ccorr=$(length(C_corrections)) Mcorr=$(length(M_corrections))")

# ── §7 augmented model + PER-PARAMETER-box multiindex set ───────────────────
ext_sys = ExternalSystem((complex(0.0, 0.0), complex(0.0, 0.0)))
ZERO = spzeros(eltype(K), n_free, n_free)
model = NDOrderModel((K, C, M, ZERO),
	(quad_maps..., cube_maps..., K_corrections..., C_corrections..., M_corrections...),
	ext_sys)

# Corrected truncation: group-total in (z₁,z₂), BOX in (θ₁,θ₂).
mset = MultiindexSet([
	SVector{NVAR, Int}(a, b, c, d)
	for a in 0:max_degree_z for b in 0:max_degree_z
	for c in 0:max_degree_θ for d in 0:max_degree_θ
	if a + b ≤ max_degree_z && 1 ≤ a + b + c + d])
println("monomials: ", length(mset))

resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(master_eigenvalues), 0.05;
	external_eigenvalues = zeros(ComplexF64, N_EXT))

# ── §8 cohomological solve ───────────────────────────────────────────────────
left_modes_derivatives = left_eigenmode_orders_from_slice(
	model.linear_terms, left_eigenmodes, collect(master_eigenvalues))[:, 1:(end-1), :]
@time (W, R) = solve_cohomological_problem(model, mset, master_eigenvalues,
	master_modes, left_eigenmodes, resonance_set;
	master_modes_derivatives = master_modes_derivatives,
	left_modes_derivatives = left_modes_derivatives,
	conjugate_permutation = [2, 1, 3, 4])   # z₁↔z₂; θ₁,θ₂ real → self-conjugate

# Sanity 3: ROM θ-sensitivities vs the FEM diagnostics.
let exps = R.poly.multiindex_set.exponents
	i_z1t1 = findfirst(==(SVector(1, 0, 1, 0)), exps)
	i_z1t2 = findfirst(==(SVector(1, 0, 0, 1)), exps)
	c1 = R.poly.coefficients[1, i_z1t1]
	c2 = R.poly.coefficients[1, i_z1t2]
	@printf "sanity: ROM (1,0,1,0) = %+.6f%+.6fim  (Im ≈ ∂ω/∂θ₁)\n" real(c1) imag(c1)
	@printf "sanity: ROM (1,0,0,1) = %+.6f%+.6fim  (≈ 0)\n" real(c2) imag(c2)
end

# ── §9 save the CORRECTED reference ──────────────────────────────────────────
outdir = joinpath(@__DIR__, "results", "data")
mkpath(outdir)
open(joinpath(outdir, "R_coefficients.csv"), "w") do io
	exps = R.poly.multiindex_set.exponents
	NVAR_R = size(R.poly.coefficients, 1)
	println(io, join(["exp_$i" for i in 1:length(exps[1])], ",") * "," *
			join(["R$(i)_re,R$(i)_im" for i in 1:NVAR_R], ","))
	for (m, ex) in enumerate(exps)
		c = R.poly.coefficients[:, m]
		any(abs.(c) .> 1e-14) || continue
		println(io, join(string.(Int.(ex)), ",") * "," *
				join(["$(real(c[i])),$(imag(c[i]))" for i in 1:NVAR_R], ","))
	end
end
println("\nCorrected ROM saved to $outdir/R_coefficients.csv")
