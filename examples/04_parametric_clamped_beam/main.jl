# # Parametric structural ROM — generic pipeline
#
# Computes a geometrically parametric invariant-manifold ROM with
# `MORFEFerrite.ParametricStructural`: the reference map is additive,
# `x(θ,x₀) = x₀ + Σᵢ θᵢ ψᵢ(x₀)`, each parameter θᵢ scaling an independent
# shape field, with per-parameter (multiindex-box) θ-series truncation.
#
# **Everything problem-specific lives in `config.jl`** (mesh, material, shape
# fields, orders, bases, providers). This file is the generic pipeline — it is
# textually identical across the parametric examples (04, 07); adapting to a
# new problem means editing `config.jl` only.
#
# Smoke run:   `MORFE_FAST=1 julia --project=. main.jl`
# Full run:    `julia --project=. main.jl`
# Check:       `julia --project=. validate.jl`

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	# One-time environment setup. MORFE.jl is expected as a sibling checkout
	# (folder MORFE.jl or MORFE_jl), next to this repository or one directory
	# above it; override with ENV["MORFE_PATH"]. Collapses to plain
	# Pkg.instantiate() once the packages are registered.
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

# ## The case: everything problem-specific comes from config.jl
include(joinpath(@__DIR__, "config.jl"))

# ## FE space and boundary conditions
grid = togrid(MESH)
ip = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr = QuadratureRule{RefHexahedron}(3)
cv = CellValues(qr, ip, geo_ip)
dh = DofHandler(grid);
add!(dh, :u, ip);
close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch);
update!(ch, 0.0)
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free = length(free)
println("Total DOFs: ", ndofs(dh), "   free: ", n_free)

# ## Base operators and eigenproblem (θ = 0 configuration)
K0_full, M0_full = BASE_KM(dh, cv)
K_ref = K0_full[free, free];
M_ref = M0_full[free, free]
solver_eig = StructureModalDampingEigensolver(NEV, ALPHA, BETA)
eigenproblem = spectrum(K_ref, M_ref, solver_eig; sorter! = (args...) -> nothing)
eigenvalues, Y, X = eigenproblem.eigenvalues, eigenproblem.eigenmodes, eigenproblem.left_eigenmodes
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[:, 1, 1:ROM]
left_eigenmodes = X[:, 1:ROM]
master_modes_derivatives = zeros(ComplexF64, n_free, 2, ROM)
for r in 1:ROM
	master_modes_derivatives[:, 1, r] .= Y[:, 2, r]
	master_modes_derivatives[:, 2, r] .= master_eigenvalues[r] .* Y[:, 2, r]
end
println("master eigenvalues: ", collect(master_eigenvalues))

# ## Shape-field geometry provider (analytic, or built from the eigen data)
provider = GEOM_BUILDER(dh, free, master_modes, ndofs(dh))

# ## Parametric stiffness/mass coefficient matrices over the θ box
K_arr, M_arr = PARAM_KM(dh, cv, provider, free)
K = K_arr[1];
M = M_arr[1];
C = ALPHA * M + BETA * K
RUN_SANITY_CHECKS && SANITY(K_arr, M_arr, K_ref, master_eigenvalues, master_modes)

# ## Parametric nonlinear maps and linear K/C/M corrections
pgn_quad = PS.ParametricGeometricNonlinearity{2}(dh, cv, LAMBDA_LAME, MU_LAME,
	provider, free_to_local, n_free, BASIS_QUAD)
pgn_cube = PS.ParametricGeometricNonlinearity{3}(dh, cv, LAMBDA_LAME, MU_LAME,
	provider, free_to_local, n_free, BASIS_CUBIC)
quad_maps = PS.multilinear_maps(pgn_quad)
cube_maps = PS.multilinear_maps(pgn_cube)
K_corrections = PS.build_K_corrections(K_arr, BASIS_K)
C_corrections = PS.build_C_corrections(K_arr, M_arr, ALPHA, BETA, BASIS_K)
M_corrections = PS.build_M_corrections(M_arr, BASIS_K)
println("maps: quad=$(length(quad_maps)) cube=$(length(cube_maps)) ",
	"Kcorr=$(length(K_corrections)) Ccorr=$(length(C_corrections)) Mcorr=$(length(M_corrections))")

# ## Augmented model, multiindex set, and the cohomological solve
ext_sys = ExternalSystem(ntuple(_ -> complex(0.0, 0.0), N_EXT))
ZERO = spzeros(eltype(K), n_free, n_free)
model = NthOrderModel((K, C, M, ZERO),
	(quad_maps..., cube_maps...,
		K_corrections..., C_corrections..., M_corrections...),
	ext_sys)
mset = BUILD_MSET()
println("monomials: ", length(mset))
## The resonance policy is stated explicitly: complex-normal-form style over the
## master eigenvalues with the frozen external states; outer (non-master) modes
## are deliberately not added as resonance targets.
resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(master_eigenvalues), 0.05;
	external_eigenvalues = zeros(ComplexF64, N_EXT))
left_modes_derivatives = left_eigenmode_orders_from_slice(
	model.linear_terms, left_eigenmodes, collect(master_eigenvalues))[:, 1:(end-1), :]

## One spectral object carries the master eigenvalues and both sets of order-blocks;
## `SpectralData` applies the mirrored right/left convention. The ROM-length master
## pairing is not stated here because PERMUTATION spans the external states too, so it
## goes to the solve, which uses it verbatim.
spectral = SpectralData(; eigenvalues = master_eigenvalues,
	right_modes = master_modes, right_derivatives = master_modes_derivatives,
	left_modes = left_eigenmodes, left_blocks = Array(left_modes_derivatives))

@time (W, R) = parametrise(model, spectral, mset;
	resonance = resonance_set,
	conjugate_permutation = PERMUTATION)

# ## Save the standard result layout (data/{W.jls, R.jls, R_coefficients.csv})
MORFE.save_rom(joinpath(@__DIR__, "results"), W, R; metadata = META())
println("\nResults written to $(joinpath(@__DIR__, "results"))")
