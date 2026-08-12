# # Parametric structural ROM — generic pipeline
#
# Computes a geometrically parametric invariant-manifold ROM with
# `MORFEFerrite.ParametricGeometry`: the reference map is additive,
# `x(θ,x₀) = x₀ + Σᵢ θᵢ ψᵢ(x₀)`, each parameter θᵢ scaling an independent
# shape field, with per-parameter (multiindex-box) θ-series truncation.
#
# The model assembly lives in the module, not here: `parametric_model` expands
# an SVK structure over the coordinate transform, and `build_model` produces the
# augmented `NthOrderModel`, the reconciled `SpectralData` and the derived
# conjugate permutation. This file only states the CASE and the reduction.
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
const PG = MORFEFerrite.ParametricGeometry
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
n_free = length(free)
println("Total DOFs: ", ndofs(dh), "   free: ", n_free)

# ## The assembled parametric case and its base eigenproblem.
# `BUILD_CASE` owns the ordering the problem needs: example 04's shape field is
# built FROM the first bending mode, so it eigensolves first; example 07's
# geometry is analytic, so it assembles first and reads the θ⁰ operators back.
pcase, eigenproblem = BUILD_CASE(dh, cv, free)
println("master eigenvalues: ", collect(eigenproblem.eigenvalues[1:ROM]))
RUN_SANITY_CHECKS && SANITY(pcase, eigenproblem)
display(pcase)

# ## Augmented model, spectral data and conjugate permutation — all derived.
(; model, spectral, meta) = build_model(pcase; master = [1], spectrum = eigenproblem)
println("\nORD = $(meta.ORD),  N_EXT = $(meta.N_EXT),  nonlinear terms = $(meta.n_terms)")
println("conjugate permutation: ", meta.conjugate_permutation)

# ## The reduction. The monomial set is the CALLER's choice — `build_model`
# deliberately does not pick one, because the θ-box truncation is a modelling
# decision, not a property of the assembled model.
mset = BUILD_MSET()
println("monomials: ", length(mset))
@time (W, R) = parametrise(model, spectral, mset;
	resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05),
	conjugate_permutation = meta.conjugate_permutation)

# ## Save the standard result layout (data/{W.jls, R.jls, R_coefficients.csv})
MORFE.save_rom(joinpath(@__DIR__, "results"), W, R; metadata = META())
println("\nResults written to $(joinpath(@__DIR__, "results"))")
