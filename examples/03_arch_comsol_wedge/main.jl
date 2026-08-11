"""
Isotropic polysilicon arch, COMSOL P18 wedge mesh, St. Venant-Kirchhoff.

Computes the invariant-manifold ROM (DPIM) of the first mode pair via the
high-level `MORFEFerrite.StructuralSVK` UI, reading the mesh directly from a
COMSOL `.mphtxt` file (Dirichlet boundary given as COMSOL entity IDs). Edit
the CASE block below for your own problem; the pipeline underneath is generic
and identical across the structural examples.

Smoke run (≈ seconds, order 3):   MORFE_FAST=1 julia --project=. main.jl
Full run  (order 5):              julia --project=. main.jl
Check the result:                 julia --project=. validate.jl
Expected numbers: see README.md (asserted by validate.jl).
"""

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
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays"])
end
Pkg.instantiate()

using MORFE, MORFEFerrite
const SVK = MORFEFerrite.StructuralSVK

# ── CASE — edit this block for your own problem ──────────────────────────────
FAST = get(ENV, "MORFE_FAST", "0") == "1"

MESH = joinpath(@__DIR__, "arch_2_force.mphtxt")        # Gmsh .msh or COMSOL .mphtxt
MATERIAL = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3)
DAMPING = SVK.RayleighDamping(α = 0.0, β = 0.0)
DIRICHLET = Set([1, 11])       # COMSOL entity IDs of the arch feet (raw IDs 0 and 10)
FE_ORDER = 2
QUAD_ORDER = 4 # should read quadrature order instead; "QUAD" also means quadratic
MASTER = [1]                   # master conjugate mode pairs
ORDER = FAST ? 3 : 5           # parametrisation order
FORCING = nothing              # or SVK.HarmonicForcing(mode = 1, amplitude = 0.03);
# a vector of them gives multi-harmonic forcing (N_EXT = 2 per forcing)

# ── PIPELINE — generic; no need to edit ──────────────────────────────────────
# ## Assemble the mechanical model (K, M, C on free DOFs + SVK nonlinearity)
model = SVK.mechanical_model(MESH;
	material = MATERIAL, damping = DAMPING, dirichlet = DIRICHLET,
	fe_order = FE_ORDER, quad_order = QUAD_ORDER)

# ## Compute the invariant-manifold ROM (eigenproblem + cohomological solve)
rom = SVK.parametrise(model; master = MASTER, order = ORDER, forcing = FORCING)

# ## Report the realified reduced dynamics and save the standard result layout
SVK.print_equations(rom)
SVK.save_rom(rom, joinpath(@__DIR__, "results"))
println("\nResults written to $(joinpath(@__DIR__, "results"))")
