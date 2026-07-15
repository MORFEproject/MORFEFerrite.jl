"""
Clamped-clamped beam, St. Venant-Kirchhoff, Ferrite backend — high-level UI
via `MORFEFerrite.StructuralSVK`.

The fully explicit construction of the same ROM lives in `low_level.jl`;
both produce identical reduced dynamics (enforced by `test/StructuralSVK/`).
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
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays"])
end
Pkg.instantiate()

using MORFE, MORFEFerrite, Ferrite, FerriteGmsh, Arpack, LinearMaps
const SVK = MORFEFerrite.StructuralSVK

beam = SVK.mechanical_model(
	joinpath(@__DIR__, "clamped_clamped_beam.msh");
	material = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
	damping = SVK.RayleighDamping(
		α = 0.5370828278264171 / 100.0,
		β = 1.0 / (0.5370828278264171 * 100.0),
	),
	dirichlet = "Dirichlet",
	fe_order = 2, quad_order = 3,
)

rom = SVK.parametrise(beam; master = [1], order = 9)

# Near-resonant harmonic forcing (load shaped like mode 1, at mode 1's natural
# frequency) — adds two external states (N_EXT = 2). Uncomment to use:
# rom = SVK.parametrise(beam; master = [1], order = 9,
#     forcing = SVK.HarmonicForcing(mode = 1, amplitude = 0.03))

SVK.print_equations(rom)
SVK.save_rom(rom, joinpath(@__DIR__, "results"))
println("\nResults written to $(joinpath(@__DIR__, "results"))")
