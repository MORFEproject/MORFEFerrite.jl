# Reusable dev-path bootstrap for MORFEFerrite environments.
#
# MORFE (and MORFEFerrite) are unregistered, so consumers must dev them by path
# rather than relying on `[sources]` (which needs Julia ≥ 1.11). Call
# `bootstrap_dev(project_dir)` from a script that has `Pkg.activate`d its own
# environment; it dev-installs MORFE and MORFEFerrite from their sibling repos.

using Pkg: Pkg

"""
    bootstrap_dev(; morfe, morfeferrite)

Dev-install MORFE and MORFEFerrite by path into the active project. Defaults
resolve the sibling checkouts relative to this file.
"""
function bootstrap_dev(;
	morfe::AbstractString = normpath(joinpath(@__DIR__, "..", "..", "..", "MORFE_jl")),
	morfeferrite::AbstractString = normpath(joinpath(@__DIR__, "..")))
	specs = Pkg.PackageSpec[]
	haskey(Pkg.project().dependencies, "MORFE") ||
		push!(specs, Pkg.PackageSpec(path = morfe))
	haskey(Pkg.project().dependencies, "MORFEFerrite") ||
		push!(specs, Pkg.PackageSpec(path = morfeferrite))
	isempty(specs) || Pkg.develop(specs)
	return nothing
end
