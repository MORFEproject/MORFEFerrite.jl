"""
`MORFEFerrite.StructuralSVK` — high-level "mesh → ROM" UI for St. Venant-Kirchhoff
structural models with the Ferrite backend, autonomous or with near-resonant
harmonic forcing.

    using MORFE, MORFEFerrite
    const SVK = MORFEFerrite.StructuralSVK
    beam = SVK.mechanical_model(mesh; material, damping, dirichlet, fe_order, quad_order)

    (; model, spectral, meta) = build_model(beam; master = [1], expansion_order = 9)
    W, R = parametrise(model, spectral, 9;
        resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))
    rom = SVK.InvariantManifoldROM(W, R, meta; master = [1], order = 9)

**There is one `parametrise` and it is MORFE's.** This module contributes
`build_model` — the single contract every physics backend implements — and the
result container above; it does not wrap the reduction. The Ferrite SVK geometric
nonlinearity factory is `svk_nonlinearity` and the linear stiffness/mass assembler
is `svk_assemble_KM!` (a concrete `MORFE.FEMMultilinearMap` backend).
"""
module StructuralSVK

using MORFE
import MORFE: save_rom, spectrum
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using LinearAlgebra, SparseArrays, Serialization, Printf
using StaticArrays
using ..Common: load_comsol_grid, AbstractAssembledModel, Common
import ..Common: build_model, summary_entries

# Ferrite SVK backend: FerriteGeometricNonlinearity <: MORFE.FEMMultilinearMap{2}
# and the linear assemble_KM!.
include("ferrite_assembly.jl")

"""
    svk_nonlinearity(degree, dh, cv, free_to_local, n_free, λ, μ; max_unique_cols = degree)

Construct a Ferrite-backed St. Venant-Kirchhoff geometric nonlinearity term of the
given polynomial `degree` (2 = quadratic, 3 = cubic) as a `MORFE.FEMMultilinearMap{2}`.
"""
svk_nonlinearity(degree::Integer, args...; kwargs...) =
	FerriteGeometricNonlinearity{Int(degree)}(args...; kwargs...)

"""
    svk_assemble_KM!(K, M, dh, cv, λ, μ, ρ)

Assemble the linear stiffness `K` and mass `M` matrices with the Ferrite SVK backend.
"""
svk_assemble_KM!(args...; kwargs...) = assemble_KM!(args...; kwargs...)


include("types.jl")

# Material-dispatching forms: one call site works for isotropic and anisotropic.
svk_nonlinearity(degree::Integer, dh, cv, free_to_local, n_free,
	material::Union{SVKMaterial, AnisotropicMaterial}; kwargs...) =
	FerriteGeometricNonlinearity{Int(degree)}(dh, cv, free_to_local, n_free,
		stress_model(material); kwargs...)

svk_assemble_KM!(K, M, dh, cv, material::Union{SVKMaterial, AnisotropicMaterial}) =
	assemble_KM!(K, M, dh, cv, stress_model(material), Float64(material.ρ))
include("rayleigh_solver.jl")
include("mechanical_model.jl")
include("build_model.jl")
include("parametrise.jl")
include("postprocess.jl")

# SVK's implementation of the ParametricGeometry kernel interface, and the entry
# point that expands an SVK structure over a parametric coordinate transform.
include("pullback_kernel.jl")
include("parametric_model.jl")

export SVKMaterial, AnisotropicMaterial, CubicCrystal, rotate_voigt, voigt_stiffness,
	RayleighDamping, HarmonicForcing,
	AssembledMechanicalModel, InvariantManifoldROM, RayleighEigensolver,
	mechanical_model, spectrum, eigenfrequencies, print_mode_table,
	resonances, print_resonances, real_dynamics, print_equations, save_rom,
	svk_nonlinearity, svk_assemble_KM!,
	SVKPullbackKernel, parametric_model, base_operators

end # module StructuralSVK
