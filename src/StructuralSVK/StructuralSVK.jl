"""
`MORFEFerrite.StructuralSVK` — Ferrite-backed St. Venant-Kirchhoff structural
models for autonomous or harmonically forced invariant-manifold reductions.

    using MORFE, MORFEFerrite
    const SVK = MORFEFerrite.StructuralSVK
    beam = SVK.mechanical_model(mesh; material, damping, dirichlet, fe_order, quad_order)

    (; model, spectral, meta) = build_model(beam;
        master = [1], expansion_order = 9)
    W, R = MORFE.parametrise(model, spectral, 9;
        resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))

This module assembles the mechanical case, implements the shared [`build_model`](@ref)
contract, and provides spectral and resonance-inspection helpers. `build_model` returns
the `MORFE.NthOrderModel` and `MORFE.SpectralData`; `MORFE.parametrise` performs the
physics-independent reduction and returns `(W, R)`.

The low-level Ferrite entry points are [`svk_nonlinearity`](@ref), which constructs a
concrete `MORFE.FEMMultilinearMap{2}`, and [`svk_assemble_KM!`](@ref), which assembles
the linear stiffness and mass matrices.
"""
module StructuralSVK

import MORFE
using MORFE: AbstractEigensolver, ExternalSystem, MultilinearMap, NthOrderModel,
             SpectralData, StructureModalDampingEigensolver, all_multiindices_up_to,
             left_eigenmode_orders_from_slice, n_internal,
             resonance_set_from_complex_normal_form_style, resonant_multiindices
import MORFE: spectrum
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using LinearAlgebra, SparseArrays, Printf
using StaticArrays
using ..Common: load_comsol_grid, AbstractAssembledModel, Common
import ..Common: build_model, summary_entries

# Ferrite SVK backend: FerriteGeometricNonlinearity <: MORFE.FEMMultilinearMap{2}
# and the linear assemble_KM!.
include("ferrite_assembly.jl")

"""
    svk_nonlinearity(degree, dh, cv, free_to_local, n_free, λ, μ;
                     max_unique_cols = degree, fully_asymmetric = false)
    svk_nonlinearity(degree, dh, cv, free_to_local, n_free, material;
                     max_unique_cols = degree, fully_asymmetric = false)

Construct a Ferrite-backed St. Venant-Kirchhoff geometric nonlinearity term of the
given polynomial `degree` (`2` for the quadratic form, `3` for the cubic form) as a
`MORFE.FEMMultilinearMap{2}`. The constitutive law may be supplied as Lamé constants
`λ`, `μ` or as an `SVKMaterial`/`AnisotropicMaterial`.

`free_to_local` maps global Ferrite DOFs into the `n_free`-component state. The
`max_unique_cols` cache must accommodate the column batches used by the reduction;
`fully_asymmetric` is forwarded to MORFE's multilinear-term symmetry policy.
"""
function svk_nonlinearity(degree::Integer, args...; kwargs...)
    FerriteGeometricNonlinearity{Int(degree)}(args...; kwargs...)
end

"""
    svk_assemble_KM!(K, M, dh, cv, λ, μ, ρ)
    svk_assemble_KM!(K, M, dh, cv, material)

Assemble the three-dimensional linear stiffness `K` and mass `M` matrices in place
with the Ferrite SVK backend. Supply either Lamé constants and density or an
`SVKMaterial`/`AnisotropicMaterial`. `K` and `M` must be preallocated with the
sparsity pattern of `dh`; the function returns `nothing`.
"""
svk_assemble_KM!(args...; kwargs...) = assemble_KM!(args...; kwargs...)

include("types.jl")

# Material-dispatching forms: one call site works for isotropic and anisotropic.
function svk_nonlinearity(degree::Integer, dh, cv, free_to_local, n_free,
        material::Union{SVKMaterial, AnisotropicMaterial}; kwargs...)
    FerriteGeometricNonlinearity{Int(degree)}(dh, cv, free_to_local, n_free,
        stress_model(material); kwargs...)
end

function svk_assemble_KM!(K, M, dh, cv, material::Union{SVKMaterial, AnisotropicMaterial})
    assemble_KM!(K, M, dh, cv, stress_model(material), Float64(material.ρ))
end
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
       AssembledMechanicalModel, RayleighEigensolver,
       mechanical_model, spectrum, eigenfrequencies, print_mode_table, probe_dof,
       resonances, print_resonances,
       svk_nonlinearity, svk_assemble_KM!,
       SVKPullbackKernel, parametric_model, base_operators

end # module StructuralSVK
