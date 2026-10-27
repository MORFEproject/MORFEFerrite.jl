"""
`MORFEFerrite.StructuralSVK` — Ferrite-backed St. Venant-Kirchhoff structural
models for autonomous or harmonically forced invariant-manifold reductions.

    using MORFE, MORFEFerrite
    const SVK = MORFEFerrite.StructuralSVK
    beam = SVK.mechanical_model(mesh; material, damping, dirichlet, fe_order, quad_order)

    (; model, spectral, meta) = build_model(beam; master = [1])
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
using MORFE: ExternalSystem, MultilinearMap, NthOrderModel,
             SpectralData, StructureModalDampingEigensolver, all_multiindices_up_to,
             n_internal, resonance_set_from_complex_normal_form_style,
             resonant_multiindices
import MORFE: spectrum
# Arpack is not called here directly: loading it activates MORFE's Arpack
# extension, which implements StructureModalDampingEigensolver, the default of
# `spectrum`.
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using LinearAlgebra, SparseArrays, Printf
using StaticArrays
using ..Common: load_comsol_grid, AbstractAssembledModel, Common
import ..Common: build_model, summary_entries

# Ferrite SVK backend: FerriteGeometricNonlinearity <: MORFE.FEMMultilinearMap{2}
# and the linear assemble_KM!.
include("ferrite_assembly.jl")

include("types.jl")

"""
    svk_nonlinearity(degree, dof_handler, cell_values, dof_to_free, n_free, material;
                     fully_asymmetric = false)
    svk_nonlinearity(degree, dof_handler, cell_values, dof_to_free, n_free, λ, μ;
                     fully_asymmetric = false)

Construct a Ferrite-backed St. Venant-Kirchhoff geometric nonlinearity term as a
[`FerriteGeometricNonlinearity`](@ref), a `MORFE.FEMMultilinearMap{2}`.

# Arguments
- `degree`: `2` for the quadratic form, `3` for the cubic form.
- `dof_handler::Ferrite.DofHandler`: a closed handler with one displacement field.
- `cell_values::Ferrite.CellValues`: built from that field's vector Lagrange interpolation,
  `ip^3` (checked), and the quadrature rule to integrate with.
- `dof_to_free::Dict{Int, Int}`: global DOF number → index in the free state vector;
  constrained DOFs are absent.
- `n_free::Int`: the number of free DOFs, the length of the state vector.
- `material`: an [`SVKMaterial`](@ref) or [`AnisotropicMaterial`](@ref); or the Lamé
  constants `λ`, `μ`.

`fully_asymmetric` is forwarded to MORFE's multilinear-term symmetry policy.
[`mechanical_model`](@ref) builds the discretisation arguments and returns them in its
`info` as `dof_handler`, `cell_values`, `dof_to_free` and `n_dofs`.
"""
function svk_nonlinearity(degree::Integer, dof_handler, cell_values, dof_to_free, n_free,
        material::Union{SVKMaterial, AnisotropicMaterial}; kwargs...)
    FerriteGeometricNonlinearity{Int(degree)}(
        dof_handler, cell_values, dof_to_free, n_free,
        stress_model(material); kwargs...)
end

function svk_nonlinearity(degree::Integer, dof_handler, cell_values, dof_to_free, n_free,
        λ::Real, μ::Real; kwargs...)
    FerriteGeometricNonlinearity{Int(degree)}(
        dof_handler, cell_values, dof_to_free, n_free,
        λ, μ; kwargs...)
end

function svk_nonlinearity(degree::Integer, dof_handler, cell_values, dof_to_free, n_free,
        stress::AbstractStress; kwargs...)
    FerriteGeometricNonlinearity{Int(degree)}(
        dof_handler, cell_values, dof_to_free, n_free,
        stress; kwargs...)
end

"""
    svk_assemble_KM!(K, M, dof_handler, cell_values, material)
    svk_assemble_KM!(K, M, dof_handler, cell_values, λ, μ, ρ)

Assemble the three-dimensional linear stiffness `K` and mass `M` matrices in place
with the Ferrite SVK backend, over all global DOFs of `dof_handler`. `K` and `M` must be
preallocated with its sparsity pattern, `allocate_matrix(dof_handler)`; `cell_values`
holds the displacement interpolation and the quadrature rule. Supply either an
`SVKMaterial`/`AnisotropicMaterial` or Lamé constants and density. Returns `nothing`.
"""
function svk_assemble_KM!(K, M, dof_handler, cell_values,
        material::Union{SVKMaterial, AnisotropicMaterial})
    assemble_KM!(K, M, dof_handler, cell_values, stress_model(material),
        Float64(material.ρ))
end

function svk_assemble_KM!(K, M, dof_handler, cell_values, λ::Real, μ::Real, ρ::Real)
    assemble_KM!(K, M, dof_handler, cell_values, λ, μ, ρ)
end
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
       AssembledMechanicalModel,
       mechanical_model, spectrum, eigenfrequencies, print_mode_table, probe_dof,
       resonances, print_resonances,
       svk_nonlinearity, svk_assemble_KM!,
       SVKPullbackKernel, parametric_model, base_operators

end # module StructuralSVK
