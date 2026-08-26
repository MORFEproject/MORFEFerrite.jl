"""
`MORFEFerrite.ParametricGeometry` — parametric mesh coordinate transforms.

A **physics-blind** module: it implements the general parametric coordinate
transform `x(θ,x₀) = Σ_α x_α(x₀) θ^α` as a multivariate θ-power series — the
determinant, adjugate and inverse determinant of its Jacobian — and expands any
physics' multilinear maps over it.

```
J(θ,x₀) = Σ_α J_α(x₀) θ^α             det/adj are then exact polynomials
∇u  →  ∇u · adj J(θ)                  gradient pullback
dΩ  →  dΩ · (1/det J(θ))^p            weak-form weighting
```

**Dimension-general.** `d` is carried by the tensor type of the Jacobian
coefficients and appears in no signature: `determinant_series` is App. A.1's
multilinear expansion over the columns and `adjugate_series` is App. A.2's
recurrence, so no cofactor formula is hardcoded. 2D and 3D are both gated
end-to-end (`test_moved_mesh_fom_2d.jl`, `test_series_algebra_2d.jl`).

**One caveat on App. A.2**: the appendix derives the adjugate recurrence
assuming `J₀ = I`. A curved reference configuration has `J₀ ≠ I` — example 07's
arch is exactly that — so the implemented form inverts the leading term,
`A_σ = J₀⁻¹(c_σ I − Σ J_α A_{σ−α})`, reducing to the published one when `J₀ = I`.

The affine map `x₀ + Σᵢ θᵢψᵢ(x₀)` is the common case and has its own
`jacobian_series` method; the polynomial form takes `multiindex => J_α` pairs.
The θ-series live over a MORFE `MultiindexSet` **box** (per-parameter
truncation); the single-parameter arch and the multi-parameter beam are
instances of the same engine. `θ` here is the `μ` of the theory write-up.

## Choose the per-parameter bounds from the geometry, not by symmetry

The box is per-parameter *precisely* so unequal parameters need not be truncated
alike, and getting this wrong is the difference between a converged model and a
plausible-looking wrong one. Two things set the bounds:

- **The exact polynomial degrees.** `adj J` has degree `≤ (d−1)·deg J` and
  `det J` degree `≤ d·deg J`, **per parameter as well as in total**, so a form's
  integrand has degree = (number of gradient factors) × `deg adj J`. For an
  affine map that is 2 / 3 / 4 for the stiffness / quadratic / cubic form when
  `∇ψ` is nilpotent — example 07 uses exactly `[2], [3], [4]` and is lossless.
  Note the bounds scale with `deg J`: a map that is quadratic in `θᵢ` needs
  `2d` on that axis, not `d`.
- **The reciprocal.** If `det J ≢ 1` the `1/det J` series is infinite and its
  truncation, not the polynomial degrees, dominates. That parameter needs many
  more terms than a volume-preserving one. Example 04's `[4,4]` → `[8,2]` cost
  two extra terms and was 555× more accurate.

## The expansion has a radius, and it is checked

`1/det J` is expanded about `det J = 1`, so it converges only where
`|det J − 1| < 1`, and the transform separately needs `det J > 0`. Outside that
radius **no truncation order converges** while the assembled model still looks
well formed. [`report_geometry_validity`](@ref) measures the range and
`build_model` prints it; [`geometry_validity_at`](@ref) gives the error at one θ.

## Validate against a moved mesh, never against an archived coefficient file

The reference that means anything is a FOM whose geometry actually changed:
freeze θ, displace the mesh nodes to `x₀ + Σᵢθᵢψᵢ(x₀)`, reassemble with the
physics' ordinary non-parametric code and compare. Same topology ⇒ same DOF
numbering ⇒ entry-by-entry comparison with no gauge. See
`test/ParametricGeometry/test_moved_mesh_fom.jl`. Judge accuracy on **modal**
quantities: `‖ΔK‖_F/‖K‖_F` of 1e-05 has corresponded to a 267 % error in the
fundamental frequency, because the Frobenius norm is dominated by stiff
directions and the frequency by the softest mode.

**It connects to a physics module through [`AbstractPullbackKernel`](@ref)**, and
names no material, stress law or strain measure anywhere. A physics implements
the QP integrand; this module owns the coordinate transform, the geometry cache,
the assembly loops, the shared-input cache, the `MultilinearMap` wrapping and
the `build_model` contract. Adding a second physics is one new kernel type — see
`StructuralSVK.SVKPullbackKernel`.
"""
module ParametricGeometry

using ..Common: AbstractAssembledModel
import ..Common: build_model

# General multivariate θ-series algebra (det/adj/reciprocal over a MultiindexSet box).
include("geometry_parameter_series.jl")

# How 1/det J is obtained — Method 4 (power series) or Method 3 (auxiliary field).
# Included before the cache, which dispatches on it.
include("inverse_determinant.jl")

# The coordinate transform itself: geometry providers and the per-QP series cache.
include("pullback.jl")

# THE SEAM: what a physics must implement.
include("kernel.jl")

# Physics-blind assembly driver: cell loops, sweeps, MultilinearMap wrapping.
include("driver.jl")

# Linear-operator θ-corrections, generalised over modal arity.
include("corrections.jl")

# The assembled model and the build_model contract.
include("types.jl")

# Measured (never assumed) detection of θ-multiindices that contribute nothing.
include("diagnostics.jl")

include("build_model.jl")

export GeometryParameterBasis, nterms, jacobian_series, det_adj_series,
	determinant_series, adjugate_series, position_of, complex_tensor_type,
	reciprocal_series, inv_det_power, poly_mul, poly_dot, poly_contract,
	poly_mul!, poly_dot!, poly_contract!, convolve_weight_accumulate!,
	∇adj_series!, series_extent, Tens3, Tens3C, SymT3C,
	PullbackCache, inv_det_power_series, adj_tensor_type,
	AbstractInverseDeterminant, PowerSeriesInverseDet, AuxiliaryFieldInverseDet,
	build_inverse_determinant,
	AbstractPullbackKernel, QPContext, ∇adj_series,
	det_weight_power, qp_prepare, qp_integrand!, linear_qp_series!,
	ParametricDiscretisation, ParametricMap, multilinear_maps, sweep_all!, free_dof_map,
	build_linear_corrections, assemble_linear_series!,
	ParametricOperator, AssembledParametricModel, model_order, build_model,
	zero_coefficient_report, report_zero_coefficients,
	series_value, geometry_validity_at, geometry_validity_report,
	report_geometry_validity, inverse_determinant_comparison

end # module ParametricGeometry
