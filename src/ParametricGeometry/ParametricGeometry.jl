"""
`MORFEFerrite.ParametricGeometry` — parametric mesh coordinate transforms.

A **physics-blind** module: it implements the general parametric coordinate
transform `x(θ,x₀) = x₀ + Σᵢ θᵢ ψᵢ(x₀)` as a multivariate θ-power series — the
determinant, adjugate and inverse determinant of its Jacobian — and expands any
physics' multilinear maps over it.

```
J(θ,x₀) = J₀(x₀) + Σᵢ θᵢ ∇ψᵢ(x₀)      additive, so det/adj are exact polynomials
∇u  →  ∇u · adj J(θ)                  gradient pullback
dΩ  →  dΩ · (1/det J(θ))^p            weak-form weighting
```

The θ-series live over a MORFE `MultiindexSet` **box** (per-parameter
truncation); the single-parameter arch and the multi-parameter beam are
instances of the same engine.

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
	reciprocal_series, inv_det_power, poly_mul, poly_dot, poly_contract,
	PullbackCache, inv_det_power_series,
	AbstractPullbackKernel, QPContext, ∇adj_series,
	det_weight_power, qp_prepare, qp_integrand!, linear_qp_series!,
	ParametricDiscretisation, ParametricMap, multilinear_maps, sweep_all!,
	build_linear_corrections, assemble_linear_series!,
	ParametricOperator, AssembledParametricModel, model_order, build_model,
	zero_coefficient_report, report_zero_coefficients

end # module ParametricGeometry
