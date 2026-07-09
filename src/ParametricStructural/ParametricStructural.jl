"""
`MORFEFerrite.ParametricStructural` — geometric-parameter (θ) structural ROMs.

General multivariate parametric formulation: the reference map is
`x(θ,x₀) = x₀ + Σᵢ θᵢ ψᵢ(x₀)` (each `θᵢ` scaling an independent shape field
`ψᵢ`), so the Jacobian series is additive, `J(θ,x₀) = J₀(x₀) + Σᵢ θᵢ ∇ψᵢ(x₀)`.
The θ-series (`det J`, `adj J`, `1/det J`, weak-form brackets) is a multivariate
polynomial over a MORFE `MultiindexSet` box (per-parameter truncation). The
single-parameter arch and multi-parameter beam are instances of the same engine.

Builds on `StructuralSVK` for the St. Venant-Kirchhoff material.
"""
module ParametricStructural

# General multivariate θ-series algebra (det/adj/reciprocal over a MultiindexSet box).
include("theta_series.jl")

# General parametric geometric nonlinearity (quadratic + cubic) + MultilinearMap wrappers.
include("nonlinearity.jl")

export ThetaBasis, nterms, jacobian_series, det_adj_series,
	reciprocal_series, inv_det_power, poly_mul, poly_dot, poly_contract,
	ParametricGeometricNonlinearity, multilinear_maps,
	evaluate_theta_quadratic!, evaluate_theta_cubic!

end # module ParametricStructural
