# =====================================================================
# THE SEAM.
#
# `ParametricGeometry` owns the coordinate transform and the assembly driver;
# a physics module owns the integrand. This file states the contract between
# them and nothing else — no material, no stress law, no strain measure
# appears anywhere in this module.
# =====================================================================

"""
	AbstractPullbackKernel{DEG}

A physics' integrand, expressed over a parametric coordinate transform.

`DEG` is the arity in the field variable: `2` for a quadratic form `g(u₁,u₂)`,
`3` for a cubic `h(u₁,u₂,u₃)`, `0` for a purely geometric (linear-operator) kernel.

A physics module implements a subtype and the methods below; `ParametricGeometry`
then owns everything around them — the cell loop, the gather/scatter, the
geometry cache, the all-coefficients sweep, the shared-input cache, and the
wrapping of each θ-coefficient as a MORFE `MultilinearMap`.

## Required for a nonlinear kernel (`DEG ≥ 2`)

	det_weight_power(k) -> Int

The `p` in the `(1/det J)^p` weighting of this form's integrand. The driver
multiplies by it; the kernel must not.

	qp_prepare(k, ctx, ∇u_adj::NTuple{DEG, Vector{Tens3}}) -> state

Whatever the integrand needs once per quadrature point, independent of the test
function — stresses, strain cross-terms. Returning it separately is what keeps
that work out of the inner basis-function loop.

	qp_integrand!(integ, k, ctx, state, ∇N_adj::Vector{Tens3}) -> integ

The θ-series of the integrand for one test function, written into `integ`
(length `nterms(ctx.basis)`), **before** the determinant weighting.

## Required for a linear-operator kernel

	linear_qp_series!(a_ser, b_ser, k, ctx, i, j) -> nothing

The θ-series of the two operator entries (for a structure: stiffness and mass)
coupling shape functions `i` and `j` at this quadrature point, already weighted.

## Note on the gradients

`∇u_adj` and `∇N_adj` arrive **already contracted with the adjugate series** —
that is, `∇u·adj J(θ)` rather than `∇u`. Contracting is the coordinate
transform's job, so the kernel receives the pulled-back gradient and writes its
weak form exactly as it would in the reference configuration.
"""
abstract type AbstractPullbackKernel{DEG} end

"""
	QPContext(cv, q, basis, adj, det, inv_det)

What a kernel sees at one quadrature point: the `CellValues` (already
`reinit!`-ed) and the point index, so it can reach shape values; the
[`GeometryParameterBasis`](@ref) its series live in; and the geometry's `adj J`, `det J` and
`1/det J` series at this point.

A nonlinear kernel normally leaves the determinant weighting to the driver (see
[`det_weight_power`](@ref)) and never touches `det`/`inv_det`; a linear-operator
kernel weights its own entries, since stiffness and mass carry different powers.
"""
struct QPContext{Nθ, CV}
	cv::CV
	q::Int
	basis::GeometryParameterBasis{Nθ}
	adj::Vector{Tens3}
	det::Vector{Float64}
	inv_det::Vector{Float64}
end

nterms(ctx::QPContext) = nterms(ctx.basis)

# --- the pullback of a gradient --------------------------------------
# ∇u (θ-independent) contracted with the adj-series → a θ-series. This is the
# coordinate transform applied to a gradient, and it is the ONLY place the
# transform touches a field quantity.
∇adj_series(∇u, adj_ser::Vector) = [∇u ⋅ a for a in adj_ser]

# --- interface fallbacks ---------------------------------------------
det_weight_power(k::AbstractPullbackKernel) = throw(MethodError(det_weight_power, (k,)))

function qp_prepare(k::AbstractPullbackKernel, ctx, ∇u_adj)
	throw(MethodError(qp_prepare, (k, ctx, ∇u_adj)))
end

function qp_integrand!(integ, k::AbstractPullbackKernel, ctx, state, ∇N_adj)
	throw(MethodError(qp_integrand!, (integ, k, ctx, state, ∇N_adj)))
end

function linear_qp_series!(a_ser, b_ser, k::AbstractPullbackKernel, ctx, i, j)
	throw(MethodError(linear_qp_series!, (a_ser, b_ser, k, ctx, i, j)))
end
