# =====================================================================
# StructuralSVK's implementation of the ParametricGeometry kernel interface.
#
# This is the PHYSICS half of the parametric seam: the Green-Lagrange strain
# measure, the constitutive law, and the weak form. `ParametricGeometry` owns
# the coordinate transform, the assembly loops and the MultilinearMap wrapping,
# and never names any of the below.
#
# SVK internal virtual work, pulled back through x(θ,x₀):
#     W_int = ∫ S : δE · (1/det J)^p dV₀,   ε_adj(u) = sym(∇₀u · adj J(θ))
# The gradients arrive already contracted with adj J, so these integrands are
# textually the reference-configuration ones — which is exactly why they agree
# with `accumulate_qp!` at J = I.
# =====================================================================

using Ferrite
using Tensors
using ..ParametricGeometry: AbstractPullbackKernel, QPContext, GeometryParameterBasis, nterms,
	poly_mul, poly_dot, poly_contract
import ..ParametricGeometry: det_weight_power, qp_prepare, qp_integrand!,
	linear_qp_series!

"""
	SVKPullbackKernel{DEG, S}(stress, ρ)

St. Venant-Kirchhoff physics over a parametric coordinate transform.

- `DEG = 2` — the quadratic elastic form `g(u₁,u₂;θ)`
- `DEG = 3` — the cubic form `h(u₁,u₂,u₃;θ)`
- `DEG = 0` — the linear operators (stiffness and mass)

`stress` is any [`AbstractStress`](@ref) — the same object the non-parametric
backend uses, so an anisotropic or cubic-crystal material works parametrically
with no further code. `ρ` is only read by the `DEG = 0` kernel.
"""
struct SVKPullbackKernel{DEG, S <: AbstractStress} <: AbstractPullbackKernel{DEG}
	stress::S
	ρ::Float64
end

SVKPullbackKernel{DEG}(stress::S, ρ::Real = 0.0) where {DEG, S <: AbstractStress} =
	SVKPullbackKernel{DEG, S}(stress, Float64(ρ))

SVKPullbackKernel{DEG}(material) where {DEG} =
	SVKPullbackKernel{DEG}(stress_model(material), Float64(material.ρ))

# The quadratic integrand carries (1/det J)², the cubic (1/det J)³ — one inverse
# determinant per displacement gradient in the form.
det_weight_power(::SVKPullbackKernel{2}) = 2
det_weight_power(::SVKPullbackKernel{3}) = 3

# --- strain measure (physics) ----------------------------------------
# Green-Lagrange cross term series sym(¼(∇uAᵀ∇uB + ∇uBᵀ∇uA)) — the series form
# of `_E_nl` in ferrite_assembly.jl.
function _E_nl_series(A_ser::Vector, B_ser::Vector, basis::GeometryParameterBasis)
	AB = poly_dot([transpose(g) for g in A_ser], B_ser, basis)
	BA = poly_dot([transpose(g) for g in B_ser], A_ser, basis)
	return [symmetric(0.25 * (AB[m] + BA[m])) for m in 1:nterms(basis)]
end

_σ_series(ser::Vector, s::AbstractStress) = [_σ(E, s) for E in ser]

# --- quadratic form --------------------------------------------------
function qp_prepare(k::SVKPullbackKernel{2}, ctx::QPContext,
	∇u_adj::NTuple{2, <:Vector})
	b = ctx.basis
	∇u1a, ∇u2a = ∇u_adj
	σ_u1 = _σ_series([symmetric(g) for g in ∇u1a], k.stress)
	σ_u2 = _σ_series([symmetric(g) for g in ∇u2a], k.stress)
	σE12 = _σ_series(_E_nl_series(∇u1a, ∇u2a, b), k.stress)
	return (; ∇u1a, ∇u2a, σ_u1, σ_u2, σE12)
end

function qp_integrand!(integ, k::SVKPullbackKernel{2}, ctx::QPContext, st,
	∇N_adj::Vector)
	b = ctx.basis
	ε_v = [symmetric(g) for g in ∇N_adj]
	t1 = poly_contract(ε_v, st.σE12, b)
	S2 = [symmetric(A) for A in poly_dot([transpose(g) for g in st.∇u1a], ∇N_adj, b)]
	t2 = poly_contract(S2, st.σ_u2, b)
	S3 = [symmetric(A) for A in poly_dot([transpose(g) for g in st.∇u2a], ∇N_adj, b)]
	t3 = poly_contract(S3, st.σ_u1, b)
	@inbounds for m in 1:nterms(b)
		integ[m] = t1[m] + 0.5 * (t2[m] + t3[m])
	end
	return integ
end

# --- cubic form ------------------------------------------------------
function qp_prepare(k::SVKPullbackKernel{3}, ctx::QPContext,
	∇u_adj::NTuple{3, <:Vector})
	b = ctx.basis
	∇u1a, ∇u2a, ∇u3a = ∇u_adj
	σE23 = _σ_series(_E_nl_series(∇u2a, ∇u3a, b), k.stress)
	σE13 = _σ_series(_E_nl_series(∇u1a, ∇u3a, b), k.stress)
	σE12 = _σ_series(_E_nl_series(∇u1a, ∇u2a, b), k.stress)
	return (; ∇u1a, ∇u2a, ∇u3a, σE23, σE13, σE12)
end

function qp_integrand!(integ, k::SVKPullbackKernel{3}, ctx::QPContext, st,
	∇N_adj::Vector)
	b = ctx.basis
	S1 = [symmetric(A) for A in poly_dot([transpose(g) for g in st.∇u1a], ∇N_adj, b)]
	S2 = [symmetric(A) for A in poly_dot([transpose(g) for g in st.∇u2a], ∇N_adj, b)]
	S3 = [symmetric(A) for A in poly_dot([transpose(g) for g in st.∇u3a], ∇N_adj, b)]
	t1 = poly_contract(S1, st.σE23, b)
	t2 = poly_contract(S2, st.σE13, b)
	t3 = poly_contract(S3, st.σE12, b)
	@inbounds for m in 1:nterms(b)
		integ[m] = (t1[m] + t2[m] + t3[m]) / 3
	end
	return integ
end

# --- linear operators ------------------------------------------------
# K_α = θ^α coeff of ∫ ε_adj(v) ⊡ σ(ε_adj(u)) · (1/det J) dV₀
# M_α = θ^α coeff of ∫ ρ (u·v) · det J dV₀
function linear_qp_series!(k_ser, m_ser, k::SVKPullbackKernel{0}, ctx::QPContext,
	i::Int, j::Int)
	b = ctx.basis
	cv, q = ctx.cv, ctx.q
	∇Ni = shape_gradient(cv, q, i)
	∇Nj = shape_gradient(cv, q, j)
	ε_adj_i = [symmetric(∇Ni ⋅ a) for a in ctx.adj]
	ε_adj_j = [symmetric(∇Nj ⋅ a) for a in ctx.adj]
	σ_adj_j = _σ_series(ε_adj_j, k.stress)
	K_ser = poly_mul(poly_contract(ε_adj_i, σ_adj_j, b), ctx.inv_det, b)
	NiNj = shape_value(cv, q, i) ⋅ shape_value(cv, q, j)
	@inbounds for m in 1:nterms(b)
		k_ser[m] = K_ser[m]
		m_ser[m] = k.ρ * NiNj * ctx.det[m]
	end
	return nothing
end
