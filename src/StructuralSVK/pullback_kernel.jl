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
	poly_mul, poly_dot, poly_contract, poly_mul!, poly_dot!, poly_contract!,
	Tens3, Tens3C, SymT3C
import ..ParametricGeometry: det_weight_power, qp_prepare, qp_integrand!,
	linear_qp_series!

# =====================================================================
# Reusable per-quadrature-point storage.
#
# `qp_integrand!` runs once per BASIS FUNCTION per quadrature point per cell —
# the innermost loop of the whole reduction — and every intermediate it needs is
# a length-L series. Allocating them there cost example 04 7.8 G allocations
# (4.5 TiB) for one FAST run. The kernel is constructed once per `ParametricMap`
# and used by one sweep at a time, so it can simply own the buffers.
#
# Sized lazily: the θ-basis is not known when the kernel is built.
# =====================================================================
const _SymT3 = SymmetricTensor{2, 3, Float64, 6}

struct SVKScratch
	∇ut::NTuple{3, Vector{Tens3C}}       # transposed pulled-back gradients
	εu::NTuple{3, Vector{SymT3C}}        # sym(∇uᵢ·adj)
	σu::NTuple{3, Vector{SymT3C}}        # σ of the above
	σE::NTuple{3, Vector{SymT3C}}        # σ(E_nl) cross terms
	S::NTuple{3, Vector{SymT3C}}         # sym(∇uᵢᵀ·∇N) per basis function
	t::NTuple{3, Vector{ComplexF64}}     # scalar contraction accumulators
	tmp::NTuple{2, Vector{Tens3C}}       # E_nl working space
	# The DEG = 0 (linear-operator) path is real, and runs nbf² times per
	# quadrature point — 7.1 M calls for example 04 — so it needs its own
	# Float64 buffers rather than the complex ones above.
	lin_ε::NTuple{2, Vector{_SymT3}}
	lin_σ::Vector{_SymT3}
	lin_c::NTuple{2, Vector{Float64}}
end

SVKScratch() = SVKScratch(
	ntuple(_ -> Tens3C[], 3), ntuple(_ -> SymT3C[], 3), ntuple(_ -> SymT3C[], 3),
	ntuple(_ -> SymT3C[], 3), ntuple(_ -> SymT3C[], 3), ntuple(_ -> ComplexF64[], 3),
	ntuple(_ -> Tens3C[], 2),
	ntuple(_ -> _SymT3[], 2), _SymT3[], ntuple(_ -> Float64[], 2))

function _fit!(s::SVKScratch, L::Int)
	length(s.tmp[1]) == L && return s
	for g in (s.∇ut, s.εu, s.σu, s.σE, s.S, s.t, s.tmp), v in g
		resize!(v, L)
	end
	return s
end

function _fit_lin!(s::SVKScratch, L::Int)
	length(s.lin_σ) == L && return s
	for g in (s.lin_ε, s.lin_c), v in g
		resize!(v, L)
	end
	resize!(s.lin_σ, L)
	return s
end

# σ(E) applied elementwise to a strain series, in place.
@inline function _σ_series!(out, ser, s::AbstractStress)
	@inbounds for m in eachindex(ser)
		out[m] = _σ(ser[m], s)
	end
	return out
end

# Green-Lagrange cross term sym(¼(∇uAᵀ∇uB + ∇uBᵀ∇uA)) as a series, in place.
# `At`/`Bt` are the ALREADY TRANSPOSED series — transposing here would put the
# work back inside the caller's loop, which is exactly what this avoids.
@inline function _E_nl_series!(out, At, A, Bt, B, basis, tmp1, tmp2)
	poly_dot!(tmp1, At, B, basis)
	poly_dot!(tmp2, Bt, A, basis)
	@inbounds for m in eachindex(out)
		out[m] = symmetric(0.25 * (tmp1[m] + tmp2[m]))
	end
	return out
end

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
	scratch::SVKScratch
end

SVKPullbackKernel{DEG}(stress::S, ρ::Real = 0.0) where {DEG, S <: AbstractStress} =
	SVKPullbackKernel{DEG, S}(stress, Float64(ρ), SVKScratch())

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
# Everything independent of the test function is built ONCE here, including the
# transposed gradient series: `qp_integrand!` used to rebuild those for every
# basis function, i.e. `nbf` times per quadrature point (81 for a hex27).
function qp_prepare(k::SVKPullbackKernel{2}, ctx::QPContext,
	∇u_adj::NTuple{2, <:Vector})
	b = ctx.basis
	L = nterms(b)
	s = _fit!(k.scratch, L)
	∇u1a, ∇u2a = ∇u_adj
	@inbounds for m in 1:L
		s.∇ut[1][m] = transpose(∇u1a[m])
		s.∇ut[2][m] = transpose(∇u2a[m])
		s.εu[1][m] = symmetric(∇u1a[m])
		s.εu[2][m] = symmetric(∇u2a[m])
	end
	_σ_series!(s.σu[1], s.εu[1], k.stress)
	_σ_series!(s.σu[2], s.εu[2], k.stress)
	_E_nl_series!(s.εu[3], s.∇ut[1], ∇u1a, s.∇ut[2], ∇u2a, b, s.tmp[1], s.tmp[2])
	_σ_series!(s.σE[1], s.εu[3], k.stress)
	return (; ∇u1a, ∇u2a, s)
end

function qp_integrand!(integ, k::SVKPullbackKernel{2}, ctx::QPContext, st,
	∇N_adj::Vector)
	b = ctx.basis
	s = st.s
	@inbounds for m in eachindex(∇N_adj)
		s.S[1][m] = symmetric(∇N_adj[m])                 # ε(v)
	end
	poly_contract!(s.t[1], s.S[1], s.σE[1], b)
	poly_dot!(s.tmp[1], s.∇ut[1], ∇N_adj, b)
	@inbounds for m in eachindex(s.tmp[1])
		s.S[2][m] = symmetric(s.tmp[1][m])
	end
	poly_contract!(s.t[2], s.S[2], s.σu[2], b)
	poly_dot!(s.tmp[2], s.∇ut[2], ∇N_adj, b)
	@inbounds for m in eachindex(s.tmp[2])
		s.S[3][m] = symmetric(s.tmp[2][m])
	end
	poly_contract!(s.t[3], s.S[3], s.σu[1], b)
	@inbounds for m in 1:nterms(b)
		integ[m] = s.t[1][m] + 0.5 * (s.t[2][m] + s.t[3][m])
	end
	return integ
end

# --- cubic form ------------------------------------------------------
function qp_prepare(k::SVKPullbackKernel{3}, ctx::QPContext,
	∇u_adj::NTuple{3, <:Vector})
	b = ctx.basis
	L = nterms(b)
	s = _fit!(k.scratch, L)
	∇u1a, ∇u2a, ∇u3a = ∇u_adj
	@inbounds for m in 1:L
		s.∇ut[1][m] = transpose(∇u1a[m])
		s.∇ut[2][m] = transpose(∇u2a[m])
		s.∇ut[3][m] = transpose(∇u3a[m])
	end
	_E_nl_series!(s.εu[1], s.∇ut[2], ∇u2a, s.∇ut[3], ∇u3a, b, s.tmp[1], s.tmp[2])
	_σ_series!(s.σE[1], s.εu[1], k.stress)                   # σE23
	_E_nl_series!(s.εu[2], s.∇ut[1], ∇u1a, s.∇ut[3], ∇u3a, b, s.tmp[1], s.tmp[2])
	_σ_series!(s.σE[2], s.εu[2], k.stress)                   # σE13
	_E_nl_series!(s.εu[3], s.∇ut[1], ∇u1a, s.∇ut[2], ∇u2a, b, s.tmp[1], s.tmp[2])
	_σ_series!(s.σE[3], s.εu[3], k.stress)                   # σE12
	return (; ∇u1a, ∇u2a, ∇u3a, s)
end

function qp_integrand!(integ, k::SVKPullbackKernel{3}, ctx::QPContext, st,
	∇N_adj::Vector)
	b = ctx.basis
	s = st.s
	@inbounds for r in 1:3
		poly_dot!(s.tmp[1], s.∇ut[r], ∇N_adj, b)
		for m in eachindex(s.tmp[1])
			s.S[r][m] = symmetric(s.tmp[1][m])
		end
		poly_contract!(s.t[r], s.S[r], s.σE[r], b)
	end
	@inbounds for m in 1:nterms(b)
		integ[m] = (s.t[1][m] + s.t[2][m] + s.t[3][m]) / 3
	end
	return integ
end

# --- linear operators ------------------------------------------------
# K_α = θ^α coeff of ∫ ε_adj(v) ⊡ σ(ε_adj(u)) · (1/det J) dV₀
# M_α = θ^α coeff of ∫ ρ (u·v) · det J dV₀
function linear_qp_series!(k_ser, m_ser, k::SVKPullbackKernel{0}, ctx::QPContext,
	i::Int, j::Int)
	b = ctx.basis
	L = nterms(b)
	s = _fit_lin!(k.scratch, L)
	cv, q = ctx.cv, ctx.q
	∇Ni = shape_gradient(cv, q, i)
	∇Nj = shape_gradient(cv, q, j)
	@inbounds for m in 1:L
		a = ctx.adj[m]
		s.lin_ε[1][m] = symmetric(∇Ni ⋅ a)
		s.lin_ε[2][m] = symmetric(∇Nj ⋅ a)
	end
	_σ_series!(s.lin_σ, s.lin_ε[2], k.stress)
	poly_contract!(s.lin_c[1], s.lin_ε[1], s.lin_σ, b)
	poly_mul!(s.lin_c[2], s.lin_c[1], ctx.inv_det, b)
	NiNj = shape_value(cv, q, i) ⋅ shape_value(cv, q, j)
	@inbounds for m in 1:L
		k_ser[m] = s.lin_c[2][m]
		m_ser[m] = k.ρ * NiNj * ctx.det[m]
	end
	return nothing
end
