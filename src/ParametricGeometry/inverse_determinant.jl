# =====================================================================
# HOW 1/det J IS OBTAINED — the one place Methods 3 and 4 differ.
#
# The theory (Table 1) lists four treatments of J⁻¹. Two of them survive the
# adjugate identity J⁻¹ = adj J / det J, which makes adj J an exact polynomial
# and leaves a single scalar, det J⁻¹, as the only non-polynomial quantity:
#
#   METHOD 4  `PowerSeriesInverseDet`     — expand 1/det J as a power series in θ.
#             Pointwise, exact within its truncation, needs no extra unknowns.
#             Converges only where |det J − 1| < 1 (App. A.3); see
#             `geometry_validity_report`.
#
#   METHOD 3  `AuxiliaryFieldInverseDet`  — introduce a scalar FE field s ≈ 1/det J
#             and enforce s·det J = 1 weakly. Accuracy is set by the
#             discretisation of s rather than by a truncation order.
#
# Everything downstream — adj J, det J, the (1/det J)^p powers, every physics
# kernel, the driver, the corrections, `build_model` — is identical either way.
# Both strategies are asked for the same thing and return the same object, so
# switching methods is a keyword, not a code path.
#
# ── Why Method 3 is solved OFFLINE here ──────────────────────────────────────
#
# The constraint s·det J = 1 contains no displacement: det J is a known function
# of (x₀, θ). The s-block therefore decouples COMPLETELY from the physics, and a
# monolithic solve carrying s as a full-order unknown would reproduce exactly the
# recurrence below in its s-rows — the z̃-dependent coefficients of s are
# provably zero. Solving it here is thus not an approximation of Method 3; it is
# Method 3, minus a descriptor eigenproblem, minus one DOF per node in the FOM,
# and minus the higher-arity forms that carrying s through the physics would
# force (a cubic form would become a 6-linear map in (u,u,u,s,s,s)).
#
# The monolithic variant remains a clean extension point: implement
# `build_inverse_determinant` for a new strategy type, and nothing else moves.
# =====================================================================

using Ferrite
using LinearAlgebra: cholesky, Diagonal, Symmetric
using SparseArrays: SparseMatrixCSC

"""
	AbstractInverseDeterminant

How a [`PullbackCache`](@ref) obtains the θ-series of `1/det J` at each
quadrature point. See [`PowerSeriesInverseDet`](@ref) (Method 4, the default)
and [`AuxiliaryFieldInverseDet`](@ref) (Method 3).
"""
abstract type AbstractInverseDeterminant end

"""
	PowerSeriesInverseDet()

**Method 4** — expand `1/det J` directly as a θ-power series (theory App. A.3).

Pointwise and exact within the θ-box: no auxiliary field, no extra unknowns, no
spatial discretisation error. The expansion is geometric about `det J = 1`, so it
converges only where `|det J − 1| < 1`; outside that radius NO truncation order
converges. [`report_geometry_validity`](@ref) measures the radius and
`build_model` prints it.
"""
struct PowerSeriesInverseDet <: AbstractInverseDeterminant end

"""
	AuxiliaryFieldInverseDet(ip; qr, lump = false, tol = 1e-10)

**Method 3** — represent `1/det J` by a scalar FE field `s` on the interpolation
`ip`, with `s · det J = 1` enforced weakly.

Expanding `s = Σ_κ s_κ(x₀) θ^κ` with `s_κ ∈ V_h` and collecting `θ^γ` gives a
triangular recurrence in graded-lex order:

	γ = 0 :  A₀ s₀ = b                          b_i  = ∫ Nᵢ dΩ₀
	γ > 0 :  A₀ s_γ = − Σ_{0<β≤γ} A_β s_{γ−β}   A_β  = ∫ Nᵢ c_β Nⱼ dΩ₀

where `c_β` are the coefficients of the (exactly polynomial) `det J` series.
`A₀` is the mass matrix weighted by `c₀ = det J₀`, which is the plain mass matrix
only when the reference configuration is undeformed (`J₀ = I`); a curved
reference has `c₀ ≠ 1` and the weight matters. `A₀` is SPD exactly when
`det J₀ > 0`, i.e. under the orientation condition the transform already needs.

One factorisation of `A₀` serves every multiindex, so the whole expansion costs
one assembly sweep, one Cholesky and `L−1` backsolves — negligible next to a
single order of the reduction.

`ip` is **the** accuracy knob: unlike Method 4, whose error is a truncation
order, Method 3's error is the FE interpolation error of `1/det J`. Raising the
order of `ip` drives it towards Method 4's pointwise-exact answer.

`qr` must be the quadrature rule the physics `CellValues` uses, so `c_β` is read
at the points the cache already holds. `lump = true` replaces `A₀` by its
row-sum diagonal. `tol` is the tolerance of the `s₀ ≡ 1` consistency check,
which is exact to round-off whenever `J₀ = I`.

Note `(1/det J)^p` is formed as the `p`-th power of the interpolant `s`, not as a
separately projected field — a different (and untaken) Method-3 variant.
"""
struct AuxiliaryFieldInverseDet{IP, QR} <: AbstractInverseDeterminant
	ip::IP
	qr::QR
	lump::Bool
	tol::Float64
end

AuxiliaryFieldInverseDet(ip; qr, lump::Bool = false, tol::Real = 1e-10) =
	AuxiliaryFieldInverseDet{typeof(ip), typeof(qr)}(ip, qr, lump, Float64(tol))

"""
	build_inverse_determinant(strategy, dh, cv, basis, det) -> Vector{Vector{Vector{Float64}}}

The θ-series of `1/det J`, indexed `[cell][qp]`, from the already-computed `det J`
series in the same layout. The single seam between Methods 3 and 4.
"""
function build_inverse_determinant end

# ── Method 4 ─────────────────────────────────────────────────────────────────
build_inverse_determinant(::PowerSeriesInverseDet, dh, cv,
	basis::GeometryParameterBasis, det) =
	[[reciprocal_series(det[ci][q], basis) for q in eachindex(det[ci])]
	 for ci in eachindex(det)]

# ── Method 3 ─────────────────────────────────────────────────────────────────

# Which θ-multiindices carry any det coefficient at all. For an affine map in d
# dimensions only |β| ≤ d·deg J can be nonzero, so this is a handful of matrices
# rather than one per basis term — measured, not assumed, so a degenerate
# geometry costs nothing extra.
function _det_support(det, L::Int; tol::Float64 = 0.0)
	keep = falses(L)
	for ci in eachindex(det), q in eachindex(det[ci])
		d = det[ci][q]
		@inbounds for m in 1:L
			abs(d[m]) > tol && (keep[m] = true)
		end
	end
	keep[1] = true                      # c₀ always participates: it IS the operator
	return findall(keep)
end

function build_inverse_determinant(s::AuxiliaryFieldInverseDet, dh, cv,
	basis::GeometryParameterBasis, det)
	grid = Ferrite.get_grid(dh)
	dh_s = DofHandler(grid)
	add!(dh_s, :s, s.ip)
	close!(dh_s)
	cv_s = CellValues(s.qr, s.ip)
	getnquadpoints(cv_s) == getnquadpoints(cv) || throw(ArgumentError(
		"AuxiliaryFieldInverseDet: the auxiliary quadrature has $(getnquadpoints(cv_s)) " *
		"points but the physics CellValues has $(getnquadpoints(cv)) — pass the same " *
		"`qr` the physics uses, so det J is read where it was computed"))

	L = nterms(basis)
	support = _det_support(det, L)
	pos = Dict(m => k for (k, m) in enumerate(support))   # basis index → matrix slot

	# ── one sweep: every A_β and the right-hand side b, together ──
	A = [allocate_matrix(dh_s) for _ in support]
	asm = [start_assemble(a) for a in A]
	b = zeros(Float64, ndofs(dh_s))
	nbf = getnbasefunctions(cv_s)
	Ae = [zeros(nbf, nbf) for _ in support]
	be = zeros(nbf)

	for (ci, cell) in enumerate(CellIterator(dh_s))
		for k in eachindex(Ae)
			fill!(Ae[k], 0.0)
		end
		fill!(be, 0.0)
		reinit!(cv_s, cell)
		for q in 1:getnquadpoints(cv_s)
			dΩ₀ = getdetJdV(cv_s, q)
			d = det[ci][q]
			for i in 1:nbf
				Ni = shape_value(cv_s, q, i)
				be[i] += Ni * dΩ₀
				for j in 1:nbf
					NiNj = Ni * shape_value(cv_s, q, j) * dΩ₀
					@inbounds for (k, m) in enumerate(support)
						Ae[k][i, j] += NiNj * d[m]
					end
				end
			end
		end
		dofs = celldofs(cell)
		for k in eachindex(Ae)
			assemble!(asm[k], dofs, Ae[k])
		end
		for (i, dof) in pairs(dofs)
			b[dof] += be[i]
		end
	end

	# ── triangular solve in graded-lex order, ONE factorisation ──
	A₀ = A[pos[1]]
	F = s.lump ? Diagonal(vec(sum(A₀; dims = 2))) : cholesky(Symmetric(A₀))
	sk = Vector{Vector{Float64}}(undef, L)
	sk[1] = F \ b

	# `1 ∈ V_h` for any Lagrange space, so when the reference configuration is
	# undeformed (c₀ ≡ 1) the γ = 0 equation is the projection of the constant 1
	# and must return it to round-off. A sharp, free check on the whole assembly.
	if all(abs(det[ci][q][1] - 1) <= 1e-12
		   for ci in eachindex(det) for q in eachindex(det[ci]))
		dev = maximum(abs, sk[1] .- 1)
		dev <= s.tol || @warn "AuxiliaryFieldInverseDet: s₀ deviates from 1 by $dev " *
							  "although det J₀ ≡ 1 — the auxiliary mass matrix or its " *
							  "quadrature is inconsistent with the physics CellValues."
	end

	rhs = zeros(Float64, ndofs(dh_s))
	D = basis.diff                                   # position of γ − β, 0 if outside
	@inbounds for k in 2:L
		fill!(rhs, 0.0)
		for m in support
			m == 1 && continue                       # β = 0 is the operator on the left
			j = D[k, m]
			j == 0 && continue
			rhs .-= A[pos[m]] * sk[j]                # graded-lex ⟹ sk[j] is ready
		end
		sk[k] = F \ rhs
	end

	# ── evaluate the coefficient fields at the quadrature points ──
	inv = [Vector{Vector{Float64}}(undef, length(det[ci])) for ci in eachindex(det)]
	se = zeros(Float64, getnbasefunctions(cv_s))
	for (ci, cell) in enumerate(CellIterator(dh_s))
		reinit!(cv_s, cell)
		dofs = celldofs(cell)
		for q in 1:getnquadpoints(cv_s)
			inv[ci][q] = Vector{Float64}(undef, L)
		end
		for k in 1:L
			for (i, dof) in pairs(dofs)
				se[i] = sk[k][dof]
			end
			for q in 1:getnquadpoints(cv_s)
				inv[ci][q][k] = function_value(cv_s, q, se)
			end
		end
	end
	return inv
end
