# =====================================================================
# Vanishing-coefficient detection — physics-blind, assumption-free.
#
# A θ-multiindex can contribute nothing to the model for two very different
# reasons, and neither is knowable in advance:
#
#   · the GEOMETRY does not reach that degree. The sinusoidal arch has
#     ∇ψ² = 0 and det J ≡ 1, so most of its adj/det series is identically zero
#     above degree 1 — a θ-basis truncated higher is simply wasted work.
#   · the CONFIGURATION has a symmetry that annihilates it. A shape field that
#     is antisymmetric under a reflection the reference configuration respects
#     makes every odd power of that parameter integrate to zero.
#
# This module asserts NEITHER. It measures what was actually assembled and
# reports which multiindices came out negligible, leaving the interpretation to
# whoever knows the problem. That is deliberate: hardcoding a parity rule would
# be correct for one family of shape fields and silently wrong for the next.
#
# Thresholds are RELATIVE to the largest coefficient of the same quantity, so
# the report is scale-free and does not depend on the material's units.
# =====================================================================

using LinearAlgebra: norm
using SparseArrays: nnz
using Printf: @sprintf

# =====================================================================
# Method 4's VALIDITY — the condition the theory states and the code never checked.
#
# Expanding 1/det J as a power series about det J = 1 is a geometric series, so
# it converges only inside
#
#     |det J(θ, x₀) − 1| < 1        (equivalently 0 < det J < 2)
#
# at EVERY quadrature point, and the transform additionally requires det J > 0
# for invertibility (the determinant cannot change sign). This is Method 4's
# entire advantage over a Neumann series for J⁻¹: a scalar bound on local volume
# change, evaluable pointwise, instead of a spectral radius.
#
# Outside that radius NO truncation order converges — the series is divergent,
# not merely inaccurate, and the assembled model still looks perfectly ordinary.
# A uniform axial stretch ψ = x₁e₁ has det J = 1 + θ, so θ = 1 sits exactly ON
# the boundary: measured against a moved-mesh reference, its fundamental
# frequency is wrong by the same 6.1e+01 at every truncation from 2 to 12.
#
# Reported as a RADIUS rather than a pass/fail at a guessed θ, because θ's scale
# is set by however the caller normalised its shape field, and only the caller
# knows the range of interest. Silent when det J ≡ 1 (a volume-preserving
# transform such as the sinusoidal arch is unconditionally convergent) — the
# common good case should produce no output.
# =====================================================================

# Value of a scalar θ-series at a concrete θ.
function series_value(ser::AbstractVector{<:Real}, basis::GeometryParameterBasis, θ)
	v = 0.0
	@inbounds for (m, α) in enumerate(basis.mset.exponents)
		c = ser[m]
		iszero(c) && continue
		p = 1.0
		for i in eachindex(α)
			α[i] == 0 && continue
			p *= θ[i]^α[i]
		end
		v += c * p
	end
	return v
end

# Stride the (cell, qp) grid down to at most `max_points` probes. The radius scan
# evaluates this set many times, and a boundary located on a subsample is then
# re-verified against every point, so the stride costs speed and not correctness.
function _probe_points(cache::PullbackCache, max_points::Int)
	pts = Tuple{Int, Int}[]
	total = sum(length(cache.det[ci]) for ci in eachindex(cache.det); init = 0)
	total == 0 && return pts
	stride = max(1, cld(total, max_points))
	k = 0
	for ci in eachindex(cache.det), q in eachindex(cache.det[ci])
		k += 1
		k % stride == 0 && push!(pts, (ci, q))
	end
	isempty(pts) && push!(pts, (1, 1))
	return pts
end

"""
	geometry_validity_at(cache, θ; points = nothing) -> NamedTuple

Measure Method 4's validity conditions at one concrete θ.

Returns `(; det_min, det_max, max_deviation, reciprocal_residual, worst_cell,
worst_qp, convergent, orientation_ok)`:

- `max_deviation`       — `max |det J(θ) − 1|` over the probed points. Must be `< 1`.
- `det_min`             — must be `> 0`; a non-positive determinant means the
  coordinate transform has folded the mesh and is not invertible.
- `reciprocal_residual` — `max |det J(θ) · (1/det J)(θ) − 1|`, the TRUNCATED
  series measured against its own defining identity. This is the honest error of
  the expansion at this θ, computed from what the cache actually holds.

`points` restricts the sweep to a `(cell, qp)` subset; `nothing` sweeps all.
"""
function geometry_validity_at(cache::PullbackCache, θ; points = nothing)
	b = cache.basis
	det_min, det_max, max_dev, resid = Inf, -Inf, 0.0, 0.0
	worst_cell, worst_qp = 0, 0
	pts = points === nothing ?
		  ((ci, q) for ci in eachindex(cache.det) for q in eachindex(cache.det[ci])) :
		  points
	for (ci, q) in pts
		D = series_value(cache.det[ci][q], b, θ)
		R = series_value(cache.inv_det[ci][q], b, θ)
		det_min = min(det_min, D)
		det_max = max(det_max, D)
		resid = max(resid, abs(D * R - 1))
		dev = abs(D - 1)
		if dev > max_dev
			max_dev, worst_cell, worst_qp = dev, ci, q
		end
	end
	return (; det_min, det_max, max_deviation = max_dev, reciprocal_residual = resid,
		worst_cell, worst_qp, convergent = max_dev < 1, orientation_ok = det_min > 0)
end

_axis_θ(::GeometryParameterBasis{Nθ}, i, t) where {Nθ} =
	ntuple(k -> k == i ? t : 0.0, Nθ)

_ok_at(cache, θ, pts) =
	(r = geometry_validity_at(cache, θ; points = pts); r.convergent && r.orientation_ok)

# Largest |t| along ±eᵢ that keeps every probed point convergent and unfolded.
# `Inf` means "valid at least out to `cap`" — a volume-preserving transform never
# leaves the radius, and reporting a finite number for it would be misleading.
function _axis_radius(cache::PullbackCache, i::Int, sgn::Float64, pts;
	cap::Float64 = 4.0, rtol::Float64 = 1e-3)
	_ok_at(cache, _axis_θ(cache.basis, i, sgn * cap), pts) && return Inf
	lo, hi = 0.0, cap
	while hi - lo > rtol * max(hi, 1.0)
		mid = 0.5 * (lo + hi)
		_ok_at(cache, _axis_θ(cache.basis, i, sgn * mid), pts) ? (lo = mid) : (hi = mid)
	end
	return lo
end

"""
	geometry_validity_report(cache; max_probe_points = 5000, cap = 4.0) -> NamedTuple

The θ-range over which Method 4's expansion is valid, measured per parameter.

Returns `(; radius_pos, radius_neg, unconditional, det_at_reference)`, where
`radius_pos[i]` / `radius_neg[i]` is the largest `θᵢ` along `±eᵢ` (all other
parameters zero) for which every probed quadrature point satisfies both
`|det J − 1| < 1` and `det J > 0`. `Inf` means the scan reached `cap` without
leaving the radius.

`unconditional` is `true` when `det J ≡ 1` everywhere in θ — a volume-preserving
transform, for which the reciprocal series is the single term `1` and no
convergence question arises.

Axis-wise, because a full box scan costs `2^Nθ` corners and the axes already
expose the parameter responsible. For a specific θ of interest — including cross
terms — call [`geometry_validity_at`](@ref) directly.

Diagnostic only: nothing here changes the model.
"""
function geometry_validity_report(cache::PullbackCache{Nθ};
	max_probe_points::Int = 5000, cap::Real = 4.0) where {Nθ}
	pts = _probe_points(cache, max_probe_points)

	# det J ≡ 1 ⟺ every det coefficient beyond the constant vanishes and the
	# constant is 1. Then 1/det J = 1 exactly and the expansion is exact.
	unconditional = true
	for (ci, q) in pts
		d = cache.det[ci][q]
		if abs(d[1] - 1) > 1e-12 || any(abs(d[m]) > 1e-12 for m in 2:length(d))
			unconditional = false
			break
		end
	end

	ref = geometry_validity_at(cache, ntuple(_ -> 0.0, Nθ); points = pts)
	radius_pos = unconditional ? fill(Inf, Nθ) :
				 [_axis_radius(cache, i, +1.0, pts; cap = Float64(cap)) for i in 1:Nθ]
	radius_neg = unconditional ? fill(Inf, Nθ) :
				 [_axis_radius(cache, i, -1.0, pts; cap = Float64(cap)) for i in 1:Nθ]
	return (; radius_pos, radius_neg, unconditional, det_at_reference = ref)
end

"""
	report_geometry_validity(cache; kwargs...) -> NamedTuple

Emit the measured convergence radius of the inverse-determinant expansion, and
return the underlying [`geometry_validity_report`](@ref).

Silent for a volume-preserving transform (`det J ≡ 1`), which is unconditionally
convergent. Errors — not warns — when the REFERENCE configuration itself is
already folded (`det J ≤ 0` at `θ = 0`), because every series in the cache is
then built about an invalid point.
"""
function report_geometry_validity(cache::PullbackCache{Nθ}; kwargs...) where {Nθ}
	r = geometry_validity_report(cache; kwargs...)
	ref = r.det_at_reference
	ref.orientation_ok || error("ParametricGeometry: the REFERENCE configuration is " *
								"folded — min det J = $(ref.det_min) ≤ 0 at θ = 0 " *
								"(cell $(ref.worst_cell), qp $(ref.worst_qp)). The " *
								"coordinate transform is not invertible there and every " *
								"θ-series is expanded about an invalid point.")
	r.unconditional && return r      # det J ≡ 1: exact, nothing to say

	fmt(i) = "θ$i ∈ (" * (isinf(r.radius_neg[i]) ? "-∞" : @sprintf("%.3g", -r.radius_neg[i])) *
			 ", " * (isinf(r.radius_pos[i]) ? "+∞" : @sprintf("%.3g", r.radius_pos[i])) * ")"
	@info """
	ParametricGeometry: measured validity range of the inverse-determinant expansion \
	(|det J − 1| < 1 and det J > 0 at every probed quadrature point):
	  $(join([fmt(i) for i in 1:Nθ], ",  "))
	  OUTSIDE this range the geometric series for 1/det J DIVERGES — raising the θ-truncation \
	does not help, and the assembled model will still look well formed. Inside it, accuracy is \
	set by the truncation; check it with `geometry_validity_at(cache, θ).reciprocal_residual` \
	at the θ you care about."""
	return r
end

# =====================================================================
# METHOD 3 vs METHOD 4 — the measurement behind the choice.
#
# The theory's Table 1 prefers Method 4 (a θ-power series for 1/det J) over
# Method 3 (an auxiliary FE field s with s·det J = 1 enforced weakly). The two
# fail in DIFFERENT ways, and this is what distinguishes them:
#
#   Method 4  is pointwise exact within its truncation and has NO spatial
#             discretisation error, but its expansion is geometric about
#             det J = 1 and diverges outside |det J − 1| < 1.
#   Method 3  has no radius at all, but 1/det J is forced into V_h, so it carries
#             the interpolation error of that space forever — raising the
#             θ-truncation cannot remove it.
#
# Each method's `reciprocal_residual` (max |det J · s − 1| at the quadrature
# points) is its OWN error against the defining identity, so comparing the two
# residuals states the trade-off directly, without either method being treated as
# ground truth.
# =====================================================================

"""
	inverse_determinant_comparison(a::PullbackCache, b::PullbackCache, θ; points = nothing)

Compare two inverse-determinant strategies at one concrete θ.

`a` and `b` must be built on the same mesh and quadrature (same `(cell, qp)`
layout); their θ-bases may differ, since each series is evaluated in its own.
Typically `a` is a [`PowerSeriesInverseDet`](@ref) cache (Method 4) and `b` an
[`AuxiliaryFieldInverseDet`](@ref) one (Method 3).

Returns `(; max_abs, max_rel, rms, residual_a, residual_b, worst_cell, worst_qp)`:

- `max_abs` / `max_rel` / `rms` — how far apart the two `1/det J` fields are at
  the quadrature points, which is exactly where the difference enters the
  assembled operators.
- `residual_a` / `residual_b` — each method measured against `det J · s = 1`,
  its own defining identity. Neither method is used as the reference for the
  other; they are each compared with the truth they are both approximating.

Diagnostic only: nothing here changes a model.
"""
function inverse_determinant_comparison(a::PullbackCache, b::PullbackCache, θ;
	points = nothing)
	length(a.det) == length(b.det) || throw(ArgumentError(
		"inverse_determinant_comparison: caches have $(length(a.det)) and " *
		"$(length(b.det)) cells — they must share a mesh and quadrature rule"))
	ba, bb = a.basis, b.basis
	max_abs, max_rel, sq, n = 0.0, 0.0, 0.0, 0
	res_a, res_b = 0.0, 0.0
	worst_cell, worst_qp = 0, 0
	pts = points === nothing ?
		  ((ci, q) for ci in eachindex(a.det) for q in eachindex(a.det[ci])) : points
	for (ci, q) in pts
		Da = series_value(a.det[ci][q], ba, θ)
		Db = series_value(b.det[ci][q], bb, θ)
		Sa = series_value(a.inv_det[ci][q], ba, θ)
		Sb = series_value(b.inv_det[ci][q], bb, θ)
		res_a = max(res_a, abs(Da * Sa - 1))
		res_b = max(res_b, abs(Db * Sb - 1))
		d = abs(Sa - Sb)
		sq += d^2
		n += 1
		if d > max_abs
			max_abs, worst_cell, worst_qp = d, ci, q
		end
		max_rel = max(max_rel, d / max(abs(Sa), eps()))
	end
	return (; max_abs, max_rel, rms = n == 0 ? 0.0 : sqrt(sq / n),
		residual_a = res_a, residual_b = res_b, worst_cell, worst_qp)
end

"""
	zero_coefficient_report(m::AssembledParametricModel; rtol = 1e-12) -> NamedTuple

Which θ-multiindices contribute nothing to the assembled model.

Returns `(; geometry, operators, maps, exponents)`:

- `geometry[i]`  — `true` when multiindex `i` is negligible in **every** geometry
  series (`adj J`, `det J`) at **every** quadrature point. The coordinate
  transform simply does not reach that degree.
- `operators[i]` — `true` when multiindex `i` is negligible in every linear
  operator's θ^α coefficient matrix. The transform may reach that degree while
  the weak form still annihilates it.
- `maps[i]`      — `true` when multiindex `i` is negligible in every nonlinear
  form, **probed at a pseudo-random state**. A form that is identically zero
  gives zero for any input; a form that is not gives a nonzero result for a
  random input with probability 1. Set `probe = false` to skip (it costs one
  sweep per map).
- `exponents`    — the multiindices themselves, for reporting.

The three are reported separately because they call for different responses: a
geometry gap means the θ-basis can shrink at no cost, whereas an operator or form
that annihilates a degree the geometry reaches is a property of the weak form or
of a symmetry, and is worth understanding before it is relied on.

`rtol` is applied against the largest coefficient of the same quantity, so a
model in newtons and one in millinewtons give the same answer.

Diagnostic only: nothing here changes the model. Use it to see whether a θ-basis
is larger than the geometry justifies, or whether a symmetry you expected is
actually present in the discretisation.
"""
function zero_coefficient_report(m::AssembledParametricModel; rtol::Real = 1e-12,
	probe::Bool = true)
	b = basis(m.pd)
	L = nterms(b)
	cache = m.pd.cache

	# ── Geometry: max |coefficient| over every cell and quadrature point ──
	geo_scale = zeros(Float64, L)
	for ci in eachindex(cache.adj), q in eachindex(cache.adj[ci])
		adj_ser = cache.adj[ci][q]
		det_ser = cache.det[ci][q]
		@inbounds for i in 1:L
			geo_scale[i] = max(geo_scale[i], norm(adj_ser[i]), abs(det_ser[i]))
		end
	end
	geo_max = maximum(geo_scale; init = 0.0)
	geometry = [geo_max > 0 && geo_scale[i] <= rtol * geo_max for i in 1:L]

	# ── Linear operators: Frobenius norm of each θ^α coefficient matrix ──
	op_scale = zeros(Float64, L)
	for op in m.operators, i in 1:L
		A = op.arrays[i]
		A === nothing && continue
		op_scale[i] = max(op_scale[i], nnz(A) == 0 ? 0.0 : norm(A))
	end
	op_max = maximum(op_scale; init = 0.0)
	operators = [op_max > 0 && op_scale[i] <= rtol * op_max for i in 1:L]

	# ── Nonlinear forms: one sweep per map at a deterministic pseudo-random
	#    state. A form that is identically zero in θ^α returns zero whatever the
	#    input; one that is not returns nonzero for a random input almost surely.
	#    The state is generated from a fixed formula rather than `rand` so the
	#    report is reproducible run to run.
	maps = falses(L)
	if probe && !isempty(m.maps)
		n = m.pd.n_free
		map_scale = zeros(Float64, L)
		for pm in m.maps
			DEG = ndims(pm)
			mb = basis(pm.pd)
			nterms(mb) == L || continue      # a form on a different truncation
			us = ntuple(k -> ComplexF64[cis(0.31 * k * j) * (1 + 0.001j) for j in 1:n], DEG)
			A = sweep_all!(zeros(ComplexF64, n, L), pm, us)
			for i in 1:L
				map_scale[i] = max(map_scale[i], maximum(abs, view(A, :, i); init = 0.0))
			end
		end
		mx = maximum(map_scale; init = 0.0)
		maps = [mx > 0 && map_scale[i] <= rtol * mx for i in 1:L]
	end

	return (; geometry, operators, maps, exponents = b.mset.exponents)
end

"""
	report_zero_coefficients(m::AssembledParametricModel; rtol = 1e-12, io = stderr)

Emit one `@info` block naming the θ-multiindices that contribute nothing, and
return the underlying [`zero_coefficient_report`](@ref).

Silent when every multiindex carries something — the common case should not
produce output.
"""
function report_zero_coefficients(m::AssembledParametricModel; rtol::Real = 1e-12,
	probe::Bool = true)
	r = zero_coefficient_report(m; rtol = rtol, probe = probe)
	geo = findall(r.geometry)
	ops = findall(r.operators)
	mps = findall(r.maps)
	all_zero = union(geo, ops, mps)
	isempty(all_zero) && return r

	fmt(idx) = join([string(Tuple(r.exponents[i])) for i in idx], ", ")
	# Reported in three groups because they call for different responses. A
	# multiindex the geometry never reaches is a truncation the caller could
	# shrink for free; one the geometry reaches but the weak form annihilates is a
	# property of the physics — often a symmetry — and is worth understanding
	# rather than tidying away.
	only_ops = setdiff(ops, geo)
	only_maps = setdiff(mps, geo)
	lines = String[]
	isempty(geo) ||
		push!(lines, "  · absent from the GEOMETRY series (θ^α beyond the transform's " *
					 "degree — the θ-basis is larger than this geometry requires): $(fmt(geo))")
	isempty(only_ops) ||
		push!(lines, "  · reached by the geometry but annihilated by the LINEAR OPERATORS: " *
					 "$(fmt(only_ops))")
	isempty(only_maps) ||
		push!(lines, "  · reached by the geometry but annihilated by the NONLINEAR FORMS " *
					 "(probed at a pseudo-random state): $(fmt(only_maps))")

	@info """
	ParametricGeometry: $(length(all_zero)) of $(nterms(basis(m.pd))) θ-multiindices \
	contribute nothing to the assembled model (relative threshold $(rtol)).
	$(join(lines, "\n"))
	  Measured, not assumed — no symmetry is inferred and none is enforced. The first \
	group can be truncated away at no cost to the result; the others are telling you \
	something about the physics, and are worth explaining before they are relied on."""
	return r
end
