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
