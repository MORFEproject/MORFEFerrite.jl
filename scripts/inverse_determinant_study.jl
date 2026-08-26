# Method 3 vs Method 4 for 1/det J — the measurements behind the choice.
#
# The theory (Table 1) adopts Method 4, a θ-power series for the reciprocal of
# the determinant, over Method 3, an auxiliary FE field s with s·det J = 1
# enforced weakly. This script produces the numbers that support that, rather
# than asserting it in prose. It is mesh-light and runs in seconds:
#
#   julia --project=. scripts/inverse_determinant_study.jl
#
# Three tables:
#
#   TABLE 1  Method 3 → Method 4 as V_h is enriched. Method 4 has no spatial
#            discretisation error, so it is the limit Method 3 approaches; the
#            gap that remains at a given V_h order is Method 3's permanent cost,
#            which no θ-truncation can remove.
#   TABLE 2  What each method costs to build.
#   TABLE 3  Method 4's radius. This is the other side of the trade: outside
#            |det J − 1| < 1 the expansion DIVERGES, and raising the truncation
#            order does not help. Method 3 has no such limit.
#
# The geometry matters. A uniform stretch has a SPATIALLY CONSTANT det J, and
# every Lagrange space contains the constants exactly — Method 3 is then exact at
# order 1 and the comparison shows nothing. The wavy field below makes det J vary
# in space, which is the regime the theory means by "accuracy depends on the
# spatial discretisation of s".

using MORFEFerrite
using Ferrite, Tensors, Printf
using LinearAlgebra: norm

const PG = MORFEFerrite.ParametricGeometry
const T3 = Tensor{2, 3, Float64, 9}

function setup(; nel = (2, 2, 1), order = 2, qorder = 3)
	grid = generate_grid(Hexahedron, nel, Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
	ip = Lagrange{RefHexahedron, order}()^3
	qr = QuadratureRule{RefHexahedron}(qorder)
	cv = CellValues(qr, ip, Lagrange{RefHexahedron, 1}())
	dh = DofHandler(grid)
	add!(dh, :u, ip)
	close!(dh)
	return (; dh, cv, qr)
end

# det J = 1 + θ sin(x₁): non-polynomial in x₀, so V_h has to resolve it.
wavy(x) = (one(T3), T3((i, j) -> (i == 1 && j == 1) ? sin(x[1]) : 0.0))
# det J = 1 + θ: spatially constant — Method 3 is exact here at any order.
stretch(x) = (one(T3), T3((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0))

const S = setup()
# A larger mesh for the cost table only: on the accuracy mesh both methods build
# in under a millisecond and the timings round to zero, which says nothing.
const S_COST = setup(; nel = (8, 8, 4))
const BASIS = PG.GeometryParameterBasis([8])
const Θ = (0.2,)

m4(geom, basis = BASIS) = PG.PullbackCache(S.dh, S.cv, geom, basis)
m3(geom, vh; basis = BASIS) = PG.PullbackCache(S.dh, S.cv, geom, basis;
	inverse_determinant = PG.AuxiliaryFieldInverseDet(
		Lagrange{RefHexahedron, vh}(); qr = S.qr))

# ── TABLE 1 ──────────────────────────────────────────────────────────────────
println("\nTABLE 1 — Method 3 → Method 4 as V_h is enriched   (θ = $(Θ[1]))")
println("The residual columns are each method against det J · s = 1, its OWN")
println("defining identity — neither method is treated as ground truth.\n")
println("  geometry   V_h    max|s₃−s₄|      rms       resid M4     resid M3")
println("  " * "-"^68)
for (name, geom) in (("wavy", wavy), ("stretch", stretch))
	ref = m4(geom)
	for vh in (1, 2)
		c = PG.inverse_determinant_comparison(ref, m3(geom, vh), Θ)
		@printf("  %-9s  %-4d  %.3e   %.3e   %.3e   %.3e\n",
			name, vh, c.max_abs, c.rms, c.residual_a, c.residual_b)
	end
end
println("""
  Read: for `wavy` the order-1 gap is Method 3's interpolation error and it does
  not shrink with the θ-truncation — only with V_h. For `stretch`, det J is
  spatially constant, so Method 3 is exact at order 1 and the columns collapse:
  a comparison run on that geometry would wrongly suggest the methods agree.""")

# ── TABLE 2 ──────────────────────────────────────────────────────────────────
let ncell = getncells(S_COST.dh.grid), nqp = getnquadpoints(S_COST.cv)
	println("\nTABLE 2 — cost of building the cache")
	println("  wavy geometry, $(PG.nterms(BASIS)) θ-terms, $ncell cells × $nqp qp = " *
			"$(ncell * nqp) quadrature points\n")
	println("  method                       time [ms]    alloc [MiB]")
	println("  " * "-"^52)
	c4(g) = PG.PullbackCache(S_COST.dh, S_COST.cv, g, BASIS)
	c3(g, vh) = PG.PullbackCache(S_COST.dh, S_COST.cv, g, BASIS;
		inverse_determinant = PG.AuxiliaryFieldInverseDet(
			Lagrange{RefHexahedron, vh}(); qr = S_COST.qr))
	c4(wavy)                                     # warm up
	t = @timed c4(wavy)
	@printf("  Method 4 (power series)      %9.1f    %9.1f\n", 1e3 * t.time, t.bytes / 2^20)
	for vh in (1, 2)
		c3(wavy, vh)
		t = @timed c3(wavy, vh)
		@printf("  Method 3 (V_h order %d)       %9.1f    %9.1f\n", vh,
			1e3 * t.time, t.bytes / 2^20)
	end
end
println("""
  Method 3 pays one assembly sweep plus one Cholesky of the c₀-weighted mass
  matrix and L−1 backsolves; Method 4 pays a scalar recurrence per quadrature
  point. Both are negligible beside a single DPIM order — cost is NOT what
  separates them.""")

# ── TABLE 3 ──────────────────────────────────────────────────────────────────
println("\nTABLE 3 — Method 4's radius: |det J − 1| < 1 (theory App. A.3)\n")
let
	c = m4(stretch)                              # det J = 1 + θ ⟹ radius exactly 1
	r = PG.geometry_validity_report(c)
	@printf("  measured radius (stretch, det J = 1+θ): θ ∈ (%.4f, %.4f)\n",
		-r.radius_neg[1], r.radius_pos[1])
	println("  analytic radius                       : θ ∈ (-1.0000, 1.0000)\n")
	println("  reciprocal residual vs truncation order, at three θ:\n")
	println("     MAXT       θ=0.20       θ=0.50       θ=1.00")
	println("     " * "-"^45)
	for maxt in (2, 4, 6, 8, 12)
		b = PG.GeometryParameterBasis([maxt])
		cc = m4(stretch, b)
		@printf("     %-4d   %.4e   %.4e   %.4e\n", maxt,
			PG.geometry_validity_at(cc, (0.2,)).reciprocal_residual,
			PG.geometry_validity_at(cc, (0.5,)).reciprocal_residual,
			PG.geometry_validity_at(cc, (1.0,)).reciprocal_residual)
	end
	println("""
  Read: the θ=1.00 column does not improve with the truncation order — it sits
  exactly ON the radius, where the geometric series diverges. This is Method 4's
  one genuine weakness and the reason the validity report runs by default.
  Method 3 has no radius; its error is set by V_h alone (Table 1).""")
end

# ── the arch case: where the question does not arise ─────────────────────────
println("\nA volume-preserving transform sidesteps the whole comparison:")
let
	shear(x) = (one(T3), T3((i, j) -> (i == 2 && j == 1) ? 1.0 : 0.0))  # nilpotent
	c4 = m4(shear)
	c3 = m3(shear, 1)
	cmp = PG.inverse_determinant_comparison(c4, c3, (0.9,))
	@printf("  nilpotent ∇ψ (det J ≡ 1): max|s₃−s₄| = %.3e, unconditional = %s\n",
		cmp.max_abs, PG.geometry_validity_report(c4).unconditional)
	println("""  Both methods return the single term 1, exactly. Example 07's arch is
  this case, which is why its geometry expansion is lossless.""")
end
