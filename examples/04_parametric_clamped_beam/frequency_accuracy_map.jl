# How accurately does the parametric model predict the FIRST BENDING FREQUENCY
# across the (θ₁, θ₂) box?
#
#   MORFE_FAST=1 julia --project=. frequency_accuracy_map.jl
#   NX=41 NY=41 julia --project=. frequency_accuracy_map.jl      # finer, slower
#
# ── The reference is exact, and that is the whole point ──────────────────────
#
# The obvious reference — move the mesh nodes to x₀ + θ₁ψ₁ + θ₂ψ₂ and reassemble
# — does NOT work here. ψ₂ = h₀sin(πx₁/L)e₂ is not polynomial, so a node-moved
# mesh realises only the geometric map's QUADRATIC INTERPOLANT of the sine
# (~0.3 % shape error on this 10-element mesh). That mismatch would appear as a
# θ₂-proportional error floor which no truncation order removes, and it would
# swamp the truncation error this map is about.
#
# Instead the reference freezes θ and assembles with the EXACT Jacobian, through
# a series with a single term:
#
#     b0    = GeometryParameterBasis([0, 0])     # L = 1: only the zero multiindex
#     geom  = x₀ -> [SVector(0,0) => J_EXACT(x₀, θ)]
#
# `reciprocal_series` of a one-term series is exactly 1/p₀, so NOTHING is
# truncated: det J, adj J and 1/det J are all exact at that θ. The difference
# against the parametric model is therefore θ-truncation and nothing else.
#
# ── The first bending mode is TRACKED, not assumed to be ω₁ ──────────────────
#
# At θ₂ = 20 the rise is 10 % of the span and the arch stiffens enough to
# reorder modes. Each point identifies the lowest reference mode whose
# transverse (e₂) component dominates, then pairs the parametric mode to it by
# MAC — both sides share the mesh and DOF numbering, so MAC is direct and
# gauge-free. The chosen indices are written out, so a reordering is visible in
# the data rather than silently changing what is being compared.

using Pkg: Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MORFE, MORFEFerrite
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using SparseArrays, LinearAlgebra, Printf, Tensors, StaticArrays
const PG = MORFEFerrite.ParametricGeometry
const SVK = MORFEFerrite.StructuralSVK
const Tens3 = Tensor{2, 3, Float64, 9}

include(joinpath(@__DIR__, "config.jl"))

# ── grid over the intended validity box ──────────────────────────────────────
#
# CLUSTERED ABOUT THE EXPANSION POINT θ = 0. The series is centred there, so
# that is where the error structure is steepest: the truncation error grows like
# θ₁^(GEOM_T1+1), which means it collapses through several decades over a narrow
# band and then flattens onto the numerical floor. A uniform grid resolves that
# transition worst exactly where it is most interesting. It is also where the
# physics moves fastest — the arch does most of its stiffening over the first
# quarter of the θ₂ range.
#
# `CLUSTER = 2` is deliberate rather than maximal: the error is already at the
# floor below |θ₁| ≈ 0.05, so clustering harder than this spends points where
# nothing is resolvable.
const NX = parse(Int, get(ENV, "NX", "53"))   # odd ⇒ θ₁ = 0 is on the grid exactly
const NY = parse(Int, get(ENV, "NY", "41"))
const CLUSTER = parse(Float64, get(ENV, "CLUSTER", "2.0"))

_cluster_sym(b, n, p) = [b * sign(s) * abs(s)^p for s in range(-1, 1; length = n)]
_cluster_one(b, n, p) = [b * u^p for u in range(0, 1; length = n)]

const Θ1 = _cluster_sym(0.2, NX, CLUSTER)     # includes θ₁ = 0 exactly (odd NX)
const Θ2 = _cluster_one(20.0, NY, CLUSTER)
const NEV_MAP = 8                     # physical modes to look at per point

# ── FE space (identical to main.jl) ──────────────────────────────────────────
grid = togrid(MESH)
ip = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr = QuadratureRule{RefHexahedron}(3)
cv = CellValues(qr, ip, geo_ip)
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch)
update!(ch, 0.0)
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
println("Total DOFs: ", ndofs(dh), "   free: ", length(free))

# Component of each FREE dof: the field is a single Lagrange{...}^3, so the three
# components are consecutive in the global numbering.
const COMP = [(d - 1) % 3 + 1 for d in free]

# ── the parametric model: assembled ONCE, then evaluated at each θ ───────────
pcase, _ = BUILD_CASE(dh, cv, free)
const K_ARR = pcase.operators[1].arrays          # already restricted to `free`
const M_ARR = pcase.operators[3].arrays
const EXPS = BASIS_K.mset.exponents
println("parametric linear series: $(length(K_ARR)) θ-terms over box $(Tuple(BASIS_K.bounds))")

function parametric_KM(θ)
	K = spzeros(length(free), length(free))
	M = spzeros(length(free), length(free))
	for (m, α) in enumerate(EXPS)
		c = θ[1]^α[1] * θ[2]^α[2]
		iszero(c) && continue
		K .+= c .* K_ARR[m]
		M .+= c .* M_ARR[m]
	end
	return K, M
end

# ── the exact, untruncated reference at a frozen θ ───────────────────────────
const B0 = PG.GeometryParameterBasis([0, 0])
const F2L = PG.free_dof_map(ndofs(dh), free)
const STRESS = SVK.stress_model(MATERIAL)

function exact_KM(θ)
	geom = x₀ -> [SVector(0, 0) => J_EXACT(x₀, θ)]
	cache = PG.PullbackCache(dh, cv, geom, B0)
	pd = PG.ParametricDiscretisation(dh, cv, F2L, length(free), cache)
	K = [allocate_matrix(dh)]
	M = [allocate_matrix(dh)]
	PG.assemble_linear_series!(K, M, pd, SVK.SVKPullbackKernel{0}(STRESS, RHO))
	return K[1][free, free], M[1][free, free]
end

# ── modal helpers ────────────────────────────────────────────────────────────
# The eigensolver returns conjugate pairs in adjacent slots; odd indices are the
# distinct physical modes.
function modes_of(K, M)
	ep = spectrum(K, M, StructureModalDampingEigensolver(NEV_MAP, ALPHA, BETA);
		sorter! = (args...) -> nothing)
	idx = 1:2:length(ep.eigenvalues)
	ω = [abs(ep.eigenvalues[i]) for i in idx]
	Φ = [ep.eigenmodes[:, 1, i] for i in idx]
	p = sortperm(ω)
	return ω[p], Φ[p]
end

# Gauge-free modal assurance criterion.
mac(a, b) = abs(dot(a, b))^2 / (real(dot(a, a)) * real(dot(b, b)))

# Fraction of a mode's energy in Cartesian component `c`.
function comp_fraction(φ, c)
	tot = sum(abs2, φ)
	tot == 0 && return 0.0
	return sum(abs2(φ[j]) for j in eachindex(φ) if COMP[j] == c) / tot
end

# The first BENDING mode: lowest frequency whose transverse (e₂) motion
# dominates. The section is 10 (y) × 24 (z), so e₂ is the soft direction and at
# θ = 0 this is the fundamental — but the arch can reorder modes, which is
# exactly why this is searched for rather than taken as index 1.
function first_bending(ω, Φ)
	for k in eachindex(ω)
		comp_fraction(Φ[k], 2) > 0.5 && return k
	end
	return 1
end

# ── sweep ────────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "results", "data")
mkpath(outdir)
csv = joinpath(outdir, "frequency_accuracy_map.csv")

# The reference configuration: straight beam at the nominal span. Every reported
# frequency change is relative to THIS, which is also the point the θ-series is
# expanded about.
let Kr, Mr
	global OMEGA0
	Kr, Mr = exact_KM((0.0, 0.0))
	ωr, Φr = modes_of(Kr, Mr)
	OMEGA0 = ωr[first_bending(ωr, Φr)]
end
@printf("reference ω₀ (straight beam, nominal span) = %.10g\n", OMEGA0)

t0 = time()
open(csv, "w") do io
	println(io, "theta1,theta2,dL_over_L_pct,rise_over_L_pct," *
				"rise_over_Lact_pct,omega_ref,omega_rom,rel_err,domega_pct,idx_ref,idx_rom,mac")
	n = 0
	for θ2 in Θ2, θ1 in Θ1
		θ = (θ1, θ2)
		Kr, Mr = exact_KM(θ)
		Kp, Mp = parametric_KM(θ)
		ωr, Φr = modes_of(Kr, Mr)
		ωp, Φp = modes_of(Kp, Mp)

		ir = first_bending(ωr, Φr)
		# Pair by MAC rather than by index, so a reordering on the parametric
		# side is followed instead of silently comparing different modes.
		macs = [mac(Φr[ir], Φp[k]) for k in eachindex(Φp)]
		ip_, mbest = argmax(macs), maximum(macs)

		err = abs(ωp[ip_] - ωr[ir]) / ωr[ir]
		# Physical axes: θ₁ IS ΔL/L (the span goes L → (1+θ₁)L), and the rise is
		# θ₂·h₀, so the rise ratio is θ₂·(h₀/L). Written here rather than derived
		# in the plotting script, so the h₀/L constant lives in exactly one place.
		dL_pct = 100 * θ1
		rise_pct = 100 * θ2 * h0_L_ratio
		# …and normalised by the ACTUAL span L(θ) = (1+θ₁)L₀. The transverse
		# displacement does not depend on θ₁ but the span does, so the same
		# physical rise is a larger fraction of a shortened beam.
		rise_act_pct = rise_pct / (1 + θ1)
		dω_pct = 100 * (ωr[ir] - OMEGA0) / OMEGA0
		@printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.12g,%.12g,%.6e,%.10g,%d,%d,%.6f\n",
			θ1, θ2, dL_pct, rise_pct, rise_act_pct,
			ωr[ir], ωp[ip_], err, dω_pct, ir, ip_, mbest)
		n += 1
		if n % 50 == 0
			@printf("  %4d/%d points  (%.1f s elapsed)\n", n, NX * NY, time() - t0)
			flush(io)
			flush(stdout)
		end
	end
end
@printf("\n%d × %d grid done in %.1f s → %s\n", NX, NY, time() - t0, csv)

# ── the two checks that must hold, printed rather than assumed ───────────────
rows = [split(l, ',') for l in readlines(csv)[2:end]]
val(r, i) = parse(Float64, r[i])
const C_ERR, C_DW, C_IDX, C_MAC = 8, 9, 10, 12
at_origin = [r for r in rows if val(r, 1) == 0.0 && val(r, 2) == 0.0]
col_t1_0 = [r for r in rows if val(r, 1) == 0.0]
@printf("θ = (0,0)          : rel_err = %.3e   (must be ~0: same operators)\n",
	isempty(at_origin) ? NaN : val(at_origin[1], C_ERR))
@printf("θ₁ = 0 column      : max rel_err = %.3e over %d points\n",
	isempty(col_t1_0) ? NaN : maximum(val(r, C_ERR) for r in col_t1_0), length(col_t1_0))
println("""
  The expansion is EXACT along θ₁ = 0 for any θ₂: ∇ψ₂ is nilpotent and adj J is
  degree 1 in θ₂, so the bound of 2 on the stiffness loses nothing. Measured
  directly, the operators there agree ENTRYWISE to 6e-16 of max|K| and M agrees
  to 0.0 exactly.

  That column should therefore read ~1e-9, NOT ~1e-16. This is the NUMERICAL
  FLOOR of the comparison, not truncation: max|K| is set by axial stiffness
  while the bending eigenvalue sits ~1e-6 below it, so double-precision roundoff
  at the operator scale is amplified by ~10⁶ at the modal scale. A deterministic
  dense eigensolve reproduces it, so it is conditioning and not solver
  tolerance. Errors below ~1e-9 are not resolvable on this problem — read the
  map above that level.""")
@printf("worst point        : rel_err = %.3e\n", maximum(val(r, C_ERR) for r in rows))
@printf("Δω/ω range         : %+.1f %% … %+.1f %%\n",
	minimum(val(r, C_DW) for r in rows), maximum(val(r, C_DW) for r in rows))
@printf("min MAC            : %.6f   (low values mean mode tracking is struggling)\n",
	minimum(val(r, C_MAC) for r in rows))
reord = count(r -> val(r, C_IDX) != 1, rows)
@printf("mode reorderings   : %d of %d points had the bending mode away from index 1\n",
	reord, length(rows))
