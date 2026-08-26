# What do the geometries in the parameter range actually look like?
#
#   julia --project=. geometry_gallery.jl        (then: python3 figures_geometry.py)
#
# The accuracy map says how well the ROM predicts the first bending frequency over
# the (ΔL/L₀, h/L(θ)) box, but nothing about what that box CONTAINS. This emits the
# outlines so the configurations can be drawn: one panel per sampled parameter
# pair, the ACTUAL geometry over the reference geometry, at true scale.
#
# ── Actual, not "deformed" ───────────────────────────────────────────────────
#
# These are different beams produced by a geometry parametrisation, not one beam
# responding to a load. The word throughout this example is ACTUAL.
#
# ── Drawn from the analytic map, not from a moved mesh ───────────────────────
#
# `sine_bend_displacement` is the same function the model integrates, so the
# picture and the physics cannot drift apart. Deliberately NOT a node-moved mesh:
# ψ₂ = h₀sin(πx₁/L₀)e₂ is not polynomial, so moving nodes would realise only the
# geometric map's quadratic interpolant of the sine (~0.3 % shape error on this
# mesh) — i.e. it would draw something the model does not use. The reference
# element boundaries are instead carried THROUGH the analytic map, which shows the
# discretisation without misrepresenting the geometry.

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

# ── what to sample ───────────────────────────────────────────────────────────
#
# The grid is chosen by the CALLER, in PERCENT OF THE REFERENCE LENGTH, because
# that is the language the figure is read in. `figures_geometry.py` sets these
# and re-invokes this script whenever they change, so the grid can be edited in
# one place without the constants h₀ and L₀ having to exist in two.
#
#   GALLERY_DL_PCT   ΔL/L₀  [%]   → θ₁ = pct/100                (exact, ΔL/L₀ ≡ θ₁)
#   GALLERY_H_PCT    h/L₀   [%]   → θ₂ = pct/100 / (h₀/L₀)
#
# Note h/L₀ is the ROW label — constant along a row. The per-panel h/L(θ), which
# divides by the ACTUAL length, varies along a row and is emitted separately.
_pct_list(key, default) =
	(v = strip(get(ENV, key, "")); isempty(v) ? default :
	 [parse(Float64, strip(t)) for t in split(v, ',') if !isempty(strip(t))])

const DL_PCT = _pct_list("GALLERY_DL_PCT", [-20.0, 0.0, 20.0])
const H_PCT = _pct_list("GALLERY_H_PCT", [0.0, 2.5, 5.0, 7.5, 10.0])

const Θ1_SAMPLES = DL_PCT ./ 100                       # columns: span change
const Θ2_SAMPLES = H_PCT ./ 100 ./ h0_L_ratio          # rows: arch rise = θ₂·h₀

@printf("grid: ΔL/L₀ = %s %%   h/L₀ = %s %%\n",
	join(DL_PCT, ", "), join(H_PCT, ", "))

# The section in the drawing plane, and the mesh's element layout along it. The
# mesh is 10×2×2 over [0,1000]×[0,10]×[0,24], so in the x₁–x₂ view there are 11
# element stations along the span and 3 across the thickness.
const Y_LO, Y_HI = 0.0, 10.0
const N_ELEM_X, N_ELEM_Y = 10, 2
const NPTS = 241                                     # samples per spanwise curve

# ── the map, straight from the model's own shape fields ──────────────────────
# x(θ,x₀) = x₀ + θ₁·(∇ψ₁·x₀) + θ₂·ψ₂(x₀), with ψ₂ the isochoric sine bend.
function actual_point(x₀₁, x₀₂, θ)
	x₀ = Vec{3, Float64}((x₀₁, x₀₂, 0.0))
	w = sine_bend_displacement(x₀, H0, L_SPAN)       # the model's own ψ₂
	p = x₀ + θ[1] * (∇ψ₁ ⋅ x₀) + θ[2] * w
	return (p[1], p[2])
end

# A spanwise curve at constant reference height, and a through-thickness segment
# at a constant reference station.
span_curve(x₀₂, θ) =
	[actual_point(s, x₀₂, θ) for s in range(0.0, L_SPAN; length = NPTS)]
station_segment(x₀₁, θ) =
	[actual_point(x₀₁, y, θ) for y in range(Y_LO, Y_HI; length = 2)]

# Every polyline of one configuration, tagged so the plotting side can style the
# boundary differently from the interior element edges.
function polylines(θ)
	out = Tuple{String, Vector{Tuple{Float64, Float64}}}[]
	for (k, y) in enumerate(range(Y_LO, Y_HI; length = N_ELEM_Y + 1))
		kind = (k == 1 || k == N_ELEM_Y + 1) ? "boundary" : "element"
		push!(out, (kind, span_curve(y, θ)))
	end
	for (k, s) in enumerate(range(0.0, L_SPAN; length = N_ELEM_X + 1))
		kind = (k == 1 || k == N_ELEM_X + 1) ? "boundary" : "element"
		push!(out, (kind, station_segment(s, θ)))
	end
	return out
end

# ── emit ─────────────────────────────────────────────────────────────────────
outdir = joinpath(@__DIR__, "results", "data")
mkpath(outdir)
csv = joinpath(outdir, "geometry_gallery.csv")

open(csv, "w") do io
	println(io, "row,col,theta1,theta2,dL_over_L0_pct,h_over_L0_pct," *
				"h_over_L_pct,part,kind,curve,x,y")
	curve = 0
	for (r, θ2) in enumerate(Θ2_SAMPLES), (c, θ1) in enumerate(Θ1_SAMPLES)
		θ = (θ1, θ2)
		dL_pct = 100 * θ1                                  # ΔL/L₀ ≡ θ₁
		span = (1 + θ1) * L_SPAN                           # the ACTUAL length L(θ)
		h_pct = 100 * θ2 * H0 / span                       # rise over the ACTUAL length
		h0_pct = 100 * θ2 * H0 / L_SPAN                    # rise over the REFERENCE length
		for (part, θp) in (("reference", (0.0, 0.0)), ("actual", θ))
			for (kind, pts) in polylines(θp)
				curve += 1
				for (x, y) in pts
					@printf(io, "%d,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%s,%s,%d,%.10g,%.10g\n",
						r, c, θ1, θ2, dL_pct, h0_pct, h_pct, part, kind, curve, x, y)
				end
			end
		end
	end
end
@printf("wrote %s\n", csv)

# ── the checks that must hold, printed rather than assumed ───────────────────
for (r, θ2) in enumerate(Θ2_SAMPLES), (c, θ1) in enumerate(Θ1_SAMPLES)
	θ = (θ1, θ2)
	pts = reduce(vcat, [p for (_, p) in polylines(θ)])
	xs = [p[1] for p in pts]
	ys = [p[2] for p in pts]
	span = maximum(xs)
	rise = maximum(ys) - Y_HI                        # top surface lifts by the rise
	# Recomputed FROM THE EMITTED POINTS, not from the inputs — that is what makes
	# it a check rather than a restatement.
	h_pct = 100 * rise / span
	ok_span = isapprox(span, (1 + θ1) * L_SPAN; rtol = 1e-12)
	ok_rise = isapprox(rise, θ2 * H0; atol = 1e-9)
	ok_h = isapprox(h_pct, 100 * θ2 * H0 / ((1 + θ1) * L_SPAN); rtol = 1e-10)
	(ok_span && ok_rise && ok_h) ||
		error("panel (row $r, col $c) inconsistent: span $span, rise $rise")
end
println("span, rise and h/L(θ) verified against the emitted points for all " *
		"$(length(Θ1_SAMPLES) * length(Θ2_SAMPLES)) panels")

# At θ = (0,0) the map must be the IDENTITY. Comparing the reference polylines
# against the actual ones at θ = 0 would be vacuous — both call the same function
# with the same argument — so the check is against the reference COORDINATES.
let worst = 0.0
	for x₀₂ in (Y_LO, 0.5(Y_LO + Y_HI), Y_HI),
		x₀₁ in range(0.0, L_SPAN; length = NPTS)

		p = actual_point(x₀₁, x₀₂, (0.0, 0.0))
		worst = max(worst, abs(p[1] - x₀₁), abs(p[2] - x₀₂))
	end
	worst == 0.0 ||
		error("θ = (0,0) is not the identity map (max deviation $worst)")
	println("θ = (0,0) is exactly the identity map (deviation 0)")
end
