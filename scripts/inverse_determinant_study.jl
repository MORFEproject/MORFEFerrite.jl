# Method 3 vs Method 4 for 1/det J — plot-ready measurements.
#
# This study writes the data behind the inverse-determinant comparison instead
# of rounding a few selected values into console tables:
#
#   julia --project=. scripts/inverse_determinant_study.jl
#   python3 scripts/plot_inverse_determinant_study.py
#
# Generated CSVs and figures live under
# `scripts/results/inverse_determinant_study/` and are ignored by git.
#
# The mesh sequence refines only x₁ because det J = 1 + θ sin(x₁) is constant in
# the transverse directions. With a Q1 vector carrier, doubling the number of
# x₁ nodes gives an exact 1:2:4 sequence of global model DOFs. The carrier does
# not set Method 3's accuracy: that remains the separately reported scalar V_h
# order (1 or 2).

using MORFEFerrite
using Ferrite, Tensors, Printf

const PG = MORFEFerrite.ParametricGeometry
const T3 = Tensor{2, 3, Float64, 9}

const BASIS = PG.GeometryParameterBasis([8])
const VH_ORDERS = (1, 2)
const THETA_SAMPLES = collect(range(-0.95, 0.95; step = 0.025))
const COST_REPETITIONS = 7
const OUTPUT_ROOT = joinpath(@__DIR__, "results", "inverse_determinant_study")
const DATA_DIR = joinpath(OUTPUT_ROOT, "data")

const MESH_SPECS = (
	(label = "small", nel = (11, 5, 5)),
	(label = "medium", nel = (23, 5, 5)),
	(label = "large", nel = (47, 5, 5)),
)

function setup(spec; qorder = 3)
	grid = generate_grid(
		Hexahedron, spec.nel, Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
	ip = Lagrange{RefHexahedron, 1}()^3
	qr = QuadratureRule{RefHexahedron}(qorder)
	cv = CellValues(qr, ip, Lagrange{RefHexahedron, 1}())
	dh = DofHandler(grid)
	add!(dh, :u, ip)
	close!(dh)
	return (; label = spec.label, nel = spec.nel, dh, cv, qr,
		n_cells = getncells(grid), model_dofs = ndofs(dh))
end

function auxiliary_dofs(s, vh)
	dh = DofHandler(Ferrite.get_grid(s.dh))
	add!(dh, :s, Lagrange{RefHexahedron, vh}())
	close!(dh)
	return ndofs(dh)
end

# det J = 1 + θ sin(x₁): non-polynomial in x₀, so V_h must resolve it.
wavy(x) = (one(T3), T3((i, j) -> (i == 1 && j == 1) ? sin(x[1]) : 0.0))
# det J = 1 + θ: every Lagrange space contains its spatially constant inverse
# series, so Method 3 and Method 4 collapse to roundoff on a shared truncation.
stretch(x) = (one(T3), T3((i, j) -> (i == 1 && j == 1) ? 1.0 : 0.0))
# det J ≡ 1: a volume-preserving control for which no reciprocal expansion is
# needed at all.
shear(x) = (one(T3), T3((i, j) -> (i == 2 && j == 1) ? 1.0 : 0.0))

m4(s, geom, basis = BASIS) = PG.PullbackCache(s.dh, s.cv, geom, basis)
m3(s, geom, vh; basis = BASIS) = PG.PullbackCache(s.dh, s.cv, geom, basis;
	inverse_determinant = PG.AuxiliaryFieldInverseDet(
		Lagrange{RefHexahedron, vh}(); qr = s.qr))

function verify_mesh_sequence(setups)
	dofs = getproperty.(setups, :model_dofs)
	dofs == [1296, 2592, 5184] ||
		error("unexpected model-DOF sequence: $dofs")
	dofs[2] == 2dofs[1] || error("medium model DOFs are not 2× small")
	dofs[3] == 2dofs[2] || error("large model DOFs are not 2× medium")
	length(THETA_SAMPLES) == 77 || error("expected 77 θ samples")
	isapprox(first(THETA_SAMPLES), -0.95; atol = 1e-14) ||
		error("θ sweep does not start at -0.95")
	isapprox(last(THETA_SAMPLES), 0.95; atol = 1e-14) ||
		error("θ sweep does not end at 0.95")
	any(isapprox(θ, 0.2; atol = 1e-14) for θ in THETA_SAMPLES) ||
		error("θ sweep does not contain the original θ=0.2 comparison")
	return dofs
end

function write_accuracy(path, setups)
	at_point = Dict{Tuple{String, String, Int}, NamedTuple}()
	nrows = 0
	open(path, "w") do io
		println(io, "geometry,mesh,nel_x,nel_y,nel_z,n_cells,model_dofs," *
			"vh_order,auxiliary_dofs,theta,max_abs,max_rel,rms,residual_m4," *
			"residual_m3,worst_cell,worst_qp")
		for s in setups, (geometry, geom) in (("wavy", wavy), ("stretch", stretch))
			ref = m4(s, geom)
			for vh in VH_ORDERS
				approx = m3(s, geom, vh)
				ns = auxiliary_dofs(s, vh)
				for θ in THETA_SAMPLES
					c = PG.inverse_determinant_comparison(ref, approx, (θ,))
					values = (c.max_abs, c.max_rel, c.rms, c.residual_a, c.residual_b)
					all(isfinite, values) || error(
						"non-finite accuracy result for $(s.label), $geometry, V$vh, θ=$θ")
					@printf(io,
						"%s,%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.17e,%.17e,%.17e,%.17e,%.17e,%d,%d\n",
						geometry, s.label, s.nel..., s.n_cells, s.model_dofs,
						vh, ns, θ, c.max_abs, c.max_rel, c.rms,
						c.residual_a, c.residual_b, c.worst_cell, c.worst_qp)
					nrows += 1
					if isapprox(θ, 0.2; atol = 1e-14)
						at_point[(s.label, geometry, vh)] = c
					end
				end
			end
		end
	end
	nrows == 2 * length(setups) * length(VH_ORDERS) * length(THETA_SAMPLES) ||
		error("unexpected accuracy row count: $nrows")
	verify_accuracy_trends(at_point, setups)
	return nrows
end

function verify_accuracy_trends(at_point, setups)
	for vh in VH_ORDERS, metric in (:max_abs, :rms)
		values = [getproperty(at_point[(s.label, "wavy", vh)], metric) for s in setups]
		(values[1] > values[2] > values[3]) || error(
			"$metric does not decrease under mesh refinement for V$vh: $values")
	end
	for s in setups, metric in (:max_abs, :rms)
		v1 = getproperty(at_point[(s.label, "wavy", 1)], metric)
		v2 = getproperty(at_point[(s.label, "wavy", 2)], metric)
		v2 < v1 || error("V2 does not improve $metric on the $(s.label) mesh")
	end
	stretch_gap = maximum(at_point[(s.label, "stretch", vh)].max_abs
		for s in setups for vh in VH_ORDERS)
	stretch_gap < 1e-12 || error("constant-stretch control gap is $stretch_gap")
	return stretch_gap
end

function build_method(s, method, vh)
	method == "method4" && return m4(s, wavy)
	method == "method3" || error("unknown inverse-determinant method: $method")
	return m3(s, wavy, vh)
end

function measure_build(f)
	GC.gc()
	t = @timed f()
	return (; elapsed = t.time, gc_time = t.gctime, bytes = t.bytes)
end

function write_cost(path, setups)
	methods = ((method = "method4", vh = 0),
		(method = "method3", vh = 1), (method = "method3", vh = 2))
	nrows = 0
	open(path, "w") do io
		println(io, "mesh,nel_x,nel_y,nel_z,n_cells,model_dofs,method," *
			"vh_order,auxiliary_dofs,repetition,elapsed_seconds,gc_seconds," *
			"allocated_bytes")
		for s in setups, spec in methods
			# One explicit per-configuration warm-up keeps compilation and first-use
			# setup out of all seven recorded measurements.
			build_method(s, spec.method, spec.vh)
			ns = spec.vh == 0 ? 0 : auxiliary_dofs(s, spec.vh)
			for repetition in 1:COST_REPETITIONS
				m = measure_build(() -> build_method(s, spec.method, spec.vh))
				all(isfinite, (m.elapsed, m.gc_time)) ||
					error("non-finite timing for $(s.label), $(spec.method), V$(spec.vh)")
				@printf(io, "%s,%d,%d,%d,%d,%d,%s,%d,%d,%d,%.17e,%.17e,%d\n",
					s.label, s.nel..., s.n_cells, s.model_dofs, spec.method,
					spec.vh, ns, repetition, m.elapsed, m.gc_time, m.bytes)
				nrows += 1
			end
		end
	end
	expected = length(setups) * length(methods) * COST_REPETITIONS
	nrows == expected || error("unexpected cost row count: $nrows (expected $expected)")
	return nrows
end

function write_radius(path, s)
	base = m4(s, stretch)
	report = PG.geometry_validity_report(base)
	lower = -report.radius_neg[1]
	upper = report.radius_pos[1]
	nrows = 0
	boundary_residuals = Float64[]
	open(path, "w") do io
		println(io, "maxt,nterms,theta,reciprocal_residual," *
			"measured_radius_lower,measured_radius_upper," *
			"analytic_radius_lower,analytic_radius_upper")
		for maxt in 1:16
			basis = PG.GeometryParameterBasis([maxt])
			cache = m4(s, stretch, basis)
			for θ in THETA_SAMPLES
				residual = PG.geometry_validity_at(cache, (θ,)).reciprocal_residual
				isfinite(residual) || error("non-finite radius residual at MAXT=$maxt, θ=$θ")
				@printf(io, "%d,%d,%.6f,%.17e,%.17e,%.17e,-1.0,1.0\n",
					maxt, PG.nterms(basis), θ, residual, lower, upper)
				nrows += 1
			end
			# Keep the convergence-boundary check even though θ=1 lies outside the
			# plotted interval requested for the radius sweep.
			push!(boundary_residuals,
				PG.geometry_validity_at(cache, (1.0,)).reciprocal_residual)
		end
	end
	nrows == 16 * length(THETA_SAMPLES) || error("unexpected radius row count: $nrows")
	all(isapprox(r, 1.0; atol = 1e-12) for r in boundary_residuals) ||
		error("θ=1 residual should remain one at every truncation order")
	return (; nrows, lower, upper)
end

function verify_volume_preserving(s)
	ref = m4(s, shear)
	approx = m3(s, shear, 1)
	comparison = PG.inverse_determinant_comparison(ref, approx, (0.9,))
	unconditional = PG.geometry_validity_report(ref).unconditional
	comparison.max_abs < 1e-12 || error(
		"volume-preserving Method 3/4 gap is $(comparison.max_abs)")
	unconditional || error("volume-preserving transform was not reported unconditional")
	return comparison.max_abs
end

function main()
	mkpath(DATA_DIR)
	setups = [setup(spec) for spec in MESH_SPECS]
	dofs = verify_mesh_sequence(setups)

	accuracy_path = joinpath(DATA_DIR, "accuracy.csv")
	cost_path = joinpath(DATA_DIR, "cost.csv")
	radius_path = joinpath(DATA_DIR, "radius.csv")

	@printf("inverse-determinant mesh study: model DOFs = %s\n", join(dofs, " / "))
	accuracy_rows = write_accuracy(accuracy_path, setups)
	cost_rows = write_cost(cost_path, setups)
	radius = write_radius(radius_path, first(setups))
	volume_gap = verify_volume_preserving(first(setups))

	println("wrote plot-ready datasets:")
	@printf("  %s  (%d rows)\n", accuracy_path, accuracy_rows)
	@printf("  %s  (%d rows)\n", cost_path, cost_rows)
	@printf("  %s  (%d rows)\n", radius_path, radius.nrows)
	@printf("controls: radius θ ∈ (%.4f, %.4f); volume-preserving gap %.3e\n",
		radius.lower, radius.upper, volume_gap)
	println("plot with: python3 scripts/plot_inverse_determinant_study.py")
end

main()
