# Regenerate the blessed reference_data/ for the parametric examples (04, 07).
#
#   MORFE_FAST=1 julia scripts/regenerate_parametric_references.jl     # FAST profile
#   julia scripts/regenerate_parametric_references.jl                  # FULL profile
#
# ⚠ RUN THE FULL PROFILE ON OTHER HARDWARE. Example 04 FAST alone is ~14 minutes
# on the development machine; FULL is z ≤ 9, θ ≤ 4 and far heavier. Run one
# example at a time — this script does, deliberately, and does not parallelise.
#
# ── What blessing means here, and what it does not ───────────────────────────
#
# These CSVs are CHANGE DETECTORS, not correctness references. Re-blessing them
# proves only that a future run reproduces this one. The thing that establishes
# correctness is `test/ParametricGeometry/test_moved_mesh_fom.jl` (and its 2D
# counterpart): the parametric transform against a FOM whose geometry actually
# moved. **Do not re-bless without that gate passing** — the archived references
# were once re-blessed against each other, which is exactly how example 04 stayed
# wrong for so long while looking validated.
#
# The script therefore refuses to write anything until the moved-mesh gate has
# been run and passed in this session.

using Dates
using Pkg: Pkg

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FAST = get(ENV, "MORFE_FAST", "0") == "1"
const PROFILE = FAST ? "FAST" : "FULL"

# (example directory, blessed filename for this profile, the truncation it encodes)
const TARGETS = [
	("04_parametric_clamped_beam",
		FAST ? "R_coefficients_corrected_z3_t2.csv" : "R_coefficients_full_z9_t4.csv",
		FAST ? "z ≤ 3, θ ≤ 2" : "z ≤ 9, θ ≤ 4"),
	("07_parametric_arch",
		FAST ? "R_coefficients_fast_z5_t3.csv" : "R_coefficients_full_z11_t7.csv",
		FAST ? "z ≤ 5, θ ≤ 3" : "z ≤ 11, θ ≤ 7"),
]

function gate!()
	@info "Running the moved-mesh FOM gate before blessing anything (3D and 2D)."
	# Built as a Julia string and interpolated as ONE argument: escaping quotes
	# inside backticks is fragile and silently produces the wrong argv.
	script = """
	using Test, MORFEFerrite
	cd($(repr(ROOT)))
	@testset "gate" begin
		include("test/ParametricGeometry/test_moved_mesh_fom.jl")
		include("test/ParametricGeometry/test_moved_mesh_fom_2d.jl")
		include("test/ParametricGeometry/test_series_algebra_2d.jl")
	end
	"""
	cmd = `$(Base.julia_cmd()) --project=$ROOT -e $script`
	success(pipeline(cmd; stdout = stdout, stderr = stderr)) || error(
		"The moved-mesh FOM gate FAILED. Nothing was blessed. Fix the transform " *
		"before touching reference_data/ — a reference blessed over a broken gate " *
		"is worse than no reference.")
	@info "Gate passed."
	return nothing
end

function run_example(dir)
	ex = joinpath(ROOT, "examples", dir)
	@info "Running example $dir at the $PROFILE profile (this is the slow part)."
	env = copy(ENV)
	env["MORFE_FAST"] = FAST ? "1" : "0"
	cmd = setenv(`$(Base.julia_cmd()) --project=$ex $(joinpath(ex, "main.jl"))`, env)
	success(pipeline(cmd; stdout = stdout, stderr = stderr)) ||
		error("example $dir failed; nothing blessed")
	return joinpath(ex, "results", "data", "R_coefficients.csv")
end

function bless(dir, fname, trunc, fresh)
	refdir = joinpath(ROOT, "examples", dir, "reference_data")
	mkpath(refdir)
	dest = joinpath(refdir, fname)
	isfile(fresh) || error("no fresh results at $fresh")
	cp(fresh, dest; force = true)
	@info "Blessed $dir → reference_data/$fname"

	# Provenance is not optional. A reference whose origin is unrecorded is what
	# CLAUDE.md had to warn about for both of these examples.
	commit = try
		strip(read(`git -C $ROOT rev-parse --short HEAD`, String))
	catch
		"unknown"
	end
	open(joinpath(refdir, "PROVENANCE.md"), "a") do io
		println(io, """

		## $(fname) — regenerated $(Dates.format(now(), "yyyy-mm-dd"))

		- profile: **$PROFILE** ($trunc)
		- code: commit `$commit`
		- validated by: `test/ParametricGeometry/test_moved_mesh_fom.jl` (3D) and
		  `test_moved_mesh_fom_2d.jl`, both passing at the time of blessing — the
		  parametric transform against a FOM whose geometry actually moved.
		- this file is a CHANGE DETECTOR. It proves reproducibility, not
		  correctness; the gate above is what establishes correctness.""")
	end
	return dest
end

gate!()
if !FAST
	@warn """FULL profile requested. This is heavy — example 04 FAST alone is ~14 min
	on the development machine. If you are on that machine, stop now and set
	MORFE_FAST=1, or move this to other hardware."""
end
for (dir, fname, trunc) in TARGETS
	bless(dir, fname, trunc, run_example(dir))
end
@info "Done. Review the diff in examples/*/reference_data/ before committing."
