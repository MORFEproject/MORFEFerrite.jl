# Compare a fresh FULL run against the blessed reference. Exit nonzero on mismatch.
#   julia --project=. main.jl && julia --project=. validate.jl
using Pkg: Pkg
Pkg.activate(@__DIR__)
using MORFE
include(joinpath(@__DIR__, "config.jl"))

# FAST changes the MESH (see config.jl), so its coefficients are a different
# discretisation — not comparable to the reference. FAST checks that the
# pipeline runs; only FULL runs are validated numerically.
FAST && (println("FAST profile: coarse mesh, numerical validation skipped " *
				 "(run without MORFE_FAST to validate)."); exit(0))

fresh = joinpath(@__DIR__, "results", @sprintf("Re%.2f_ord%d", Re₀, MAX_ORD),
	"data", "R_coefficients.csv")
reference = joinpath(@__DIR__, "reference_data", "R_coefficients_ref.csv")

isfile(fresh) || error("No fresh results at $fresh — run main.jl first.")
# Gauge-invariant mode: raw R coefficients are NOT comparable across runs (the
# Arpack eigenvector gauge differs), so compare the eigenvalue row, the rows
# linear in the external parameter η′, and the Im/Re ratio of the leading
# nonlinear term — all invariant under z → e^{iφ}z.
pass, dev, report = MORFE.compare_rom_coefficients(fresh, reference;
	mode = :gauge_invariant, rtol = 1e-6)
println(report)
pass || error("Example 05 deviates from its reference (max rel dev $dev).")
println("Example 05 validation passed (gauge-invariant).")

# To bless a new reference after a deliberate, reviewed change:
#   cp results/Re49.03_ord9/data/R_coefficients.csv reference_data/R_coefficients_ref.csv
#   git add reference_data/ && git commit -m "Bless example 05 reference"
