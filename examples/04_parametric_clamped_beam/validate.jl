# Compare a fresh run against the blessed reference. Exit nonzero on mismatch.
#   MORFE_FAST=1 julia --project=. main.jl && MORFE_FAST=1 julia --project=. validate.jl
using Pkg: Pkg
Pkg.activate(@__DIR__)
using MORFE

FAST = get(ENV, "MORFE_FAST", "0") == "1"
fresh = joinpath(@__DIR__, "results", "data", "R_coefficients.csv")
# The blessed reference is the corrected general-formulation run at z ≤ 3, θ ≤ 2.
# A FULL run (z ≤ 9, θ ≤ 4) is a graded superset: the comparison matches rows by
# exponent over the SHARED monomials, so both profiles validate against it.
reference = joinpath(@__DIR__, "reference_data", "R_coefficients_corrected_z3_t2.csv")

isfile(fresh) || error("No fresh results at $fresh — run main.jl first.")
pass, dev, report = MORFE.compare_rom_coefficients(fresh, reference;
	mode = :exact, rtol = 1e-10)
println(report)
pass || error("Example 04 deviates from its reference (max rel dev $dev).")
println("Example 04 validation passed$(FAST ? " (FAST profile)" : "").")

# To bless a new reference after a deliberate, reviewed change:
#   cp results/data/R_coefficients.csv reference_data/R_coefficients_corrected_z3_t2.csv
#   git add reference_data/ && git commit -m "Bless example 04 reference"
