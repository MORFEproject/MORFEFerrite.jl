# Compare a fresh run against the blessed reference. Exit nonzero on mismatch.
#   MORFE_FAST=1 julia --project=. main.jl && MORFE_FAST=1 julia --project=. validate.jl
using Pkg: Pkg
Pkg.activate(@__DIR__)
using MORFE

FAST = get(ENV, "MORFE_FAST", "0") == "1"
fresh = joinpath(@__DIR__, "results", "data", "R_coefficients.csv")
# The DPIM solve is graded, so an order-3 (FAST) run is an exact truncation of
# the order-5 (FULL) one: both validate against the FULL reference, compared
# over the shared monomials.
reference = joinpath(@__DIR__, "reference_data", "R_coefficients_ref.csv")

isfile(fresh) || error("No fresh results at $fresh — run main.jl first.")
pass, dev, report = MORFE.compare_rom_coefficients(fresh, reference;
	mode = :exact, rtol = 1e-8)
println(report)
pass || error("Example 03 deviates from its reference (max rel dev $dev).")
println("Example 03 validation passed$(FAST ? " (FAST profile)" : "").")

# To bless a new reference after a deliberate, reviewed change:
#   cp results/data/R_coefficients.csv reference_data/R_coefficients_ref.csv
#   git add reference_data/ && git commit -m "Bless example 03 reference"
