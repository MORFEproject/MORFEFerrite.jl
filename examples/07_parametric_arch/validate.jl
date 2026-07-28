# Compare a fresh run against the blessed reference. Exit nonzero on mismatch.
#   MORFE_FAST=1 julia --project=. main.jl && MORFE_FAST=1 julia --project=. validate.jl
using Pkg: Pkg
Pkg.activate(@__DIR__)
using MORFE

FAST = get(ENV, "MORFE_FAST", "0") == "1"
fresh = joinpath(@__DIR__, "results", "data", "R_coefficients.csv")

# Two references, see reference_data/PROVENANCE.md:
#  · FAST profile → the current-code blessed run (tight tolerance).
#  · FULL profile → the historic reference; its near-resonant high-order rows
#    carry cross-run Arpack gauge amplification, so the tolerance is loose.
reference, rtol = FAST ?
				  (joinpath(@__DIR__, "reference_data", "R_coefficients_fast_z5_t3.csv"), 1e-10) :
				  (joinpath(@__DIR__, "reference_data", "R_coefficients_ref.csv"), 1e-3)

isfile(fresh) || error("No fresh results at $fresh — run main.jl first.")
pass, dev, report = MORFE.compare_rom_coefficients(fresh, reference;
	mode = :exact, rtol = rtol)
println(report)
pass || error("Example 07 deviates from its reference (max rel dev $dev).")
println("Example 07 validation passed$(FAST ? " (FAST profile)" : "").")

# To bless a new reference after a deliberate, reviewed change:
#   cp results/data/R_coefficients.csv reference_data/R_coefficients_fast_z5_t3.csv
#   git add reference_data/ && git commit -m "Bless example 07 FAST reference"
