# Compare a fresh run against the blessed reference. Exit nonzero on mismatch.
#   Set `order = 3` or `order = 9` in clamped_beam.ipynb, then execute it.
#   jupyter nbconvert --execute --to notebook --inplace clamped_beam.ipynb
#   MORFE_FAST=1 julia --project=. validate.jl
using Pkg: Pkg
Pkg.activate(@__DIR__)
using MORFE

FAST = get(ENV, "MORFE_FAST", "0") == "1"
fresh = joinpath(@__DIR__, "results", "data", "R_coefficients.csv")
# The DPIM solve is graded, so an order-3 (FAST) run is an exact truncation of
# the order-9 (FULL) one: both validate against the FULL reference, compared
# over the shared monomials.
reference = joinpath(@__DIR__, "reference_data", "R_coefficients_ref.csv")
probe_reference = joinpath(
    @__DIR__, "reference_data", "W_node289_y_coefficients_ref.csv")

isfile(fresh) || error("No fresh results at $fresh — execute clamped_beam.ipynb first.")
# Arpack's eigenvector gauge differs slightly across runs; compare the shared
# order-3/order-9 coefficients with a tolerance above that cross-run noise.
pass, dev, report = MORFE.compare_rom_coefficients(fresh, reference;
    mode = :exact, rtol = 1e-8)
println(report)
pass || error("Example 01 deviates from its reference (max rel dev $dev).")

# The website backbone needs the full order-9 resonant row and the complete
# position-map row at its highlighted probe node.
ref_exponents, _ = MORFE.read_rom_coefficients(reference)
@assert any(r -> Tuple(ref_exponents[r, :]) == (5, 4), axes(ref_exponents, 1))
@assert isfile(probe_reference)
@assert length(readlines(probe_reference)) == 55 # header + 54 monomials, degrees 1:9
println("Example 01 validation passed$(FAST ? " (FAST profile)" : "").")

# To bless a new reference after a deliberate, reviewed change:
#   Re-export R with drop_below = 0.0 so its degree-9 resonant row is retained.
#   git add reference_data/ && git commit -m "Bless example 01 reference"
