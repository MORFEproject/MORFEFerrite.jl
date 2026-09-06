# Compare a fresh run against the blessed reference. Exit nonzero on mismatch.
#   python3 -m nbconvert --execute --to notebook --inplace clamped_beam.ipynb
#   MORFE_FAST=1 julia --project=. validate.jl
using Pkg: Pkg
Pkg.activate(@__DIR__)
using MORFE
using MORFE.Polynomials: DensePolynomial
using StaticArrays: SVector

FAST = get(ENV, "MORFE_FAST", "0") == "1"
data = joinpath(@__DIR__, "results", "data")
fresh = joinpath(data, "R_coefficients.csv")
reference = joinpath(@__DIR__, "reference_data", "R_coefficients_ref.csv")
probe_reference = joinpath(
    @__DIR__, "reference_data", "W_node289_y_coefficients_ref.csv")

isfile(fresh) ||
    error("No fresh results at $fresh. Execute clamped_beam.ipynb first.")
# Arpack's eigenvector gauge differs slightly across runs; compare with a
# tolerance above that cross-run noise.
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

# ---------------------------------------------------------------------------
# The backbone the website draws.
#
# Compared through `cycle_amplitude` rather than coefficient by coefficient. A
# rescaling of the master eigenvector by exp(iψ) multiplies the (a,b) coefficient
# of W by exp(i(a-b)ψ), so the raw rows are only reproducible while the
# eigensolver keeps picking the same phase. Half the peak-to-peak excursion over
# a full cycle is a rigid shift of that signal in φ, so it is gauge-free, and it
# is exactly the quantity FIG 02 plots.
# ---------------------------------------------------------------------------
backbone = joinpath(data, "backbone.csv")
@assert isfile(backbone) "No backbone at $backbone. Execute clamped_beam.ipynb first."

"Rebuild a scalar polynomial in (z₁, z̄₁) from an `exp_1,exp_2,re,im` table."
function read_probe_polynomial(path)
    rows = [split(line, ',') for line in readlines(path)[2:end]]
    exps = [SVector(parse(Int, r[1]), parse(Int, r[2])) for r in rows]
    coeffs = [complex(parse(Float64, r[3]), parse(Float64, r[4])) for r in rows]
    mset = MultiindexSet(exps)
    # MultiindexSet stores graded-lex order, which the CSV need not follow.
    order = [findfirst(==(e), mset.exponents) for e in exps]
    out = zeros(ComplexF64, length(mset.exponents))
    out[order] .= coeffs
    return DensePolynomial(out, mset)
end

u_ref = read_probe_polynomial(probe_reference)
rows = [split(line, ',') for line in readlines(backbone)[2:end]]
@assert length(rows) == 4 * 341 "backbone.csv has $(length(rows)) rows, expected 1364"

# The stored order-9 curve is the one the reference polynomial can reproduce; the
# lower orders are truncations of it, checked by the package's own tests.
checked = 0
worst = 0.0
for r in rows
    parse(Int, r[1]) == 9 || continue
    ρ = parse(Float64, r[2])
    ρ in (0.0, 20.0, 40.0, 60.0, 85.0) || continue
    expected = cycle_amplitude(u_ref, ρ)
    got = parse(Float64, r[5])
    global worst = max(worst, abs(got - expected) / max(expected, 1e-12))
    global checked += 1
end
@assert checked == 5 "expected 5 sampled radii on the order-9 curve, checked $checked"
worst < 1e-10 || error("Backbone deviates from the probe reference (max rel dev $worst).")
println("Backbone matches the probe reference at $checked radii (max rel dev $worst).")

println("Example 01 validation passed$(FAST ? " (FAST profile)" : "").")

# To bless a new reference after a deliberate, reviewed change:
#   The notebook writes results/data/W_probe_coefficients.csv and exports R with
#   drop_below = 0.0, so copy both over reference_data/ and commit.
#   git add reference_data/ && git commit -m "Bless example 01 reference"
