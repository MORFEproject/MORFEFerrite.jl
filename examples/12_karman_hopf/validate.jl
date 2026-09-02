# Compare a fresh run against the blessed reference. Exit nonzero on mismatch.
#   Set `order = 3` or `order = 9` in karman_hopf.ipynb, then execute it.
#   jupyter nbconvert --execute --to notebook --inplace karman_hopf.ipynb
#   julia --project=. validate.jl
using Pkg: Pkg
Pkg.activate(@__DIR__)
using MORFEFerrite
const NSE = MORFEFerrite.FluidNavierStokes

fresh = joinpath(@__DIR__, "results", "data", "R_coefficients.csv")
reference = joinpath(@__DIR__, "reference_data", "karman_invariants_ref.txt")

isfile(fresh) || error("No fresh results at $fresh — execute karman_hopf.ipynb first.")
isfile(reference) || error("No reference at $reference.")

# Physical invariants, not raw coefficients. The ARPACK eigenvector gauge differs run to
# run, so R's coefficients are not comparable between runs at all; σ, ω, c101 and the
# c210 ratio are.
inv = NSE.rom_invariants(NSE.load_rom_poly(fresh))
ref = NSE.read_invariants(reference)
λmag = hypot(ref["σ"], ref["ω"])

# The DPIM solve is graded, so an order-3 run is an exact truncation of the order-9
# reference. Every invariant here is degree ≤ 3 — c101 is monomial (1,0,1) and c210 is
# (2,1,0) — so both orders validate against the same file.
#
# σ is compared ABSOLUTELY, against the magnitude of λ: at the expansion point the growth
# rate is 4e-3 against a frequency of 16.9, so a relative test on σ would demand six
# digits of a quantity that is numerically zero by construction.
checks = [("σ", inv.σ, ref["σ"], :abs, 1e-6 * λmag),
    ("ω", inv.ω, ref["ω"], :rel, 1e-6),
    ("c101_re", inv.c101_re, ref["c101_re"], :rel, 1e-5),
    ("c101_im", inv.c101_im, ref["c101_im"], :rel, 1e-5),
    ("c210_ratio", inv.c210_ratio, ref["c210_ratio"], :rel, 1e-4),
    ("c210_re", inv.c210_re, ref["c210_re"], :rel, 1e-3),
    ("c210_im", inv.c210_im, ref["c210_im"], :rel, 1e-3)]

println("NVAR = $(inv.nvar), $(inv.criticality)")
println(rpad("quantity", 12), rpad("this run", 22), rpad("reference", 22),
    rpad("dev", 11), "tol")
fails = String[]
for (name, got, want, kind, tol) in checks
    dev = kind === :abs ? abs(got - want) : abs(got - want) / max(abs(want), eps())
    dev <= tol || push!(fails, name)
    println(rpad(name, 12), rpad(got, 22), rpad(want, 22),
        rpad(round(dev; sigdigits = 3), 11), tol, dev <= tol ? "  ✓" : "  ✗")
end

isempty(fails) ||
    error("Example 12 deviates from its reference in: $(join(fails, ", ")).")

# c210 carries the gauge: it scales as |mode scale|², so a run made with a different
# `scale =` in the notebook fails here even though the physics is identical. That is the
# intent — see reference_data/PROVENANCE.md.
inv.criticality == "supercritical" ||
    error("Hopf bifurcation is $(inv.criticality); the Kármán street is supercritical.")
println("\nExample 12 validation passed.")

# To bless a new reference after a deliberate, reviewed change:
#   Set order = 9 in karman_hopf.ipynb and execute it, then
#   NSE.write_invariants(reference, NSE.rom_invariants(NSE.load_rom_poly(fresh)))
#   git add reference_data/ && git commit -m "Bless example 12 reference"
