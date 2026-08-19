# Compare a fresh run against THE reference. Exit nonzero on mismatch.
#   julia --project=. main.jl && julia --project=. validate.jl
#   julia --project=. validate.jl --bless     # re-bless the reference from this run
#
# There is ONE reference, `reference_data/karman_invariants_ref.txt`, and it validates a
# run with OR without promoted outer modes. That is possible because it stores physical
# invariants rather than raw R coefficients:
#
#   · raw R is not comparable across runs at all — the Arpack eigenvector gauge differs
#     (z → e^{iφ}z), which is why the previous reference could only ever be compared
#     "gauge-invariantly" on a hand-picked subset of rows;
#   · promoting outer modes into the master set is a CHANGE OF COORDINATES: the ROM
#     gains columns and shares no monomials with an un-promoted one. Two such runs
#     describe the same physics and must validate against the same file.
#
# invariants.jl does the work, slaving any promoted coordinates before reading off λ,
# c₁₀₁ and the effective Landau coefficient. See its header for why slaving (rather than
# zeroing) is the only correct reduction.
using Pkg: Pkg
Pkg.activate(@__DIR__)
using Printf
using Dates: now
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "invariants.jl"))

const BLESS = "--bless" in ARGS
const REF = joinpath(@__DIR__, "reference_data", "karman_invariants_ref.txt")
const FRESH = joinpath(RESULTS_DIR, "data", "R_coefficients.csv")

# FAST changes the MESH (see config.jl), so its coefficients are a different
# discretisation — not comparable to the reference. FAST checks that the pipeline runs;
# only FULL runs are validated numerically.
FAST && !BLESS && (println("FAST profile: coarse mesh, numerical validation skipped " *
						   "(run without MORFE_FAST to validate)."); exit(0))
FAST && BLESS && error("Refusing to bless a reference from the FAST profile — its mesh " *
					   "is a different discretisation. Re-run without MORFE_FAST.")

isfile(FRESH) || error("No fresh results at $FRESH — run main.jl first.")

# Say HOW OLD the artefact is relative to the source that produced it. Nothing here
# re-runs the solve: this script validates whatever main.jl last wrote, so a stale
# artefact reports a confident numerical FAIL that looks exactly like a regression.
# That has already cost one debugging session — the order-9 artefact predated a fix to
# the conjugate pairing in FluidNavierStokes/build_model.jl, so it still carried the
# pre-fix c₂₁₀ while the fixed code had never produced an order-9 run at all.
let
	src = joinpath(@__DIR__, "..", "..", "src", "FluidNavierStokes")
	srcs = [joinpath(root, f)
			for (root, _, fs) in walkdir(src) for f in fs if endswith(f, ".jl")]
	newest = isempty(srcs) ? 0.0 : maximum(mtime, srcs)
	@printf("Validating %s\n  written %.1f days ago\n", FRESH, (time() - mtime(FRESH)) / 86400)
	mtime(FRESH) < newest && @warn """
	STALE ARTEFACT: this run predates the current FluidNavierStokes source, so it was
	NOT produced by the code you are validating. A failure below says nothing about the
	current state — re-run main.jl first."""
end

poly = load_rom_poly(FRESH)
inv = rom_invariants(poly)
@printf("  NVAR = %d (%d promoted outer mode%s)\n",
	inv.nvar, inv.n_promoted, inv.n_promoted == 1 ? "" : "s")

if BLESS
	write_invariants(REF, inv;
		note = @sprintf("Blessed from order-%d run, NVAR=%d, MODE_SCALE=%g, %s.",
			MAX_ORD, inv.nvar, MODE_SCALE, string(now())))
	println("Blessed $REF")
	exit(0)
end

isfile(REF) || error("No reference at $REF — bless one with `validate.jl --bless`.")
ref = read_invariants(REF)

# Tolerances, per quantity and per physical meaning:
#
#   σ  is ~0.004 against |λ| ≈ 16.9 because Re₀ sits essentially AT the bifurcation, so
#      a relative test on σ alone would demand 4 more digits than the eigensolve carries.
#      It is checked absolutely, scaled by |λ| — the natural scale of the eigenvalue.
#   c210_re/im scale as |c|² under z → cz, so they are only comparable at a FIXED
#      MODE_SCALE. They get a loose check; c210_ratio is the strict one.
#   c210_ratio = Im/Re is gauge-free (the |c|² cancels) and is the quantity whose SIGN
#      the conjugate-pairing bug flipped. It is the sharpest single check here.
#
# A PROMOTED run carries an extra, legitimate error: the reference is un-promoted, and the
# promoted ROM expands each y_k only to core degree PROMOTED_CORE_ORD, so its mean-flow
# response is truncated where the un-promoted run resolves it through W to full order. That
# is a modelling difference, not a regression — measured at 4.3e-3 on c₂₁₀ at order 9, while
# λ and c₁₀₁ still match to 1e-6. The nonlinear tolerances are relaxed accordingly; the
# LINEAR ones are not, because promotion cannot touch them.
λmag = hypot(ref["σ"], ref["ω"])
promoted = inv.n_promoted > 0
c210_tol = promoted ? 1e-2 : 1e-3
ratio_tol = promoted ? 1e-2 : 1e-4
promoted && println("  promoted run: nonlinear tolerances relaxed to $(c210_tol) for the\n" *
					"  PROMOTED_CORE_ORD truncation of y_k (λ and c₁₀₁ are not relaxed).")
checks = [
	("σ", inv.σ, ref["σ"], :abs, 1e-6 * λmag),
	("ω", inv.ω, ref["ω"], :rel, 1e-6),
	("c101_re", inv.c101_re, ref["c101_re"], :rel, 1e-5),
	("c101_im", inv.c101_im, ref["c101_im"], :rel, 1e-5),
	("c210_ratio", inv.c210_ratio, ref["c210_ratio"], :rel, ratio_tol),
	("c210_re", inv.c210_re, ref["c210_re"], :rel, c210_tol),
	("c210_im", inv.c210_im, ref["c210_im"], :rel, c210_tol),
]

println("\n  quantity      this run           reference          deviation   tol       ")
println("  " * "─"^70)
fails = String[]
for (name, got, want, kind, tol) in checks
	dev = kind === :abs ? abs(got - want) : abs(got - want) / max(abs(want), eps())
	ok = dev <= tol
	ok || push!(fails, name)
	@printf("  %-12s %+.10g   %+.10g   %8.2e  %8.2e %s  %s\n",
		name, got, want, dev, tol, kind === :abs ? "abs" : "rel", ok ? "✓" : "✗ FAIL")
end

if !isempty(fails)
	error("Example 05 deviates from its reference in: " * join(fails, ", "))
end
println("\nExample 05 validation passed — ", inv.criticality,
	", ", inv.n_promoted == 0 ? "un-promoted" : "$(inv.n_promoted) promoted mode(s)",
	" (the same reference validates both).")
