# compare_runs.jl — do two runs describe the same physics?
#
#   julia --project=. compare_runs.jl                # auto-discover, current profile
#   julia --project=. compare_runs.jl dirA dirB      # explicit
#
# Promoting resonant outer modes into the master set is a CHANGE OF COORDINATES: the
# mean-flow distortion becomes an explicit coordinate y_k instead of being folded into the
# manifold W. The two ROMs therefore share no monomials and their raw R coefficients are
# not comparable — the promoted run's ż₁ carries a z₁²z̄₁ coefficient that is missing
# everything the y_k route contributes.
#
# What MUST agree is the physics. invariants.jl slaves the promoted coordinates (they solve
# R_k = 0 on the slow manifold, and R is exactly affine in them) and reads off λ, c₁₀₁ and
# the EFFECTIVE Landau coefficient, all of which are promotion-invariant. That is the
# comparison made here.
using Pkg: Pkg
Pkg.activate(@__DIR__)
using Printf
include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "invariants.jl"))

"Free-DOF count a run was computed with, from summary.txt; `nothing` if absent."
function run_n_free(run_dir::AbstractString)
	path = joinpath(run_dir, "summary.txt")
	isfile(path) || return nothing
	for line in eachline(path)
		m = match(r"^\s*n_free\s*:\s*(\d+)", line)
		m === nothing || return parse(Int, m.captures[1])
	end
	return nothing
end

run_dirs = if length(ARGS) >= 2
	[abspath(a) for a in ARGS]
else
	# Default: the un-promoted run for this profile and its `_promoted` sibling. Both are
	# derived from RESULTS_DIR so the profile (FAST/FULL) always matches.
	base = PROMOTE_RESONANT ? replace(RESULTS_DIR, r"_promoted$" => "") : RESULTS_DIR
	filter(isdir, [base, base * "_promoted"])
end

length(run_dirs) >= 2 ||
	error("need two run directories; found $(length(run_dirs)). Run main.jl both ways:\n" *
		  "  julia --project=. main.jl\n  MORFE_PROMOTE=1 julia --project=. main.jl")

# A run is tied to its mesh. Comparing across discretisations measures mesh refinement, not
# promotion, so refuse rather than report a meaningless number.
meshes = unique(filter(!isnothing, run_n_free.(run_dirs)))
length(meshes) <= 1 ||
	error("these runs use different meshes ($(join(meshes, " vs ")) free DOFs). " *
		  "That difference is discretisation, not promotion — compare like with like.")

invs = NamedTuple[]
for d in run_dirs
	csv = joinpath(d, "data", "R_coefficients.csv")
	isfile(csv) || error("no data/R_coefficients.csv in $d — run main.jl first")
	push!(invs, rom_invariants(load_rom_poly(csv)))
end

println("\nComparing reduced dynamics — promotion-invariant quantities")
for (d, i) in zip(run_dirs, invs)
	@printf("  %-34s NVAR = %d  (%d promoted)\n", basename(d), i.nvar, i.n_promoted)
end

ref = invs[1]
rows = [("σ", :σ, :abs), ("ω", :ω, :rel),
	("c101_re", :c101_re, :rel), ("c101_im", :c101_im, :rel),
	("c210_re", :c210_re, :rel), ("c210_im", :c210_im, :rel),
	("c210_ratio", :c210_ratio, :rel)]

hdr = @sprintf("\n  %-12s %-22s %-22s %-11s", "quantity", basename(run_dirs[1]),
	basename(run_dirs[2]), "deviation")
println(hdr)
println("  " * "─"^(length(hdr) - 3))
λmag = hypot(ref.σ, ref.ω)
for (name, key, kind) in rows
	a, b = getproperty(invs[1], key), getproperty(invs[2], key)
	dev = kind === :abs ? abs(a - b) / λmag : abs(a - b) / max(abs(a), eps())
	@printf("  %-12s %+-22.12g %+-22.12g %.2e %s\n", name, a, b, dev,
		kind === :abs ? "(abs/|λ|)" : "(rel)")
end

println()
for (d, i) in zip(run_dirs, invs)
	@printf("  %-34s %s\n", basename(d), i.criticality)
end

# Two DIFFERENT standards apply, and conflating them mislabels ordinary truncation as a bug.
#
# λ at the bifurcation is the MASTER EIGENVALUE itself: at ρ → 0, η′ = 0 the promoted
# coordinates slave to zero, so nothing of the promotion survives. It must be exact.
λdev = max(abs(invs[1].σ - invs[2].σ) / λmag, abs(invs[1].ω - invs[2].ω) / abs(invs[1].ω))
@printf("\n  λ at the bifurcation agrees to %.2e — ", λdev)
println(λdev < 1e-12 ? "exact, as it must be: at ρ→0, η′=0 the promoted\n" *
					   "    coordinates slave to zero, so promotion leaves no trace here." :
		"SUSPICIOUS. This is the master eigenvalue and\n" *
		"    cannot depend on the coordinate system — suspect the mode selection.")

# c₁₀₁ and c₂₁₀ are invariant only in EXACT arithmetic. Both are computed with the promoted
# coordinates slaved, and y_k is expanded only to core degree PROMOTED_CORE_ORD — so y_k(η′)
# and y_k(ρ²) are truncated. The residual measures that truncation, which is the actual
# quantity of interest: how much a first-order treatment of those modes differs from folding
# them into W.
c101dev = max(abs(invs[1].c101_re - invs[2].c101_re) / abs(invs[1].c101_re),
	abs(invs[1].c101_im - invs[2].c101_im) / abs(invs[1].c101_im))
c210dev = max(abs(invs[1].c210_re - invs[2].c210_re) / abs(invs[1].c210_re),
	abs(invs[1].c210_im - invs[2].c210_im) / abs(invs[1].c210_im))
@printf("  c₁₀₁ (Reynolds sensitivity)     agrees to %.2e\n", c101dev)
@printf("  c₂₁₀ (effective Landau coeff.)  agrees to %.2e\n", c210dev)
println("""
    These two are invariant in exact arithmetic only. Both are read off WITH the promoted
    coordinates slaved, and y_k is expanded to core degree $PROMOTED_CORE_ORD, so its
    responses to η′ and to ρ² are truncated. The residual is that truncation — a RESULT,
    not a failure. The physical claim is criticality, and it must match.""")

invs[1].criticality == invs[2].criticality ||
	error("the two runs disagree on CRITICALITY ($(invs[1].criticality) vs " *
		  "$(invs[2].criticality)). They cannot both be right — promotion is a change of " *
		  "coordinates and cannot change the sign of Re c₂₁₀.")
