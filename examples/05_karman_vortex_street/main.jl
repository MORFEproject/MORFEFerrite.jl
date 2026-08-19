"""
	main.jl — Kármán vortex street DPIM run (STEP 1 of 3).

What this computes
──────────────────
The 2D incompressible flow past a cylinder loses stability at Re_c ≈ 49 through a Hopf
bifurcation — the Kármán vortex street. This script reduces the ~58 000-DOF
finite-element model to a SINGLE complex ODE (a Stuart–Landau equation)

	ż₁ = λ z₁ + c₁₀₁ z₁ η′ + c₂₁₀ z₁|z₁|² + …

by the Direct Parametrisation of Invariant Manifolds (DPIM). The Reynolds number is
carried along as an extra parametric coordinate η′ = 1/Re − 1/Re₀, so ONE run at the
expansion point Re₀ describes the whole bifurcation neighbourhood.

How to run
──────────
	julia --project=. main.jl          # ≈ 7 min (Apple Silicon): 35 s setup + order-9 solve

Outputs land in results/Re{Re₀}_ord{MAX_ORD}/ (ROM, lift polynomial, TKE Gram, CSVs;
see README.md). One run at MAX_ORD suffices for the whole order-convergence study —
the cohomological solve is graded, so lower orders are exact truncations.

Then:   julia --project=. solve_rom.jl     (limit-cycle branch per truncation order)
		python3 compare_orders.py          (lift & avg-TKE vs Re, one curve per order)

All parameters in config.jl.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	# MORFE.jl is expected as a sibling checkout (folder MORFE.jl or MORFE_jl),
	# either next to this repository or one directory above it; override with
	# ENV["MORFE_PATH"] if it lives elsewhere.
	morfe = get(ENV, "MORFE_PATH", "")
	if isempty(morfe)
		cands = [joinpath(@__DIR__, "..", "..", "..", n) for n in ("MORFE.jl", "MORFE_jl")]
		append!(cands, [joinpath(@__DIR__, "..", "..", "..", "..", n) for n in ("MORFE.jl", "MORFE_jl")])
		morfe = first(filter(isdir, cands))
	end
	Pkg.develop([
		Pkg.PackageSpec(path = morfe),
		Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..")),
	])
	Pkg.add([
		"Ferrite", "FerriteGmsh", "Gmsh",
		"Arpack", "LinearMaps",
		"StaticArrays", "KLU",
	])
end
Pkg.instantiate()

using MORFE
using MORFEFerrite
# The fluid backend, now including the whole "mesh → ROM" layer: `fluid_model`
# assembles (FEM setup, Newton base flow, linearised operators, η′ coupling) and
# `parametrise` reduces (Hopf eigenproblem, model, cohomological solve). The
# eigensolver lives here too — this file used to include its own copy.
# Mesh *generation* (Gmsh) stays example-local in fem/mesh.jl.
using MORFEFerrite.FluidNavierStokes
using Ferrite
using FerriteGmsh
using Gmsh
using LinearAlgebra
using SparseArrays
using StaticArrays: SVector
using Printf
using Serialization
using DelimitedFiles

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fem", "mesh.jl"))
include(joinpath(@__DIR__, "exports.jl"))

# RESULTS_DIR comes from config.jl, so validate.jl reads exactly what this writes.
const DATA_DIR = joinpath(RESULTS_DIR, "data")
mkpath(DATA_DIR)
mkpath(joinpath(RESULTS_DIR, "figures"))

_log = open(joinpath(RESULTS_DIR, "summary.log"), "w")
_out = TeeIO(stdout, _log)   # everything printed to _out also lands in summary.log

println(_out, _sep)
println(_out, "Kármán Vortex Street DPIM  (Re₀ = $Re₀,  order = $MAX_ORD)")
println(_out, "  results → $RESULTS_DIR")
println(_out, _sep)

# ─────────────────────────────────────────────────────────────────────────────
# 1 — Mesh: Turek–Schäfer channel (2.2 × 0.41 m) with a Ø 0.1 m cylinder
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[1/6] Generating Turek–Schäfer mesh ...")
r_mesh = @timed generate_mesh(;
	h_cyl = MESH_H_CYL,
	h_wake = MESH_H_WAKE,
	h_bulk = MESH_H_BULK,
)
meshfile = r_mesh.value

# ─────────────────────────────────────────────────────────────────────────────
# 2 — FEM setup: P2/P1 Taylor-Hood velocity/pressure spaces.
#     Dirichlet BCs (Poiseuille inlet, no-slip walls/cylinder) are eliminated;
#     the DPIM operates on the remaining `free_dpim` DOFs, where the perturbation
#     around the base flow vanishes on every prescribed boundary.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[2/6] Assembling the fluid model at Re₀ = $Re₀ ...")
r_model = @timed fluid_model(meshfile; Re = Re₀)
case = r_model.value
show(_out, MIME"text/plain"(), case)
println(_out)
fom = case.fom
s₀_full = case.s₀_full

# ─────────────────────────────────────────────────────────────────────────────
# 3 — Eigenproblem, and the CHOICE of modes.
#
#     Shift-invert ARPACK on A_lin y = λ B₁ y returns the whole computed spectrum
#     with left and right eigenvectors. Which of those modes span the manifold and
#     which are off-manifold targets is decided HERE, in the driver — the module
#     has no policy about it.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[3/6] Shift-invert ARPACK eigenproblem ...")
r_eig = @timed solve_hopf_eigenproblem(-case.B₀, case.B₁;
	nev = EIG_NEV, sigma_re = EIG_SIGMA_RE, sigma_im = EIG_SIGMA_IM,
	target_freq = EIG_TARGET_FREQ)
eig = r_eig.value

# ── Conjugates are NOT in the computed spectrum ──────────────────────────────
# The shift σ = sigma_re + i·sigma_im is COMPLEX, so ARPACK returns the modes near σ
# only: a strongly oscillatory mode's conjugate sits near conj(σ) and is never
# computed. (Near-real modes are the exception — both halves are about equally far
# from σ, so both appear.) Conjugates are therefore synthesised analytically here,
# which is exact for the real A, B of this problem: λ̄ is an eigenvalue with
# eigenvector φ̄.
# Julia 1.12 is strict about calling a function defined in the same top-level scope,
# so this is a `let` block rather than a helper: the work happens once, inline.
eig = let e = eig
	λ = collect(ComplexF64, e.eigenvalues)
	Φ = Matrix{ComplexF64}(e.right_modes)
	for k in eachindex(e.eigenvalues)
		# A REAL mode is UNPAIRED — it is its own conjugate and gets no partner.
		# Synthesising one would duplicate the coordinate. The test is RELATIVE:
		# ARPACK's numerical zero is ~1e-7 of the magnitude, not machine epsilon.
		abs(imag(λ[k])) <= REAL_MODE_RTOL * abs(λ[k]) && continue
		# Already present? (near-real pairs sit about equally far from σ, so ARPACK
		# returns both halves of those.)
		any(l -> abs(l - conj(λ[k])) <= REAL_MODE_RTOL * abs(λ[k]), λ) && continue
		push!(λ, conj(λ[k]))
		Φ = hcat(Φ, conj.(e.right_modes[:, k]))
	end
	(; eigenvalues = λ, right_modes = Φ,
		hopf_index = e.hopf_index, conjugate_index = e.conjugate_index)
end
@printf(_out, "  %d modes after adding analytic conjugates\n", length(eig.eigenvalues))

# The Hopf pair spans the manifold; every other mode is an outer target.
i_hopf = eig.hopf_index
i_hopf_conj = argmin(abs.(eig.eigenvalues .- conj(eig.eigenvalues[i_hopf])))
hopf = [i_hopf, i_hopf_conj]
outer_all = setdiff(eachindex(eig.eigenvalues), hopf)

# ── Pass 1: which outer modes actually resonate? ─────────────────────────────
# Detection needs a model and spectral data, so a throwaway pair is built with the
# Hopf pair alone. Nothing from this pass reaches the final ROM — only the verdict.
mset_probe = all_multiindices_up_to(2 + N_EXT, MAX_ORD; min_degree = 1)
probe = build_model(case, eig; master = hopf, outer = outer_all,
	n_monomials = length(mset_probe), scale = MODE_SCALE,
	conjugate_atol = REAL_MODE_RTOL)
RESONANCE_TOL = RESONANCE_TOL_FACTOR * abs(eig.eigenvalues[i_hopf])
@printf(_out, "  detuning tolerance = %.4f  (%.3g · |λ_Hopf|)\n",
	RESONANCE_TOL, RESONANCE_TOL_FACTOR)
rset_probe = MORFE.build_resonance_set(probe.model, mset_probe, probe.spectral,
	ResonanceConfig(style = :complex_normal_form, tol = RESONANCE_TOL,
		outer_targets = true, warn_outer = false))

resonant_local = [j for j in eachindex(outer_all)
						if !isempty(resonant_multiindices(rset_probe, 2 + j))]

# ── Pass 1.5: can each resonant candidate actually CARRY a coordinate? ───────
# Resonance says a mode interacts; it does not say the mode is usable as a master
# coordinate. That is decided by the bilinear pairing α = ψᵀB₁φ, the denominator of the
# SymmetricBiorthogonal gauge (both sides ÷ √α).
#
# B₁ is SINGULAR here — 6636 of 57860 rows are P1 pressure DOFs — so this is a descriptor
# pencil B₁ẋ = −B₀x. A vector in B₁'s null space is an algebraic incompressibility mode
# with no dynamics, and its α vanishes. A near-zero α therefore means the mode is
# defective, unconverged onto the essential spectrum, or purely algebraic: ż = λz cannot
# be obtained by projecting with ψᵀB₁, because that projection is null. Promoting it adds
# a coordinate with no equation and amplifies its eigenvector by 1/√α.
#
# The threshold is RELATIVE to the Hopf pair's own α, measured in this same run, since the
# absolute value depends on the mesh and on Arpack's vector scaling.
α_hopf = abs(probe.meta.bilinear_pairings[1])
@printf(_out, "\n  bilinear pairing α = ψᵀB₁φ  (|α_Hopf| = %.3e, keep ≥ %.3g·|α_Hopf|)\n",
	α_hopf, PROMOTE_ALPHA_RTOL)
@printf(_out, "    %-5s %-26s %-11s %-11s %s\n", "mode", "λ", "|α|", "|α|/|α_H|", "verdict")
usable_local = Int[]
for j in resonant_local
	k = outer_all[j]
	local α
	try
		(_, _, α) = left_eigenvector(-case.B₀, case.B₁,
			eig.eigenvalues[k], eig.right_modes[:, k]; scale = MODE_SCALE)
	catch err
		@printf(_out, "    %-5d %-26s %-11s %-11s %s\n", k,
			@sprintf("%+.6f %+.6fi", real(eig.eigenvalues[k]), imag(eig.eigenvalues[k])),
			"—", "—", "DROP (adjoint solve failed: $(typeof(err)))")
		continue
	end
	ratio = abs(α) / α_hopf
	keep = ratio >= PROMOTE_ALPHA_RTOL
	keep && push!(usable_local, j)
	@printf(_out, "    %-5d %-26s %-11.3e %-11.3e %s\n", k,
		@sprintf("%+.6f %+.6fi", real(eig.eigenvalues[k]), imag(eig.eigenvalues[k])),
		abs(α), ratio, keep ? "keep" : "DROP (degenerate — no dynamics to reduce)")
end
if length(usable_local) < length(resonant_local)
	@printf(_out, "  dropped %d of %d resonant candidates on the α test\n",
		length(resonant_local) - length(usable_local), length(resonant_local))
end
resonant_local = usable_local
# Closed under conjugation: a pair is one physical mode, and `SpectralData` rejects
# a pairing that splits one across the master/outer boundary. A real mode is its
# own conjugate and adds nothing here.
# Closed under conjugation: a complex mode and its conjugate are ONE physical mode
# and must be promoted together, or the manifold is not conjugation-invariant and the
# ROM has no real realisation. A REAL mode is its own conjugate — it contributes a
# single coordinate, with conj_perm[i] = i.
# The α filter above cannot split a pair: for a conjugate pair α_p = conj(α_k), so |α| is
# identical and both halves get the same verdict (the 1e-3 threshold leaves three orders
# of margin against measurement noise). The closure below then pulls in a kept mode's
# partner unconditionally, which is what keeps the master set conjugation-invariant.
promoted = Int[]
for j in resonant_local
	k = outer_all[j]
	push!(promoted, k)
	# A real mode is unpaired: it contributes ONE coordinate, with conj_perm[i] = i.
	abs(imag(eig.eigenvalues[k])) <= REAL_MODE_RTOL * abs(eig.eigenvalues[k]) && continue
	p = argmin(abs.(eig.eigenvalues .- conj(eig.eigenvalues[k])))
	p in hopf || push!(promoted, p)
end
promoted = sort(unique(promoted))

@printf(_out, "  %d of %d outer modes resonate at tol = %.4f → promoting %d\n",
	length(resonant_local), length(outer_all), RESONANCE_TOL, length(promoted))
for k in promoted
	@printf(_out, "    mode %2d:  λ = %+.6f %+.6f·i\n",
		k, real(eig.eigenvalues[k]), imag(eig.eigenvalues[k]))
end

# ── Pass 2: the real model, with the resonant modes promoted to master ──────
# The promoted coordinates are expanded to FIRST ORDER only: the monomial set gets
# their unit vectors and nothing else, so the manifold is linear in those
# directions and they never couple to z₁, z̄₁ or η′. That is what keeps a mode
# whose direction would otherwise be solved through a near-singular operator in
# the ROM, without paying for a full expansion in it.
master = vcat(hopf, promoted)
outer = setdiff(eachindex(eig.eigenvalues), master)
ROM_FULL = length(master)
NVAR_FULL = ROM_FULL + N_EXT

# ── The monomial set ────────────────────────────────────────────────────────
# Order MAX_ORD in (z₁, z̄₁, η′), and order 1 in each promoted coordinate: a
# monomial may carry AT MOST ONE promoted coordinate, to the first power.
#
# The mixed terms are the point. The promoted modes are driven quadratically by the
# oscillation (ẏ_k gets a z₁z̄₁ term — that is the mean-flow distortion), and the
# distortion acts back on the oscillator through z₁·y_k. Keeping only the bare unit
# vectors y_k would compute the distortion and then give it no path back to ż₁,
# which removes the dominant stabilising contribution to c₂₁₀ and turns this
# supercritical Hopf subcritical. z₁·y_k is what closes the loop.
#
# y_k² and higher are still excluded — that is what "order 1 in the promoted
# coordinates" means, and it is what keeps the set small.
mset_exps = SVector{NVAR_FULL, Int}[]
_core(a, b, c, p) = SVector{NVAR_FULL, Int}(ntuple(
	i -> i == 1 ? a : i == 2 ? b : i == NVAR_FULL ? c : (i == 2 + p ? 1 : 0), NVAR_FULL))
for a in 0:MAX_ORD, b in 0:MAX_ORD, c in 0:MAX_ORD
	core = a + b + c
	core <= MAX_ORD || continue
	core >= 1 && push!(mset_exps, _core(a, b, c, 0))          # no promoted coordinate
	# One promoted coordinate, first power — but only multiplying a LOW-order core
	# monomial. See PROMOTED_CORE_ORD: the mean-flow loop needs core degree ≤ 2, and
	# mixing beyond that multiplies the monomial count by the core size.
	core <= PROMOTED_CORE_ORD || continue
	core + 1 <= MAX_ORD || continue
	for p in 1:length(promoted)
		push!(mset_exps, _core(a, b, c, p))
	end
end
mset = MultiindexSet(mset_exps)
n_mixed = count(e -> sum(@view e[3:(NVAR_FULL-1)]) > 0, mset_exps)

println(_out, "\n[3/6] Building the model (ROM = $ROM_FULL, NVAR = $NVAR_FULL) ...")
r_build = @timed build_model(case, eig;
	master = master, outer = outer, n_monomials = length(mset), scale = MODE_SCALE,
	conjugate_atol = REAL_MODE_RTOL)
(; model, spectral, meta) = r_build.value
@printf(_out, "  W will be %.0f MB  (%d free DOFs × %d monomials × 16 B)\n",
	case.info.n_free_dpim * length(mset) * 16 / 1024^2, case.info.n_free_dpim, length(mset))
println(_out, "  $(length(mset)) monomials " *
			  "($(length(mset) - n_mixed) in (z₁,z̄₁,η′) ≤ $MAX_ORD, " *
			  "$n_mixed carrying one promoted coordinate)")
println(_out, "  conjugate permutation: $(meta.conjugate_permutation)")

# ─────────────────────────────────────────────────────────────────────────────
# 4 — Solve the cohomological equations for R(z, η′) and W(z, η′).
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[4/6] Solving cohomological equations (order $MAX_ORD) ...")
r_dpim = @timed parametrise(model, spectral, mset;
	resonance = ResonanceConfig(style = :complex_normal_form,
		tol = RESONANCE_TOL,
		# Detection on the FULL complex eigenvalues — the growth rate is part of the
		# distance, which is what separates a genuinely resonant mode from one that
		# merely shares a frequency while being strongly damped.
		eigenvalue_projection = :full,
		outer_targets = true),
	conjugate_permutation = meta.conjugate_permutation)
(W, R) = r_dpim.value

master_eigenvalues = collect(eig.eigenvalues[master])
# Display/export set: the positive-imaginary half, most weakly damped first.
_disp = sort(findall(λ -> imag(λ) > 0.1, eig.eigenvalues); by = i -> abs(real(eig.eigenvalues[i])))
all_eigenvalues = eig.eigenvalues[_disp]
all_modes = eig.right_modes[:, _disp]

# ─────────────────────────────────────────────────────────────────────────────
# 5 — Report the reduced dynamics: the first row ż₁ = R₁(z₁, z̄₁, η′) is the
#     Stuart-Landau equation (row 2 is its conjugate, row 3 is η̇′ = 0).
#     Supercritical Hopf ⇔ Re(c₂₁₀) < 0 for the [2,1,0] monomial.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[5/6] Reduced dynamics — first row ż₁ = R₁(z₁, z̄₁, η′) ...")
export_reduced_dynamics(_out, DATA_DIR, R, master_eigenvalues;
	re0 = Re₀, ord = MAX_ORD, nvar = NVAR)

# ─────────────────────────────────────────────────────────────────────────────
# 6 — Save the ROM and the two physical observables that compare_orders.py
#      evaluates along the limit cycle:
#        · lift polynomial  L(z) = L0 + Σ (lᵀW_α) z^α   (pressure lift on the cylinder)
#        · TKE Gram matrix  G = W_velᵀ M_vel W_vel / |Ω| (period-averaged kinetic energy)
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[6/6] Saving ROM and exporting observables ...")

# `external_system` records the external coordinates the ROM was written in — the
# model has one, so it is passed rather than defaulted away. `Re0`/`max_order` are
# NOT repeated here: `write_summary` records `order` and FluidNavierStokes records
# `Re0`, and duplicated keys in summary.txt break naive parsers.
MORFE.save_rom(RESULTS_DIR, W, R;
	external_system = meta.external_system,
	metadata = ["example" => "05_karman_vortex_street", "fast" => FAST])

# Lift functional: l picks the pressure traction −p·n_y on the cylinder boundary.
(l_free, L0_lift) = lift_functional(case)
L_coeffs_lift = export_lift_polynomial(_out, DATA_DIR, W, l_free, L0_lift)

# TKE Gram: the L×L matrix that lets Python evaluate ⟨TKE⟩ without FOM-sized data.
(M_vel, vel_rows, area) = prepare_energy_gram(fom)
write_energy_gram(DATA_DIR, W, M_vel, vel_rows, area)
println(_out, "  tke_gram_re.csv, tke_gram_im.csv, tke_avector.csv  written to data/")

export_vtk_bundle(DATA_DIR, fom, s₀_full, master_eigenvalues, all_eigenvalues, all_modes)

export_lift_csv(_out, DATA_DIR, mset, L0_lift, L_coeffs_lift)
csv_path = joinpath(DATA_DIR, "R_coefficients.csv")
EXPORT_MATLAB && export_matlab_model(_out, DATA_DIR, csv_path; re0 = Re₀, ord = MAX_ORD)

# The summary skeleton (sizes, order, every `*_time_s` stage) is shared across
# physics modules; FluidNavierStokes contributes its own rows through
# `Common.summary_entries`. Appends to the summary.txt `MORFE.save_rom` wrote.
write_summary(_out, joinpath(RESULTS_DIR, "summary.txt"), case;
	rom = meta,
	# NVAR_FULL, not the config's NVAR: a promoted run has more coordinates, and the
	# title is the first thing read when telling two runs apart.
	title = "Kármán Vortex Street DPIM — Summary  " *
			"(Re₀ = $Re₀, order = $MAX_ORD, NVAR = $NVAR_FULL)",
	# The mesh and the solve are timed HERE, not inside a module: mesh generation is
	# example-local and the reduction is MORFE's `parametrise`, so neither shows up in
	# `case.info` or `meta`.
	metadata = ["mesh_time_s" => @sprintf("%.3f", r_mesh.time),
		"solve_time_s" => @sprintf("%.3f", r_dpim.time),
		"resonance_eigenvalues" => "full", "resonance_tol" => RESONANCE_TOL,
		"outer_targets" => true,
		# The coordinate system this run used. `nvar` is what makes two runs
		# incomparable at the raw-R level, so downstream scripts group on it.
		"promote_resonant" => PROMOTE_RESONANT,
		"nvar" => NVAR_FULL,
		"n_promoted" => length(promoted),
		"promoted_modes" => isempty(promoted) ? "none" : join(promoted, " "),
		"promote_alpha_rtol" => PROMOTE_ALPHA_RTOL,
		"mesh_h_cyl" => MESH_H_CYL, "mesh_h_wake" => MESH_H_WAKE,
		"mesh_h_bulk" => MESH_H_BULK, "fast" => FAST])

close(_log)
