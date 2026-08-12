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
using Printf
using Serialization
using DelimitedFiles

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "fem", "mesh.jl"))
include(joinpath(@__DIR__, "exports.jl"))

# Results directory (deterministic name — the same config overwrites the previous run)
const RESULTS_DIR = joinpath(@__DIR__, "results", @sprintf("Re%.2f_ord%d", Re₀, MAX_ORD))
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
# 3 — Master modes: the Hopf pair. Shift-invert ARPACK on A_lin y = λ B₁ y finds
#     the least-damped complex-conjugate eigenpair λ₁,₂ = σ₀ ± iω₀ (σ₀ ≈ 0 near
#     Re_c, ω₀ ≈ 16.86 rad/s — the shedding frequency). Its right/left
#     eigenvectors seed the two master coordinates z₁, z̄₁ of the manifold.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[3/6] Hopf eigenproblem and model assembly ...")
mset = all_multiindices_up_to(NVAR, MAX_ORD; min_degree = 1)
r_build = @timed build_model(case;
	n_monomials = length(mset),
	nev = EIG_NEV,
	sigma_re = EIG_SIGMA_RE,
	sigma_im = EIG_SIGMA_IM,
	target_freq = EIG_TARGET_FREQ,
	scale = 1e-2)
(; model, spectral, meta) = r_build.value
println(_out, "  $(length(mset)) monomials (NVAR=$NVAR, order ≤ $MAX_ORD)")
println(_out, "  conjugate permutation: $(meta.conjugate_permutation)")

# ─────────────────────────────────────────────────────────────────────────────
# 4 — Run parametrise to solve the cohomological equations and produce the 
#     reduced dynamics R(z, z̄, η′) and the transformation W(z, z̄, η′) from the 
#     reduced coordinates to the full FEM state.
# ─────────────────────────────────────────────────────────────────────────────

println(_out, "\n[4/6] Solving cohomological equations (order $MAX_ORD) ...")
r_dpim = @timed parametrise(model, spectral, mset;
	resonance = ResonanceConfig(style = :complex_normal_form,
		tol_relative = 0.05, eigenvalue_projection = :imaginary_part_only),
	conjugate_permutation = meta.conjugate_permutation)
(W, R) = r_dpim.value

master_eigenvalues = collect(meta.spectrum.eigenvalues)
all_eigenvalues = meta.spectrum.all_eigenvalues
all_modes = meta.spectrum.all_modes

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
	title = "Kármán Vortex Street DPIM — Summary  " *
			"(Re₀ = $Re₀, order = $MAX_ORD, NVAR = $NVAR)",
	# The mesh and the solve are timed HERE, not inside a module: mesh generation is
	# example-local and the reduction is MORFE's `parametrise`, so neither shows up in
	# `case.info` or `meta`.
	metadata = ["mesh_time_s" => @sprintf("%.3f", r_mesh.time),
		"solve_time_s" => @sprintf("%.3f", r_dpim.time),
		"resonance_eigenvalues" => "imaginary_part_only", "resonance_tol_relative" => 0.05,
		"mesh_h_cyl" => MESH_H_CYL, "mesh_h_wake" => MESH_H_WAKE,
		"mesh_h_bulk" => MESH_H_BULK, "fast" => FAST])

close(_log)
