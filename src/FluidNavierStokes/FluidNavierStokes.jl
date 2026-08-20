"""
`MORFEFerrite.FluidNavierStokes` — incompressible Navier-Stokes DPIM backend
(Taylor-Hood P2/P1, cylinder-flow class of problems), promoted from
`examples/05_karman_vortex_street`.

Pipeline (see the Kármán example driver):

	fom = setup_fem(meshfile)                       # Ferrite P2/P1 spaces + BCs
	s0 = solve_steady_state(fom; Re0)              # Newton base flow
	ops = assemble_linear_operators(s0, fom; Re0)   # linearised operators
	conv = FluidConvection(fom; max_unique_cols)     # f₂(s,s) FEMMultilinearMap{1}
	g1 = make_param_coupling(K_visc_free)          # −D·η′·K_raw·u′ (Re-parametric)
	h0 = make_base_forcing(h₀_vec_free)            # base-flow forcing direction

Mesh *generation* (Gmsh) deliberately stays example-local; `setup_fem` only
reads a `.msh` via FerriteGmsh.
"""
module FluidNavierStokes

using MORFE
using MORFE.Polynomials: evaluate   # rom_analysis.jl evaluates R at a point
using Ferrite, FerriteGmsh
using LinearAlgebra, SparseArrays, DelimitedFiles
using Serialization: serialize   # observables.jl writes the VTK/lift bundles
using KLU
using ..Common: AbstractAssembledModel, Common
import ..Common: build_model, summary_entries

# Include order mirrors the Kármán driver (steady state feeds the linearisation).
include("fem_setup.jl")
include("steady_state.jl")
include("linear_operators.jl")
include("fluid_maps.jl")
include("energy_gram.jl")
include("eigensolver.jl")

# The "mesh → ROM" layer. There is deliberately no `parametrise` here: the only
# `parametrise` is MORFE's, and a caller reduces with
# `parametrise(model, spectral, order_or_mset; resonance = ResonanceConfig(...))`.
include("types.jl")
include("fluid_model.jl")
include("build_model.jl")
include("observables.jl")

# Post-processing of a finished ROM: slaving, limit-cycle continuation, the
# promotion-invariant physical quantities, and the convergence diagnostics. Kept out of
# the example because three copies of the slaving algorithm had drifted apart there.
include("rom_analysis.jl")
# Full-order time integration and periodic orbits — the DNS reference.
include("fom_orbit.jl")

export setup_fem, U_MEAN, U_MAX
export solve_steady_state, assemble_steady_nse!, compute_drag_lift
export assemble_linear_operators, check_linearisation, compute_pressure_lift_weights
export FluidConvection, make_param_coupling, make_base_forcing, assemble_K_visc
export velocity_dof_mask, assemble_velocity_mass_full, domain_area,
	prepare_energy_gram, write_energy_gram
export solve_hopf_eigenproblem, AbstractModeNormalisation,
	SymmetricBiorthogonal, LeftBiorthogonal, NoNormalisation
# Exported because MODE SELECTION belongs to the driver, not to this module, and choosing
# which modes can carry a master coordinate requires their bilinear pairing α = ψᵀB₁φ.
# See the warning in its docstring: never call it for both halves of a conjugate pair.
export left_eigenvector
export AssembledFluidModel, fluid_model, build_model
export lift_functional, lift_polynomial
export export_reduced_dynamics, export_lift_polynomial, export_lift_csv, export_vtk_bundle

# ROM post-processing (rom_analysis.jl)
export ROMPoly, load_rom_poly, slaved_R1, truncate_dynamics
export rom_po_residual, rom_po_frequency, rom_hopf_eta,
	rom_palc_tangent, rom_palc_step, trace_limit_cycle_branch,
	roots_at_re, sweep_branch_in_re
export rom_invariants, read_invariants, write_invariants
export homological_denominators, manifold_ratio_test, modal_growth

# Full-order periodic orbits (fom_orbit.jl)
export eval_perturbation_convection!, build_imex_operators, integrate_orbit!,
	find_periodic_orbit, seed_from_branch, measure_orbit

end # module FluidNavierStokes
