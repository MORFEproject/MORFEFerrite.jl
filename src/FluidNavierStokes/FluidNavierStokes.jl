"""
`MORFEFerrite.FluidNavierStokes` — incompressible Navier-Stokes DPIM backend
(Taylor-Hood P2/P1, cylinder-flow class of problems), promoted from
`examples/05_karman_vortex_street`.

Pipeline (see the Kármán example driver):

	fom  = setup_fem(meshfile)                       # Ferrite P2/P1 spaces + BCs
	s0   = solve_steady_state(fom; Re0)              # Newton base flow
	ops  = assemble_linear_operators(s0, fom; Re0)   # linearised operators
	conv = FluidConvection(fom; max_unique_cols)     # f₂(s,s) FEMMultilinearMap{1}
	g1   = make_param_coupling(K_visc_free)          # −D·η′·K_raw·u′ (Re-parametric)
	h0   = make_base_forcing(h₀_vec_free)            # base-flow forcing direction

Mesh *generation* (Gmsh) deliberately stays example-local; `setup_fem` only
reads a `.msh` via FerriteGmsh.
"""
module FluidNavierStokes

using MORFE
using Ferrite, FerriteGmsh
using LinearAlgebra, SparseArrays, DelimitedFiles
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

export setup_fem, U_MEAN, U_MAX
export solve_steady_state, assemble_steady_nse!, compute_drag_lift
export assemble_linear_operators, check_linearisation, compute_pressure_lift_weights
export FluidConvection, make_param_coupling, make_base_forcing, assemble_K_visc
export velocity_dof_mask, assemble_velocity_mass_full, domain_area,
	prepare_energy_gram, write_energy_gram
export solve_hopf_eigenproblem, AbstractModeNormalisation,
	SymmetricBiorthogonal, LeftBiorthogonal, NoNormalisation
export AssembledFluidModel, fluid_model, build_model
export lift_functional, lift_polynomial

end # module FluidNavierStokes
