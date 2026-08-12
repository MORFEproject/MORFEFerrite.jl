"""
MORFEFerrite — Ferrite.jl FEM backends and high-level UIs for MORFE.jl.

Umbrella package organised into per-physics submodules over a FEM-backend-agnostic
MORFE core (which owns the `FEMMultilinearMap` interface):

- `Common`             — shared Ferrite backend layer (mesh IO, Paraview/VTK export).
- `ParametricGeometry` — physics-blind parametric mesh coordinate transforms (θ-series
  over det/adj/inv-det). Defines the kernel interface a physics implements.
- `StructuralSVK`      — St. Venant-Kirchhoff "mesh → ROM" UI (autonomous / harmonic
  forcing), and the SVK parametric kernel.
- `FluidNavierStokes`  — incompressible cylinder-flow DPIM.

The physics modules depend on `ParametricGeometry`, not the other way round: it is a
lower layer, like `Common`.

VTK export (`write_paraview_*`) is provided by the `MORFEFerriteWriteVTKExt`
extension, activated by `using WriteVTK`.
"""
module MORFEFerrite

using MORFE

# Shared Ferrite backend layer (mesh IO, Paraview/VTK export stubs).
include("common/Common.jl")
using .Common

# Physics-blind parametric coordinate transforms. Included BEFORE the physics
# modules: it defines the kernel interface they implement.
include("ParametricGeometry/ParametricGeometry.jl")
using .ParametricGeometry

# Per-physics submodules.
include("StructuralSVK/StructuralSVK.jl")
using .StructuralSVK

include("FluidNavierStokes/FluidNavierStokes.jl")
using .FluidNavierStokes

# Re-export the Common public API. `build_model` is the single contract every
# physics module implements, so it belongs at the top level.
export load_comsol_grid, AbstractAssembledModel, build_model
# The shared run-summary writer and its per-physics dispatch seam.
export write_summary, summary_entries
export write_paraview_mesh, write_paraview_modes,
	write_paraview_manifold, write_paraview_deformation

# Re-export the StructuralSVK public API. `save_rom` is intentionally NOT
# re-exported here: it clashes with the `save_rom(dirs, W, R)` helper that the
# low-level example includes from `examples/common/results_io.jl`. Reach it as
# `MORFEFerrite.save_rom` or `MORFEFerrite.StructuralSVK.save_rom`.
export SVKMaterial, RayleighDamping, HarmonicForcing,
	mechanical_model, parametrise, real_dynamics, print_equations

end # module MORFEFerrite
