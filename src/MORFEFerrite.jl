"""
MORFEFerrite — Ferrite.jl FEM backends and high-level UIs for MORFE.jl.

Umbrella package organised into per-physics submodules over a FEM-backend-agnostic
MORFE core (which owns the `FEMMultilinearMap` interface):

- `Common`            — shared Ferrite backend layer (mesh IO, Paraview/VTK export).
- `StructuralSVK`     — St. Venant-Kirchhoff "mesh → ROM" UI (autonomous / harmonic forcing).
- `ParametricStructural` — geometric-parameter (θ-series) structural ROMs (uses `StructuralSVK`).
- `FluidNavierStokes` — incompressible cylinder-flow DPIM.

VTK export (`write_paraview_*`) is provided by the `MORFEFerriteWriteVTKExt`
extension, activated by `using WriteVTK`.
"""
module MORFEFerrite

using MORFE

# Shared Ferrite backend layer (mesh IO, Paraview/VTK export stubs).
include("common/Common.jl")
using .Common

# Per-physics submodules.
include("StructuralSVK/StructuralSVK.jl")
using .StructuralSVK

include("ParametricStructural/ParametricStructural.jl")
using .ParametricStructural

# Later migration stages:
# include("FluidNavierStokes/FluidNavierStokes.jl");       using .FluidNavierStokes

# Re-export the Common public API.
export load_comsol_grid
export write_paraview_mesh, write_paraview_modes,
	write_paraview_manifold, write_paraview_deformation

# Re-export the StructuralSVK public API. `save_rom` is intentionally NOT
# re-exported here: it clashes with the `save_rom(dirs, W, R)` helper that the
# low-level example includes from `examples/common/results_io.jl`. Reach it as
# `MORFEFerrite.save_rom` or `MORFEFerrite.StructuralSVK.save_rom`.
export SVKMaterial, RayleighDamping, HarmonicForcing,
	mechanical_model, parametrise, real_dynamics, print_equations

end # module MORFEFerrite
