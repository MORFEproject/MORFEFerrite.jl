# MORFEFerrite.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia 1.10+](https://img.shields.io/badge/Julia-1.10+-9558B2.svg)](https://julialang.org/downloads/)

Ferrite.jl FEM backends and high-level user interfaces for
[**MORFE.jl**](https://github.com/MORFEproject/MORFE.jl) — the Direct
Parametrisation of Invariant Manifolds (DPIM) for nonlinear model order
reduction. MORFE owns the DPIM solver and the abstract `FEMMultilinearMap`
backend interface; this package implements that interface for concrete physics
domains as first-class submodules.

## Submodules

| Submodule | Provides |
| --------- | -------- |
| `StructuralSVK` | St. Venant-Kirchhoff model construction: `mechanical_model`, `build_model`, `SVKMaterial`, `RayleighDamping`, `HarmonicForcing`, plus the Ferrite geometric-nonlinearity backend (`svk_nonlinearity`, `svk_assemble_KM!`) |
| `ParametricStructural` | General multi-parameter geometric ROMs: additive map `x(θ,x₀) = x₀ + Σᵢ θᵢψᵢ(x₀)` with per-parameter (multiindex-box) θ-series truncation |
| `FluidNavierStokes` | Incompressible cylinder-flow DPIM: Taylor-Hood setup, Newton base flow, linearised operators, convection `FEMMultilinearMap` |
| `Common.MeshIO` | COMSOL/Abaqus/Gmsh conversion (`comsol_to_gmsh`, `abaqus_to_gmsh`, `gmsh_to_comsol` and linear variants) plus COMSOL-to-Ferrite loading (`load_comsol_grid`); all functions are also exported at package top level |
| `Common` | Shared assembled-model contracts and Paraview/VTK export (`write_paraview_*`, activated by `using WriteVTK`) |

## Installation

MORFE is registered in Julia's General registry. MORFEFerrite is currently
installed from GitHub:

```julia
using Pkg
Pkg.add("MORFE")
Pkg.add(url = "https://github.com/MORFEproject/MORFEFerrite.jl.git")
```

## Quick start

```julia
using MORFE, MORFEFerrite
SVK = MORFEFerrite.StructuralSVK

beam = SVK.mechanical_model("beam.msh";
    material  = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping   = SVK.RayleighDamping(α = 5.4e-3, β = 1.9e-2),
    dirichlet = "Dirichlet")

(; model, spectral, meta) = build_model(beam;
    master = [1], expansion_order = 7)
W, R = parametrise(model, spectral, 7;
    resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))
```

## Examples

Runnable examples live in their own repository,
[MORFEExamples](https://github.com/MORFEproject/MORFEExamples). It carries one shared
Julia environment and one Jupyter kernel for every example, so `julia setup.jl` there is
the only setup step. Two of its examples use this package:

| Example | Model |
| ------- | ----- |
| [`from_a_mesh_to_a_rom/`](https://github.com/MORFEproject/MORFEExamples/tree/main/from_a_mesh_to_a_rom) | Clamped-clamped SVK beam, and the backbone curve read off its reduced dynamics |
| [`karman_vortex_street/`](https://github.com/MORFEproject/MORFEExamples/tree/main/karman_vortex_street) | Kármán vortex street: Hopf bifurcation to a Stuart-Landau ROM |

Each is documented as a tutorial on the
[MORFE website](https://morfeproject.github.io/tutorials/).

Research working copies of the other cases (parametric arch, MEMS micromirror, turbine
blade, the outer-mode promotion study) stay in this checkout under `examples/` but are
not tracked: they carry meshes, result archives and environments a package clone has no
use for. Tag `examples-before-untrack` is the last commit that tracked them.

## Tests

```julia
using Pkg; Pkg.test("MORFEFerrite")
```

The suite is self-contained: mesh-conversion fixtures live in `test/fixtures/mesh/`, so
it does not read from `examples/`.

The clamped beam validates an order-9 run against its conservative reference. From a
MORFEExamples checkout:

```bash
cd from_a_mesh_to_a_rom
python3 -m nbconvert --execute --to notebook --inplace from_a_mesh_to_a_rom.ipynb
MORFE_FAST=1 julia validate.jl
```

## License

MIT — see [LICENSE](LICENSE).
