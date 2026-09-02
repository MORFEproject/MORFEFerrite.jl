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

Self-contained, runnable examples live under [`examples/`](examples/). Example
01 resolves MORFE from the registry and uses MORFEFerrite from the current
checkout; its README contains the one-time environment setup command. Some of
the other examples still use the older sibling-checkout development workflow.

| Folder | Model |
| ------ | ----- |
| [`01_clamped_beam_ferrite/`](examples/01_clamped_beam_ferrite/) | Clamped-clamped SVK beam — minimal notebook using the common MORFE API |
| [`03_arch_comsol_wedge/`](examples/03_arch_comsol_wedge/) | Polysilicon arch, COMSOL P18 wedge mesh |
| [`04_parametric_clamped_beam/`](examples/04_parametric_clamped_beam/) | Two-parameter ROM (axial stretch + bending-mode arch) |
| [`05_karman_vortex_street/`](examples/05_karman_vortex_street/) | Kármán vortex street — outer-mode promotion study, diagnostics and DNS reference |
| [`07_parametric_arch/`](examples/07_parametric_arch/) | Single-parameter sinusoidal arch |
| [`08_mems_micromirror/`](examples/08_mems_micromirror/) | MEMS scanning micromirror from CAD |
| [`12_karman_hopf/`](examples/12_karman_hopf/) | Kármán vortex street — Hopf bifurcation to a Stuart-Landau ROM, minimal notebook using the common MORFE API |
| [`mesh_import/`](examples/mesh_import/) | COMSOL/Abaqus/Gmsh conversion examples and reusable source fixtures |

## Tests

```julia
using Pkg; Pkg.test("MORFEFerrite")
```

Example 01 validates both its committed order-3 output and an order-9 run
against the conservative order-9 reference:

```bash
MORFE_ORDER=9 jupyter nbconvert --execute --to notebook --inplace \
  examples/01_clamped_beam_ferrite/clamped_beam.ipynb
julia --project=examples/01_clamped_beam_ferrite \
  examples/01_clamped_beam_ferrite/validate.jl
```

## License

MIT — see [LICENSE](LICENSE).
