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
| `StructuralSVK` | St. Venant-Kirchhoff "mesh → ROM" UI: `mechanical_model`, `parametrise`, `SVKMaterial`, `RayleighDamping`, `HarmonicForcing`, plus the Ferrite geometric-nonlinearity backend (`svk_nonlinearity`, `svk_assemble_KM!`) |
| `ParametricStructural` | General multi-parameter geometric ROMs: additive map `x(θ,x₀) = x₀ + Σᵢ θᵢψᵢ(x₀)` with per-parameter (multiindex-box) θ-series truncation |
| `FluidNavierStokes` | Incompressible cylinder-flow DPIM: Taylor-Hood setup, Newton base flow, linearised operators, convection `FEMMultilinearMap` |
| `Common` | COMSOL `.mphtxt` mesh reading (`load_comsol_grid`) and Paraview/VTK export (`write_paraview_*`, activated by `using WriteVTK`) |

## Installation

Neither package is registered yet; install both from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MORFEproject/MORFE.jl.git")
Pkg.add(url="https://github.com/MORFEproject/MORFEFerrite.jl.git")
```

## Quick start

```julia
using MORFE, MORFEFerrite
SVK = MORFEFerrite.StructuralSVK

beam = SVK.mechanical_model("beam.msh";
    material  = SVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
    damping   = SVK.RayleighDamping(α = 5.4e-3, β = 1.9e-2),
    dirichlet = "Dirichlet")

rom = SVK.parametrise(beam; master = [1], order = 7)
SVK.print_equations(rom)
```

## Examples

Self-contained, runnable examples under [`examples/`](examples/) — each
bootstraps its own environment (clone MORFE.jl next to this repository, or set
`ENV["MORFE_PATH"]`):

| Folder | Model |
| ------ | ----- |
| [`01_clamped_beam_ferrite/`](examples/01_clamped_beam_ferrite/) | Clamped-clamped SVK beam — high-level UI and the fully explicit low-level pipeline |
| [`03_arch_comsol_wedge/`](examples/03_arch_comsol_wedge/) | Polysilicon arch, COMSOL P18 wedge mesh |
| [`04_parametric_clamped_beam/`](examples/04_parametric_clamped_beam/) | Two-parameter ROM (axial stretch + bending-mode arch) |
| [`05_karman_vortex_street/`](examples/05_karman_vortex_street/) | Kármán vortex street — Hopf bifurcation to a Stuart-Landau ROM |
| [`07_parametric_arch/`](examples/07_parametric_arch/) | Single-parameter sinusoidal arch |
| [`08_mems_micromirror/`](examples/08_mems_micromirror/) | MEMS scanning micromirror from CAD |

## Tests

```julia
using Pkg; Pkg.test("MORFEFerrite")   # small in-memory SVK equivalence gates
```

Full-mesh equivalence gates (high-level ≡ low-level ROM) run via the example
environment:

```bash
julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl
```

## License

MIT — see [LICENSE](LICENSE).
