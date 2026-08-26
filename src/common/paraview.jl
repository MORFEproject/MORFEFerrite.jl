"""
Module `ParaviewExport` — stubs for Paraview/VTK export of Ferrite grids.

The actual implementations live in `ext/MORFEFerriteWriteVTKExt.jl` and are
activated automatically when `WriteVTK` is loaded (`using WriteVTK`).
"""
module ParaviewExport

export write_paraview_mesh, write_paraview_modes, write_paraview_manifold,
	write_paraview_deformation, write_paraview_p2p1,
	write_paraview_p2p1_phase_animation

const _NEED_VTK = "requires WriteVTK.jl. Load it with `using WriteVTK` to " *
	"activate the MORFEFerriteWriteVTKExt extension."

"""
    write_paraview_mesh(filename, grid; dh=nothing, prescribed_dofs=nothing)

Write the undeformed Ferrite mesh to `filename.vtu`.

When `dh` and `prescribed_dofs` (= `ch.prescribed_dofs`) are supplied, the
following point-data arrays are embedded so that clicking any node in Paraview
shows its metadata:

  node_id            — 1-indexed node number (Int32)
  x, y, z            — coordinates as named scalar fields (Float64)
  dof_x, dof_y, dof_z — free DOF indices (Int32, same as reduced K/M vectors); -1 = constrained
"""
write_paraview_mesh(args...; kwargs...) = error("write_paraview_mesh $_NEED_VTK")

"""
    write_paraview_p2p1(filename, fom; state=nothing, mode=nothing,
        mode_dofs=fom.free_dpim, normalize_mode=true, align_mode_phase=true,
        visual_amplitude=0.1,
        reynolds=nothing, eigenvalue=nothing, diagnostics=NamedTuple())

Write a two-dimensional Taylor--Hood P2/P1 solution or complex mode to
`filename.vtu`.

Unlike a geometry-only export, this constructs the six-node quadratic triangle
defined by the velocity space. Velocity is therefore written at its true P2
nodes; the P1 pressure is evaluated at those same points (corner values and
edge averages). The resulting file can be opened directly in ParaView.

- `state` is an optional full mixed velocity-pressure vector.
- `mode` may be a full vector or a reduced vector. A reduced vector is scattered
  through `mode_dofs` (normally `fom.free_dpim` or `fom.free`).
- a complex mode is phase-aligned and velocity-normalised by default. Real and
  imaginary parts, amplitudes, and two perturbed-state snapshots are exported.
- scalar vorticity is recovered at the P2 visualization nodes by area-weighted
  averaging of adjacent element gradients. The file contains steady vorticity,
  mode real/imaginary/amplitude, and steady plus `visual_amplitude` times each
  mode phase as `total_vorticity_real` and `total_vorticity_imag`.
- set `align_mode_phase=false` for a mode already phase-aligned by a continuation
  algorithm; amplitude normalisation can remain enabled independently.
- scalar `reynolds`, `eigenvalue`, and entries of `diagnostics` are stored as
  cell data for provenance.

The convenience method expects the fields returned by
`FluidNavierStokes.setup_fem`. A lower-level method accepting `grid`, `dh`, and
the local velocity/pressure DOF ranges is also available.
"""
write_paraview_p2p1(args...; kwargs...) = error("write_paraview_p2p1 $_NEED_VTK")

"""
    write_paraview_p2p1_phase_animation(filename, fom; state, mode,
        phase_frames=24, visual_amplitude=0.1, kwargs...)

Write a PVD animation of the diagnostic linear field
`state + visual_amplitude*real(exp(im*phase)*mode)`. Each VTU contains
`total_velocity` and `total_vorticity`.
"""
write_paraview_p2p1_phase_animation(args...; kwargs...) = error(
	"write_paraview_p2p1_phase_animation $_NEED_VTK")

"""
    write_paraview_modes(outdir, grid, dh, eigenvalues, Y, free; n_modes=10, prefix="mode")

Write the first `n_modes` eigenmodes to `outdir/mode_kk.vtu` and collect them
in a PVD file `outdir/modes.pvd` (one "time" frame per mode).

- `Y`: position eigenvectors, shape `(n_free, ORD, n_eig)` from `spectrum`.
- `free`: sorted vector of free (unconstrained) global DOF indices.
"""
write_paraview_modes(args...; kwargs...) = error("write_paraview_modes $_NEED_VTK")

"""
    write_paraview_manifold(outdir, grid, dh, W, free; alpha_indices=nothing)

Write manifold coefficient columns of `W` to individual `.vtu` files.
`alpha_indices` selects which monomials to export (all by default).
"""
write_paraview_manifold(args...; kwargs...) = error("write_paraview_manifold $_NEED_VTK")

"""
    write_paraview_deformation(outdir, grid, dh, W, free, r_amplitude;
                                theta=0.0, label="deformation")

Write a single `.vtu` at the backbone point `z₁ = r_amplitude · exp(i·theta)` with
two vector point-data fields:

  - `u_total`      — full deformation `Re[W(z₁, z̄₁)]`
  - `u_nonlinear`  — nonlinear part `Re[W(z₁, z̄₁) − W_lin(z₁, z̄₁)]`

`theta = 0` (default) gives a real displacement aligned with the eigenvector.
"""
write_paraview_deformation(args...; kwargs...) = error("write_paraview_deformation $_NEED_VTK")

end # module ParaviewExport
