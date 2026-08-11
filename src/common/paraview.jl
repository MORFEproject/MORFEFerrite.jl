"""
Module `ParaviewExport` — stubs for Paraview/VTK export of Ferrite grids.

The actual implementations live in `ext/MORFEFerriteWriteVTKExt.jl` and are
activated automatically when `WriteVTK` is loaded (`using WriteVTK`).
"""
module ParaviewExport

export write_paraview_mesh, write_paraview_modes, write_paraview_manifold,
	write_paraview_deformation

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
