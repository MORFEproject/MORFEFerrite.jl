# =====================================================================
# Observables on the reduced model.
#
# These sit next to `write_energy_gram` (energy_gram.jl), which already does the
# same job for kinetic energy: take a parametrisation and project it onto a
# functional of the flow, so a post-processing script never needs FOM-sized data.
# =====================================================================

using LinearAlgebra: dot

"""
	lift_functional(m::AssembledFluidModel) -> (l_free, L0)

The lift functional restricted to the DPIM free DOFs, and the base flow's own
lift.

`l_free` is the weight vector such that `l_freeᵀ · u′` is the perturbation lift;
`L0 = l_freeᵀ · s₀` is the steady contribution the perturbation adds to. Together
they give `L(t) = L0 + l_freeᵀ·u′(t)`.

Pressure traction only (`−p·n_y` on the cylinder), matching
[`compute_pressure_lift_weights`](@ref) — the viscous shear contribution is
deliberately omitted there, so it is omitted here too.
"""
function lift_functional(m::AssembledFluidModel)
	l_free = compute_pressure_lift_weights(m.fom)[m.fom.free_dpim]
	L0 = dot(l_free, real.(m.s₀_full[m.fom.free_dpim]))
	return l_free, L0
end

"""
	lift_polynomial(W, l_free) -> (L_coeffs, mset)

Project the lift weights onto the parametrisation: `L_α = lᵀ W_α`, one complex
coefficient per monomial, plus the monomial set they are indexed by.

The product is **bilinear, not sesquilinear** — `transpose`, not `adjoint`. `W`'s
coefficients are complex because the reduced coordinates are, but `l_free` is a
real functional of the flow; conjugating it would silently give the lift of the
conjugate manifold.

Evaluating the resulting polynomial at `(z, z̄, η′)` reproduces the perturbation
lift without touching a FOM-sized vector, which is what lets the Python
post-processing work from CSV alone.
"""
function lift_polynomial(W, l_free::AbstractVector)
	C = MORFE.ParametrisationMethod.coefficients(W)     # (FOM, ORD, L); ORD = 1 here
	W1 = @view(C[:, 1, :])                              # (FOM, L)
	L_coeffs = vec(transpose(W1) * l_free)              # (L,)
	return L_coeffs, MORFE.ParametrisationMethod.multiindex_set(W)
end


# ─────────────────────────────────────────────────────────────────────────────
# Observable writers — the CSV/serialised artefacts a run leaves behind for the
# plotting layer. Moved verbatim from examples/05_karman_vortex_street/exports.jl.
#
# `export_rom` and `export_matlab_model` were dropped on the way: the first is
# superseded by `MORFE.save_rom`, and the second drove a matcont/COCO exporter that was
# permanently disabled (`EXPORT_MATLAB = false`) and is now deleted.
#
# These take an `out::IO` for the human-readable log line. A notebook passes `stdout`.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Reduced-dynamics report (complex form, equation 1)
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_reduced_dynamics(out, data_dir, R, master_eigenvalues; re0, ord, nvar)

Write `reduced_dynamics.txt` with the first row ż₁ = R₁(z₁, z̄₁, η′) in complex form
(equation 2 is the conjugate, equation 3 the parameter η̇′ = 0) and echo the nonzero
monomials to `out`.
"""
function export_reduced_dynamics(out::IO, data_dir::AbstractString, R,
		master_eigenvalues; re0, ord, nvar)
	exps = R.poly.multiindex_set.exponents

	open(joinpath(data_dir, "reduced_dynamics.txt"), "w") do io
		println(io, "Kármán Vortex Street — Reduced Dynamics (complex form, equation 1)")
		@printf(io, "Re₀ = %.4f,  DPIM order = %d,  NVAR = %d\n", re0, ord, nvar)
		println(io, "")
		println(io, "Hopf eigenvalues:")
		for (i, λ) in enumerate(master_eigenvalues)
			@printf(io, "  λ[%d] = %+.10f %+.10f i\n", i, real(λ), imag(λ))
		end
		println(io, "")
		println(io, "ż₁ = R₁(z₁, z̄₁, η′) — nonzero monomials  [z₁-pow, z̄₁-pow, η′-pow]:")
		for m in eachindex(exps)
			c = R.poly.coefficients[1, m]
			abs(c) > 1e-14 || continue
			@printf(io, "  %-14s : %+.10e %+.10e·i\n", string(exps[m]), real(c), imag(c))
		end
		println(io, "")
		println(io, "Equation 2 is the complex conjugate; equation 3 is the parameter, η̇′ = 0.")
	end

	println(out, "\nReduced dynamics ż₁ = R₁ — nonzero monomials:")
	for m in eachindex(exps)
		c = R.poly.coefficients[1, m]
		abs(c) > 1e-12 || continue
		@printf(out, "  %-14s : %s\n", string(exps[m]), string(round(c; sigdigits = 6)))
	end
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Lift polynomial
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_lift_polynomial(out, data_dir, W, l_free, L0) -> L_coeffs

Project the lift weight vector onto the parametrisation, L_α = lᵀW_α (bilinear —
the adjoint would conjugate), serialise `lift_polynomial.jls`, and return the
coefficient vector for the CSV export.
"""
function export_lift_polynomial(out::IO, data_dir::AbstractString, W, l_free, L0)
	# The projection itself lives in FluidNavierStokes (observables.jl), next to
	# write_energy_gram — this function is only the serialisation around it.
	(L_coeffs, mset_l) = lift_polynomial(W, l_free)
	serialize(joinpath(data_dir, "lift_polynomial.jls"),
		(; L0 = L0, L_coeffs = L_coeffs, mset = mset_l))
	@printf(out, "  Lift polynomial: L0 = %.6f, %d coefficients\n", L0, length(L_coeffs))
	return L_coeffs
end

# ─────────────────────────────────────────────────────────────────────────────
# Coefficient CSVs (consumed by the example's figures.py)
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_lift_csv(out, data_dir, mset, L0, L_coeffs)

Write `L_coefficients.csv` (constant base-flow row + lift polynomial).
`R_coefficients.csv` is written by `MORFE.save_rom` in the driver.
"""
function export_lift_csv(out::IO, data_dir::AbstractString, mset, L0, L_coeffs)
	let
		lift_csv_path = joinpath(data_dir, "L_coefficients.csv")
		L_exps = mset.exponents
		n_L_rows = 0
		open(lift_csv_path, "w") do io
			header = join(["exp_$i" for i in 1:length(L_exps[1])], ",") * ",L_re,L_im"
			println(io, header)
			println(io, join(zeros(Int, length(L_exps[1])), ",") * ",$(L0),0.0")   # constant (base-flow lift)
			for (m, ex) in enumerate(L_exps)
				c = L_coeffs[m]
				abs(c) > 1e-14 || continue
				println(io, join(string.(Int.(ex)), ",") * ",$(real(c)),$(imag(c))")
				n_L_rows += 1
			end
		end
		println(out, "  L_coefficients.csv  ($n_L_rows polynomial rows, L0 = $(round(L0; sigdigits=6)))")
	end
	return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# VTK bundle + run summary
# ─────────────────────────────────────────────────────────────────────────────

"""
	export_vtk_bundle(data_dir, fom, s0_full, master_eigenvalues, all_eigenvalues, all_modes)

Serialise a plain-array bundle (mesh, DOF maps, base flow, eigenmodes — no Ferrite
types) for external ParaView/VTU export tooling.
"""
function export_vtk_bundle(data_dir::AbstractString, fom, s0_full,
		master_eigenvalues, all_eigenvalues, all_modes)
	_nn = Ferrite.getnnodes(fom.grid)
	vtk_data = (;
		node_coords = Float64[fom.grid.nodes[i].x[c] for c in 1:2, i in 1:_nn],
		cell_connectivity = [collect(Int32, c.nodes) for c in fom.grid.cells],
		cell_dofs = [collect(Ferrite.celldofs(cell)) for cell in Ferrite.CellIterator(fom.dh)],
		free_dpim = fom.free_dpim,
		ndofs_total = Ferrite.ndofs(fom.dh),
		n_vel_dofs_per_cell = fom.n_vel_dofs_per_cell,
		s0_free_dpim = s0_full[fom.free_dpim],
		master_eigenvalues = master_eigenvalues,
		all_eigenvalues = all_eigenvalues,
		all_modes = Matrix{ComplexF64}(all_modes),
	)
	serialize(joinpath(data_dir, "vtk_data.jls"), vtk_data)
	return nothing
end
