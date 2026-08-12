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

