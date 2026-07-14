"""
    time_integration.jl — IMEX θ-method FOM integrator for the perturbation NSE.

Integrates

    B₁ ṡ = −B₀ s + f₂(s,s) + η′ K_visc s + η′ h₀_vec

in the free_dpim DOF subspace with an implicit-explicit θ-method:
  • linear part (−B₀ s + η′ K_visc s) : implicit with weight θ
  • quadratic convection f₂(s,s)      : explicit (evaluated at the current step)

The LHS matrix B₁/Δt + θ(B₀ − η′ K_visc) is constant per (η′, T, Δt) and is
factorised once with KLU.  `K_visc` and `h₀_vec` follow main.jl's convention:
both already scaled by −_CYL_D, and h₀ built from the rectangular free×ALL
block of K_raw times the FULL base flow (the prescribed inlet DOFs carry the
Poiseuille profile — see stage 5 of main.jl).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Element-level assembly of f₂(s, s) = −∫ φ · (u·∇u) dΩ
# ─────────────────────────────────────────────────────────────────────────────

"""
    eval_perturbation_convection!(accum, s_free, fom)

Assemble the quadratic perturbation convection into `accum` (length n_free_dpim):

    accum[k] += −∫_Ω  φᵢ · (∇u · u)  dΩ     (velocity DOFs k only)

where u is the velocity part of `s_free` extracted element-by-element via
`fom.free_to_local_dpim`.  `accum` is zeroed on entry.
"""
function eval_perturbation_convection!(
    accum::Vector{Float64},
    s_free::Vector{Float64},
    fom,
)
    fill!(accum, 0.0)

    n_dpc = ndofs_per_cell(fom.dh)
    n_vel = fom.n_vel_dofs_per_cell

    Fe = zeros(Float64, n_dpc)
    u_e = zeros(Float64, n_vel)

    for element in CellIterator(fom.dh)
        reinit!(fom.cv_vel, element)
        dofs = celldofs(element)
        vel_dofs = dofs[fom.dof_range_u]   # global velocity DOF indices for this cell

        # Extract velocity values from the free-DOF state vector
        fill!(u_e, 0.0)
        for (i, d) in enumerate(vel_dofs)
            li = get(fom.free_to_local_dpim, d, 0)
            li != 0 && (u_e[i] = s_free[li])
        end

        fill!(Fe, 0.0)
        for q in 1:getnquadpoints(fom.cv_vel)
            dΩ = getdetJdV(fom.cv_vel, q)
            u_q = function_value(fom.cv_vel, q, u_e)   # Vec{2,Float64}
            ∇u_q = function_gradient(fom.cv_vel, q, u_e)   # Tensor{2,2,Float64}

            # f₂(s,s) = −(u·∇)u;  in Ferrite: (∇u_q ⋅ u_q)[i] = Σⱼ ∂_j uᵢ · uⱼ = (u·∇u)ᵢ
            conv = -(∇u_q ⋅ u_q)   # Vec{2,Float64}

            for i in 1:n_vel
                ri = fom.dof_range_u[i]
                φᵢ = shape_value(fom.cv_vel, q, i)   # Vec{2,Float64}
                Fe[ri] += (φᵢ ⋅ conv) * dΩ
            end
        end

        # Scatter element residual into free-DOF accumulator
        for (r, d) in enumerate(dofs)
            li = get(fom.free_to_local_dpim, d, 0)
            li != 0 && (accum[li] += Fe[r])
        end
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# IMEX θ-method operators and orbit stepping
# ─────────────────────────────────────────────────────────────────────────────

"""
    build_imex_operators(B₀, B₁, K_visc, η_prime, T, Δt; θ = 0.5, γ = 0.0)
    -> (L_klu, RHS_M, Δt_exact, n_steps)

Assemble and KLU-factorise the θ-method operators for one orbit of period `T`,
with the step count rounded so that `n_steps · Δt_exact == T` exactly.
`γ > 0` adds artificial damping `γ B₁` to the implicit operator, shifting every
eigenvalue left by γ — used for transient suppression during spin-up.
"""
function build_imex_operators(B₀, B₁, K_visc, η_prime, T, Δt; θ = 0.5, γ = 0.0)
    n_steps = max(1, round(Int, T / Δt))
    Δt_exact = T / n_steps
    inv_dt = 1.0 / Δt_exact

    A_imp = B₀ .- η_prime .* K_visc
    γ > 0.0 && (A_imp = A_imp .+ γ .* B₁)

    LHS = inv_dt .* B₁ .+ θ .* A_imp
    RHS_M = inv_dt .* B₁ .- (1.0 - θ) .* A_imp

    return klu(LHS), RHS_M, Δt_exact, n_steps
end

"""
    integrate_orbit!(s, n_steps, η_prime, fom, L_klu, RHS_M, h₀_vec, f2, rhs;
                     on_step = nothing) -> Bool

Advance `s` in place by `n_steps` IMEX steps:

    (B₁/Δt + θ A) s⁺ = (B₁/Δt − (1−θ) A) s + f₂(s,s) + η′ h₀_vec

`f2` and `rhs` are pre-allocated work vectors (length of `s`).  If given,
`on_step(step, s)` is called after every accepted step (lift recording, state
storage, norm tracking).  Returns `false` on blow-up (non-finite state).
"""
function integrate_orbit!(
    s::Vector{Float64},
    n_steps::Int,
    η_prime::Float64,
    fom,
    L_klu,
    RHS_M::AbstractMatrix,
    h₀_vec::Vector{Float64},
    f2::Vector{Float64},
    rhs::Vector{Float64};
    on_step = nothing,
)
    for step in 1:n_steps
        eval_perturbation_convection!(f2, s, fom)
        mul!(rhs, RHS_M, s)
        axpy!(1.0, f2, rhs)
        axpy!(η_prime, h₀_vec, rhs)
        ldiv!(s, L_klu, rhs)
        isfinite(dot(s, s)) || return false
        on_step === nothing || on_step(step, s)
    end
    return true
end
