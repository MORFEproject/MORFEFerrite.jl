"""
	fluid_maps.jl — FEMMultilinearMap and MultilinearMap terms for the cylinder-flow DPIM.

Three nonlinear terms are implemented:

1. FluidConvection — quadratic convective term f₂(s, s):
	   f₂(s₁, s₂) = −½[(u₁·∇)u₂ + (u₂·∇)u₁]  (half-symmetrised, so f₂(s, s) = −(u·∇)u)
   Implements FEMMultilinearMap{1} with multiindex=(2,), multiplicity_external=0.

2. make_param_coupling — parametric viscous coupling g₁(s, η′):
	   g₁(s, η′) = −D·η′ · K_raw · u′
   A simple MultilinearMap{1} backed by a pre-assembled sparse matrix; `fluid_model`
   passes K_visc = −D·K_raw, the raw stiffness scaled by −D, D = fom.reference_length.
   multiindex=(1,), multiplicity_external=1.

3. make_base_forcing — base-flow parametric forcing h₀(η′):
	   h₀(η′) = −D·η′ · K_raw[free, ALL] · s₀
   A constant-in-state MultilinearMap{1} that provides the forcing direction when
   Re departs from Re₀.  multiindex=(0,), multiplicity_external=1.
   The rectangular free×ALL block is essential: the full base flow s₀ carries the
   Poiseuille profile on the prescribed inlet DOFs.
   This term drives the external direction Φ_ext in the cohomological solve: without
   it the (0,0,1) monomial has zero RHS and Φ_ext is computed incorrectly, making all
   η′-dependent reduced-dynamics coefficients wrong.
"""

using Ferrite
using LinearAlgebra
using SparseArrays
using MORFE

# Common element kernel used by both the ordinary moved-mesh convection action
# and the parametric pullback.  Keeping the symmetric bilinear form here makes
# their signs, factors of two, prescribed-DOF treatment, and complex arithmetic
# identical by construction.
@inline function _fluid_state_value(state, dof, free, ::Val{:free}, T)
    return free == 0 ? zero(T) : state[free]
end

@inline _fluid_state_value(state, dof, _, ::Val{:full}, T) = T(state[dof])

function _accumulate_fluid_convection_pair!(accum, u1, u2, fom,
        adjugate_at, factor, layout1::Val = Val(:free), layout2::Val = Val(:free),
        same_state::Val{SAME} = Val(false)) where {SAME}
    T = promote_type(eltype(accum), eltype(u1), eltype(u2), typeof(factor))
    ue1 = zeros(T, fom.n_vel_dofs_per_cell)
    ue2 = SAME ? ue1 : similar(ue1)
    Fe = zeros(T, ndofs_per_cell(fom.dof_handler))
    for (ci, cell) in enumerate(CellIterator(fom.dof_handler))
        reinit!(fom.velocity_cell_values, cell)
        dofs = celldofs(cell)
        for (local_index, dof) in enumerate(dofs[fom.dof_range_u])
            free = get(fom.dof_to_free_dpim, dof, 0)
            ue1[local_index] = _fluid_state_value(u1, dof, free, layout1, T)
            SAME || (ue2[local_index] = _fluid_state_value(u2, dof, free, layout2, T))
        end
        fill!(Fe, zero(T))
        for q in 1:getnquadpoints(fom.velocity_cell_values)
            dOmega = getdetJdV(fom.velocity_cell_values, q)
            A = adjugate_at(ci, q)
            u1q = function_value(fom.velocity_cell_values, q, ue1)
            grad1 = function_gradient(fom.velocity_cell_values, q, ue1)
            if !isnothing(A)
                grad1 = grad1 ⋅ A
            end
            convection = if SAME
                grad1 ⋅ u1q
            else
                u2q = function_value(fom.velocity_cell_values, q, ue2)
                grad2 = function_gradient(fom.velocity_cell_values, q, ue2)
                isnothing(A) || (grad2 = grad2 ⋅ A)
                0.5 * (grad2 ⋅ u1q + grad1 ⋅ u2q)
            end
            for local_index in 1:fom.n_vel_dofs_per_cell
                row = fom.dof_range_u[local_index]
                Fe[row] -= factor *
                           (shape_value(fom.velocity_cell_values, q, local_index) ⋅
                            convection) * dOmega
            end
        end
        for (local_index, dof) in enumerate(dofs)
            row = get(fom.dof_to_free_dpim, dof, 0)
            row == 0 || (accum[row] += Fe[local_index])
        end
    end
    return accum
end

"""Internal symmetric bilinear convection action on an ordinary physical mesh."""
function _eval_perturbation_convection_pair!(accum, u1, u2, fom)
    fill!(accum, zero(eltype(accum)))
    if u1 === u2
        return _accumulate_fluid_convection_pair!(accum, u1, u2, fom,
            (ci, q) -> nothing, one(eltype(accum)), Val(:free), Val(:free),
            Val(true))
    end
    return _accumulate_fluid_convection_pair!(accum, u1, u2, fom,
        (ci, q) -> nothing, one(eltype(accum)))
end

"""Internal bilinear convection action with independently selected state layouts."""
function _eval_convection_pair_with_lifting!(accum, u1, u2, fom,
        layout1::Val, layout2::Val)
    fill!(accum, zero(eltype(accum)))
    if u1 === u2 && layout1 === layout2
        return _accumulate_fluid_convection_pair!(accum, u1, u2, fom,
            (ci, q) -> nothing, one(eltype(accum)), layout1, layout2, Val(true))
    end
    return _accumulate_fluid_convection_pair!(accum, u1, u2, fom,
        (ci, q) -> nothing, one(eltype(accum)), layout1, layout2)
end

# ─────────────────────────────────────────────────────────────────────────────
# Quadrature-point value: velocity and its gradient at one quadrature point
# ─────────────────────────────────────────────────────────────────────────────

"""
	FluidVelQP{T}

The quadrature-point value of `FluidConvection`: the velocity and its gradient at one
quadrature point, for one coefficient of the parametrisation. `scatter_qp!` writes it and
`accumulate_qp!` reads it.

# Fields
- `value::Vec{2, T}`: the velocity `(u_x, u_y)`.
- `gradient::Tensor{2, 2, T, 4}`: the velocity gradient, `gradient[i, j] = ∂_j u_i`.
  Concrete, so a buffer of these stores them inline rather than boxed.
"""
struct FluidVelQP{T}
    value::Vec{2, T}
    gradient::Tensor{2, 2, T, 4}
end

# ─────────────────────────────────────────────────────────────────────────────
# FluidConvection — FEMMultilinearMap{1} for f₂(s, s) = −(u·∇)u symmetrised
# ─────────────────────────────────────────────────────────────────────────────

"""
	FluidConvection{DH, CV} <: MORFE.FEMMultilinearMap{1}

FEM-backed quadratic convective term for 2D incompressible flow.

Assembles the half-symmetrised bilinear form (so that f₂(s, s) = −∫ φ·(u·∇)u dΩ):
	f₂(s₁, s₂) = −½ ∫ φ · [(u₁·∇)u₂ + (u₂·∇)u₁] dΩ
	 = −½ ∫ φ · [∇u₂ · u₁ + ∇u₁ · u₂] dΩ

where ∇u[i,j] = ∂_j u_i so that (∇u · v)[i] = Σ_j ∂_j u_i · v_j = (v·∇u)_i.

The state `s` holds velocity and pressure DOFs; only the velocity enters the form, and only
the velocity rows of the element vector are written. `DH` and `CV` are the concrete types
of the DOF handler and the velocity cell values.

# Fields
- `dof_handler`: the Ferrite `DofHandler` with velocity `:u` and pressure `:p`.
- `velocity_cell_values`: the Ferrite `CellValues` of the velocity field.
- `dof_to_free`: global DOF number → index in the free (perturbation) state vector, `0` at
  a constrained DOF; a dense vector, read on every gather and assembly.
- `n_free`: the number of free DOFs, the length of the state vector.
- `dof_range_u`: the local positions of the velocity DOFs within a cell's DOF vector.
- `multiindex`, `multiplicity_external`, `deg`, `fully_asymmetric`: MORFE's term metadata,
  `(2,)`, `0`, `2`, `false`.
- `qp_buffer`, `(2, n_qp)`: the arguments of the direct evaluation `MORFE.evaluate_term!`;
  MORFE's single-pass assembly keeps its own.
- `element_velocity`: one coefficient's velocity DOF values on the current cell.
"""
struct FluidConvection{DH, CV} <: MORFE.FEMMultilinearMap{1}
    dof_handler::DH
    velocity_cell_values::CV
    dof_to_free::Vector{Int}
    n_free::Int
    dof_range_u::UnitRange{Int}
    multiindex::NTuple{1, Int}
    multiplicity_external::Int
    deg::Int
    fully_asymmetric::Union{Nothing, Bool}
    qp_buffer::Matrix{FluidVelQP{ComplexF64}}
    element_velocity::Vector{ComplexF64}
end

"""
	FluidConvection(fom)

Construct a `FluidConvection` term from the discretisation `fom` returned by
[`setup_fem`](@ref). It reads `dof_handler`, `velocity_cell_values`, `dof_range_u` and the
perturbation free-DOF map `dof_to_free_dpim` / `n_free_dpim`, so the term acts on the state
vector of the reduction.
"""
function FluidConvection(fom)
    n_qp = getnquadpoints(fom.velocity_cell_values)
    zero_qp = FluidVelQP(zero(Vec{2, ComplexF64}), zero(Tensor{2, 2, ComplexF64}))
    dof_to_free = zeros(Int, ndofs(fom.dof_handler))
    for (dof, free) in fom.dof_to_free_dpim
        dof_to_free[dof] = free
    end
    return FluidConvection(
        fom.dof_handler, fom.velocity_cell_values, dof_to_free, fom.n_free_dpim,
        fom.dof_range_u, (2,), 0, 2, false,
        fill(zero_qp, 2, n_qp), zeros(ComplexF64, fom.n_vel_dofs_per_cell))
end

# ── FEMMultilinearMap interface ────────────────────────────────────────────

# The element loop, for MORFE's single-pass assembly: DOF handler and velocity cell values by
# identity, the free-DOF map by content.
function MORFE.fem_element_loop(term::FluidConvection)
    (term.dof_handler, term.velocity_cell_values, term.dof_to_free)
end
MORFE.fem_elements(term::FluidConvection) = CellIterator(term.dof_handler)
MORFE.fem_n_qp(term::FluidConvection) = getnquadpoints(term.velocity_cell_values)
MORFE.fem_ndofs_per_cell(term::FluidConvection) = ndofs_per_cell(term.dof_handler)
MORFE.fem_qp_buffer(term::FluidConvection) = term.qp_buffer
function MORFE.fem_getdetJdV(_element, q, term::FluidConvection)
    getdetJdV(term.velocity_cell_values, q)
end
function MORFE.fem_reinit!(element, term::FluidConvection)
    reinit!(term.velocity_cell_values, element)
end

"""
	MORFE.scatter_qp!(qp_values, coefficient, element, term::FluidConvection)

Gather the velocity DOFs of `coefficient`, a vector over the free DOFs, on `element`
(zero at constrained DOFs), then store the velocity value and gradient at each quadrature
point as a `FluidVelQP` in `qp_values[q]`. Pressure DOFs do not enter the convective
nonlinearity and are not read.
"""
function MORFE.scatter_qp!(qp_values, coefficient, element, term::FluidConvection)
    dofs = celldofs(element)
    u_e = term.element_velocity
    for (i, local_dof) in enumerate(term.dof_range_u)
        free = term.dof_to_free[dofs[local_dof]]
        u_e[i] = free == 0 ? zero(ComplexF64) : coefficient[free]
    end
    for q in eachindex(qp_values)
        qp_values[q] = FluidVelQP(
            function_value(term.velocity_cell_values, q, u_e),
            function_gradient(term.velocity_cell_values, q, u_e)
        )
    end
end

"""
	MORFE.accumulate_qp!(Fe, qp_args::NTuple{2}, multiplier, element, q, dΩ, term::FluidConvection)

Convective integrand at one quadrature point, with `qp_args = (qp1, qp2)` two `FluidVelQP`:

	Fe[k] += multiplier · φᵢ · (−½)[∇u₂·u₁ + ∇u₁·u₂] · dΩ   for velocity DOF k

∇u ⋅ v computes (v·∇u) via Tensor{2,2} × Vec{2} → Vec{2} contraction:
	(∇u ⋅ v)[i] = Σ_j ∂_j u_i · v_j = (v·∇u)_i
which is exactly the (v·∇)u material-derivative direction.
"""
function MORFE.accumulate_qp!(
        Fe, qp_args::NTuple{2, FluidVelQP{ComplexF64}}, multiplier, _element, q, dΩ,
        term::FluidConvection
)
    qp1, qp2 = qp_args

    # Symmetric bilinear form f₂(s₁,s₂) satisfying f₂(s,s) = f(s) = −(u·∇u).
    # MORFE assumes fully_asymmetric=false → calls f₂(W_col₁, W_col₂) once and
    # multiplies by the multinomial coefficient (e.g. ×2 for the z₁z₂ cross-term).
    # Therefore f₂ must be the HALF-symmetrised form:
    #   f₂(u₁,u₂) = −½[(u₁·∇)u₂ + (u₂·∇)u₁]
    # Check: f₂(u,u) = −½[2(u·∇u)] = −(u·∇u) ✓
    conv = -0.5 * (qp2.gradient ⋅ qp1.value + qp1.gradient ⋅ qp2.value)   # Vec{2}

    c = multiplier * dΩ

    # Velocity rows only; the pressure rows of Fe keep their initial zero.
    for (i, row) in enumerate(term.dof_range_u)
        φᵢ = shape_value(term.velocity_cell_values, q, i)   # Vec{2, Float64}
        Fe[row] += c * (φᵢ ⋅ conv)
    end
end

"""
	MORFE.assemble_element!(result, Fe, element, term::FluidConvection)

Add the element vector `Fe` to `result`, a vector over the free DOFs. The pressure entries
of `Fe` are zero, so only velocity contributions are added.
"""
function MORFE.assemble_element!(result, Fe, element, term::FluidConvection)
    dofs = celldofs(element)
    for (r, d) in enumerate(dofs)
        free = term.dof_to_free[d]
        free != 0 && (result[free] += Fe[r])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# FluidParamCoupling — MultilinearMap{1} for g₁(s, η′) = η′ · K_visc · s
# ─────────────────────────────────────────────────────────────────────────────

"""
	make_param_coupling(K_visc_free) -> MultilinearMap{1}

Create a `MultilinearMap{1}` for the parametric viscous coupling

	g₁(s, η′) = η′ · K_visc · u′        (`fluid_model` passes K_visc = −D·K_raw)

with multiindex=(1,) and multiplicity_external=1 (one state input, one η′ input).

K_visc_free is the viscosity stiffness restricted to free DOFs, pre-scaled by −D
(D = fom.reference_length) in `fluid_model` so that the term equals
−D·η′·K_raw·u′ = −(ν−ν₀)·K_raw·u′.
Its pressure rows and columns are zero, so the multiplication correctly
targets only velocity DOFs in both input and output.

MORFE passes the external argument as a basis direction — the unit SVector{1,Int}([1])
here, since this external system is diagonal.  The actual η′ scaling is handled by the
polynomial factorisation in the cohomological solver.  Hence f!(accum, s, r_ext) adds
r_ext[1] · K_visc · s = K_visc · s to accum.

Do not assume the argument is integer-valued: when an external system's linear matrix is
not upper triangular, `ExternalSystem` re-bases it and MORFE then passes the columns of the
change of basis, which are complex in general.  The `eltype(accum)` conversion below is
what keeps this term correct in that case.
"""
function make_param_coupling(K_visc_free::SparseMatrixCSC)
    K = K_visc_free   # close over the matrix

    # deg = 1 (state) + 1 (external) = 2;  f! takes (accum, s, r_ext) → nargs = 4
    function g₁!(accum, s, r_ext)
        mul!(accum, K, s, eltype(accum)(r_ext[1]), one(eltype(accum)))
    end

    return MultilinearMap(g₁!, (1,), 1; fully_asymmetric = false)
end

"""
	make_base_forcing(h₀_vec_free) -> MultilinearMap{1}

Create a `MultilinearMap{1}` for the base-flow parametric forcing

	h₀(η′) = −D·η′ · K_raw[free, ALL] · s₀

with multiindex=(0,) and multiplicity_external=1 (no state input, one η′ input).

h₀_vec_free = −D · K_raw[free, ALL] · s₀_full is a fixed vector encoding the
viscous forcing direction when Re departs from Re₀: the rectangular block acts on
the FULL base-flow vector, whose prescribed inlet DOFs carry the Poiseuille
profile.  It determines the external direction Φ_ext via the cohomological
equation for monomial (0,0,1).

MORFE passes the external argument as a basis direction — the unit SVector{1,Int}([1])
here, since this external system is diagonal — so f!(accum, r_ext) adds r_ext[1] · h₀_vec
to accum (η′ scaling is handled by the polynomial factorisation framework).

Do not assume the argument is integer-valued: a re-based external system (one whose linear
matrix was not upper triangular) is passed the columns of its change of basis, which are
complex in general.  The `eltype(accum)` conversion below is what keeps this term correct
in that case.
"""
function make_base_forcing(h₀_vec_free::AbstractVector)
    h₀ = ComplexF64.(h₀_vec_free)   # close over a complex copy
    function h₀!(accum, r_ext)
        axpy!(eltype(accum)(r_ext[1]), h₀, accum)
    end
    return MultilinearMap(h₀!, (0,), 1; fully_asymmetric = false)
end

# ─────────────────────────────────────────────────────────────────────────────
# Assemble the raw viscosity stiffness (without 1/Re factor)
# ─────────────────────────────────────────────────────────────────────────────

"""
	assemble_K_visc(fom) -> (K_visc_free, K_visc_rect)

Assemble the raw viscosity stiffness matrix K_visc (full DOF space):

	K_visc_kl = ∫ 2 ε(φ^k) : ε(φ^l) dΩ    (velocity test and trial, P2)

Pressure DOF rows and columns are zero.  Two restrictions are returned:

  K_visc_free — square free×free block, for the parametric coupling g₁ acting on
				the perturbation (which vanishes on prescribed DOFs);
  K_visc_rect — rectangular free×ALL block, for the base-flow forcing h₀ = K·u₀,
				where u₀ is nonzero on the prescribed inlet DOFs (Poiseuille).
"""
function assemble_K_visc(fom)
    K_full = allocate_matrix(fom.dof_handler)
    fill!(K_full, 0.0)

    n_dpc = ndofs_per_cell(fom.dof_handler)
    Ke = zeros(n_dpc, n_dpc)
    n_vel = fom.n_vel_dofs_per_cell
    asm = start_assemble(K_full)

    for element in CellIterator(fom.dof_handler)
        fill!(Ke, 0.0)
        reinit!(fom.velocity_cell_values, element)

        for q in 1:getnquadpoints(fom.velocity_cell_values)
            dΩ = getdetJdV(fom.velocity_cell_values, q)
            for i in 1:n_vel
                ri = fom.dof_range_u[i]
                ∇φᵢ = shape_gradient(fom.velocity_cell_values, q, i)
                for j in 1:n_vel
                    rj = fom.dof_range_u[j]
                    ∇φⱼ = shape_gradient(fom.velocity_cell_values, q, j)
                    Ke[ri, rj] += 2 * (symmetric(∇φᵢ) ⊡ symmetric(∇φⱼ)) * dΩ
                end
            end
        end

        assemble!(asm, celldofs(element), Ke)
    end

    K_visc_free = K_full[fom.free_dpim, fom.free_dpim]
    K_visc_rect = K_full[fom.free_dpim, :]
    @info "K_visc assembled: $(nnz(K_visc_free)) nonzeros in free subspace"
    return K_visc_free, K_visc_rect
end
