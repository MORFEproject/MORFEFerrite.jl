"""
Ferrite.jl FEM backend for MORFE.jl St. Venant-Kirchhoff elasticity.

Implements a three-dimensional `MORFE.FEMMultilinearMap{2}` for the quadratic
and cubic geometric internal forces and assembles the linear stiffness and mass.
Both isotropic Lamé elasticity and a general anisotropic Voigt stiffness use the
same Green-Lagrange strain expansion.

SVK internal virtual work:
	δW_int = ∫ S ⊡ δE dΩ
where:
	E  = ε(u) + 0.5∇uᵀ∇u
	S  = σ(E)
	δE = ε(δu) + sym(∇uᵀ∇δu).

For isotropy, `σ(E) = λ tr(E) I + 2μE`; the anisotropic path applies its Voigt
matrix instead. Equivalently, with the first Piola-Kirchhoff stress
`P = (I + ∇u) S`, `δW_int = ∫ P ⊡ ∇δu dΩ`. Expansion in `u` gives linear,
quadratic, and cubic internal forces; the kernels below form the quadratic and
cubic parts of `P` once per quadrature point. The multilinear maps return the
negatives of the nonlinear internal terms, matching MORFE's
`linear = nonlinear` model convention.
"""

# ── Stress models ───────────────────────────────────────────────────────────
# `_σ(ε, model)` maps a symmetric strain tensor, real or complex, to the 2nd
# Piola-Kirchhoff stress. The isotropic model routes to the original Lamé
# expression unchanged.
abstract type AbstractStress end

struct IsotropicStress <: AbstractStress
	λ::Float64
	μ::Float64
end

"""
	VoigtStress(D)

Fully anisotropic stress, `σ = D : ε`, with `D` the 6×6 Voigt stiffness in the
ordering `[11, 22, 33, 23, 13, 12]`. The input strain tensor is converted to the
engineering-shear vector `[ε₁₁, ε₂₂, ε₃₃, 2ε₂₃, 2ε₁₃, 2ε₁₂]`; the result is
returned as a symmetric second Piola-Kirchhoff stress tensor.
"""
struct VoigtStress <: AbstractStress
	D::SMatrix{6, 6, Float64, 36}
end

# Lamé stress from a strain tensor.
@inline _σ(E, λ, μ) = λ * tr(E) * one(E) + 2μ * E

@inline _σ(E, s::IsotropicStress) = _σ(E, s.λ, s.μ)

@inline function _σ(ε, s::VoigtStress)
	D = s.D
	e1 = ε[1, 1];
	e2 = ε[2, 2];
	e3 = ε[3, 3]
	e4 = 2 * ε[2, 3];
	e5 = 2 * ε[1, 3];
	e6 = 2 * ε[1, 2]
	s1 = D[1, 1]*e1 + D[1, 2]*e2 + D[1, 3]*e3 + D[1, 4]*e4 + D[1, 5]*e5 + D[1, 6]*e6
	s2 = D[2, 1]*e1 + D[2, 2]*e2 + D[2, 3]*e3 + D[2, 4]*e4 + D[2, 5]*e5 + D[2, 6]*e6
	s3 = D[3, 1]*e1 + D[3, 2]*e2 + D[3, 3]*e3 + D[3, 4]*e4 + D[3, 5]*e5 + D[3, 6]*e6
	s4 = D[4, 1]*e1 + D[4, 2]*e2 + D[4, 3]*e3 + D[4, 4]*e4 + D[4, 5]*e5 + D[4, 6]*e6
	s5 = D[5, 1]*e1 + D[5, 2]*e2 + D[5, 3]*e3 + D[5, 4]*e4 + D[5, 5]*e5 + D[5, 6]*e6
	s6 = D[6, 1]*e1 + D[6, 2]*e2 + D[6, 3]*e3 + D[6, 4]*e4 + D[6, 5]*e5 + D[6, 6]*e6
	# SymmetricTensor{2,3} takes the lower triangle column-major: (11, 21, 31, 22, 32, 33)
	return SymmetricTensor{2, 3}((s1, s6, s5, s2, s4, s3))
end

# -----------------------------------------------------------------------
# Concrete FEMMultilinearMap type
# -----------------------------------------------------------------------

"""
    FerriteGeometricNonlinearity{DEG, DH, CV, S} <: MORFE.FEMMultilinearMap{2}

Three-dimensional FEM-backed multilinear term for the negative SVK geometric
internal force.

- `DEG = 2`: quadratic form with two displacement inputs.
- `DEG = 3`: cubic form with three displacement inputs.

The `MORFE.FEMMultilinearMap{2}` supertype places the term in a second-order
model. It uses only position inputs, with `multiindex == (DEG, 0)`, and has no
external-state factors. `S` is the concrete isotropic or anisotropic stress law.

`dh`, `cv`, `free_to_local`, and `n_free` describe the constrained Ferrite
discretisation. The remaining storage is reusable evaluation state: the
quadrature-gradient buffer has size `(max_unique_cols, getnquadpoints(cv))`, and
the element vectors have `ndofs_per_cell(dh)` entries. A term instance is
mutable through these buffers and is intended for one evaluation sweep at a time.
"""
struct FerriteGeometricNonlinearity{DEG, DH, CV, S} <: MORFE.FEMMultilinearMap{2}
	dh::DH
	cv::CV
	free_to_local::Dict{Int, Int}
	n_free::Int
	stress::S                      # IsotropicStress or VoigtStress (see below)
	multiindex::NTuple{2, Int}
	multiplicity_external::Int
	deg::Int
	fully_asymmetric::Union{Nothing, Bool}
	∇W_qp::Matrix{Tensor{2, 3, ComplexF64, 9}}   # concrete: Tensor{2, 3, ComplexF64} boxes
	Fe::Vector{ComplexF64}
	u_e::Vector{ComplexF64}
end

"""
    FerriteGeometricNonlinearity{DEG}(dh, cv, free_to_local, n_free, λ, μ;
                                       max_unique_cols = DEG,
                                       fully_asymmetric = false)
    FerriteGeometricNonlinearity{DEG}(dh, cv, free_to_local, n_free, stress;
                                       max_unique_cols = DEG,
                                       fully_asymmetric = false)

Construct a degree-`DEG` term from Lamé constants or an internal stress-law
object. `free_to_local` maps global cell DOFs to the free state. Buffers are
preallocated from `dh`, `cv`, and `max_unique_cols`; the latter must cover the
largest column batch evaluated by MORFE. `fully_asymmetric` is stored with the
same symmetry-policy meaning as on `MORFE.MultilinearMap`.
"""
function FerriteGeometricNonlinearity{DEG}(
	dh::DH, cv::CV,
	free_to_local::Dict{Int, Int}, n_free::Int,
	λ::Real, μ::Real;
	max_unique_cols::Int = DEG,
	fully_asymmetric::Union{Nothing, Bool} = false) where {DEG, DH, CV}
	return FerriteGeometricNonlinearity{DEG}(dh, cv, free_to_local, n_free,
		IsotropicStress(Float64(λ), Float64(μ));
		max_unique_cols = max_unique_cols, fully_asymmetric = fully_asymmetric)
end

function FerriteGeometricNonlinearity{DEG}(
	dh::DH, cv::CV,
	free_to_local::Dict{Int, Int}, n_free::Int,
	stress::S;
	max_unique_cols::Int = DEG,
	fully_asymmetric::Union{Nothing, Bool} = false) where {DEG, DH, CV, S <: AbstractStress}
	n_qp = getnquadpoints(cv)
	n_dofs = ndofs_per_cell(dh)
	∇W_qp = Matrix{Tensor{2, 3, ComplexF64, 9}}(undef, max_unique_cols, n_qp)
	Fe = Vector{ComplexF64}(undef, n_dofs)
	u_e = Vector{ComplexF64}(undef, n_dofs)
	return FerriteGeometricNonlinearity{DEG, DH, CV, S}(
		dh, cv, free_to_local, n_free, stress,
		(DEG, 0), 0, DEG, fully_asymmetric, ∇W_qp, Fe, u_e)
end

# -----------------------------------------------------------------------
# FEMMultilinearMap interface
# -----------------------------------------------------------------------

MORFE.fem_elements(t::FerriteGeometricNonlinearity) = CellIterator(t.dh)

MORFE.fem_n_qp(t::FerriteGeometricNonlinearity) = getnquadpoints(t.cv)

MORFE.fem_ndofs_per_cell(t::FerriteGeometricNonlinearity) = ndofs_per_cell(t.dh)

MORFE.fem_qp_buffer(t::FerriteGeometricNonlinearity) = t.∇W_qp

MORFE.fem_getdetJdV(_element, q, t::FerriteGeometricNonlinearity) = getdetJdV(t.cv, q)

# Called once per element in _replay_fem_split! before the scatter loop (O1).
MORFE.fem_reinit!(element, t::FerriteGeometricNonlinearity) = reinit!(t.cv, element)

"""
	MORFE.scatter_qp!(∇W_col, W_free, element, t)

Scatter `W_free` into the cell DOF ordering, inserting zero at constrained DOFs,
then write the complex displacement gradient at each quadrature point to
`∇W_col[q]`. `t.cv` must already have been reinitialized for `element` through
`MORFE.fem_reinit!`.
"""
function MORFE.scatter_qp!(∇W_col, W_free, element, t::FerriteGeometricNonlinearity)
	dofs = celldofs(element)
	u_e = t.u_e
	for (i, d) in enumerate(dofs)
		local_idx = get(t.free_to_local, d, 0)
		u_e[i] = local_idx == 0 ? zero(ComplexF64) : W_free[local_idx]
	end
	for q in eachindex(∇W_col)
		∇W_col[q] = function_gradient(t.cv, q, u_e)
	end
end



# Symmetric Green-Lagrange cross term for two gradients.
@inline _E_nl(∇u1, ∇u2) = symmetric(
	Tensor{2, 3}(0.25 * (transpose(∇u1) ⋅ ∇u2 + transpose(∇u2) ⋅ ∇u1)))

"""
	MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{2}, mult, element, q, dΩ, t)

Quadratic geometric nonlinearity integrand at one quadrature point:

	Fe_r -= mult * ∇φ_r ⊡ P₂(∇u1, ∇u2) * dΩ
	P₂(∇u1, ∇u2) = σ(E_nl(∇u1, ∇u2)) + ½(∇u1 ⋅ σ(ε(∇u2)) + ∇u2 ⋅ σ(ε(∇u1)))

`P₂` is the polarised quadratic part of the first Piola-Kirchhoff stress `(I + ∇u) σ(E)`,
with `E_nl(A, B) = sym(0.25(AᵀB + BᵀA))`. It equals the quadratic part of `S ⊡ δE` term by
term, because `sym(Aᵀ∇φ) ⊡ σ = ∇φ ⊡ (A ⋅ σ)` for a symmetric `σ`. Formed once per quadrature
point, it leaves one double contraction per test function. The gradients are complex and
the form must stay multilinear, so nothing here conjugates: `E_nl` writes `transpose`, and
Tensors.jl's `symmetric`, `⋅` and `⊡` do not conjugate either. The negative sign moves the
nonlinear internal force to MORFE's right-hand side.
"""
function MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{2}, mult, _element, q, dΩ,
	t::FerriteGeometricNonlinearity{2})
	∇u1, ∇u2 = ∇W_args
	s = t.stress
	P = _σ(_E_nl(∇u1, ∇u2), s) +
		0.5 * (∇u1 ⋅ _σ(symmetric(∇u2), s) + ∇u2 ⋅ _σ(symmetric(∇u1), s))
	c = mult * dΩ
	for r in 1:ndofs_per_cell(t.dh)
		Fe[r] -= c * (shape_gradient(t.cv, q, r) ⊡ P)
	end
end

"""
	MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{3}, mult, element, q, dΩ, t)

Cubic geometric nonlinearity integrand at one quadrature point:

	Fe_r -= mult * ∇φ_r ⊡ P₃(∇u1, ∇u2, ∇u3) * dΩ
	P₃ = ⅓ Σ_{(i,j,k) cyclic} ∇ui ⋅ σ(E_nl(∇uj, ∇uk))

`P₃` is the polarised cubic part of the first Piola-Kirchhoff stress, formed once per
quadrature point as in the quadratic method. The negative sign moves the nonlinear internal
force to MORFE's right-hand side.
"""
function MORFE.accumulate_qp!(Fe, ∇W_args::NTuple{3}, mult, _element, q, dΩ,
	t::FerriteGeometricNonlinearity{3})
	∇u1, ∇u2, ∇u3 = ∇W_args
	s = t.stress
	P = (∇u1 ⋅ _σ(_E_nl(∇u2, ∇u3), s) + ∇u2 ⋅ _σ(_E_nl(∇u1, ∇u3), s) +
		 ∇u3 ⋅ _σ(_E_nl(∇u1, ∇u2), s)) / 3
	c = mult * dΩ
	for r in 1:ndofs_per_cell(t.dh)
		Fe[r] -= c * (shape_gradient(t.cv, q, r) ⊡ P)
	end
end

"""
	MORFE.assemble_element!(accum, Fe, element, t)

Add the element residual `Fe`, indexed in cell-DOF order, to the free-state
accumulator `accum`. Entries corresponding to constrained global DOFs are skipped.
The function mutates `accum` and returns `nothing`.
"""
function MORFE.assemble_element!(accum, Fe, element, t::FerriteGeometricNonlinearity)
	dofs = celldofs(element)
	for (r, d) in enumerate(dofs)
		local_idx = get(t.free_to_local, d, 0)
		local_idx != 0 && (accum[local_idx] += Fe[r])
	end
end

# -----------------------------------------------------------------------
# Linear matrix assembly
# -----------------------------------------------------------------------

"""
    assemble_KM!(K, M, dh, cv, λ, μ, ρ)
    assemble_KM!(K, M, dh, cv, stress, ρ)

Assemble the three-dimensional global stiffness matrix `K` and mass matrix `M`
into preallocated sparse matrices using isotropic Lamé constants or an internal
stress-law object. The matrices must have the sparsity pattern associated with
`dh`; both are mutated and the function returns `nothing`.

	K_rs = ∫ ε(φ_r) ⊡ σ(ε(φ_s)) dΩ
	M_rs = ∫ ρ φ_r · φ_s dΩ

For the Lamé overload, `σ(ε) = λ tr(ε)I + 2με`.
"""
assemble_KM!(K, M, dh, cv, λ::Real, μ::Real, ρ::Real) =
	assemble_KM!(K, M, dh, cv, IsotropicStress(Float64(λ), Float64(μ)), Float64(ρ))

function assemble_KM!(K, M, dh, cv, stress::AbstractStress, ρ::Float64)
	n_dpc = ndofs_per_cell(dh)
	Ke = zeros(n_dpc, n_dpc)
	Me = zeros(n_dpc, n_dpc)
	asm_K = start_assemble(K)
	asm_M = start_assemble(M)

	for element in CellIterator(dh)
		fill!(Ke, 0.0)
		fill!(Me, 0.0)
		reinit!(cv, element)
		for q in 1:getnquadpoints(cv)
			dΩ = getdetJdV(cv, q)
			for r in 1:n_dpc
				δε = shape_symmetric_gradient(cv, q, r)
				Nr = shape_value(cv, q, r)
				for s in 1:n_dpc
					ε = shape_symmetric_gradient(cv, q, s)
					σ = _σ(ε, stress)
					Ke[r, s] += (δε ⊡ σ) * dΩ
					Ns = shape_value(cv, q, s)
					Me[r, s] += ρ * (Nr ⋅ Ns) * dΩ
				end
			end
		end
		assemble!(asm_K, celldofs(element), Ke)
		assemble!(asm_M, celldofs(element), Me)
	end
end
