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
    e1 = ε[1, 1]
    e2 = ε[2, 2]
    e3 = ε[3, 3]
    e4 = 2 * ε[2, 3]
    e5 = 2 * ε[1, 3]
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
    FerriteGeometricNonlinearity{DEG, DH, CV, S, CC} <: MORFE.FEMMultilinearMap{2}

Three-dimensional FEM-backed multilinear term for the negative SVK geometric
internal force.

- `DEG = 2`: quadratic form with two displacement inputs.
- `DEG = 3`: cubic form with three displacement inputs.

The `MORFE.FEMMultilinearMap{2}` supertype places the term in a second-order
model. It uses only position inputs, with `multiindex == (DEG, 0)`, and has no
external-state factors. `DH`, `CV`, `S` and `CC` are the concrete types of the DOF
handler, the cell values, the stress law and the cell cache.

# Fields

The discretisation, shared with the linear operators:
- `dof_handler`: the Ferrite `DofHandler` of the single displacement field.
- `cell_values`: the Ferrite `CellValues` of that field, reinitialised on each cell.
- `dof_to_free`: global DOF number → index in the free state vector, `0` at a
  constrained DOF; a dense vector, read on every gather and assembly.
- `n_free`: the number of free DOFs, the length of the state vector.
- `stress`: the constitutive law, an `IsotropicStress` or a `VoigtStress`.

MORFE's term metadata: `multiindex`, `multiplicity_external`, `deg`, `fully_asymmetric`.

Reusable evaluation state, so a term instance serves one evaluation sweep at a time:
- `qp_buffer`, `(DEG, n_qp)`: the displacement gradients of the direct evaluation
  `MORFE.evaluate_term!`; the single-pass assembly keeps its own.
- `nodal_displacements[j]`: one coefficient's displacement at node `j` of the current cell.
- `shape_gradients[j, q]`: the gradient `∇Nⱼ` of the scalar shape function of node `j` at
  quadrature point `q` of the current cell, from which `scatter_qp!` forms
  `∇u = Σⱼ uⱼ ⊗ ∇Nⱼ`.
- `strains`: one strain per coefficient for `MORFE.accumulate_terms_qp!`, grown on demand.
- `cell_cache`: the Ferrite `CellCache` that every traversal reuses.
"""
struct FerriteGeometricNonlinearity{DEG, DH, CV, S, CC} <: MORFE.FEMMultilinearMap{2}
    dof_handler::DH
    cell_values::CV
    dof_to_free::Vector{Int}
    n_free::Int
    stress::S
    multiindex::NTuple{2, Int}
    multiplicity_external::Int
    deg::Int
    fully_asymmetric::Union{Nothing, Bool}
    qp_buffer::Matrix{Tensor{2, 3, ComplexF64, 9}}   # concrete: Tensor{2, 3, ComplexF64} boxes
    nodal_displacements::Vector{Vec{3, ComplexF64}}
    shape_gradients::Matrix{Vec{3, Float64}}
    strains::Vector{SymmetricTensor{2, 3, ComplexF64, 6}}
    cell_cache::CC
end

"""
    FerriteGeometricNonlinearity{DEG}(dof_handler, cell_values, dof_to_free, n_free, λ, μ;
                                       fully_asymmetric = false)
    FerriteGeometricNonlinearity{DEG}(dof_handler, cell_values, dof_to_free, n_free, stress;
                                       fully_asymmetric = false)

Construct a degree-`DEG` term from Lamé constants or an internal stress-law object.

# Arguments
- `dof_handler::Ferrite.DofHandler`: a closed handler with one displacement field.
- `cell_values::Ferrite.CellValues`: built from that field's vector Lagrange interpolation,
  `ip^3` (checked), and the quadrature rule to integrate with.
- `dof_to_free::Dict{Int, Int}`: global DOF number → index in the free state vector;
  constrained DOFs are absent. [`mechanical_model`](@ref) returns it as `info.dof_to_free`.
- `n_free::Int`: the number of free DOFs.

Buffers are preallocated from `dof_handler` and `cell_values`. `fully_asymmetric` is stored
with the same symmetry-policy meaning as on `MORFE.MultilinearMap`.
"""
function FerriteGeometricNonlinearity{DEG}(
        dof_handler::DH, cell_values::CV,
        dof_to_free::Dict{Int, Int}, n_free::Int,
        λ::Real, μ::Real;
        fully_asymmetric::Union{Nothing, Bool} = false) where {DEG, DH, CV}
    return FerriteGeometricNonlinearity{DEG}(dof_handler, cell_values, dof_to_free, n_free,
        IsotropicStress(Float64(λ), Float64(μ)); fully_asymmetric = fully_asymmetric)
end

function FerriteGeometricNonlinearity{DEG}(
        dof_handler::DH, cell_values::CV,
        dof_to_free::Dict{Int, Int}, n_free::Int,
        stress::S;
        fully_asymmetric::Union{Nothing, Bool} = false) where {
        DEG, DH, CV, S <: AbstractStress}
    _check_vector_lagrange(dof_handler, cell_values)
    n_qp = getnquadpoints(cell_values)
    n_dofs = ndofs_per_cell(dof_handler)
    qp_buffer = Matrix{Tensor{2, 3, ComplexF64, 9}}(undef, DEG, n_qp)
    nodal_displacements = Vector{Vec{3, ComplexF64}}(undef, n_dofs ÷ 3)
    shape_gradients = Matrix{Vec{3, Float64}}(undef, n_dofs ÷ 3, n_qp)
    dense_dof_to_free = zeros(Int, ndofs(dof_handler))
    for (dof, free) in dof_to_free
        dense_dof_to_free[dof] = free
    end
    cell_cache = CellCache(dof_handler)
    return FerriteGeometricNonlinearity{DEG, DH, CV, S, typeof(cell_cache)}(
        dof_handler, cell_values, dense_dof_to_free, n_free, stress,
        (DEG, 0), 0, DEG, fully_asymmetric, qp_buffer, nodal_displacements,
        shape_gradients, SymmetricTensor{2, 3, ComplexF64, 6}[], cell_cache)
end

# The scatter forms ∇u = Σⱼ uⱼ ⊗ ∇Nⱼ from the scalar shape gradients, which holds for a vector
# Lagrange interpolation `ip^3`: the cell dof 3(j-1)+c has gradient e_c ⊗ ∇Nⱼ. Checked on the
# first cell, at every quadrature point.
function _check_vector_lagrange(dof_handler, cell_values)
    n = getnbasefunctions(cell_values)
    vector_lagrange = n == ndofs_per_cell(dof_handler) && n % 3 == 0
    if vector_lagrange
        reinit!(cell_values, first(CellIterator(dof_handler)))
        for q in 1:getnquadpoints(cell_values), i in 1:n

            j, c = divrem(i - 1, 3)
            G, g = shape_gradient(cell_values, q, i), shape_gradient(cell_values, q, 3j + 1)
            vector_lagrange &= all(G[r, k] == (r == c + 1 ? g[1, k] : 0)
            for r in 1:3, k in 1:3)
        end
    end
    vector_lagrange ||
        throw(ArgumentError("the SVK term needs a single displacement field " *
                            "with a vector Lagrange interpolation, `ip^3`"))
    return nothing
end

# -----------------------------------------------------------------------
# FEMMultilinearMap interface
# -----------------------------------------------------------------------

"""
	MORFE.fem_element_loop(term::FerriteGeometricNonlinearity)

The element loop of `term`: its DOF handler and cell values, compared by identity, and its
free-DOF map, compared by content. The quadratic and cubic terms of one model share all
three, so they share one pass of MORFE's single-pass assembly. A term with its own
`CellValues` object gets its own pass: the driver's reinitialisation would not reach it.
"""
function MORFE.fem_element_loop(term::FerriteGeometricNonlinearity)
    (term.dof_handler, term.cell_values, term.dof_to_free)
end

# Reuses the term's cell cache, so a traversal allocates nothing.
function MORFE.fem_elements(term::FerriteGeometricNonlinearity)
    CellIterator(term.cell_cache, 1:getncells(term.dof_handler.grid))
end

MORFE.fem_n_qp(term::FerriteGeometricNonlinearity) = getnquadpoints(term.cell_values)

function MORFE.fem_ndofs_per_cell(term::FerriteGeometricNonlinearity)
    ndofs_per_cell(term.dof_handler)
end

MORFE.fem_qp_buffer(term::FerriteGeometricNonlinearity) = term.qp_buffer

function MORFE.fem_getdetJdV(_element, q, term::FerriteGeometricNonlinearity)
    getdetJdV(term.cell_values, q)
end

# Called once per element, before any scatter: reinitialises the cell values and extracts the
# scalar shape gradients, ∇Nⱼ(q) = row 1 of the gradient of the cell dof 3(j-1)+1, which every
# coefficient scattered in the cell then reuses.
function MORFE.fem_reinit!(element, term::FerriteGeometricNonlinearity)
    reinit!(term.cell_values, element)
    ∇N = term.shape_gradients
    for q in axes(∇N, 2), j in axes(∇N, 1)

        g = shape_gradient(term.cell_values, q, 3j - 2)
        ∇N[j, q] = Vec{3}((g[1, 1], g[1, 2], g[1, 3]))
    end
end

"""
	MORFE.scatter_qp!(qp_values, coefficient, element, term::FerriteGeometricNonlinearity)

Gather `coefficient`, a vector over the free DOFs, into the cell's nodal displacements,
inserting zero at constrained DOFs, then write the complex displacement gradient
`∇u = Σⱼ uⱼ ⊗ ∇Nⱼ` at each quadrature point to `qp_values[q]`: the 27 scalar shape
gradients of a quadratic hexahedron rather than the 81 full tensors of
`function_gradient`, with the same result. `MORFE.fem_reinit!` must already have been
called for `element`.
"""
function MORFE.scatter_qp!(qp_values, coefficient, element,
        term::FerriteGeometricNonlinearity)
    dofs = celldofs(element)
    u, ∇N = term.nodal_displacements, term.shape_gradients
    free(d) = (l = term.dof_to_free[d]; l == 0 ? zero(ComplexF64) : coefficient[l])
    for j in eachindex(u)
        u[j] = Vec{3}((free(dofs[3j - 2]), free(dofs[3j - 1]), free(dofs[3j])))
    end
    checkbounds(∇N, eachindex(u), eachindex(qp_values))
    @inbounds for q in eachindex(qp_values)
        ∇u = zero(Tensor{2, 3, ComplexF64, 9})
        for j in eachindex(u)
            ∇u += otimes(u[j], ∇N[j, q])   # `⊗` clashes with LinearMaps
        end
        qp_values[q] = ∇u
    end
end

# Symmetric Green-Lagrange cross term for two gradients, sym(¼(∇u1ᵀ∇u2 + ∇u2ᵀ∇u1)): one
# product, since ∇u2ᵀ∇u1 is the transpose of ∇u1ᵀ∇u2.
@inline _E_nl(∇u1, ∇u2) = 0.5 * symmetric(transpose(∇u1) ⋅ ∇u2)

# Polarised quadratic and cubic parts of the first Piola-Kirchhoff stress; see the
# `accumulate_qp!` methods below.
@inline _P2(∇u1, ∇u2, s) = _σ(_E_nl(∇u1, ∇u2), s) +
                           0.5 * (∇u1 ⋅ _σ(symmetric(∇u2), s) + ∇u2 ⋅ _σ(symmetric(∇u1), s))

@inline _P3(∇u1, ∇u2, ∇u3, s) = (∇u1 ⋅ _σ(_E_nl(∇u2, ∇u3), s) +
                                 ∇u2 ⋅ _σ(_E_nl(∇u1, ∇u3), s) +
                                 ∇u3 ⋅ _σ(_E_nl(∇u1, ∇u2), s)) / 3

# Fe_r -= c ∇φ_r ⊡ P for every test function of the element: the one pass over the test
# functions per call, shared by `accumulate_qp!` and `contract_qp!`. Bounds are checked
# once, before the loop, so the loop runs unchecked: run on its own, as when the stresses
# of all quadrature points are formed first, it then costs half as much (81 dofs: 324 →
# 172 ns per point).
@inline function _test_P!(Fe, P, c, q, term)
    n = ndofs_per_cell(term.dof_handler)
    checkbounds(Fe, 1:n)
    (n <= getnbasefunctions(term.cell_values) &&
     1 <= q <= getnquadpoints(term.cell_values)) ||
        throw(BoundsError(term.cell_values, (q, n)))
    @inbounds for r in 1:n
        Fe[r] -= c * (shape_gradient(term.cell_values, q, r) ⊡ P)
    end
end

"""
	MORFE.accumulate_qp!(Fe, qp_args::NTuple{2}, multiplier, element, q, dΩ, term)

Quadratic geometric nonlinearity integrand at one quadrature point, with
`qp_args = (∇u1, ∇u2)`:

	Fe_r -= multiplier * ∇φ_r ⊡ P₂(∇u1, ∇u2) * dΩ
	P₂(∇u1, ∇u2) = σ(E_nl(∇u1, ∇u2)) + ½(∇u1 ⋅ σ(ε(∇u2)) + ∇u2 ⋅ σ(ε(∇u1)))

`P₂` is the polarised quadratic part of the first Piola-Kirchhoff stress `(I + ∇u) σ(E)`,
with `E_nl(A, B) = sym(0.25(AᵀB + BᵀA))`. It equals the quadratic part of `S ⊡ δE` term by
term, because `sym(Aᵀ∇φ) ⊡ σ = ∇φ ⊡ (A ⋅ σ)` for a symmetric `σ`. Formed once per quadrature
point, it leaves one double contraction per test function. The gradients are complex and
the form must stay multilinear, so nothing here conjugates: `E_nl` writes `transpose`, and
Tensors.jl's `symmetric`, `⋅` and `⊡` do not conjugate either. The negative sign moves the
nonlinear internal force to MORFE's right-hand side.
"""
function MORFE.accumulate_qp!(Fe, qp_args::NTuple{2}, multiplier, _element, q, dΩ,
        term::FerriteGeometricNonlinearity{2})
    ∇u1, ∇u2 = qp_args
    _test_P!(Fe, _P2(∇u1, ∇u2, term.stress), multiplier * dΩ, q, term)
end

"""
	MORFE.accumulate_qp!(Fe, qp_args::NTuple{3}, multiplier, element, q, dΩ, term)

Cubic geometric nonlinearity integrand at one quadrature point, with
`qp_args = (∇u1, ∇u2, ∇u3)`:

	Fe_r -= multiplier * ∇φ_r ⊡ P₃(∇u1, ∇u2, ∇u3) * dΩ
	P₃ = ⅓ Σ_{(i,j,k) cyclic} ∇ui ⋅ σ(E_nl(∇uj, ∇uk))

`P₃` is the polarised cubic part of the first Piola-Kirchhoff stress, formed once per
quadrature point as in the quadratic method. The negative sign moves the nonlinear internal
force to MORFE's right-hand side.
"""
function MORFE.accumulate_qp!(Fe, qp_args::NTuple{3}, multiplier, _element, q, dΩ,
        term::FerriteGeometricNonlinearity{3})
    ∇u1, ∇u2, ∇u3 = qp_args
    _test_P!(Fe, _P3(∇u1, ∇u2, ∇u3, term.stress), multiplier * dΩ, q, term)
end

"""
	MORFE.fem_flux_zero(term::FerriteGeometricNonlinearity)

Opt the SVK terms into flux summation: the flux is the polarised part of the first
Piola-Kirchhoff stress, `P₂` or `P₃`, which `accumulate_qp!` tests against the gradients of
all the element's shape functions. MORFE's single-pass assembly then sums
`multiplier * P` over the factorisations of both terms at each quadrature point and tests
the sum once, so a factorisation costs one stress and no pass over the test functions.
"""
MORFE.fem_flux_zero(::FerriteGeometricNonlinearity) = zero(Tensor{2, 3, ComplexF64, 9})

"""
	MORFE.flux_qp(qp_args, element, q, term::FerriteGeometricNonlinearity)

The stress `P₂(∇u1, ∇u2)` or `P₃(∇u1, ∇u2, ∇u3)` of the `accumulate_qp!` methods, without the
multiplier, the weight and the test functions.
"""
function MORFE.flux_qp(qp_args::NTuple{2}, _element, _q,
        term::FerriteGeometricNonlinearity{2})
    _P2(qp_args[1], qp_args[2], term.stress)
end

function MORFE.flux_qp(qp_args::NTuple{3}, _element, _q,
        term::FerriteGeometricNonlinearity{3})
    _P3(qp_args[1], qp_args[2], qp_args[3], term.stress)
end

"""
	MORFE.contract_qp!(Fe, P, element, q, dΩ, term::FerriteGeometricNonlinearity)

`Fe_r -= ∇φ_r ⊡ P * dΩ` for every test function of the element, as in `accumulate_qp!`.
"""
function MORFE.contract_qp!(Fe, P, _element, q, dΩ, term::FerriteGeometricNonlinearity)
    _test_P!(Fe, P, dΩ, q, term)
end

"""
	MORFE.accumulate_terms_qp!(flux, Fe, ∇u, external_arguments, entries, terms, element, q, dΩ)

The stresses of all quadratic and cubic factorisations at one quadrature point, summed with
the linearity of `σ`:

	Σ m P₂(∇uₐ, ∇u_b) + Σ m P₃(∇uₐ, ∇u_b, ∇u_c) = σ(E) + Σᵣ ∇uᵣ ⋅ σ(Aᵣ)

`∇u[r]` is the displacement gradient of the `r`-th distinct coefficient at the point, the
`coefficient_values` of MORFE's interface. `E = Σ m E_nl(∇uₐ, ∇u_b)` runs over the quadratic
factorisations. `Aᵣ` collects the strains that `∇uᵣ` multiplies, one for each slot it fills:
`m/2 ε(∇u_b)` in a quadratic factorisation `(r, b)`, and `m/3 E_nl(∇u_b, ∇u_c)` in a cubic
one `(r, b, c)`. Each coefficient then costs one constitutive evaluation and one product
with its gradient, shared by the quadratic and cubic terms, where `P₂` and `P₃` cost one of
each per slot of every factorisation. The terms must share their stress law.
"""
function MORFE.accumulate_terms_qp!(flux::Tensor{2, 3, ComplexF64, 9}, _Fe, ∇u,
        _external_arguments, entries::Tuple,
        terms::Tuple{FerriteGeometricNonlinearity, Vararg{FerriteGeometricNonlinearity}},
        _element, _q, _dΩ)
    s = first(terms).stress
    all(t -> t.stress === s, terms) ||
        throw(ArgumentError("SVK terms evaluated together must share their stress law"))
    A = first(terms).strains
    n = length(∇u)
    length(A) < n && resize!(A, n)
    fill!(view(A, 1:n), zero(eltype(A)))
    E = _collect_strains!(zero(eltype(A)), A, ∇u, entries, terms)
    flux += _σ(E, s)
    for r in 1:n
        flux += ∇u[r] ⋅ _σ(A[r], s)
    end
    return flux
end

# Add each term's factorisations to `E` and `A`, term by term, unrolled at compile time.
@inline _collect_strains!(E, A, ∇u, ::Tuple{}, ::Tuple{}) = E

@inline function _collect_strains!(E, A, ∇u, entries::Tuple, terms::Tuple)
    E = _collect_strains!(E, A, ∇u, first(entries), first(terms))
    return _collect_strains!(E, A, ∇u, Base.tail(entries), Base.tail(terms))
end

function _collect_strains!(E, A, ∇u, entries::AbstractVector,
        ::FerriteGeometricNonlinearity{2})
    for e in entries
        a, b = e.factors
        m = e.multiplier
        E += m * _E_nl(∇u[a], ∇u[b])
        A[a] += (m / 2) * symmetric(∇u[b])
        A[b] += (m / 2) * symmetric(∇u[a])
    end
    return E
end

function _collect_strains!(E, A, ∇u, entries::AbstractVector,
        ::FerriteGeometricNonlinearity{3})
    for e in entries
        a, b, c = e.factors
        w = e.multiplier / 3
        A[a] += w * _E_nl(∇u[b], ∇u[c])
        A[b] += w * _E_nl(∇u[a], ∇u[c])
        A[c] += w * _E_nl(∇u[a], ∇u[b])
    end
    return E
end

"""
	MORFE.assemble_element!(result, Fe, element, term::FerriteGeometricNonlinearity)

Add the element vector `Fe`, indexed in cell-DOF order, to `result`, a vector over the free
DOFs. Entries of constrained global DOFs are skipped. The function mutates `result` and
returns `nothing`.
"""
function MORFE.assemble_element!(result, Fe, element, term::FerriteGeometricNonlinearity)
    dofs = celldofs(element)
    for (r, d) in enumerate(dofs)
        free = term.dof_to_free[d]
        free != 0 && (result[free] += Fe[r])
    end
end

# -----------------------------------------------------------------------
# Linear matrix assembly
# -----------------------------------------------------------------------

"""
    assemble_KM!(K, M, dof_handler, cell_values, λ, μ, ρ)
    assemble_KM!(K, M, dof_handler, cell_values, stress, ρ)

Assemble the three-dimensional global stiffness matrix `K` and mass matrix `M`
into preallocated sparse matrices using isotropic Lamé constants or an internal
stress-law object. The matrices are over all global DOFs of `dof_handler`, with the
sparsity pattern of `allocate_matrix(dof_handler)`; `cell_values` holds the displacement
interpolation and the quadrature rule. Both matrices are mutated and the function returns
`nothing`.

	K_rs = ∫ ε(φ_r) ⊡ σ(ε(φ_s)) dΩ
	M_rs = ∫ ρ φ_r · φ_s dΩ

For the Lamé overload, `σ(ε) = λ tr(ε)I + 2με`.
"""
function assemble_KM!(K, M, dof_handler, cell_values, λ::Real, μ::Real, ρ::Real)
    assemble_KM!(K, M, dof_handler, cell_values, IsotropicStress(Float64(λ), Float64(μ)),
        Float64(ρ))
end

function assemble_KM!(K, M, dof_handler, cell_values, stress::AbstractStress, ρ::Float64)
    n_dofs = ndofs_per_cell(dof_handler)
    Ke = zeros(n_dofs, n_dofs)
    Me = zeros(n_dofs, n_dofs)
    asm_K = start_assemble(K)
    asm_M = start_assemble(M)

    for element in CellIterator(dof_handler)
        fill!(Ke, 0.0)
        fill!(Me, 0.0)
        reinit!(cell_values, element)
        for q in 1:getnquadpoints(cell_values)
            dΩ = getdetJdV(cell_values, q)
            for r in 1:n_dofs
                δε = shape_symmetric_gradient(cell_values, q, r)
                Nr = shape_value(cell_values, q, r)
                for s in 1:n_dofs
                    ε = shape_symmetric_gradient(cell_values, q, s)
                    σ = _σ(ε, stress)
                    Ke[r, s] += (δε ⊡ σ) * dΩ
                    Ns = shape_value(cell_values, q, s)
                    Me[r, s] += ρ * (Nr ⋅ Ns) * dΩ
                end
            end
        end
        assemble!(asm_K, celldofs(element), Ke)
        assemble!(asm_M, celldofs(element), Me)
    end
end
