"""
    SVKMaterial(; E, ν, ρ)

St. Venant-Kirchhoff material: Young's modulus `E`, Poisson ratio `ν`,
density `ρ`, and derived Lamé constants
`λ = Eν/((1+ν)(1-2ν))`, `μ = E/(2(1+ν))`. With the Green-Lagrange strain and
this linear elastic stress law, the internal force has quadratic and cubic
displacement terms and no higher-degree terms.
"""
struct SVKMaterial{T}
    E::T
    ν::T
    ρ::T
    λ::T
    μ::T
end
function SVKMaterial(; E, ν, ρ)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2 * (1 + ν))
    return SVKMaterial(promote(E, ν, ρ, λ, μ)...)
end

"""
    RayleighDamping(; α, β)

Rayleigh damping coefficients defining `C = α M + β K`. The constructor promotes
`α` and `β` to a common numeric type.
"""
struct RayleighDamping{T}
    α::T
    β::T
end
RayleighDamping(; α, β) = RayleighDamping(promote(α, β)...)

"""
    HarmonicForcing(; mode, amplitude, Ω = nothing)

Harmonic load specification for
`f(t) = amplitude · M*ϕ_mode · cos(Ω*t)`. `mode` is a positive physical
mode-pair index used only for the load shape. If `Ω` is omitted, `build_model`
uses `abs(λ[2mode-1])` from the resolved spectrum.

`build_model` accepts either one of these or a vector of them (multi-harmonic
excitation); each element adds its own pair of external states with eigenvalues
±iΩ, so `N_EXT = 2 · length(forcing)`. `mode` need not be a `master` pair — it
only supplies the load shape. MORFE's parametrisation may separately warn when a
monomial frequency is near an eigenvalue left off the manifold, because that
off-manifold solve is ill-conditioned independently of the forcing shape.
"""
struct HarmonicForcing{T}
    mode::Int
    amplitude::T
    Ω::Union{Nothing, T}
end
function HarmonicForcing(; mode, amplitude, Ω = nothing)
    HarmonicForcing(mode, amplitude, Ω === nothing ? nothing :
                                     convert(typeof(amplitude), Ω))
end

"""
	AssembledMechanicalModel <: AbstractAssembledModel

Assembled three-dimensional second-order model on the free DOFs. Its linear
operators are `K`, `C`, and `M`; its FEM multilinear maps accumulate the negative
quadratic and cubic internal forces, so MORFE's model form represents
`M*ü + C*u̇ + K*u + f_int,nl(u) = f_ext`.

`term_factory(degree, max_cols)` lazily creates the degree-`2` or degree-`3`
`MORFE.FEMMultilinearMap{2}` with storage for `max_cols` batched columns.
`material`, `damping`, and `info` retain backend data used by eigensolvers,
post-processing, and summaries.

## Indexing `B`

The linear operators live in one field, in **derivative order**, so `B[k + 1]` is the
coefficient of the `k`-th time derivative:

	m.B[1]   # B₀ = K, stiffness
	m.B[2]   # B₁ = C, damping
	m.B[3]   # B₂ = M, mass

This is the order `MORFE.NthOrderModel.linear_terms` wants, and `m.B` is handed to it
unchanged. It is also the reason the field exists: the operators used to be three separate
fields declared `K, M, C` while the model tuple was `(K, C, M)`, so the declaration order
and the operator order disagreed and every call site had to restate the mapping. It is
stated once now, here.

`m.K`, `m.C` and `m.M` remain available as read-only properties — for a structural problem
the physics names read better than an index — and are exactly `B[1]`, `B[2]`, `B[3]`.
"""
struct AssembledMechanicalModel{TB, F, MAT, DMP} <: AbstractAssembledModel
    B::TB
    term_factory::F
    nonlinear_degrees::Tuple{Vararg{Int}}
    material::MAT
    damping::DMP
    info::NamedTuple
end

# The physics names, kept because `m.M` says more than `m.B[3]` in a structural context.
# Derivative order throughout: B₀ = K, B₁ = C, B₂ = M.
function Base.getproperty(m::AssembledMechanicalModel, s::Symbol)
    s === :K && return getfield(m, :B)[1]
    s === :C && return getfield(m, :B)[2]
    s === :M && return getfield(m, :B)[3]
    return getfield(m, s)
end

Base.propertynames(::AssembledMechanicalModel, private::Bool = false) = (:B, :term_factory,
    :nonlinear_degrees, :material, :damping, :info, :K, :C, :M)

_material_summary(m) = "SVK  E=$(m.E)  ν=$(m.ν)  ρ=$(m.ρ)"

function Base.show(io::IO, ::MIME"text/plain", m::AssembledMechanicalModel)
    println(io, "AssembledMechanicalModel ($(m.info.backend))")
    println(io, "  free DOFs : $(m.info.n_dofs) (of $(m.info.n_dofs_total))")
    println(io,
        "  material  : " * _material_summary(m.material))
    print(io, "  damping   : Rayleigh  α=$(m.damping.α)  β=$(m.damping.β)")
end

# The rows only this physics has; the shared skeleton (sizes, order, stage
# timings) comes from Common.write_summary.
function Common.summary_entries(m::AssembledMechanicalModel, meta::NamedTuple)
    master = [(i + 1) ÷ 2 for i in meta.master_indices[1:2:end]]
    rows = Pair{String, Any}[
        "model" => "geometrically nonlinear structure (St. Venant-Kirchhoff)",
        "material" => _material_summary(m.material),
        "damping" => "Rayleigh  α=$(m.damping.α)  β=$(m.damping.β)",
        "master_pairs" => master,
        "master_eigenvalues" => meta.spectrum.eigenvalues[meta.master_indices],
        "n_monomials" => meta.n_monomials
    ]
    isempty(meta.forcings) ? push!(rows, "forcing" => "none (autonomous)") :
    push!(rows,
        "forcing" => join(
            ["mode $(f.mode) a=$(f.amplitude) Ω=$Ω"
             for (f, Ω) in zip(meta.forcings, meta.Ω)], "; "))
    return rows
end

function Common.summary_entries(m::AssembledMechanicalModel, ::Nothing)
    Pair{String, Any}[
        "model" => "geometrically nonlinear structure (St. Venant-Kirchhoff)",
        "material" => _material_summary(m.material)]
end

# ── Anisotropic materials ───────────────────────────────────────────────────
# `SVKMaterial` above is the isotropic special case. For crystalline materials
# (silicon, quartz, …) the stress law needs the full stiffness tensor.

"""
    AnisotropicMaterial(D, ρ)

St. Venant-Kirchhoff material with a general 6×6 Voigt stiffness `D`, converted
to `SMatrix{6,6,Float64}`, and density `ρ`, converted to `Float64`. The ordering
is `[11, 22, 33, 23, 13, 12]` and the strain vector uses engineering shear
components `[ε₁₁, ε₂₂, ε₃₃, 2ε₂₃, 2ε₁₃, 2ε₁₂]`.

Use [`CubicCrystal`](@ref) for cubic crystals given by `c₁₁, c₁₂, c₄₄`.
"""
struct AnisotropicMaterial
    D::SMatrix{6, 6, Float64, 36}
    ρ::Float64
end
function AnisotropicMaterial(D::AbstractMatrix, ρ::Real)
    AnisotropicMaterial(SMatrix{6, 6, Float64}(D), Float64(ρ))
end

# Voigt index of a tensor index pair: (1,1)→1 (2,2)→2 (3,3)→3 (2,3)→4 (1,3)→5 (1,2)→6
@inline _voigt_index(i::Int, j::Int) = i == j ? i : (i + j == 5 ? 4 : (i + j == 4 ? 5 : 6))

const _VOIGT_PAIRS = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

"""
    rotate_voigt(D, Q) -> SMatrix{6,6}

Rotate a Voigt stiffness matrix by a 3×3 matrix `Q`, expected to be orthogonal.
Only its size is validated.

`Q` maps crystal axes to lab axes: if `D` is expressed in the crystal frame, the
result is expressed in the lab frame (pass `Q'` for the opposite convention).

Implemented by expanding `D` to the 4th-order stiffness `C_ijkl`, applying the
tensor transformation `C'_ijkl = Q_ip Q_jq Q_kr Q_ls C_pqrs`, and contracting
back — so no Bond-matrix convention (and its factor-of-2 traps) is involved.
"""
function rotate_voigt(D::AbstractMatrix, Q::AbstractMatrix)
    size(Q) == (3, 3) || throw(ArgumentError("rotation must be 3×3, got $(size(Q))"))
    C = Array{Float64}(undef, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C[i, j, k, l] = D[_voigt_index(i, j), _voigt_index(k, l)]
    end
    Cr = zeros(Float64, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        acc = 0.0
        for p in 1:3, q in 1:3, r in 1:3, s in 1:3
            acc += Q[i, p] * Q[j, q] * Q[k, r] * Q[l, s] * C[p, q, r, s]
        end
        Cr[i, j, k, l] = acc
    end
    Dr = Matrix{Float64}(undef, 6, 6)
    @inbounds for a in 1:6, b in 1:6

        (i, j) = _VOIGT_PAIRS[a]
        (k, l) = _VOIGT_PAIRS[b]
        Dr[a, b] = Cr[i, j, k, l]
    end
    return SMatrix{6, 6, Float64}(Dr)
end

"""
    CubicCrystal(; c11, c12, c44, ρ, rotation = nothing) -> AnisotropicMaterial

Cubic crystal (silicon, germanium, …) from its three independent constants, with
the crystal optionally rotated into the lab frame by `rotation` (a 3×3 matrix, or
an angle in radians about the z axis).

`rotation = nothing` leaves the crystal axes unchanged; a scalar is interpreted as
an angle in radians about the z axis, and a matrix is passed to [`rotate_voigt`](@ref).

The isotropic limit is `c11 = λ + 2μ`, `c12 = λ`, `c44 = μ`; it reproduces the
constitutive response of an `SVKMaterial` with those Lamé constants.
"""
function CubicCrystal(; c11::Real, c12::Real, c44::Real, ρ::Real, rotation = nothing)
    D0 = zeros(Float64, 6, 6)
    for i in 1:3, j in 1:3

        D0[i, j] = (i == j) ? Float64(c11) : Float64(c12)
    end
    D0[4, 4] = D0[5, 5] = D0[6, 6] = Float64(c44)
    Q = rotation === nothing ? nothing :
        (rotation isa Real ?
         [cos(rotation) -sin(rotation) 0.0; sin(rotation) cos(rotation) 0.0; 0.0 0.0 1.0] :
         Matrix{Float64}(rotation))
    D = Q === nothing ? SMatrix{6, 6, Float64}(D0) : rotate_voigt(D0, Q)
    return AnisotropicMaterial(D, Float64(ρ))
end

"""
    voigt_stiffness(material) -> SMatrix{6,6}

Return the `SMatrix{6,6,Float64}` Voigt stiffness of a supported material.
`AnisotropicMaterial` returns its stored matrix; `SVKMaterial` constructs the
isotropic matrix from its Lamé constants. This is useful for inspection and for
cross-checking anisotropic input.
"""
voigt_stiffness(m::AnisotropicMaterial) = m.D
function voigt_stiffness(m::SVKMaterial)
    λ, μ = Float64(m.λ), Float64(m.μ)
    D = zeros(Float64, 6, 6)
    for i in 1:3, j in 1:3

        D[i, j] = (i == j) ? λ + 2μ : λ
    end
    D[4, 4] = D[5, 5] = D[6, 6] = μ
    return SMatrix{6, 6, Float64}(D)
end

# Stress model + density for either material — the single dispatch point used by
# `svk_nonlinearity`, `svk_assemble_KM!` and `mechanical_model`.
stress_model(m::SVKMaterial) = IsotropicStress(Float64(m.λ), Float64(m.μ))
stress_model(m::AnisotropicMaterial) = VoigtStress(m.D)

function _material_summary(m::AnisotropicMaterial)
    "anisotropic SVK  D₁₁=$(m.D[1,1])  D₁₂=$(m.D[1,2])  D₄₄=$(m.D[4,4])  ρ=$(m.ρ)"
end
