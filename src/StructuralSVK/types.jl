"""
    SVKMaterial(; E, ν, ρ)

St. Venant-Kirchhoff material: Young's modulus `E`, Poisson ratio `ν`,
density `ρ`. Lamé constants `λ`, `μ` are derived. The SVK model generates
quadratic and cubic nonlinearities in the displacement — nothing higher.
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

Rayleigh damping `C = α M + β K`.
"""
struct RayleighDamping{T}
    α::T
    β::T
end
RayleighDamping(; α, β) = RayleighDamping(promote(α, β)...)

"""
    HarmonicForcing(; mode, amplitude, Ω = nothing)

Harmonic load `f(t) = amplitude · M·ϕ_mode · cos(Ω t)`: shaped like mode
`mode`, oscillating at that same mode's natural frequency unless `Ω` is given.

`parametrise` accepts either one of these or a vector of them (multi-harmonic
excitation); each element adds its own pair of external states with eigenvalues
±iΩ, so `N_EXT = 2 · length(forcing)`. `mode` need not be a `master` pair — it
only supplies the load shape — but `parametrise` warns if Ω is near-resonant
with a mode left off the manifold, which makes that direction's solve
ill-conditioned however the load is shaped.
"""
struct HarmonicForcing{T}
    mode::Int
    amplitude::T
    Ω::Union{Nothing, T}
end
HarmonicForcing(; mode, amplitude, Ω = nothing) =
    HarmonicForcing(mode, amplitude, Ω === nothing ? nothing : convert(typeof(amplitude), Ω))

"""
Assembled second-order mechanical model `M ü + C u̇ + K u = f_nl(u) (+ forcing)`,
restricted to free DOFs, with a lazy factory for the FEM nonlinear terms:
`term_factory(degree, max_cols)` → `FEMMultilinearMap`.
"""
struct AssembledMechanicalModel{TK, TM, TC, F, MAT, DMP} <: AbstractAssembledModel
    K::TK
    M::TM
    C::TC
    term_factory::F
    nonlinear_degrees::Tuple{Vararg{Int}}
    material::MAT
    damping::DMP
    info::NamedTuple
end

_material_summary(m) = "SVK  E=$(m.E)  ν=$(m.ν)  ρ=$(m.ρ)"

function Base.show(io::IO, ::MIME"text/plain", m::AssembledMechanicalModel)
    println(io, "AssembledMechanicalModel ($(m.info.backend))")
    println(io, "  free DOFs : $(m.info.n_dofs) (of $(m.info.n_dofs_total))")
    println(io,
        "  material  : " * _material_summary(m.material))
    print(io, "  damping   : Rayleigh  α=$(m.damping.α)  β=$(m.damping.β)")
end

"""
Invariant-manifold ROM for an SVK structure: `W` (parametrisation), `R` (reduced
dynamics), `eigenvalues`, `master` (mode pairs), `order`, `forcing` (a
`Vector{<:HarmonicForcing}`, empty when autonomous), and `info` (a NamedTuple whose
`Ω` is the matching vector of forcing frequencies).

A **result container, not an entry point.** There is one `parametrise` and it is
MORFE's; this packages its output together with the physics metadata that
`real_dynamics`, `print_equations` and `save_rom` need — `master` and `N_EXT` in
particular, which cannot be recovered from `W` and `R` alone.

Build it from `build_model`'s `meta`:

```julia
(; model, spectral, meta) = build_model(case; master = [1], expansion_order = order)
W, R = parametrise(model, spectral, order;
                   resonance = ResonanceConfig(style = :complex_normal_form, tol = 0.05))
rom = InvariantManifoldROM(W, R, meta; master = [1], order = order)
```
"""
struct InvariantManifoldROM{TW, TR, T, FRC}
    W::TW
    R::TR
    eigenvalues::Vector{Complex{T}}
    master::Vector{Int}
    order::Int
    forcing::FRC
    info::NamedTuple
end

"""
	InvariantManifoldROM(W, R, meta; master, order, info = (;))

Package a solve against the `meta` that [`build_model`](@ref) returned. Pulls the
spectrum, the forcing records, `N_EXT` and the timings from `meta`; `info` merges
extra rows on top (a caller's own `solve_time_s`, say).
"""
function InvariantManifoldROM(W, R, meta::NamedTuple; master::Vector{Int}, order::Int,
        info::NamedTuple = (;))
    base = (; N_EXT = meta.N_EXT, Ω = meta.Ω, eig_time_s = meta.eig_time_s)
    merged = haskey(meta, :case_info) ? (; meta.case_info..., base..., info...) :
             (; base..., info...)
    return InvariantManifoldROM(W, R, collect(meta.spectrum.eigenvalues), master, order,
        meta.forcings, merged)
end

# The rows only this physics has; the shared skeleton (sizes, order, stage
# timings) comes from Common.write_summary.
function Common.summary_entries(m::AssembledMechanicalModel, rom::InvariantManifoldROM)
    rows = Pair{String, Any}[
        "model" => "geometrically nonlinear structure (St. Venant-Kirchhoff)",
        "material" => _material_summary(m.material),
        "damping" => "Rayleigh  α=$(m.damping.α)  β=$(m.damping.β)",
        "master_pairs" => rom.master,
        "master_eigenvalues" => rom.eigenvalues[reduce(
            vcat, [[2p - 1, 2p] for p in rom.master])],
        "n_monomials" => rom.info.n_monomials,
    ]
    isempty(rom.forcing) ? push!(rows, "forcing" => "none (autonomous)") :
    push!(rows, "forcing" => join(["mode $(f.mode) a=$(f.amplitude) Ω=$Ω"
                                   for (f, Ω) in zip(rom.forcing, rom.info.Ω)], "; "))
    return rows
end

Common.summary_entries(m::AssembledMechanicalModel, ::Nothing) =
    Pair{String, Any}["model" => "geometrically nonlinear structure (St. Venant-Kirchhoff)",
        "material" => _material_summary(m.material)]

function Base.show(io::IO, ::MIME"text/plain", rom::InvariantManifoldROM)
    ROM = 2 * length(rom.master)
    println(io, "InvariantManifoldROM")
    println(io, "  master pairs : $(rom.master)  (ROM = $ROM, N_EXT = $(rom.info.N_EXT))")
    for p in rom.master
        λ = rom.eigenvalues[2p - 1]
        println(io, "    pair $p: λ = $λ   (f = $(abs(λ) / 2π))")
    end
    if isempty(rom.forcing)
        println(io, "  forcing      : none (autonomous)")
    else
        for (k, (f, Ω)) in enumerate(zip(rom.forcing, rom.info.Ω))
            label = k == 1 ? "  forcing      : " : "                 "
            println(io,
                "$label[$k] mode $(f.mode), amplitude $(f.amplitude), Ω = $Ω")
        end
    end
    println(io, "  order        : $(rom.order)   ($(rom.info.n_monomials) monomials)")
    print(io, "  solve time   : $(round(rom.info.solve_time_s; digits = 2)) s")
end

# ── Anisotropic materials ───────────────────────────────────────────────────
# `SVKMaterial` above is the isotropic special case. For crystalline materials
# (silicon, quartz, …) the stress law needs the full stiffness tensor.

"""
    AnisotropicMaterial(D, ρ)

St. Venant-Kirchhoff material with a general 6×6 Voigt stiffness `D` (ordering
`[11, 22, 33, 23, 13, 12]`, engineering shear strains) and density `ρ`.

Use [`CubicCrystal`](@ref) for cubic crystals given by `c₁₁, c₁₂, c₄₄`.
"""
struct AnisotropicMaterial
	D::SMatrix{6, 6, Float64, 36}
	ρ::Float64
end
AnisotropicMaterial(D::AbstractMatrix, ρ::Real) =
	AnisotropicMaterial(SMatrix{6, 6, Float64}(D), Float64(ρ))

# Voigt index of a tensor index pair: (1,1)→1 (2,2)→2 (3,3)→3 (2,3)→4 (1,3)→5 (1,2)→6
@inline _voigt_index(i::Int, j::Int) =
	i == j ? i : (i + j == 5 ? 4 : (i + j == 4 ? 5 : 6))

const _VOIGT_PAIRS = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

"""
    rotate_voigt(D, Q) -> SMatrix{6,6}

Rotate a Voigt stiffness matrix by the orthogonal matrix `Q`.

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
    CubicCrystal(; c11, c12, c44, ρ, rotation = I) -> AnisotropicMaterial

Cubic crystal (silicon, germanium, …) from its three independent constants, with
the crystal optionally rotated into the lab frame by `rotation` (a 3×3 matrix, or
an angle in radians about the z axis).

The isotropic limit is `c11 = λ + 2μ`, `c12 = λ`, `c44 = μ` — then this material
reproduces `SVKMaterial(λ, μ)` exactly (asserted by the test suite).
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

Voigt stiffness of any supported material (`SVKMaterial` returns its isotropic
matrix), useful for inspection and for cross-checking anisotropic input.
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

_material_summary(m::AnisotropicMaterial) =
	"anisotropic SVK  D₁₁=$(m.D[1,1])  D₁₂=$(m.D[1,2])  D₄₄=$(m.D[4,4])  ρ=$(m.ρ)"
