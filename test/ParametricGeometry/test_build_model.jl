# Contract tests for the `build_model` layer.
#
# These check the SHAPE of what build_model returns, not the physics — the
# numbers are gated by test_kernel_equivalence.jl. What matters here is that the
# contract in src/common/assembled_model.jl actually holds: a NamedTuple, an ORD
# derived rather than assumed, a SpectralData reconciled against THAT order, and
# a conjugate permutation derived rather than written out.

using Test
using MORFE, MORFEFerrite
using Ferrite, Tensors, StaticArrays
using LinearAlgebra, SparseArrays

const PGb = MORFEFerrite.ParametricGeometry
const SVKb = MORFEFerrite.StructuralSVK
const Tens3b = Tensor{2, 3, Float64, 9}

function _bm_setup()
    grid = generate_grid(Hexahedron, (2, 1, 1), Vec(0.0, 0.0, 0.0), Vec(2.0, 1.0, 1.0))
    ip = Lagrange{RefHexahedron, 2}()^3
    qr = QuadratureRule{RefHexahedron}(3)
    cv = CellValues(qr, ip, Lagrange{RefHexahedron, 1}())
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> zeros(3), [1, 2, 3]))
    close!(ch); update!(ch, 0.0)
    free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
    return dh, cv, free
end

# One geometry parameter (odd N_EXT — the case an adjacent-pairs permutation
# formula cannot express), then two.
_geom_1p(x₀) = (one(Tens3b), Tens3b((i, j) -> (i == 1 && j == 1) ? 0.5 : 0.0))
_geom_2p(x₀) = (one(Tens3b),
    Tens3b((i, j) -> (i == 1 && j == 1) ? 0.5 : 0.0),
    Tens3b((i, j) -> (i == 2 && j == 1) ? 0.2 : 0.0))

@testset "ParametricGeometry build_model contract" begin
    dh, cv, free = _bm_setup()
    material = SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3)
    damping = RayleighDamping(α = 0.0, β = 0.0)

    for (label, geom, n_par) in (("1 parameter (odd N_EXT)", _geom_1p, 1),
                                 ("2 parameters", _geom_2p, 2))
        @testset "$label" begin
            basis = PGb.GeometryParameterBasis(fill(2, n_par))
            pcase = SVKb.parametric_model(dh, cv, geom;
                geometry_parameter_basis = basis, material = material,
                damping = damping, free = free)
            @test pcase isa MORFEFerrite.AbstractAssembledModel

            K0, M0 = SVKb.base_operators(pcase)
            ep = spectrum(K0, M0, StructureModalDampingEigensolver(6, 0.0, 0.0);
                sorter! = (args...) -> nothing)

            out = build_model(pcase; master = [1], spectrum = ep)

            # ── The contract: a NamedTuple with these three fields ──────────
            @test out isa NamedTuple
            @test issubset((:model, :spectral, :meta), keys(out))
            (; model, spectral, meta) = out
            @test model isa MORFE.NthOrderModel
            @test spectral isa MORFE.SpectralData

            # ── ORD is DERIVED from the operators' arities, not assumed. A
            #    parametric mass is a correction on the highest derivative, so a
            #    second-order structure needs ORD = 3 and a zero fourth block.
            @test PGb.model_order(pcase) == 3
            @test length(model.linear_terms) == 4
            @test nnz(model.linear_terms[4]) == 0

            # ── SpectralData is reconciled against THAT model's order ───────
            @test size(spectral.master.right_blocks, 2) == 3
            @test size(spectral.master.left_blocks, 2) == 3

            # ── The θ states are frozen: N_EXT external variables, eigenvalue 0
            @test model.external_system !== nothing
            @test length(model.external_system.eigenvalues) == n_par
            @test all(iszero, model.external_system.eigenvalues)
            @test meta.N_EXT == n_par

            # ── The permutation is DERIVED and spans the external states ────
            #    z₁ ↔ z₂, then each real θ self-paired: [2, 1, 3, 4, …].
            @test meta.conjugate_permutation == vcat([2, 1], collect(3:(2 + n_par)))

            # ── build_model builds no reduction concepts ────────────────────
            @test !haskey(meta, :mset)
            @test !haskey(meta, :resonance)
        end
    end
end

@testset "build_model is free of reduction concepts (source grep)" begin
    # The invariant from MODULE_ARCHITECTURE.md §9: in a finished module,
    # build_model mentions none of these. Cheaper to enforce than to re-litigate.
    for f in ("ParametricGeometry/build_model.jl", "StructuralSVK/build_model.jl")
        src = read(joinpath(@__DIR__, "..", "..", "src", f), String)
        # Strip comments and docstrings — prose may legitimately name them.
        code = join([l for l in split(src, '\n') if !startswith(strip(l), "#")], '\n')
        code = replace(code, r"\"\"\".*?\"\"\""s => "")
        @test !occursin("resonance_set_from_", code)
        @test !occursin("ResonanceSet(", code)
        @test !occursin("validate_multiindex_set", code)
        @test !occursin("all_multiindices_up_to", code)
    end
end
