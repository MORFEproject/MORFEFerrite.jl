# The SVK terms opt into MORFE's flux summation: the single-pass assembly sums the first
# Piola-Kirchhoff stresses of all factorisations of both terms at a quadrature point and
# tests the sum against the shape-function gradients once. Checked against
# `accumulate_qp!` at single quadrature points, and, monomial by monomial, against a
# closure that calls `accumulate_qp!` once per factorisation and traverses the mesh itself.

using MORFE
using MORFEFerrite
using MORFE.MultilinearTerms: build_multilinear_terms_cache, compute_multilinear_terms!
using MORFE.Polynomials: DensePolynomial
using Ferrite, LinearAlgebra
using Test

# The closure route of the benchmark: an SVK term behind a plain `MultilinearMap`.
struct _SVKPerFactorisation{T, B, F}
    t::T
    ∇u::B
    Fe::F
end

function _SVKPerFactorisation(t)
    _SVKPerFactorisation(t,
        similar(MORFE.fem_qp_buffer(t), t.deg, MORFE.fem_n_qp(t)),
        zeros(ComplexF64, MORFE.fem_ndofs_per_cell(t)))
end

function (u::_SVKPerFactorisation)(res, args::Vararg{Any, N}) where {N}
    t = u.t
    for element in MORFE.fem_elements(t)
        MORFE.fem_reinit!(element, t)
        for k in 1:N
            MORFE.scatter_qp!(view(u.∇u, k, :), args[k], element, t)
        end
        fill!(u.Fe, 0)
        for q in 1:MORFE.fem_n_qp(t)
            MORFE.accumulate_qp!(u.Fe, ntuple(k -> u.∇u[k, q], Val(N)), 1, element, q,
                MORFE.fem_getdetJdV(element, q, t), t)
        end
        MORFE.assemble_element!(res, u.Fe, element, t)
    end
    return res
end

@testset "SVK flux summation" begin
    grid = generate_grid(Hexahedron, (3, 1, 1), Vec(0.0, 0.0, 0.0), Vec(60.0, 5.0, 3.0))
    addfacetset!(grid, "Dirichlet", x -> abs(x[1]) < 1e-8 || abs(x[1] - 60.0) < 1e-8)
    case = MORFEFerrite.StructuralSVK.mechanical_model(grid;
        material = MORFEFerrite.StructuralSVK.SVKMaterial(E = 160e3, ν = 0.22, ρ = 2.32e-3),
        damping = MORFEFerrite.StructuralSVK.RayleighDamping(α = 0.0, β = 0.0),
        dirichlet = "Dirichlet")
    # (z₁, z̄₁) to order 5: cubic factorisations with multipliers 1, 3 and 6.
    mset = build_multiindex_set(5, 2)
    terms = Tuple(case.term_factory(d) for d in (2, 3))

    @testset "scatter from the scalar shape gradients ≡ function_gradient" begin
        t = terms[1]
        N = size(case.B[1], 1)
        n_qp = MORFE.fem_n_qp(t)
        ws = [randn(ComplexF64, N) for _ in 1:3]
        vals = Vector{Tensor{2, 3, ComplexF64, 9}}(undef, n_qp)
        u_e = zeros(ComplexF64, MORFE.fem_ndofs_per_cell(t))
        identical = true
        for cell in MORFE.fem_elements(t), w in ws

            MORFE.fem_reinit!(cell, t)
            MORFE.scatter_qp!(vals, w, cell, t)
            for (i, d) in enumerate(celldofs(cell))
                l = t.dof_to_free[d]
                u_e[i] = l == 0 ? zero(ComplexF64) : w[l]
            end
            identical &= all(q -> vals[q] == function_gradient(t.cell_values, q, u_e), 1:n_qp)
        end
        @test identical
        # the term's own buffer holds only the direct evaluation's arguments
        @test size(MORFE.fem_qp_buffer(t)) == (2, n_qp)
        # a scalar interpolation cannot carry the displacement field
        scalar_cell_values = CellValues(QuadratureRule{RefHexahedron}(2), Lagrange{
            RefHexahedron, 1}())
        @test_throws ArgumentError MORFEFerrite.StructuralSVK.svk_nonlinearity(
            2, case.info.dof_handler, scalar_cell_values, case.info.dof_to_free, t.n_free,
            1.0, 1.0)
    end

    @testset "one quadrature point: summed stresses ≡ accumulate_qp!" begin
        cell = first(CellIterator(case.info.dof_handler))
        for t in terms
            @test MORFE.fem_flux_zero(t) === zero(Tensor{2, 3, ComplexF64, 9})
            MORFE.fem_reinit!(cell, t)
            n = MORFE.fem_ndofs_per_cell(t)
            for q in (1, MORFE.fem_n_qp(t))
                dΩ = MORFE.fem_getdetJdV(cell, q, t)
                Fe_accumulated = zeros(ComplexF64, n)
                Fe_contracted = zeros(ComplexF64, n)
                flux = MORFE.fem_flux_zero(t)
                for m in (1, 2, 6)
                    args = ntuple(_ -> Tensor{2, 3}(randn(ComplexF64, 3, 3)), t.deg)
                    MORFE.accumulate_qp!(Fe_accumulated, args, m, cell, q, dΩ, t)
                    flux += m * MORFE.flux_qp(args, cell, q, t)
                end
                MORFE.contract_qp!(Fe_contracted, flux, cell, q, dΩ, t)
                @test norm(Fe_accumulated) > 0
                @test norm(Fe_contracted - Fe_accumulated) <= 1e-13 * norm(Fe_accumulated)
            end
        end
    end

    @testset "one quadrature point: joint stress of both terms ≡ Σ m P₂ + Σ m P₃" begin
        # Repeated factors, every multiplier of the symmetric enumeration, and a coefficient
        # (4) that only the cubic term reads.
        cell = first(CellIterator(case.info.dof_handler))
        FE = MORFE.MultilinearTerms.FEMEntry
        entries = (FE{2, 0}[FE{2, 0}(2, (1, 2), ()), FE{2, 0}(1, (3, 3), ())],
            FE{3, 0}[FE{3, 0}(6, (1, 2, 4), ()), FE{3, 0}(3, (3, 3, 1), ()),
                FE{3, 0}(1, (4, 4, 4), ())])
        vals = [Tensor{2, 3}(randn(ComplexF64, 3, 3)) for _ in 1:5]
        D = randn(6, 6)
        anisotropic = map(
            t -> MORFEFerrite.StructuralSVK.FerriteGeometricNonlinearity{t.deg}(
                t.dof_handler, t.cell_values, case.info.dof_to_free, t.n_free,
                MORFEFerrite.StructuralSVK.VoigtStress(D + D')),
            terms)
        for ts in (terms, anisotropic), k in (1:2, 1:1, 2:2)

            reference = MORFE.fem_flux_zero(ts[1])
            for i in k, e in entries[i]

                reference += e.multiplier *
                             MORFE.flux_qp(map(j -> vals[j], e.factors), cell, 1, ts[i])
            end
            joint = MORFE.accumulate_terms_qp!(MORFE.fem_flux_zero(ts[1]), nothing,
                view(vals, 1:4), (), entries[k], ts[k], cell, 1, 1.0)
            # largest entry: Tensors.jl's `norm` does not conjugate complex entries
            @test maximum(abs, reference.data) > 0
            @test maximum(abs, (joint - reference).data) <=
                  1e-13 * maximum(abs, reference.data)
        end
        @test_throws ArgumentError MORFE.accumulate_terms_qp!(
            MORFE.fem_flux_zero(terms[1]), nothing, view(vals, 1:4), (), entries,
            (terms[1], anisotropic[2]), cell, 1, 1.0)
    end

    @testset "the quadratic and cubic terms share one element loop" begin
        loop(t) = MORFE.fem_element_loop(t)
        @test isequal(loop(terms[1]), loop(terms[2]))
        # equal cell values in another object: the driver's reinitialisation would not
        # reach them, so the term gets its own loop
        separate = MORFEFerrite.StructuralSVK.svk_nonlinearity(
            3, case.info.dof_handler, deepcopy(terms[1].cell_values),
            case.info.dof_to_free, terms[1].n_free, 1.0, 1.0)
        @test !isequal(loop(separate), loop(terms[1]))
    end

    @testset "every monomial: single-pass flux sum ≡ one accumulate_qp! per factorisation" begin
        N = size(case.B[1], 1)
        W = Parametrisation(DensePolynomial(zeros(ComplexF64, N, 2, length(mset)), mset), 0)
        W.poly.coefficients .= randn(ComplexF64, size(W.poly.coefficients))
        closures = map(
            t -> MultilinearMap(_SVKPerFactorisation(t); multiindex = t.multiindex,
                multiplicity_external = 0, fully_asymmetric = false),
            terms)
        models = (NthOrderModel(case.B, terms), NthOrderModel(case.B, closures))
        caches = map(m -> build_multilinear_terms_cache(m, W), models)
        @test length(caches[1].fem_groups) == 1   # both maps in one pass
        worst, largest = 0.0, 0.0
        for l in eachindex(mset.exponents)
            sum(mset.exponents[l]) >= 2 || continue
            single_pass, closure = map(models, caches) do m, c
                res = zeros(ComplexF64, N)
                compute_multilinear_terms!(res, m, l, W, c)
                res
            end
            worst = max(worst, norm(single_pass - closure) / norm(closure))
            largest = max(largest, norm(closure))
        end
        @test largest > 0
        @test worst < 1e-12
    end
end
