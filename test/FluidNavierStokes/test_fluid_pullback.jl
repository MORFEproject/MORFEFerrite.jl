using Test
using Tensors
using StaticArrays: SVector
using Ferrite
using LinearAlgebra
using MORFE
using MORFEFerrite

const PGF = MORFEFerrite.ParametricGeometry
const FNF = MORFEFerrite.FluidNavierStokes

function eval_series(a, μ)
    return sum(a[k] * μ^PGF.GeometryParameterBasis([length(a)-1]).mset.exponents[k][1]
               for k in eachindex(a))
end

function _convection_test_grid()
    grid = generate_grid(Triangle, (3, 2), Vec((0.0, 0.0)), Vec((1.0, 0.4)))
    addfacetset!(grid, "Inlet", x -> abs(x[1]) < 1e-12)
    addfacetset!(grid, "Outlet", x -> abs(x[1]-1.0) < 1e-12)
    addfacetset!(grid, "Farfield", x -> abs(x[2]-0.4) < 1e-12)
    addfacetset!(grid, "Obstacle", x -> abs(x[2]) < 1e-12 &&
        0.30 < x[1] < 0.70)
    return grid
end

function _affine_moved_grid(grid, G, mu)
    moved = deepcopy(grid)
    for index in eachindex(moved.nodes)
        X = grid.nodes[index].x
        moved.nodes[index] = Node(X + mu*(G ⋅ X))
    end
    return moved
end

function _parametric_convection_pair(pm, basis, u, v, mu)
    result = zeros(ComplexF64, length(u))
    for (index, exponent) in enumerate(basis.mset.exponents)
        FNF._apply_convection_coefficient!(
            result, pm, index, u, v, mu^exponent[1])
    end
    return result
end

function _series_action(arrays, basis, vector, mu)
    result = zeros(promote_type(eltype(vector), Float64), length(vector))
    for (index, exponent) in enumerate(basis.mset.exponents)
        mul!(result, arrays[index], vector, mu^exponent[1], one(eltype(result)))
    end
    return result
end

function _series_vector(arrays, basis, mu; skip_constant=false)
    result = zeros(eltype(first(arrays)), length(first(arrays)))
    for (index, exponent) in enumerate(basis.mset.exponents)
        skip_constant && iszero(exponent[1]) && continue
        result .+= mu^exponent[1] .* arrays[index]
    end
    return result
end

function _tiny_parametric_fluid(; freestream=(eps(Float64), 0.0))
    grid = _convection_test_grid()
    bc = FNF.UniformFreestreamBC(freestream)
    fom = FNF.setup_fem(grid; obstacle_tag="Obstacle", reference_length=0.1,
        quadrature_order=6, boundary_conditions=bc)
    state = zeros(ndofs(fom.dh))
    apply!(state, fom.ch_full)
    B0, B1 = FNF.assemble_linear_operators(state, fom; Re0=100.0)
    K, Krect = FNF.assemble_K_visc(fom)
    Kscaled = -fom.reference_length .* K
    h0 = -fom.reference_length .* (Krect*state)
    info = (; n_free=fom.n_free, n_free_dpim=fom.n_free_dpim,
        reference_length=fom.reference_length, obstacle_tag=fom.obstacle_tag)
    case = FNF.AssembledFluidModel(fom, 100.0, state, (B0, B1),
        Kscaled, Krect, h0, info)
    # Match the certified production reciprocal order.  Convection remains
    # exactly affine regardless of this requested viscosity-series order.
    basis = PGF.GeometryParameterBasis([12])
    I2 = one(Tensor{2,2,Float64})
    G = Tensor{2,2}((0.35, -0.25, 0.40, -0.12))
    pm = FNF.parametric_model(case, x -> (I2, G);
        geometry_parameter_basis=basis, reynolds_scale=1e-3,
        include_reynolds=true)
    return (; grid, bc, fom, case, basis, I2, G, pm)
end

@testset "fluid quadratic pullback equals moved-mesh convection" begin
    grid = _convection_test_grid()
    bc = FNF.UniformFreestreamBC((1.0, 0.0))
    fom0 = FNF.setup_fem(grid; obstacle_tag="Obstacle", reference_length=0.1,
        quadrature_order=6, boundary_conditions=bc)
    basis = PGF.GeometryParameterBasis([4])
    I2 = one(Tensor{2,2,Float64})
    G = Tensor{2,2}((0.35, -0.25, 0.40, -0.12))
    cache = PGF.PullbackCache(fom0.dh, fom0.cv_vel, x -> (I2, G), basis)
    structure = FNF._enforce_affine_2d_fluid_structure!(cache)
    pm = (; base=(; fom=fom0), cache)

    @test structure.applied
    @test structure.adj_discarded > 0
    @test structure.adj_discarded <= structure.tolerance
    @test structure.det_discarded <= structure.tolerance

    n = fom0.n_free_dpim
    u = ComplexF64[
        sin(0.17index) + 0.3cos(0.11index) +
        im*(0.2sin(0.07index)-0.1cos(0.19index)) for index in 1:n]
    v = ComplexF64[
        cos(0.13index) - 0.2sin(0.23index) +
        im*(0.15cos(0.05index)+0.25sin(0.29index)) for index in 1:n]

    # In two dimensions adj(I+mu*G) is affine. Higher cached coefficients must
    # therefore vanish exactly, independently of the requested series order.
    inactive_adjugate_is_zero = true
    for cell in eachindex(cache.adj), q in eachindex(cache.adj[cell])
        for (index, exponent) in enumerate(basis.mset.exponents)
            exponent[1] >= 2 || continue
            inactive_adjugate_is_zero &= iszero(cache.adj[cell][q][index])
        end
    end
    @test inactive_adjugate_is_zero

    for mu in (-0.1, 0.0, 0.1)
        F = I2 + mu*G
        exact_adjugate = det(F)*inv(F)
        max_adjugate_error = 0.0
        for cell in eachindex(cache.adj), q in eachindex(cache.adj[cell])
            represented = sum(mu^exponent[1]*cache.adj[cell][q][index]
                for (index, exponent) in enumerate(basis.mset.exponents))
            max_adjugate_error = max(max_adjugate_error,
                norm(represented-exact_adjugate))
        end
        @test max_adjugate_error < 1e-13

        moved = _affine_moved_grid(grid, G, mu)
        fomm = FNF.setup_fem(moved; obstacle_tag="Obstacle", reference_length=0.1,
            quadrature_order=6, boundary_conditions=bc)
        @test fomm.free_dpim == fom0.free_dpim
        exact = zeros(ComplexF64, n)
        FNF._eval_perturbation_convection_pair!(exact, u, v, fomm)
        self_fast = zeros(ComplexF64, n)
        self_general = zeros(ComplexF64, n)
        FNF._eval_perturbation_convection_pair!(self_fast, u, u, fomm)
        FNF._accumulate_fluid_convection_pair!(self_general, u, u, fomm,
            (cell, q) -> nothing, 1.0, Val(:free), Val(:free), Val(false))
        @test self_fast ≈ self_general rtol=1e-13 atol=1e-13
        pulled_back = _parametric_convection_pair(pm, basis, u, v, mu)
        relative = norm(pulled_back-exact)/max(norm(exact), eps(Float64))
        @test relative < 1e-11
        @test _parametric_convection_pair(pm, basis, u, v, mu) ≈
              _parametric_convection_pair(pm, basis, v, u, mu) rtol=1e-13 atol=1e-13
    end
end

@testset "non-affine polynomial fluid geometry remains unmodified" begin
    grid = _convection_test_grid()
    fom = FNF.setup_fem(grid; obstacle_tag="Obstacle", reference_length=0.1,
        quadrature_order=6,
        boundary_conditions=FNF.UniformFreestreamBC((1.0, 0.0)))
    basis = PGF.GeometryParameterBasis([4])
    I2 = one(Tensor{2,2,Float64})
    G = Tensor{2,2}((0.20, -0.10, 0.08, -0.05))
    H = Tensor{2,2}((0.04, 0.03, -0.02, 0.01))
    provider = x -> [SVector(0) => I2, SVector(1) => G, SVector(2) => H]
    cache = PGF.PullbackCache(fom.dh, fom.cv_vel, provider, basis)
    before_adj = deepcopy(cache.adj)
    before_det = deepcopy(cache.det)
    structure = FNF._enforce_affine_2d_fluid_structure!(cache)
    @test !structure.applied
    @test structure.adj_discarded > structure.tolerance
    @test cache.adj == before_adj
    @test cache.det == before_det
end

@testset "fluid pullback component actions and prescribed lifting" begin
    tiny = _tiny_parametric_fluid()
    (; grid, bc, fom, basis, G, pm) = tiny
    n = fom.n_free_dpim
    velocity_rows = FNF.velocity_dof_mask(fom)[fom.free_dpim]
    u = ComplexF64[sin(0.17i)+im*0.2cos(0.09i) for i in 1:n]
    p = ComplexF64[cos(0.13i)-im*0.15sin(0.21i) for i in 1:n]
    u[.!velocity_rows] .= 0
    p[velocity_rows] .= 0
    nu0 = fom.reference_length/tiny.case.Re₀

    for mu in (-0.1, 0.0, 0.1)
        moved = _affine_moved_grid(grid, G, mu)
        fomm = FNF.setup_fem(moved; obstacle_tag="Obstacle",
            reference_length=0.1, quadrature_order=6,
            boundary_conditions=bc)
        zero_state = zeros(ndofs(fomm.dh))
        B0m, B1m = FNF.assemble_linear_operators(zero_state, fomm; Re0=100.0)
        Km, _ = FNF.assemble_K_visc(fomm)
        mass_error = norm(_series_action(pm.B1_geometry, basis, u, mu)-B1m*u) /
            max(norm(B1m*u), eps(Float64))
        visc_error = norm(_series_action(pm.viscosity, basis, u, mu)-Km*u) /
            max(norm(Km*u), eps(Float64))
        Pparam_u = _series_action(pm.nonconvective, basis, u, mu) -
            nu0*_series_action(pm.viscosity, basis, u, mu)
        Pparam_p = _series_action(pm.nonconvective, basis, p, mu) -
            nu0*_series_action(pm.viscosity, basis, p, mu)
        Pm = B0m-nu0*Km
        @test mass_error < 1e-11
        @test visc_error < 1e-9
        @test norm(Pparam_u-Pm*u)/max(norm(Pm*u), eps(Float64)) < 1e-11
        @test norm(Pparam_p-Pm*p)/max(norm(Pm*p), eps(Float64)) < 1e-11
    end


    # Exercise the actual NthOrderModel maps, not just their source arrays.
    # This catches missing factorials, external-component routing, map signs,
    # and the ORD=2 mass correction in one small deterministic problem.
    mode = copy(u)
    right = reshape(hcat(mode, conj.(mode)), n, 1, 2)
    left = reshape(hcat(mode, conj.(mode)), n, 1, 2)
    spectrum = Spectrum(DefaultEigensolver(),
        ComplexF64[-1+2im, -1-2im], right, left)
    built = FNF.build_model(pm; spectrum, master=[1])
    @test built.meta.ORD == 2
    @test all(iszero, built.model.external_system.eigenvalues)
    mu = 0.07
    xi = -0.03
    x = 2e-3 .* u
    xdot = -1e-3 .* p
    model_action = zeros(ComplexF64, n)
    maximum_degree = maximum(term.deg for term in built.model.nonlinear_terms)
    for degree in 1:maximum_degree
        evaluate_nonlinear_terms!(model_action, built.model, degree,
            (x, xdot), ComplexF64[mu, xi])
    end
    B0mu_x = _series_action(pm.B0_geometry, basis, x, mu)
    B1mu_xdot = _series_action(pm.B1_geometry, basis, xdot, mu)
    Vmu_x = _series_action(pm.viscosity, basis, x, mu)
    expected = -(B0mu_x-pm.B0_geometry[1]*x) -
        (B1mu_xdot-pm.B1_geometry[1]*xdot) +
        _parametric_convection_pair(pm, basis, x, x, mu) +
        _series_vector(pm.h_geometry, basis, mu; skip_constant=true) -
        xi*fom.reference_length*pm.reynolds_scale.*Vmu_x +
        xi.*_series_vector(pm.h_reynolds, basis, mu)
    @test norm(model_action-expected)/max(norm(expected), eps(Float64)) < 1e-12

    @testset "mixed Reynolds-geometry maps use symmetric polarisation" begin
        # MORFE applies the permutation count associated with a canonical
        # external-factor tuple.  A map representing mu^k*xi must therefore
        # be the symmetric polarisation of that monomial.  Assigning xi to a
        # distinguished argument slot overcounts the coefficient by k+1 even
        # though direct evaluation at r_1=...=r_m looks correct.
        x1 = ComplexF64[0.37-0.21im]
        A1 = ComplexF64[1.9;;]
        for k in 0:4
            mm = k+1
            mset = all_multiindices_up_to(4, mm+1; min_degree=1)
            Wtest, _ = MORFE.ParametrisationMethod.create_parametrisation_method_objects(
                mset, 2, 1, 2, 2, ComplexF64)
            iz = findfirst(==(SVector(1, 0, 0, 0)), mset.exponents)
            Wtest.poly.coefficients[:, 1, iz] .= x1
            closure = Base.invokelatest(
                FNF._pf_reynolds_linear, Val(mm), A1)
            term = MultilinearMap(
                closure, (1, 0), mm; fully_asymmetric=false)
            Z = zeros(ComplexF64, 1, 1)
            coefficient_model = NthOrderModel(
                (Z, Z, Z), (term,), ExternalSystem((0im, 0im)))
            exponent = SVector(1, 0, k, 1)
            coefficient = MORFE.MultilinearTerms.compute_multilinear_terms(
                coefficient_model, exponent, Wtest)
            @test coefficient ≈ -(A1*x1) atol=1e-14 rtol=1e-14

            direct = zeros(ComplexF64, 1)
            r = ComplexF64[0.13, -0.08]
            MORFE.MultilinearMaps.evaluate_term!(
                direct, term, (x1, zero(x1)), r)
            @test direct ≈ -(r[1]^k*r[2]).*(A1*x1) atol=1e-14 rtol=1e-14
        end
    end

    # Prescribed freestream values enter the base-state convection, whereas a
    # perturbation remains zero on every prescribed DOF.  Compare both mixed
    # and full/full actions with ordinary assembly on the moved mesh.
    lifted_grid = _convection_test_grid()
    lifted_bc = FNF.UniformFreestreamBC((1.0, 0.0))
    lifted_fom = FNF.setup_fem(lifted_grid; obstacle_tag="Obstacle",
        reference_length=0.1, quadrature_order=6,
        boundary_conditions=lifted_bc)
    base = zeros(ndofs(lifted_fom.dh)); apply!(base, lifted_fom.ch_full)
    free_state = ComplexF64[sin(0.12i)+im*0.1cos(0.17i) for i in 1:n]
    for mu in (-0.1, 0.1)
        moved = _affine_moved_grid(lifted_grid, G, mu)
        fomm = FNF.setup_fem(moved; obstacle_tag="Obstacle",
            reference_length=0.1, quadrature_order=6,
            boundary_conditions=lifted_bc)
        moved_base = zeros(ndofs(fomm.dh)); apply!(moved_base, fomm.ch_full)
        @test moved_base == base
        adjugate_at = (cell, q) -> sum(
            mu^exponent[1]*pm.cache.adj[cell][q][index]
            for (index, exponent) in enumerate(basis.mset.exponents))
        for (first_state, second_state, layout1, layout2) in (
                (base, free_state, Val(:full), Val(:free)),
                (base, base, Val(:full), Val(:full)))
            pulled = zeros(ComplexF64, n)
            FNF._accumulate_fluid_convection_pair!(pulled,
                first_state, second_state, lifted_fom, adjugate_at, 1.0,
                layout1, layout2)
            exact = zeros(ComplexF64, n)
            FNF._eval_convection_pair_with_lifting!(exact,
                first_state, second_state, fomm, layout1, layout2)
            @test norm(pulled-exact)/max(norm(exact), eps(Float64)) < 1e-11
        end
    end
end

@testset "fluid residual Hessian equals the bilinear convection action" begin
    tiny = _tiny_parametric_fluid()
    fom = tiny.fom
    n = fom.n_free_dpim
    velocity_rows = FNF.velocity_dof_mask(fom)[fom.free_dpim]
    u = [sin(0.17i)+0.2cos(0.11i) for i in 1:n]
    v = [cos(0.13i)-0.15sin(0.23i) for i in 1:n]
    u[.!velocity_rows] .= 0
    v[.!velocity_rows] .= 0
    u ./= norm(u); v ./= norm(v)
    Q = zeros(n)
    FNF._eval_perturbation_convection_pair!(Q, u, v, fom)
    matrix = allocate_matrix(fom.dh)
    residual = zeros(ndofs(fom.dh))
    function moved_residual(x)
        state = zeros(ndofs(fom.dh))
        state[fom.free_dpim] .= x
        FNF.assemble_steady_nse!(matrix, residual, state, fom, 100.0)
        value = copy(residual[fom.free_dpim])
        value[.!velocity_rows] .*= -1
        return value
    end
    errors = Float64[]
    for step in (1e-2, 3e-3, 1e-3, 3e-4)
        mixed = (moved_residual(step*(u+v)) - moved_residual(step*(u-v)) -
                 moved_residual(step*(-u+v)) + moved_residual(-step*(u+v))) /
                (4step^2)
        # The Newton residual contains +(u dot grad)u, whereas the MORFE RHS
        # convention stores Q(u,v)=-(symmetric convection action).
        push!(errors, norm(mixed+2Q)/max(norm(2Q), eps(Float64)))
    end
    @test minimum(errors) < 1e-7
end

@testset "mixed-fluid pullback identities" begin
    basis = PGF.GeometryParameterBasis([8])
    I2 = one(Tensor{2,2,Float64})
    gradv = Tensor{2,2}((1.1, -0.3, 0.7, 0.2))
    gradu = Tensor{2,2}((-0.4, 0.8, 0.1, 1.3))

    for G in (zero(I2), Tensor{2,2}((0.0, 0.0, 0.65, 0.0)),
              Tensor{2,2}((0.35, 0.0, 0.0, -0.12)))
        J = PGF.jacobian_series((I2, G), basis)
        dets, adj = PGF.det_adj_series(J, basis)
        invdet = PGF.reciprocal_series(dets, basis)
        visc = FNF._fluid_viscosity_series(gradv, gradu, adj, invdet, basis)
        divs = FNF._fluid_divergence_series(gradu, adj)
        for μ in (-0.1, 0.0, 0.1)
            F = I2 + μ * G
            A = det(F) * inv(F)
            exact_visc = 2 * (symmetric(gradv ⋅ A) ⊡ symmetric(gradu ⋅ A)) / det(F)
            exact_div = tr(gradu ⋅ A)
            @test PGF.series_value(visc, basis, [μ]) ≈ exact_visc rtol = 1e-9 atol = 1e-12
            @test PGF.series_value(divs, basis, [μ]) ≈ exact_div rtol = 1e-12 atol = 1e-12
            @test PGF.series_value(dets, basis, [μ]) ≈ det(F) rtol = 1e-13 atol = 1e-13
        end
    end

    # Geometry μ can be routed to any external component. The fluid adapter
    # reserves component 1 for μ and component 2 for ξ.
    @test PGF._expand_multiindex(SVector{1,Int}(2), [2]) == (2, 2)
end
