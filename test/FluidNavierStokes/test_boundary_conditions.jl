using Ferrite
using MORFEFerrite
using Test

const FNBC = MORFEFerrite.FluidNavierStokes

function _bc_test_grid()
    grid = generate_grid(Triangle, (10, 4), Vec((0.0, 0.0)), Vec((1.0, 0.4)))
    addfacetset!(grid, "Inlet", x -> abs(x[1]) < 1e-12)
    addfacetset!(grid, "Outlet", x -> abs(x[1]-1.0) < 1e-12)
    # One horizontal outer boundary is enough to test the policy; production
    # meshes place both horizontal curves in this same physical group.
    addfacetset!(grid, "Farfield", x -> abs(x[2]-0.4) < 1e-12)
    addfacetset!(grid, "Walls", x -> abs(x[2]-0.4) < 1e-12)
    addfacetset!(grid, "Obstacle", x -> abs(x[2]) < 1e-12 &&
                                        0.39 < x[1] < 0.61)
    return grid
end

function _tag_velocity_dofs(fom, tag)
    constraints = ConstraintHandler(fom.dof_handler)
    add!(constraints, Dirichlet(:u, getfacetset(fom.grid, tag),
        (x, _) -> Vec{2}((0.0, 0.0))))
    close!(constraints)
    update!(constraints, 0.0)
    return sort(collect(constraints.prescribed_dofs))
end

@testset "fluid boundary-condition policies" begin
    grid = _bc_test_grid()
    freestream = FNBC.UniformFreestreamBC((1.0, 0.0))
    fom = FNBC.setup_fem(grid; obstacle_tag = "Obstacle",
        reference_length = 0.1, quadrature_order = 6,
        boundary_conditions = freestream)

    inlet = _tag_velocity_dofs(fom, "Inlet")
    farfield = _tag_velocity_dofs(fom, "Farfield")
    obstacle = _tag_velocity_dofs(fom, "Obstacle")
    outlet = _tag_velocity_dofs(fom, "Outlet")
    state = zeros(ndofs(fom.dof_handler))
    apply!(state, fom.base_flow_constraints)
    for dofs in (inlet, farfield)
        @test count(==(1.0), state[dofs]) == length(dofs) ÷ 2
        @test count(iszero, state[dofs]) == length(dofs) ÷ 2
    end
    @test all(iszero, state[obstacle])
    perturbation = ones(ndofs(fom.dof_handler))
    apply!(perturbation, fom.perturbation_constraints)
    @test all(iszero, perturbation[union(inlet, farfield, obstacle)])
    @test !isempty(setdiff(outlet, fom.base_flow_constraints.prescribed_dofs))
    @test all(in(fom.free), setdiff(outlet, fom.base_flow_constraints.prescribed_dofs))
    @test fom.free == fom.free_dpim
    @test fom.boundary_conditions === freestream
    @test length(fom.model_fingerprint) == 64

    legacy_default = FNBC.setup_fem(grid; obstacle_tag = "Obstacle",
        channel_height = 0.4, quadrature_order = 6)
    legacy_explicit = FNBC.setup_fem(grid; obstacle_tag = "Obstacle",
        channel_height = 0.4, quadrature_order = 6,
        boundary_conditions = FNBC.PoiseuilleChannelBC(channel_height = 0.4))
    @test legacy_default.base_flow_constraints.prescribed_dofs ==
          legacy_explicit.base_flow_constraints.prescribed_dofs
    a = zeros(ndofs(legacy_default.dof_handler))
    apply!(a, legacy_default.base_flow_constraints)
    b = zeros(ndofs(legacy_explicit.dof_handler))
    apply!(b, legacy_explicit.base_flow_constraints)
    @test a == b
    @test legacy_default.model_fingerprint == legacy_explicit.model_fingerprint
    @test legacy_default.model_fingerprint != fom.model_fingerprint
    changed_q = FNBC.setup_fem(grid; obstacle_tag = "Obstacle",
        quadrature_order = 5, boundary_conditions = freestream)
    @test changed_q.model_fingerprint != fom.model_fingerprint

    @test_throws ArgumentError FNBC.UniformFreestreamBC((0.0, 0.0))
    @test_throws ArgumentError FNBC.PoiseuilleChannelBC(channel_height = 0.0)
end
