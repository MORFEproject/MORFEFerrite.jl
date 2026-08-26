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
    ch = ConstraintHandler(fom.dh)
    add!(ch, Dirichlet(:u, getfacetset(fom.grid, tag),
        (x, _) -> Vec{2}((0.0, 0.0))))
    close!(ch)
    update!(ch, 0.0)
    return sort(collect(ch.prescribed_dofs))
end

@testset "fluid boundary-condition policies" begin
    grid = _bc_test_grid()
    freestream = FNBC.UniformFreestreamBC((1.0, 0.0))
    fom = FNBC.setup_fem(grid; obstacle_tag="Obstacle",
        reference_length=0.1, quadrature_order=6,
        boundary_conditions=freestream)

    inlet = _tag_velocity_dofs(fom, "Inlet")
    farfield = _tag_velocity_dofs(fom, "Farfield")
    obstacle = _tag_velocity_dofs(fom, "Obstacle")
    outlet = _tag_velocity_dofs(fom, "Outlet")
    state = zeros(ndofs(fom.dh))
    apply!(state, fom.ch_full)
    for dofs in (inlet, farfield)
        @test count(==(1.0), state[dofs]) == length(dofs) ÷ 2
        @test count(iszero, state[dofs]) == length(dofs) ÷ 2
    end
    @test all(iszero, state[obstacle])
    perturbation = ones(ndofs(fom.dh))
    apply!(perturbation, fom.ch_hom)
    @test all(iszero, perturbation[union(inlet, farfield, obstacle)])
    @test !isempty(setdiff(outlet, fom.ch_full.prescribed_dofs))
    @test all(in(fom.free), setdiff(outlet, fom.ch_full.prescribed_dofs))
    @test fom.free == fom.free_dpim
    @test fom.boundary_conditions === freestream
    @test length(fom.model_fingerprint) == 64

    legacy_default = FNBC.setup_fem(grid; obstacle_tag="Obstacle",
        channel_height=0.4, quadrature_order=6)
    legacy_explicit = FNBC.setup_fem(grid; obstacle_tag="Obstacle",
        channel_height=0.4, quadrature_order=6,
        boundary_conditions=FNBC.PoiseuilleChannelBC(channel_height=0.4))
    @test legacy_default.ch_full.prescribed_dofs ==
        legacy_explicit.ch_full.prescribed_dofs
    a = zeros(ndofs(legacy_default.dh)); apply!(a, legacy_default.ch_full)
    b = zeros(ndofs(legacy_explicit.dh)); apply!(b, legacy_explicit.ch_full)
    @test a == b
    @test legacy_default.model_fingerprint == legacy_explicit.model_fingerprint
    @test legacy_default.model_fingerprint != fom.model_fingerprint
    changed_q = FNBC.setup_fem(grid; obstacle_tag="Obstacle",
        quadrature_order=5, boundary_conditions=freestream)
    @test changed_q.model_fingerprint != fom.model_fingerprint

    @test_throws ArgumentError FNBC.UniformFreestreamBC((0.0, 0.0))
    @test_throws ArgumentError FNBC.PoiseuilleChannelBC(channel_height=0.0)
end
