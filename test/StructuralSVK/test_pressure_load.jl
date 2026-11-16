# Public regression tests for uniform dead pressure on named Ferrite facet sets.

using MORFEFerrite
using Ferrite
using LinearAlgebra
using Test

const _PRESSURE_SVK = MORFEFerrite.StructuralSVK

function _pressure_test_case(celltype)
    grid = generate_grid(celltype, (1, 1, 1),
        Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
    return _PRESSURE_SVK.mechanical_model(grid;
        material = _PRESSURE_SVK.SVKMaterial(E = 1.0, ν = 0.3, ρ = 1.0),
        damping = _PRESSURE_SVK.RayleighDamping(α = 0.0, β = 0.0),
        dirichlet = "bottom", fe_order = 1, quad_order = 2)
end

function _pressure_resultant(case, force)
    result = zeros(3)
    for component in 1:3
        virtual_full = zeros(case.info.n_dofs_total)
        direction = ntuple(i -> i == component ? 1.0 : 0.0, 3)
        apply_analytical!(virtual_full, case.info.dof_handler, :u, x -> Vec(direction))
        virtual_free = zeros(case.info.n_dofs)
        for (global_dof, free_dof) in case.info.dof_to_free
            virtual_free[free_dof] = virtual_full[global_dof]
        end
        result[component] = dot(force, virtual_free)
    end
    return result
end

@testset "surface pressure" begin
    pressure = 2.0

    # This is both an exact-result test and a check that `mechanical_model`
    # retains the interpolations needed by the generic Ferrite facet assembly.
    hex_case = _pressure_test_case(Hexahedron)
    @test hasproperty(hex_case.info, :displacement_interpolation)
    @test hasproperty(hex_case.info, :geometry_interpolation)
    hex_force = _PRESSURE_SVK.surface_pressure_vector(hex_case,
        _PRESSURE_SVK.SurfacePressure(boundary = "top", pressure = pressure))
    @test length(hex_force) == hex_case.info.n_dofs
    @test _pressure_resultant(hex_case, hex_force) ≈ [0.0, 0.0, -pressure] atol = 1e-12

    # A tetrahedral grid exercises the same public API through a different
    # reference shape and facet quadrature rule. The unit top face has area one.
    tet_case = _pressure_test_case(Tetrahedron)
    tet_force = _PRESSURE_SVK.surface_pressure_vector(tet_case,
        _PRESSURE_SVK.SurfacePressure(boundary = "top", pressure = pressure))
    @test length(tet_force) == tet_case.info.n_dofs
    @test _pressure_resultant(tet_case, tet_force) ≈ [0.0, 0.0, -pressure] atol = 1e-12

    named = _PRESSURE_SVK.SurfacePressure(boundary = "top", pressure = pressure)
    @test named.boundary == "top"
    @test_throws ArgumentError _PRESSURE_SVK.SurfacePressure(pressure = pressure)
    @test_throws ArgumentError _PRESSURE_SVK.SurfacePressure(boundaries = Set([0]))
    @test_throws ArgumentError _PRESSURE_SVK.SurfacePressure(
        boundary = "top", boundaries = Set{Int}(), pressure = pressure)
end
