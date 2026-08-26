using Test

# StructuralSVK exercises the public `build_model` → `parametrise` workflow
# and inspects its `(W, R, meta)` outputs directly.
@testset "MORFEFerrite" begin
    @testset "Common" begin
        include("Common/test_paraview_2d.jl")
    end

    @testset "StructuralSVK" begin
        include("StructuralSVK/test_structural_svk.jl")
        include("StructuralSVK/test_anisotropic.jl")
        include("StructuralSVK/test_master_selection.jl")
    end

    # The physics-blind parametric coordinate transform, gated six ways: against a
    # FOM whose geometry actually moved, in 3D AND in 2D (the only gates that check
    # the transform against physics rather than against the module's own past);
    # against the series algebra's defining identities in both dimensions; against
    # non-parametric StructuralSVK at J = I; against golden values for the
    # curved-J₀ and per-form-basis regimes; and on the build_model contract.
    @testset "ParametricGeometry" begin
        include("ParametricGeometry/test_moved_mesh_fom.jl")
        include("ParametricGeometry/test_moved_mesh_fom_2d.jl")
        include("ParametricGeometry/test_series_algebra_2d.jl")
        include("ParametricGeometry/test_inverse_determinant.jl")
        include("ParametricGeometry/test_kernel_equivalence.jl")
        include("ParametricGeometry/test_kernel_golden.jl")
        include("ParametricGeometry/test_build_model.jl")
    end

    # The conjugate-symmetry invariant the fluid reduction rests on. Mesh-free by
    # design: the run that would catch a violation end-to-end is order 9 on a 58k-DOF
    # mesh, which is far too heavy to keep in the default suite.
    @testset "FluidNavierStokes" begin
        include("FluidNavierStokes/test_boundary_conditions.jl")
        include("FluidNavierStokes/test_conjugate_pairing.jl")
        include("FluidNavierStokes/test_resummation.jl")
        include("FluidNavierStokes/test_joukowski_profile.jl")
        include("FluidNavierStokes/test_fluid_pullback.jl")
    end
end
