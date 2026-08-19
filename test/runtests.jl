using Test

# High-level SVK + Ferrite UI. The small in-memory Gate A/B equivalence tests
# (high-level parametrise ≡ hand-built pipeline) run here. The full-mesh gates
# live in StructuralSVK/run_gates.jl and are run via the example environment
# (julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl).
@testset "MORFEFerrite" begin
    @testset "StructuralSVK" begin
        include("StructuralSVK/test_structural_svk.jl")
        include("StructuralSVK/test_anisotropic.jl")
        include("StructuralSVK/test_master_selection.jl")
    end

    # The physics-blind parametric coordinate transform, gated three ways:
    # against non-parametric StructuralSVK at J = I (a path that never goes
    # through ParametricGeometry at all), against golden values for the curved-J₀
    # and per-form-basis regimes, and on the build_model contract itself.
    @testset "ParametricGeometry" begin
        include("ParametricGeometry/test_kernel_equivalence.jl")
        include("ParametricGeometry/test_kernel_golden.jl")
        include("ParametricGeometry/test_build_model.jl")
    end

    # The conjugate-symmetry invariant the fluid reduction rests on. Mesh-free by
    # design: the run that would catch a violation end-to-end is order 9 on a 58k-DOF
    # mesh, which is far too heavy to keep in the default suite.
    @testset "FluidNavierStokes" begin
        include("FluidNavierStokes/test_conjugate_pairing.jl")
    end
end
