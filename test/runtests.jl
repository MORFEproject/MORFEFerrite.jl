using Test

# High-level SVK + Ferrite UI. The small in-memory Gate A/B equivalence tests
# (high-level parametrise ≡ hand-built pipeline) run here. The full-mesh gates
# live in StructuralSVK/run_gates.jl and are run via the example environment
# (julia --project=examples/01_clamped_beam_ferrite test/StructuralSVK/run_gates.jl).
@testset "MORFEFerrite" begin
    @testset "StructuralSVK" begin
        include("StructuralSVK/test_structural_svk.jl")
    end
end
