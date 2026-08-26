using Test
using Tensors
using StaticArrays: SVector
using MORFEFerrite

const PGF = MORFEFerrite.ParametricGeometry
const FNF = MORFEFerrite.FluidNavierStokes

function eval_series(a, μ)
    return sum(a[k] * μ^PGF.GeometryParameterBasis([length(a)-1]).mset.exponents[k][1]
               for k in eachindex(a))
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
