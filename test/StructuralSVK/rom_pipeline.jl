# Test fixture: the public SVK reduction pipeline, in one place.
#
# There is one `parametrise` and it is MORFE's — the package deliberately ships no
# `parametrise(::AssembledMechanicalModel)`. Every reduction has two shared stages:
#
#     (; model, spectral, meta) = build_model(case; …)
#     W, R = parametrise(model, spectral, order; resonance = ResonanceConfig(…))
#
# Spelling that out at ~30 assertion sites would bury what those tests are actually
# checking, so it lives here. This is TEST SCAFFOLDING, not API: it is not exported,
# not shipped, and the examples show the three steps directly because they are
# documentation. Its only job is to be the pipeline — if it ever grows a decision of
# its own, that decision has escaped from the code under test.
#
# The `ResonanceConfig` below must match what callers state, since it is what the
# SVK equivalence gates are measured against.

if !@isdefined(svk_solve)
    using MORFE: ResonanceConfig, parametrise
    using MORFEFerrite: build_model

    """
        svk_solve(case; master, order, forcing, resonance_tol, resonance_tol_rel,
                  eigenproblem, nev) -> (; W, R, meta, model, spectral)

    `build_model` → `MORFE.parametrise`, returning the physics-independent result
    and the backend metadata separately.
    """
    function svk_solve(case;
            master::Vector{Int} = [1],
            order::Int,
            forcing = nothing,
            resonance_tol::Real = 0.05,
            resonance_tol_rel::Union{Nothing, Real} = nothing,
            eigenproblem = nothing,
            nev::Union{Nothing, Int} = nothing)
        bm_kwargs = (; master = master, forcing = forcing, spectrum = eigenproblem,
            expansion_order = order)
        (; model, spectral, meta) = nev === nothing ?
                                    build_model(case; bm_kwargs...) :
                                    build_model(case; nev = nev, bm_kwargs...)

        config = ResonanceConfig(style = :complex_normal_form,
            tol = Float64(resonance_tol), tol_relative = resonance_tol_rel)
        t_solve = @elapsed W, R = parametrise(model, spectral, order; resonance = config)

        return (; W, R, meta = (; meta..., solve_time_s = t_solve), model, spectral)
    end
end
