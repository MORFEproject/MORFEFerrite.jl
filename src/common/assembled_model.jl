"""
	AbstractAssembledModel

Supertype for every physics backend's assembled-model object — the thing that holds the
FE spaces, the assembled operators and whatever else that physics needs, before any
reduction is chosen.

Each backend subtypes it and implements exactly one method of [`build_model`](@ref).
There is no wrapper "case" type: dispatch happens on the backend's own struct.
"""
abstract type AbstractAssembledModel end

"""
	build_model(m::AbstractAssembledModel; kwargs...) -> (NthOrderModel, SpectralData)

Turn an assembled physics model into the pair a reduction consumes.

**This is the single contract every MORFEFerrite physics module implements**, and the
returned pair is exactly what `MORFE.parametrise` takes:

```julia
model, sd = build_model(case; master = [1], forcing = …)
W, R      = parametrise(model, sd, expansion_order)
```

Keyword arguments are the backend's own — which modes are master, what forcing to apply,
how many eigenpairs to compute — because those are physics-specific. What is *not*
backend-specific is the return type, and that is the point: everything downstream of
`build_model` is shared.

Implementations must:

- return an `NthOrderModel` whose `linear_terms` and nonlinear terms are complete,
  including any external system the forcing introduces;
- return a `SpectralData` reconciled against **that** model's order (use
  `SpectralData(model, spectrum; master = …)`, which handles the reconciliation);
- leave the reduced-coordinate order equal to the order the caller asked for.
"""
function build_model end
