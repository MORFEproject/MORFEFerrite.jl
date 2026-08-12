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
	build_model(m::AbstractAssembledModel; kwargs...) -> (; model, spectral, meta)

Turn an assembled physics model into what a reduction consumes.

**This is the single contract every MORFEFerrite physics module implements.** The first
two fields are exactly what `MORFE.parametrise` takes:

```julia
(; model, spectral) = build_model(case; master = [1], forcing = …)
W, R = parametrise(model, spectral, expansion_order)
```

Keyword arguments are the backend's own — which modes are master, what forcing to apply,
how many eigenpairs to compute — because those are physics-specific. What is *not*
backend-specific is the return type, and that is the point: everything downstream of
`build_model` is shared.

The return is a `NamedTuple`, not a positional tuple, so that a backend can grow what it
reports without breaking callers who only want `model` and `spectral`.

- `model`   — an `NthOrderModel`
- `spectral` — a `SpectralData`
- `meta`    — **backend-private**. Whatever that physics needs to report afterwards
  (timings, the raw spectrum, forcing records, DOF maps). MORFE never sees it.

Implementations must:

- return an `NthOrderModel` whose `linear_terms` and nonlinear terms are complete,
  including any external system the forcing introduces — not "mostly built, the caller
  adds forcing";
- return a `SpectralData` reconciled against **that** model's order (use
  `SpectralData(model, spectrum; master = …)`, which slices when the orders match and
  extends only when the model's `ORD` is higher — do not hand-roll the extension);
- apply conditioning tweaks (mode scaling, unit changes) to the **raw arrays before**
  constructing the bundle; `SpectralData` deliberately has no `scale` field, so such
  tweaks stay visible at the call site;
- build any conjugate permutation with `full_conjugate_permutation(master_block, sys)`,
  never a literal.

Implementations must **not** build a `MultiindexSet`, a `ResonanceSet` or a resonance
policy, validate the monomial set, derive conjugate closure, warn about resonances, or
solve anything. Those are `parametrise`'s, chosen by the caller.
"""
function build_model end
