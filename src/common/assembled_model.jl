"""
	AbstractAssembledModel

Supertype for a physics backend's assembled full-order data: FE spaces, assembled
operators, material data, and any backend-specific factories needed before a reduction
is chosen.

Each concrete subtype implements [`build_model`](@ref). Dispatch is on the backend's
assembled-model type; no additional wrapper "case" type is required.
"""
abstract type AbstractAssembledModel end

"""
	build_model(m::AbstractAssembledModel; kwargs...) -> (; model, spectral, meta)

Construct the full-order model and spectral data consumed by a reduction.

**This is the single contract every MORFEFerrite physics module implements.** The first
two fields are exactly what `MORFE.parametrise` takes:

```julia
(; model, spectral) = build_model(case; master = [1], forcing = …)
W, R = parametrise(model, spectral, expansion_order)
```

Keyword arguments are backend-specific: they may select master modes, add forcing,
configure diagnostics, or request/reuse an eigensolve. The return shape is shared, so
everything downstream of `build_model` can remain physics-independent.

The return is a `NamedTuple`, not a positional tuple, so that a backend can grow what it
reports without breaking callers who only want `model` and `spectral`.

- `model`   — an `NthOrderModel`
- `spectral` — a `SpectralData`
- `meta`     — backend metadata such as timings, the raw spectrum, forcing records, or
  DOF maps. It is available to callers and reporting code but is not consumed by MORFE.

Implementations must:

- return an `NthOrderModel` whose `linear_terms` and nonlinear terms are complete,
  including any external system the forcing introduces — not "mostly built, the caller
  adds forcing";
- return a `SpectralData` reconciled against **that** model's order. Use
  `SpectralData(model, spectrum; master = …)` so MORFE owns any order reconciliation;
- apply conditioning tweaks (mode scaling, unit changes) to the **raw arrays before**
  constructing the bundle; `SpectralData` deliberately has no `scale` field, so such
  tweaks stay visible at the call site;
- provide the spectrum's actual conjugate pairing to `SpectralData`; if a full reduced
  variable permutation is needed, extend the master pairing through the model's external
  system with `full_conjugate_permutation` rather than a fixed literal.

Implementations must **not** build a `MultiindexSet`, a `ResonanceSet` or a resonance
policy, validate the monomial set, derive conjugate closure, warn about resonances, or
solve the cohomological equations. Those are `parametrise`'s responsibilities and are
chosen by the caller. A backend may solve its linear eigenproblem here, or reuse spectral
data supplied by the caller.
"""
function build_model end
