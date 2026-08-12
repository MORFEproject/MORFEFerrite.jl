# =====================================================================
# The assembled parametric model — physics-blind.
# =====================================================================

"""
	ParametricOperator(arrays, arity)

One linear operator of the base physics, expanded over the θ-basis: `arrays[i]`
is the θ^α coefficient matrix at `basis.mset.exponents[i]`, and `arity` names the
derivative slot it occupies in the model (`(1,0,0)` stiffness, `(0,1,0)` damping,
`(0,0,1)` mass for a second-order structure on the augmented `ORD = 3` model;
`(1,)` for a first-order physics).

The base coefficient `arrays[1]` (α = 0) goes into the model's `linear_terms`;
every α ≠ 0 becomes a `MultilinearMap` correction.
"""
struct ParametricOperator{N}
	arrays::Vector
	arity::NTuple{N, Int}
end

"""
	AssembledParametricModel <: AbstractAssembledModel

A base physics model expanded over a parametric mesh coordinate transform.

Holds the FE discretisation and pullback cache, the base physics' assembled
model, its θ-expanded linear operators and its nonlinear [`ParametricMap`](@ref)s.
Being physics-blind, it names none of them: the `base` field is only ever passed
back to the physics, and the operators arrive already expanded.

Build one with the physics' own entry point (`StructuralSVK.parametric_model`),
then call [`build_model`](@ref).
"""
struct AssembledParametricModel{Nθ, PD, B} <: AbstractAssembledModel
	pd::PD
	base::B                              # the physics' AbstractAssembledModel
	operators::Vector{ParametricOperator}
	maps::Vector{Any}                    # ParametricMap, one per nonlinear form
	map_arities::Vector{Tuple}
	n_geometry_parameters::Int
	info::NamedTuple
end

function AssembledParametricModel(pd::ParametricDiscretisation{Nθ}, base,
	operators::Vector{ParametricOperator}, maps::Vector, map_arities::Vector;
	info::NamedTuple = NamedTuple()) where {Nθ}
	return AssembledParametricModel{Nθ, typeof(pd), typeof(base)}(
		pd, base, operators, collect(Any, maps), collect(Tuple, map_arities), Nθ, info)
end

"""
	model_order(m::AssembledParametricModel) -> Int

The `ORD` the augmented `NthOrderModel` needs: one more than the highest
derivative slot any θ-correction occupies.

A second-order structure whose mass is parametric needs `ORD = 3` — a mass
correction has arity `(0,0,1)`, which only exists on a third-order model, and
the fourth linear block is zero. A first-order physics with only a
`(1,)`-arity correction needs no augmentation and must not get one.
"""
function model_order(m::AssembledParametricModel)
	isempty(m.operators) && return 0
	return maximum(length(op.arity) for op in m.operators)
end

function Base.show(io::IO, ::MIME"text/plain", m::AssembledParametricModel{Nθ}) where {Nθ}
	n_corr = 0
	for op in m.operators
		n_corr += count(!all(iszero, α) for α in basis(m.pd).mset.exponents)
	end
	println(io, "AssembledParametricModel")
	println(io, "  parameters : $Nθ (θ-terms: $(nterms(m.pd)))")
	println(io, "  free DOFs  : $(m.pd.n_free)")
	println(io, "  base       : $(nameof(typeof(m.base)))")
	println(io, "  operators  : $(length(m.operators)) (arities " *
				join([string(op.arity) for op in m.operators], ", ") * ")")
	print(io, "  maps       : $(length(m.maps)) nonlinear form" *
			  (length(m.maps) == 1 ? "" : "s"))
end
