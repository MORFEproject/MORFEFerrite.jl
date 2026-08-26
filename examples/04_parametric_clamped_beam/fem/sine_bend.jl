"""
	sine_bend.jl

The isochoric sinusoidal bend shared with example 07:

	w(x₀) = h₀ · sin(π x₁ / L) · e₂

x₁ is the axial coordinate along the span [0, L], h₀ the rise, e₂ the transverse
(thin-direction) unit vector. Its gradient is the rank-1 tensor

	∇w = (π h₀ / L) · cos(π x₁ / L) · (e₂ ⊗ e₁)

whose only non-zero entry sits at (2, 1). **That is what makes it isochoric**:
the tensor is nilpotent, so adding any multiple of it to a lower-triangular
Jacobian leaves the determinant untouched. In this example

	J = I + θ₁ (e₁⊗e₁) + θ₂ ∇w   ⟹   det J = 1 + θ₁,  independent of θ₂

so θ₂ contributes nothing to the volume factor and the whole non-polynomial
content of `J⁻¹` (the `1/det J` series and its radius) belongs to θ₁ alone.

Duplicated from `examples/07_parametric_arch/fem/arch_geometry.jl` rather than
included from it: the examples are deliberately self-contained, each with its own
Project.toml. Keep the two in step — they are the same field, and example 07 also
uses it as its *reference* configuration (`J₀ = I + ∇w`) rather than as an
additive shape field.
"""

using Tensors

"""
	sine_bend_displacement(x₀, h₀, L) -> Vec{3,Float64}

Transverse displacement of the sinusoidal bend at reference position `x₀`.
Only the e₂ component is non-zero. Vanishes at `x₁ = 0` and `x₁ = L`, so it is
compatible with clamping at both ends.
"""
function sine_bend_displacement(x₀::Vec{3, Float64}, h₀::Float64, L::Float64)
	return Vec{3, Float64}((0.0, h₀ * sin(π * x₀[1] / L), 0.0))
end

"""
	sine_bend_jacobian(x₀, h₀, L) -> Tens3

Gradient tensor `∇w = (π h₀ / L) cos(π x₁ / L) · (e₂ ⊗ e₁)` — nilpotent, hence
volume-preserving. This is `∇ψ₂` for the additive map of this example.
"""
function sine_bend_jacobian(x₀::Vec{3, Float64}, h₀::Float64, L::Float64)
	dw_dx1 = (π * h₀ / L) * cos(π * x₀[1] / L)
	return Tens3((i, j) -> (i == 2 && j == 1) ? dw_dx1 : 0.0)
end
