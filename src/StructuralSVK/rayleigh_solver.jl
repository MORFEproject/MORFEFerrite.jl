"""
	RayleighEigensolver <: AbstractEigensolver

Solves the undamped eigenproblem K ϕ = ω² M ϕ and recovers the damped
eigenvalues λ = ω(-ξ ± i√(1-ξ²)) using Rayleigh damping ξ = ½(α/ω + βω).

Promoted verbatim from `examples/01_clamped_beam_ferrite/low_level.jl`
(formerly `Mechanical_Problem_Solver`); behavioural equivalence is enforced
by the `structural_svk` test group.

The damping is carried as the model's own [`RayleighDamping`](@ref) rather than a
loose `(α, β)` pair. The solver's `ξ` and the model's `C = αM + βK` must describe
the same structure; passing the object makes that one value instead of two that
have to be kept in step by hand.
"""
mutable struct RayleighEigensolver <: AbstractEigensolver
	right_eig_result::Union{Nothing, Matrix}
	eigenvalues::Union{Nothing, Vector}
	nev::Int
	damping::RayleighDamping{Float64}
end

"""
	RayleighEigensolver(nev, damping::RayleighDamping)

Solver for `nev` modes of a structure with this damping. Take `damping` from the
assembled model (`m.damping`) so the spectrum matches the model's `C`.
"""
RayleighEigensolver(nev::Integer, damping::RayleighDamping) =
	RayleighEigensolver(nothing, nothing, Int(nev),
		RayleighDamping(Float64(damping.α), Float64(damping.β)))

function MORFE.SpectralDecomposition.eigensolve(model::NthOrderModel, solver::RayleighEigensolver)
	ω2,
	ϕ = eigs(model.linear_terms[1], model.linear_terms[3];
		nev = solver.nev, which = :SM)
	idx = sortperm(real(ω2))[1:solver.nev]
	ω2 = real.(ω2[idx])
	ϕ = real.(ϕ[:, idx])
	ω = sqrt.(ω2)
	FOM = size(ϕ, 1)

	λ_all = zeros(ComplexF64, 2 * solver.nev)
	for i in 1:solver.nev
		ξ = 0.5 * (solver.damping.α / ω[i] + solver.damping.β * ω[i])
		λ_all[2i-1] = ω[i] * (-ξ + sqrt(Complex(1.0 - ξ^2)) * im)
		λ_all[2i] = ω[i] * (-ξ - sqrt(Complex(1.0 - ξ^2)) * im)
	end

	evecs = Matrix{ComplexF64}(undef, 2FOM, 2 * solver.nev)
	for i in 1:solver.nev
		evecs[1:FOM, 2i-1] .= ϕ[:, i]
		evecs[1:FOM, 2i] .= ϕ[:, i]
		evecs[(FOM+1):end, 2i-1] .= λ_all[2i-1] .* ϕ[:, i]
		evecs[(FOM+1):end, 2i] .= λ_all[2i] .* ϕ[:, i]
	end
	solver.right_eig_result = evecs
	solver.eigenvalues = λ_all
	return λ_all, reshape(evecs, FOM, 2, 2 * solver.nev)
end

function MORFE.SpectralDecomposition.eigensolve_left(model::NthOrderModel, solver::RayleighEigensolver)
	@assert solver.right_eig_result !== nothing "Run solve() first"
	R = solver.right_eig_result
	FOM = size(R, 1) ÷ 2
	Y = reshape(R, FOM, 2, size(R, 2))
	@assert size(Y, 2)==2 "structural left blocks require a second-order model (ORD = 2)"
	# Analytic sesquilinear left blocks φ = [(conj(λ)M + C)ϕ; ϕ] — algebraically
	# equal to -(1/conj(λ))Kᵀϕ via the quadratic eigenrelation, but built from
	# the moderate-norm M, C instead of K (which amplifies eigensolver noise by
	# (ω_max/ω₁)²). Shared with StructureModalDampingEigensolver.
	#
	# The slice is the RIGHT position mode ϕ: the pencil is self-adjoint and ϕ is
	# real, which is what spares us an adjoint eigensolve. `apply = identity`
	# because Mᴴ = M and Cᴴ = C. The stiffness slot is never read at ORD = 2, so M
	# stands in there purely to make the tuple the right length.
	left = left_eigenmode_orders_from_slice(
		(model.linear_terms[3], model.linear_terms[2], model.linear_terms[3]),
		view(Y, :, 1, :), solver.eigenvalues; apply = identity)
	return solver.eigenvalues, left
end
