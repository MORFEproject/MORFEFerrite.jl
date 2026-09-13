# =====================================================================
# Mixed P2/P1 Navier--Stokes pullback for an affine parametric geometry.
# =====================================================================

using ..ParametricGeometry: GeometryParameterBasis, PullbackCache,
	PowerSeriesInverseDet, AbstractInverseDeterminant, nterms,
	poly_contract, poly_mul, build_linear_corrections
using MORFE: MultilinearMap, NthOrderModel, ExternalSystem, SpectralData,
	full_conjugate_permutation

# Increment this whenever the numerical coefficient content of the fluid
# pullback changes.  Example-level checkpoints include it in their problem
# fingerprint, so a corrected parametric FOM cannot silently reuse stale ROM
# coefficients.
const PARAMETRIC_FLUID_FORMULATION_VERSION = 3

"""
	_enforce_affine_2d_fluid_structure!(cache)

Validate and enforce the exact polynomial degrees of the one-parameter,
two-dimensional affine fluid map.  For `F(mu) = F0 + mu*G`, `det(F)` is
quadratic and `adj(F)` is affine.  The generic recurrence used to construct a
`PullbackCache` can leave roundoff-sized coefficients above those degrees;
those values must not be interpreted as physical convection maps merely
because they are bitwise nonzero.

This is a narrowly bounded roundoff cleanup. When all coefficients above the
affine analytical degree are at roundoff scale, the cache is identified as
affine and those coefficients are replaced by exact zeros. A genuinely
polynomial geometry provider is left unchanged; the shared fluid API continues
to support the general maps accepted by `PullbackCache`.
"""
function _enforce_affine_2d_fluid_structure!(cache::PullbackCache{1,TT}) where {TT <: Tensor{2,2}}
	basis = cache.basis
	adj_discarded = 0.0
	det_discarded = 0.0
	scale = 1.0
	for ci in eachindex(cache.adj), q in eachindex(cache.adj[ci])
		adj = cache.adj[ci][q]
		det = cache.det[ci][q]
		scale = max(scale, maximum(norm, adj), maximum(abs, det))
		for (idx, exponent) in enumerate(basis.mset.exponents)
			degree = exponent[1]
			degree >= 2 && (adj_discarded = max(adj_discarded, norm(adj[idx])))
			degree >= 3 && (det_discarded = max(det_discarded, abs(det[idx])))
		end
	end
	tolerance = 512 * eps(Float64) * scale
	applied = adj_discarded <= tolerance && det_discarded <= tolerance
	applied || return (; applied, adj_discarded, det_discarded, tolerance)

	for ci in eachindex(cache.adj), q in eachindex(cache.adj[ci])
		adj = cache.adj[ci][q]
		det = cache.det[ci][q]
		for (idx, exponent) in enumerate(basis.mset.exponents)
			degree = exponent[1]
			degree >= 2 && (adj[idx] = zero(TT))
			degree >= 3 && (det[idx] = 0.0)
		end
	end
	return (; applied, adj_discarded, det_discarded, tolerance)
end

"""
	AssembledParametricFluidModel

Two-parameter (`μ`, `ξ`) fluid model about the midpoint steady state.  Geometry
coefficients are indexed by the supplied one-dimensional box basis; `ξ` is the
normalised inverse-Reynolds coordinate.  The mass corrections occupy derivative
arity `(0,1)`, hence the built MORFE model has `ORD = 2` and a zero `B₂` block.
"""
struct AssembledParametricFluidModel{C, B, PC} <: AbstractAssembledModel
	base::C
	basis::B
	cache::PC
	B0_geometry::Vector
	B1_geometry::Vector
	viscosity::Vector
	nonconvective::Vector
	base_convection::Vector{Vector{Float64}}
	h_geometry::Vector{Vector{Float64}}
	h_reynolds::Vector{Vector{Float64}}
	reynolds_scale::Float64
	include_reynolds::Bool
	identity_errors::NamedTuple
	info::NamedTuple
end

function Base.show(io::IO, ::MIME"text/plain", m::AssembledParametricFluidModel)
	println(io, "AssembledParametricFluidModel (composition pullback, P2/P1)")
	println(io, "  coordinates : " * (m.include_reynolds ? "(μ, ξ)" : "(μ,)"))
	println(io, "  geometry box: $(m.basis.bounds)")
	println(io, "  ORD         : 2 (geometry-dependent singular mass)")
	print(io, "  midpoint reconciliation: B₀=$(m.identity_errors.B0), B₁=$(m.identity_errors.B1)")
end

"""
	compute_transformed_drag_lift(s_full, fom, geometry; μ, Re)

Full pressure-plus-viscous traction on the reference obstacle.  Nanson's
formula is applied before integration, `n dΓ = adj(F)' N dΓ_ref`; the velocity
gradient uses the same composition pullback as the volume forms.
"""
function compute_transformed_drag_lift(s_full, fom, geometry;
	μ::Real, Re::Real)
	ip_vel = Lagrange{RefTriangle,2}()^2
	ip_pres = Lagrange{RefTriangle,1}()
	ip_geo = Lagrange{RefTriangle,1}()
	qr = FacetQuadratureRule{RefTriangle}(max(6, fom.quadrature_order))
	fv_vel = FacetValues(qr, ip_vel, ip_geo)
	fv_pres = FacetValues(qr, ip_pres, ip_geo)
	Fd = 0.0; Fl = 0.0
	for (cell_idx, facet_idx) in getfacetset(fom.grid, fom.obstacle_tag)
		cell = CellCache(fom.dh); reinit!(cell, cell_idx)
		dofs = celldofs(cell)
		u_e = s_full[dofs[fom.dof_range_u]]
		p_e = s_full[dofs[fom.dof_range_p]]
		reinit!(fv_vel, cell, facet_idx); reinit!(fv_pres, cell, facet_idx)
		coords = getcoordinates(cell)
		for q in 1:getnquadpoints(fv_vel)
			dΓ0 = getdetJdV(fv_vel, q)
			N = getnormal(fv_vel, q)
			x0 = spatial_coordinate(fv_vel, q, coords)
			Js = applicable(geometry, x0, cell, fv_vel, q) ?
				geometry(x0, cell, fv_vel, q) : geometry(x0)
			G = Js[2]
			F = one(G) + Float64(μ) * G
			j = det(F); A = j * inv(F)
			nanson = transpose(A) ⋅ N
			gradx = function_gradient(fv_vel, q, u_e) ⋅ A / j
			p = function_value(fv_pres, q, p_e)
			σ = -p * one(gradx) + (fom.reference_length / Float64(Re)) *
				(gradx + transpose(gradx))
			traction_measure = σ ⋅ nanson
			Fd += traction_measure[1] * dΓ0
			Fl += traction_measure[2] * dΓ0
		end
	end
	ref = U_MEAN^2 * fom.reference_length
	return (-2Fd/ref, -2Fl/ref)
end

@inline function _fluid_viscosity_series(grad_v, grad_u, adj, invdet, basis)
	gv = [grad_v ⋅ A for A in adj]
	gu = [grad_u ⋅ A for A in adj]
	inner = poly_contract(symmetric.(gv), symmetric.(gu), basis)
	return 2 .* poly_mul(inner, invdet, basis)
end

@inline _fluid_divergence_series(grad_u, adj) = [tr(grad_u ⋅ A) for A in adj]

"""
	parametric_model(case::AssembledFluidModel, geometry; ...)

Assemble the fixed-domain composition pullback. `geometry` returns
`(I, ∇ψ)` at each quadrature point.  The external ordering is always `μ` first,
then `ξ`; both are frozen and real.
"""
function parametric_model(case::AssembledFluidModel, geometry;
	geometry_parameter_basis::GeometryParameterBasis,
	reynolds_scale::Real,
	include_reynolds::Bool = true,
	inverse_determinant::AbstractInverseDeterminant = PowerSeriesInverseDet())
	basis = geometry_parameter_basis
	length(basis.bounds) == 1 || throw(ArgumentError(
		"the profile adapter has one geometry coordinate μ; got $(length(basis.bounds))"))
	Δη = Float64(reynolds_scale)
	Δη > 0 || throw(ArgumentError("reynolds_scale = Δη must be positive"))
	fom = case.fom
	cache = PullbackCache(fom.dh, fom.cv_vel, geometry, basis;
		inverse_determinant)
	affine_structure = _enforce_affine_2d_fluid_structure!(cache)
	L = nterms(basis)

	Mfull = [allocate_matrix(fom.dh) for _ in 1:L]
	Vfull = [allocate_matrix(fom.dh) for _ in 1:L]
	Pfull = [allocate_matrix(fom.dh) for _ in 1:L]
	Cfull = [allocate_matrix(fom.dh) for _ in 1:L]
	assemblers = [(start_assemble(Mfull[k]), start_assemble(Vfull[k]),
		start_assemble(Pfull[k]), start_assemble(Cfull[k])) for k in 1:L]
	base_conv = [zeros(Float64, ndofs(fom.dh)) for _ in 1:L]
	npc = ndofs_per_cell(fom.dh)
	Me = [zeros(npc, npc) for _ in 1:L]
	Ve = [zeros(npc, npc) for _ in 1:L]
	Pe = [zeros(npc, npc) for _ in 1:L]
	Ce = [zeros(npc, npc) for _ in 1:L]
	fe = [zeros(npc) for _ in 1:L]

	for (ci, cell) in enumerate(CellIterator(fom.dh))
		foreach(A -> fill!(A, 0.0), Me); foreach(A -> fill!(A, 0.0), Ve)
		foreach(A -> fill!(A, 0.0), Pe); foreach(A -> fill!(A, 0.0), Ce)
		foreach(A -> fill!(A, 0.0), fe)
		reinit!(fom.cv_vel, cell); reinit!(fom.cv_pres, cell)
		dofs = celldofs(cell)
		u0e = case.s₀_full[dofs[fom.dof_range_u]]
		for q in 1:getnquadpoints(fom.cv_vel)
			dΩ = getdetJdV(fom.cv_vel, q)
			adj = cache.adj[ci][q]
			det = cache.det[ci][q]
			invdet = cache.inv_det[ci][q]
			u0 = function_value(fom.cv_vel, q, u0e)
			grad_u0 = function_gradient(fom.cv_vel, q, u0e)
			grad_u0_A = [grad_u0 ⋅ A for A in adj]

			for i in 1:fom.n_vel_dofs_per_cell
				ri = fom.dof_range_u[i]
				vi = shape_value(fom.cv_vel, q, i)
				grad_vi = shape_gradient(fom.cv_vel, q, i)
				div_vi = _fluid_divergence_series(grad_vi, adj)
				for j in 1:fom.n_vel_dofs_per_cell
					rj = fom.dof_range_u[j]
					uj = shape_value(fom.cv_vel, q, j)
					grad_uj = shape_gradient(fom.cv_vel, q, j)
					visc = _fluid_viscosity_series(grad_vi, grad_uj, adj, invdet, basis)
					for k in 1:L
						Me[k][ri, rj] += (vi ⋅ uj) * det[k] * dΩ
						Ve[k][ri, rj] += visc[k] * dΩ
						Ce[k][ri, rj] += vi ⋅ ((grad_uj ⋅ adj[k]) ⋅ u0 +
							grad_u0_A[k] ⋅ uj) * dΩ
					end
				end
				for m in eachindex(fom.dof_range_p)
					rm = fom.dof_range_p[m]
					qm = shape_value(fom.cv_pres, q, m)
					for k in 1:L
						Pe[k][ri, rm] -= qm * div_vi[k] * dΩ
					end
				end
				for k in 1:L
					fe[k][ri] += vi ⋅ (grad_u0_A[k] ⋅ u0) * dΩ
				end
			end
			for m in eachindex(fom.dof_range_p)
				rm = fom.dof_range_p[m]
				qm = shape_value(fom.cv_pres, q, m)
				for j in 1:fom.n_vel_dofs_per_cell
					rj = fom.dof_range_u[j]
					div_uj = _fluid_divergence_series(shape_gradient(fom.cv_vel, q, j), adj)
					for k in 1:L
						Pe[k][rm, rj] -= qm * div_uj[k] * dΩ
					end
				end
			end
		end
		for k in 1:L
			am, av, ap, ac = assemblers[k]
			assemble!(am, dofs, Me[k]); assemble!(av, dofs, Ve[k])
			assemble!(ap, dofs, Pe[k]); assemble!(ac, dofs, Ce[k])
			for (i, d) in pairs(dofs)
				base_conv[k][d] += fe[k][i]
			end
		end
	end

	free = fom.free_dpim
	ν0 = fom.reference_length / case.Re₀
	B1 = [A[free, free] for A in Mfull]
	V = [A[free, free] for A in Vfull]
	nonconv = [(ν0 * Vfull[k] + Pfull[k])[free, free] for k in 1:L]
	B0 = [(ν0 * Vfull[k] + Pfull[k] + Cfull[k])[free, free] for k in 1:L]
	hgeom = Vector{Vector{Float64}}(undef, L)
	hre = Vector{Vector{Float64}}(undef, L)
	for k in 1:L
		hgeom[k] = -(ν0 * Vfull[k] + Pfull[k])[free, :] * case.s₀_full .-
			base_conv[k][free]
		hre[k] = -(fom.reference_length * Δη) .* (Vfull[k][free, :] * case.s₀_full)
	end

	rel(A, B) = norm(A - B) / max(norm(B), eps())
	errors = (B0 = rel(B0[1], case.B[1]), B1 = rel(B1[1], case.B[2]),
		forcing = norm(hgeom[1]) / max(1.0, norm(case.s₀_full)))
	errors.forcing <= 1e-10 || throw(ArgumentError(
		"midpoint pure-parameter residual is $(errors.forcing), above 1e-10; " *
		"the steady state, pressure convention, or prescribed-DOF forcing is inconsistent"))
	errors.B0 <= 1e-11 || throw(ArgumentError(
		"pulled-back B₀ at μ=0 does not reproduce the ordinary FOM (relative error $(errors.B0))"))
	errors.B1 <= 1e-11 || throw(ArgumentError(
		"pulled-back B₁ at μ=0 does not reproduce the ordinary FOM (relative error $(errors.B1))"))

	info = (; n_free = length(free), n_terms = L, ORD = 2,
		external_order = include_reynolds ? (:mu, :xi) : (:mu,),
		Re₀ = case.Re₀, Δη, fom.reference_length, fom.obstacle_tag,
		parametric_fluid_formulation_version = PARAMETRIC_FLUID_FORMULATION_VERSION,
		affine_structure)
	return AssembledParametricFluidModel(case, basis, cache, B0, B1, V, nonconv,
		base_conv, hgeom, hre, Δη, include_reynolds, errors, info)
end

# Closure factories are generated because MORFE determines map arity from the
# callable signature; a vararg closure would hide the number of frozen factors.
# Method-4 order 12 needs thirteen external factors for the legitimate mixed
# viscosity term mu^12*xi, even though convection itself stops exactly at mu^1.
const _PF_MAX_EXTERNAL_FACTORS = 13
for mm in 0:_PF_MAX_EXTERNAL_FACTORS
	ext = [Symbol("r$i") for i in 1:mm]
	factor = mm == 0 ? :(one(eltype(res))) :
		Expr(:call, :*, [:($(ext[s])[comp[$s]]) for s in 1:mm]...)
	@eval _pf_convection(::Val{$mm}, comp, pm, idx) =
		(res, u1, u2, $(ext...)) ->
			_apply_convection_coefficient!(res, pm, idx, u1, u2, $factor)
	@eval _pf_vector(::Val{$mm}, comp, h) =
		(res, $(ext...)) -> (res .+= ($factor) .* h)
	@eval _pf_linear(::Val{$mm}, comp, A, arity) =
		(res, x, $(ext...)) -> (res .-= ($factor) .* (A * x))
	if mm > 0
		# Symmetric polarisation of mu^(mm-1)*xi.  MORFE enumerates one
		# canonical external-factor tuple and applies its permutation count;
		# consequently a mixed monomial must be represented by the symmetric
		# multilinear form, not by assigning xi to one distinguished slot.
		# This average is identical to mu^(mm-1)*xi when all arguments are the
		# same physical external vector, while its canonical coefficient is
		# exactly the unscaled Taylor coefficient.
		mixed_terms = Any[]
		for xi_slot in 1:mm
			factors = Any[:($(ext[xi_slot])[2])]
			append!(factors,
				[:($(ext[s])[1]) for s in 1:mm if s != xi_slot])
			push!(mixed_terms, Expr(:call, :*, factors...))
		end
		mixed_factor = :($(Expr(:call, :+, mixed_terms...)) / $mm)
		@eval _pf_reynolds_vector(::Val{$mm}, h) =
			(res, $(ext...)) -> (res .+= ($mixed_factor) .* h)
		@eval _pf_reynolds_linear(::Val{$mm}, A) =
			(res, x, $(ext...)) -> (res .-= ($mixed_factor) .* (A * x))
	end
end

function _apply_convection_coefficient!(res, pm, idx, u1, u2, factor)
	fom, cache = pm.base.fom, pm.cache
	if u1 === u2
		return _accumulate_fluid_convection_pair!(res, u1, u2, fom,
			(ci, q) -> cache.adj[ci][q][idx], factor, Val(:free), Val(:free),
			Val(true))
	end
	return _accumulate_fluid_convection_pair!(res, u1, u2, fom,
		(ci, q) -> cache.adj[ci][q][idx], factor)
end

function _external_components(k::Int; xi::Bool = false)
	return Tuple(vcat(fill(1, k), xi ? [2] : Int[]))
end

function _nonzero_matrix(A)
	return nnz(A) > 0
end

"""Build the augmented ORD=2 MORFE model from verified coefficient arrays."""
function build_model(pm::AssembledParametricFluidModel;
	spectrum, master::AbstractVector{Int} = [1], conjugate_permutation = nothing)
	L = nterms(pm.basis)
	zero_block = spzeros(eltype(pm.base.B[1]), size(pm.base.B[1])...)
	terms = Any[]

	# Geometry corrections to B₀ and the geometry-dependent singular B₁.
	append!(terms, build_linear_corrections(pm.B0_geometry, pm.basis, (1, 0);
		external_components = [1]))
	append!(terms, build_linear_corrections(pm.B1_geometry, pm.basis, (0, 1);
		external_components = [1]))

	for (idx, α) in enumerate(pm.basis.mset.exponents)
		k = α[1]
		# Quadratic convection, including its μ⁰ coefficient.
		if any(!iszero, pm.cache.adj[ci][q][idx]
		       for ci in eachindex(pm.cache.adj) for q in eachindex(pm.cache.adj[ci]))
			comp = _external_components(k)
			cl = Base.invokelatest(_pf_convection, Val(k), comp, pm, idx)
			push!(terms, MultilinearMap(cl, (2, 0), k; fully_asymmetric = false))
		end
		k > 0 && norm(pm.h_geometry[idx]) > 0 && begin
			comp = _external_components(k)
			cl = Base.invokelatest(_pf_vector, Val(k), comp, pm.h_geometry[idx])
			push!(terms, MultilinearMap(cl, (0, 0), k; fully_asymmetric = false))
		end
		if pm.include_reynolds
			A = (pm.base.fom.reference_length * pm.reynolds_scale) .* pm.viscosity[idx]
			if _nonzero_matrix(A)
				cl = Base.invokelatest(_pf_reynolds_linear, Val(k + 1), A)
				push!(terms, MultilinearMap(cl, (1, 0), k + 1; fully_asymmetric = false))
			end
			if norm(pm.h_reynolds[idx]) > 0
				cl = Base.invokelatest(
					_pf_reynolds_vector, Val(k + 1), pm.h_reynolds[idx])
				push!(terms, MultilinearMap(cl, (0, 0), k + 1; fully_asymmetric = false))
			end
		end
	end

	ext = ExternalSystem(pm.include_reynolds ? (0.0 + 0im, 0.0 + 0im) : (0.0 + 0im,))
	model = NthOrderModel((pm.base.B..., zero_block), Tuple(terms), ext)
	master_indices = reduce(vcat, [[2p - 1, 2p] for p in master])
	npairs = length(spectrum.eigenvalues) ÷ 2
	σ = reduce(vcat, [[2p, 2p - 1] for p in 1:npairs])
	spectral = SpectralData(model, spectrum; master = master_indices,
		conjugate_permutation = σ)
	master_pair = reduce(vcat, [[2p, 2p - 1] for p in 1:length(master)])
	perm = conjugate_permutation === nothing ?
		full_conjugate_permutation(master_pair, ext) : collect(Int, conjugate_permutation)
	return (; model, spectral, meta = (; conjugate_permutation = perm,
		ORD = 2, N_EXT = pm.include_reynolds ? 2 : 1,
		external_order = pm.info.external_order, master_indices,
		Re₀ = pm.base.Re₀, Δη = pm.reynolds_scale,
		identity_errors = pm.identity_errors, n_terms = length(terms),
		parametric_fluid_formulation_version = PARAMETRIC_FLUID_FORMULATION_VERSION))
end
