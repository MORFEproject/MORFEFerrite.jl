# =====================================================================
# The assembly entry point: mesh file → AssembledFluidModel.
# =====================================================================

"""
	fluid_model(meshfile; Re, newton_tol = 1e-10, newton_max_iter = 30,
				s_init = nothing, verbose = true) -> AssembledFluidModel

Assemble the incompressible Navier-Stokes problem about its steady base flow at
Reynolds number `Re`.

Four stages, each already implemented in this module:

1. [`setup_fem`](@ref) — P2/P1 Taylor-Hood spaces, boundary conditions, DOF maps.
2. [`solve_steady_state`](@ref) — Newton solve for the base flow.
3. [`assemble_linear_operators`](@ref) — the linearised `B₀` and the singular `B₁`.
4. [`assemble_K_visc`](@ref) — the viscous operator, restricted two ways.

Then the Reynolds-continuation scaling, which is the part that used to live in the
example driver:

```julia
K_visc .*= -D
h₀_vec  = -D .* (K_visc_rect * s₀_full)
```

`D` is the cylinder diameter — the reference length in `ν = D/Re`, so the
external coordinate is `η′ = 1/Re − 1/Re₀` and these two carry its coefficient.
The rectangular block is required for `h₀` because the base flow is **nonzero on
the prescribed inlet DOFs** (Poiseuille), which the square free × free block drops.

`meshfile` is a `.msh` read by `FerriteGmsh`. Mesh *generation* stays with the
caller: it is problem geometry, not physics. The geometry constants this module
assumes (Turek–Schäfer channel, Ø 0.1 m cylinder) must match whatever produced
`meshfile` — see `fem_setup.jl`.
"""
function fluid_model(meshfile::AbstractString;
	Re::Real,
	newton_tol::Float64 = 1e-10,
	newton_max_iter::Int = 30,
	s_init::Union{Nothing, Vector{Float64}} = nothing,
	verbose::Bool = true)
	Re₀ = Float64(Re)

	t_fem = @elapsed fom = setup_fem(String(meshfile))
	verbose && @info "FEM: $(fom.n_free) free DOFs (steady state), $(fom.n_free_dpim) (DPIM)"

	t_ss = @elapsed (_, _, s₀_full) = solve_steady_state(fom;
		Re0 = Re₀, tol = newton_tol, max_iter = newton_max_iter, s_init = s_init)

	t_ops = @elapsed (B₀, B₁) = assemble_linear_operators(s₀_full, fom; Re0 = Re₀)

	t_kv = @elapsed begin
		(K_visc, K_visc_rect) = assemble_K_visc(fom)
		# η′ = 1/Re − 1/Re₀ multiplies −D·K_raw; applying it here rather than at the
		# call site is what keeps the convention out of comments.
		K_visc .*= -_CYL_D
		h₀_vec = -_CYL_D .* (K_visc_rect * s₀_full)
	end

	info = (; n_free = fom.n_free, n_free_dpim = fom.n_free_dpim,
		backend = "Ferrite P2/P1 Taylor-Hood", meshfile = String(meshfile),
		reference_length = _CYL_D,
		fem_time_s = t_fem, steady_time_s = t_ss, ops_time_s = t_ops, kvisc_time_s = t_kv)

	return AssembledFluidModel{typeof(fom), typeof(B₀), typeof(K_visc), typeof(K_visc_rect)}(
		fom, Re₀, s₀_full, B₀, B₁, K_visc, K_visc_rect, h₀_vec, info)
end
