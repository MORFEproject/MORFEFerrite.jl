"""
	main.jl — single-parameter parametric ROM for a 3D clamped-clamped arch beam,
	using the GENERAL parametric formulation in `MORFEFerrite.ParametricStructural`.

The base configuration is a sinusoidal arch of height h₀ = h₀_L_ratio · L; a single
external parameter θ controls the deviation:

	x(θ,x₀) = x₀ + (1+θ) w(x₀)   ⇒   J(θ,x₀) = J₀(x₀) + θ J₁(x₀)

Here θ is the single shape-field parameter (N_θ = 1), so the general multivariate
engine reduces to the univariate arch series; this run reproduces the reference in
`reference_data/`. The geometry is the ONLY example-specific input — everything else
(θ-series, det/adj/1-det, nonlinearity, corrections) is the shared library.

Orders are overridable via `MORFE_MAXZ` / `MORFE_MAXT` env vars (defaults 11 / 7).
NOTE: the general θ-series kernel currently allocates in its inner loop, so the full
order-11 solve is slow; a reduced order (e.g. `MORFE_MAXZ=5 MORFE_MAXT=3`) reproduces
the reference's low-order coefficients (DPIM is graded → lower orders are exact). An
allocation-reduction pass (cf. the SVK O5 optimisation) is the pending follow-up.
"""

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop([
		Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..", "..", "..", "MORFE_jl")),
		Pkg.PackageSpec(path = joinpath(@__DIR__, "..", "..")),
	])
	Pkg.add(["Ferrite", "FerriteGmsh", "Arpack", "LinearMaps", "StaticArrays", "Tensors"])
end
Pkg.instantiate()

using MORFE, MORFEFerrite
using Ferrite, FerriteGmsh, Arpack, LinearMaps
using SparseArrays, LinearAlgebra, Printf, Tensors, StaticArrays
const PS = MORFEFerrite.ParametricStructural

# arch_geometry.jl provides arch_jacobian_pair; it uses `Tens3`.
const Tens3 = Tensor{2, 3, Float64, 9}
include(joinpath(@__DIR__, "fem", "arch_geometry.jl"))
include(joinpath(@__DIR__, "config.jl"))               # h0_L_ratio

const h₀_L_ratio = h0_L_ratio
const max_degree_z = parse(Int, get(ENV, "MORFE_MAXZ", "11"))
const max_degree_θ = parse(Int, get(ENV, "MORFE_MAXT", "7"))
const max_degree_total = max_degree_z
const N_θ_K = 2        # K(θ) exact degree 2
const N_θ_G = 3        # G(u₁,u₂;θ) exact degree 3
const N_θ_H = 4        # H(u₁,u₂,u₃;θ) exact degree 4
const ROM = 2
const N_EXT = 1
const NVAR = ROM + N_EXT

# ── §1 mesh + FE ───────────────────────────────────────────────────────────
grid = togrid(joinpath(@__DIR__, "beam_h27_10x2x2.msh"))
ip = Lagrange{RefHexahedron, 2}()^3
geo_ip = Lagrange{RefHexahedron, 2}()
qr = QuadratureRule{RefHexahedron}(3)
cv = CellValues(qr, ip, geo_ip)
dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "Dirichlet"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch); update!(ch, 0.0)
free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free))
n_free = length(free); FOM = n_free
println("Total DOFs: ", ndofs(dh), "   free: ", n_free)

# ── §2 material + arch geometry ────────────────────────────────────────────
E = 160e3; ν = 0.22; ρ = 2.32e-3
λ_lame = (E * ν) / ((1 + ν) * (1 - 2ν))
μ_lame = E / (2(1 + ν))
α = 0.0; β = 0.0
const L = 1000.0
const h₀ = h₀_L_ratio * L
geom(x₀) = arch_jacobian_pair(x₀, h₀, L)          # x₀ -> (J₀, J₁)

# ── §3 parametric linear K/M coefficient matrices ──────────────────────────
basis_K = PS.ThetaBasis([N_θ_K])
K_full = [allocate_matrix(dh) for _ in 1:PS.nterms(basis_K)]
M_full = [allocate_matrix(dh) for _ in 1:PS.nterms(basis_K)]
@time PS.assemble_parametric_K_M!(K_full, M_full, dh, cv, λ_lame, μ_lame, ρ, geom, basis_K)
K_arr = [K_full[m][free, free] for m in 1:PS.nterms(basis_K)]
M_arr = [M_full[m][free, free] for m in 1:PS.nterms(basis_K)]
K = K_arr[1]; M = M_arr[1]; C = α * M + β * K       # base arch (θ=0)

# ── §4 eigenproblem on base arch ───────────────────────────────────────────
solver_eig = StructureModalDampingEigensolver(10, α, β)
eigenproblem = solve_eigenproblem(K, M, solver_eig; sorter! = (args...) -> nothing)
(eigenvalues, Y, X) = get_eigenpairs(eigenproblem)
select_master_modes_by_sorting(eigenproblem, ROM)
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[1:ROM])
master_modes = Y[:, 1, 1:ROM]
left_eigenmodes = X[:, 1:ROM]
master_modes_derivatives = zeros(ComplexF64, FOM, 2, ROM)
for r in 1:ROM
	master_modes_derivatives[:, 1, r] .= Y[:, 2, r]
	master_modes_derivatives[:, 2, r] .= master_eigenvalues[r] .* Y[:, 2, r]
end
println("master eigenvalues: ", collect(master_eigenvalues))

# ── §5 nonlinear maps + K corrections (GENERAL library) ────────────────────
pgn_quad = PS.ParametricGeometricNonlinearity{2}(dh, cv, λ_lame, μ_lame, geom,
	free_to_local, n_free, PS.ThetaBasis([N_θ_G]))
pgn_cube = PS.ParametricGeometricNonlinearity{3}(dh, cv, λ_lame, μ_lame, geom,
	free_to_local, n_free, PS.ThetaBasis([N_θ_H]))
quad_maps = PS.multilinear_maps(pgn_quad)
cube_maps = PS.multilinear_maps(pgn_cube)
K_corrections = PS.build_K_corrections(K_arr, basis_K)
println("maps: quad=$(length(quad_maps)) cube=$(length(cube_maps)) Kcorr=$(length(K_corrections))")

# ── §6 augmented model + anisotropic multiindex set ────────────────────────
ext_sys = ExternalSystem((complex(0.0, 0.0),))       # single frozen θ
ZERO = spzeros(eltype(K), n_free, n_free)
model = NDOrderModel((K, C, M, ZERO),
	(quad_maps..., cube_maps..., K_corrections...), ext_sys)

mset = MultiindexSet([
	SVector{NVAR, Int}(a, b, c)
	for a in 0:max_degree_z for b in 0:max_degree_z for c in 0:max_degree_θ
	if a + b ≤ max_degree_z && c ≤ max_degree_θ && 1 ≤ a + b + c ≤ max_degree_total])
println("monomials: ", length(mset))

resonance_set = resonance_set_from_complex_normal_form_style(
	mset, Vector{ComplexF64}(master_eigenvalues), 0.05;
	external_eigenvalues = zeros(ComplexF64, N_EXT))

# ── §7 cohomological solve ─────────────────────────────────────────────────
left_modes_derivatives = left_eigenmode_orders_from_slice(
	model.linear_terms, left_eigenmodes, collect(master_eigenvalues))[:, 1:(end-1), :]
@time (W, R) = solve_cohomological_problem(model, mset, master_eigenvalues,
	master_modes, left_eigenmodes, resonance_set;
	master_modes_derivatives = master_modes_derivatives,
	left_modes_derivatives = left_modes_derivatives,
	conjugate_permutation = [2, 1, 3])

# ── §8 save R_coefficients.csv (for validation vs reference) ────────────────
outdir = joinpath(@__DIR__, "results", "data", @sprintf("arch_h%.3f", h₀_L_ratio))
mkpath(outdir)
open(joinpath(outdir, "R_coefficients.csv"), "w") do io
	exps = R.poly.multiindex_set.exponents
	NVAR_R = size(R.poly.coefficients, 1)
	println(io, join(["exp_$i" for i in 1:length(exps[1])], ",") * "," *
			join(["R$(i)_re,R$(i)_im" for i in 1:NVAR_R], ","))
	for (m, ex) in enumerate(exps)
		c = R.poly.coefficients[:, m]
		any(abs.(c) .> 1e-14) || continue
		println(io, join(string.(Int.(ex)), ",") * "," *
				join(["$(real(c[i])),$(imag(c[i]))" for i in 1:NVAR_R], ","))
	end
end
println("\nROM saved to $outdir/R_coefficients.csv")
