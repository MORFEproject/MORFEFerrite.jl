

using Pkg: Pkg
Pkg.activate(@__DIR__)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(@__DIR__, "../..")))
	Pkg.add(["Ferrite", "LinearMaps", "StaticArrays", "WriteVTK"])
	Pkg.add(["BifurcationKit", "Plots", "StaticArrays", "OrdinaryDiffEq"])
	Pkg.add("Arpack")
end
Pkg.instantiate()

using Ferrite, WriteVTK
using MORFE, FerriteGmsh, SparseArrays, LinearAlgebra, Arpack, LinearMaps, Serialization, StaticArrays, Printf
#using DataFrames, CSV
using StaticArrays: SVector
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component,
	each_term, similar_poly
using MORFE.Realification: realify
using BifurcationKit
using Plots
import OrdinaryDiffEq: ODEProblem, Rodas5P
import OrdinaryDiffEq: solve as odesolve
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "setup/mesh.jl"))
include(joinpath(@__DIR__, "setup/assembly.jl"))
include(joinpath(@__DIR__, "setup/logging.jl"))
#include(joinpath(@__DIR__, "setup/write_vtk.jl"))

# ── Config ────────────────────────────────────────────────────────────────────
config_path = get(ARGS, 1, joinpath(@__DIR__, "results/mode_1_order_7_cnf/config.jl"))
cfg = include(config_path)
results_dir = dirname(abspath(config_path))
master_indices = vcat([[2n-1, 2n] for n in cfg.phys_modes]...)
conjugate_permutation = [2, 1, 3, 5, 4]
ROM = length(master_indices);
N_EXT = 3
NVAR = ROM + N_EXT

out, _log = open_log(results_dir)

# ── Material (isotropic polysilicon, mm·kg·s) ─────────────────────────────────
const E = 104.0e9
const ν = 0.3
const ρ = 4400.0
const λ = E*ν / ((1+ν)*(1-2ν))
# λ = 2 * λ * μ / (λ + 2μ)   # for plane stress pass this to λ in the FEM assembly 
const μ = E / (2(1+ν))
print_header(out, cfg, ROM, N_EXT, NVAR, E, ν, ρ, λ, μ, results_dir)


# -----------------------------------------------------------------------
# 1. Mesh and FE setup
# -----------------------------------------------------------------------
const mesh_file = joinpath(@__DIR__, "blade_2.mphtxt")
isfile(mesh_file) || error("Mesh not found: $mesh_file")
grid, constrained = load_arch_mesh(mesh_file)
ip = Lagrange{RefPrism, 2}()^3;
geo_ip = Lagrange{RefPrism, 2}()
qr = QuadratureRule{RefPrism}(4)
cv = CellValues(QuadratureRule{RefPrism}(4), ip, geo_ip)

dh = DofHandler(grid);
add!(dh, :u, ip);
close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, constrained, (x, t) -> zeros(3), [1, 2, 3]));
close!(ch);
update!(ch, 0.0)

#addfacetset!(grid, "bc1", (x) -> abs(x[1] - 216.0) < 1e-10)
#addfacetset!(grid, "bc2", (x) -> abs(x[1] + 216.0) < 1e-10)


free = sort(setdiff(1:ndofs(dh), ch.prescribed_dofs))
free_to_local = Dict(d => i for (i, d) in enumerate(free));
n_free = length(free)
print_mesh_info(out, mesh_file, length(grid.cells), length(grid.nodes), ndofs(dh),
	length(ch.prescribed_dofs), n_free)

# ── Stiffness, mass, damping ──────────────────────────────────────────────────
K_full = allocate_matrix(dh);
M_full = allocate_matrix(dh)
assemble_KM!(K_full, M_full, dh, cv, λ, μ, ρ)
K = K_full[free, free]
M = M_full[free, free]
C = cfg.rayleigh_α .* M .+ cfg.rayleigh_β .* K

#I, J, V = findnz(M)
#df = DataFrame((i =I, j = J, v = V))
#CSV.write("./M.csv", df)
#I, J, V = findnz(K)
#df = DataFrame((i =I, j = J, v = V))
#CSV.write("./K.csv", df)


# ── Eigenproblem ──────────────────────────────────────────────────────────────
t_eig = @timed spectrum(K, M,
	StructureModalDampingEigensolver(cfg.neig, cfg.rayleigh_α, cfg.rayleigh_β);
	sorter! = (args...) -> nothing)
eigenproblem = t_eig.value;
eigenvalues, Y, X = eigenproblem.eigenvalues, eigenproblem.eigenmodes, eigenproblem.left_eigenmodes
print_mode_table(out, eigenvalues, master_indices)

# ── NL terms + model ──────────────────────────────────────────────────────────
mset  = all_multiindices_up_to(NVAR, cfg.max_degree; min_degree = 1)
ncols = length(mset)

scaling_eigs = 1;
Y = Y*scaling_eigs;
X = X/scaling_eigs;
master_modes = Y[:, 1, master_indices];
left_eigenmodes = X[:, master_indices];
master_eigenvalues = SVector{ROM, ComplexF64}(eigenvalues[master_indices])
n_deriv = size(eigenproblem.eigenmodes, 2) - 1
master_modes_derivatives = zeros(ComplexF64, n_free, n_deriv, ROM)
for (r, idx) in enumerate(master_indices), k in 1:n_deriv
	master_modes_derivatives[:, k, r] .= Y[:, k+1, idx];
end


# ── Export modes in VTK ──────────────────────────────────────────────────────────────
export_modes = false
if export_modes == true
	const _out = joinpath(@__DIR__, "results/modes/paraview")
	println("\n§4  Writing Paraview files to $(_out)/ …")

	write_paraview_mesh(joinpath(_out, "results/modes/mesh"), grid;
		dh = dh, prescribed_dofs = ch.prescribed_dofs)

	write_paraview_modes(_out, grid, dh, eigenvalues, Y, free; n_modes = 2*cfg.neig)
end

term_quad  = FerriteGeometricNonlinearity{2}(dh, cv, free_to_local, n_free, λ, μ; max_unique_cols = ncols)
term_cubic = FerriteGeometricNonlinearity{3}(dh, cv, free_to_local, n_free, λ, μ; max_unique_cols = ncols)


# Parameters Lorenz
scale_time = 72.5864/10.1945
σ=10.0  #ρ=28.0
β=8.0/3
βρ = 6*sqrt(2.0) # = sqrt(β*(ρ-1))

# Lorenz system rewritten around one nontrivial equilibrium and scaled to have linear freq=72.5864
# dr1/dt = scale_time*( -σ r1 +σ r2 )
# dr2/dt = scale_time*(  r1 - r2 - βρ r3 - r1*r3)
# dr3/dt = scale_time*(  + βρ r1 + βρ r2 - β r3  + r1 r2)


# the previous system is then diagonalised through the eigenvectors, resulting in the following system:
external_system = ExternalSystem(
	DensePolynomial(
		scale_time*ComplexF64[                -13.8546 0 0 0.139722 0.36342+0.251644im 0.36342-0.251644im -0.109834+0.0352947im -0.047656 -0.109834-0.0352947im
			0 0.0939556-10.1945im 0 -0.957035+1.22873im -0.139722+1.48651im 0.178149+0.379738im -0.347639-0.337818im 0.331857-0.423992im 0.116385-0.0280735im
			0 0 0.0939556+10.1945im -0.957035-1.22873im 0.178149-0.379738im -0.139722-1.48651im 0.116385+0.0280735im 0.331857+0.423992im -0.347639+0.337818im],
		all_multiindices_up_to(3, 2, min_degree = 1),
	),
)


# since the eigenvector matrix has all ones on the last variable (z-z_eq),
# the forcing expressed here as modal coordinates, results in the z=sum(r):
forcing = Tuple(map(cfg.forces) do f
	fv = real((f.amplitude) .* (M * Y[:, 1, 2*f.shape_mode-1]/scaling_eigs))
	MultilinearMap((res, r) -> (res .+= fv * sum(r)), (0, 0), 1);
end)
model = NthOrderModel((K, C, M), (term_quad, term_cubic, forcing...), external_system)


# ── Resonance set ─────────────────────────────────────────────────────────────
master_eigs = Vector{ComplexF64}(master_eigenvalues);
ext_eigs = Vector{ComplexF64}(external_system.eigenvalues)
tol_rel = cfg.resonance.tolerance_rel
tol_vec = [[tol_rel * abs(master_eigs[j]) for j in 1:ROM] for _ in 1:length(mset.exponents)]
resonance_set = if cfg.resonance.style == :cnf
	resonance_set_from_complex_normal_form_style(mset, master_eigs, tol_vec; external_eigenvalues = ext_eigs)
elseif cfg.resonance.style == :rnf
	resonance_set_from_real_normal_form_style(mset, master_eigs, conjugate_permutation, tol_vec;
		external_eigenvalues = ext_eigs)
else
	resonance_set_from_graph_style(mset, master_eigs, ext_eigs, ComplexF64[], tol_rel)
end
print_resonance_summary(out, resonance_set, mset, master_eigs, ext_eigs, tol_rel, NVAR, cfg.max_degree)

# ── Cohomological solve ───────────────────────────────────────────────────────
#if get(cfg, :check, false)
#	print(out, "\nProceed with cohomological solve? [y/N]: ")
#	readline() == "y" || (close_log(_log); exit(0))
#end
left_modes_derivatives = left_eigenmode_orders_from_slice(
	model.linear_terms, left_eigenmodes, collect(master_eigenvalues))[:, 1:(end-1), :]
spectral = SpectralData(; eigenvalues = master_eigenvalues,
	right_modes = master_modes, right_derivatives = master_modes_derivatives,
	left_modes = left_eigenmodes, left_blocks = Array(left_modes_derivatives))
t_solve = @timed solve_cohomological_problem(model, mset, spectral, resonance_set;
	conjugate_permutation = conjugate_permutation)
W, R = t_solve.value
print_R_coefficients(out, R)

# ── Save ──────────────────────────────────────────────────────────────────────
serialize(joinpath(@__DIR__, "W_Lorenz.jls"), W);
serialize(joinpath(@__DIR__, "R_Lorenz.jls"), R)
print_summary(out, cfg, n_free, eigenvalues, master_indices, cfg.max_degree, NVAR, ncols,
	t_eig, t_solve, results_dir)
close_log(_log)






