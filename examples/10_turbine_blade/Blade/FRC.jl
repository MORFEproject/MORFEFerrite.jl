Blade_dir = "/Users/alessandravizzaccaro/Documents/julia/MORFE.jl/demo/Blade"


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

# ------------------------------------------------------------------
# 1.  Load ROM
# ------------------------------------------------------------------

println("Loading ROM …")
R = deserialize(joinpath(Blade_dir, "R_Lorenz.jls"))



# ---------------------------------------------------------------------------
# Full RHS
# ---------------------------------------------------------------------------

"""
    rhs!(du, u, p, t)

Full ODE right-hand side. The c and s equations are replaced by the explicit
Ω-parametric form regardless of their position in the state vector:

    du[idx_c] = -Ω · u[idx_s]
    du[idx_s] =  Ω · u[idx_c]

All other equations come from the polynomial F.
"""

function make_rhs(idx_c, idx_s)
    function rhs!(du, u, Omega, t = 0)    # t = 0 default — works for both callers
        Omega_val = Omega isa AbstractArray ? Omega[1] : Omega
            # what follows is simply the R.poly obtained from Lorenz ,
            # but with the z calculated as follows:
            # z[1] = u[1]+u[3]*1im
            # z[2] = u[1]-u[3]*1im
            # z[4] = u[idx_c]-u[idx_s]*1im
            # z[5] = u[idx_c]+u[idx_s]*1im
            du13 =  (-0.362932 + 72.58551589911019im)*(u[1]+u[3]*1im) + 
            (6.15757652821076e-14 - 0.020665279870899812im)*(u[idx_c]+u[idx_s]*1im) + 
            (0.1574288484668464 - 123.70910348028646im)*(u[1]+u[3]*1im)^2*(u[1]-u[3]*1im) + 
            (-0.00011786334667083817 - 0.017609158554239425im)*(u[1]+u[3]*1im)^2*(u[idx_c]-u[idx_s]*1im) +
            (0.00021705640178306806 - 0.017290725279845338im)*(u[1]+u[3]*1im)*(u[1]-u[3]*1im)*(u[idx_c]+u[idx_s]*1im) +
            (9.101921043884228e-9 + 0.00014123564804243932im)*(u[1]+u[3]*1im)*(u[idx_c]-u[idx_s]*1im)*(u[idx_c]+u[idx_s]*1im) +
            (-1.7757600151754315e-5 + 1.3199383247351357e-5im)*(u[1]-u[3]*1im)*(u[idx_c]+u[idx_s]*1im)^2 +
            (7.965970699375778e-6 - 1.733188548943378e-5im)*(u[idx_c]-u[idx_s]*1im)*(u[idx_c]+u[idx_s]*1im)^2
        # to obtain the linear FRC, simply comment the lines after the second one
        # this system should be equivalent to:
        # du12=evaluate(extract_component(R.poly,1),[u[1]+u[2]*1im, u[1]-u[2]*1im, 0, u[idx_c]-u[idx_s]*1im, u[idx_c]+u[idx_s]*1im])
        # with R.poly the one obtained from R_Lorenz.jls
        # note that the evaluation for Lorenz forcing had a factor 3 in front of the forcing terms, 
        # which is not present here 
        # I don't rememeber why, but perhaps I obtained it by making the linear FRF match between the two different forcings?
        du[1] = real(du13)
        du[3] = imag(du13)
        du[idx_c] = -Omega_val * u[idx_s] + Omega_val*u[idx_c]*(beta_forcing^2 -u[idx_c]^2-u[idx_s]^2)
        du[idx_s] =  Omega_val * u[idx_c] + Omega_val*u[idx_s]*(beta_forcing^2 -u[idx_c]^2-u[idx_s]^2)
        return du
    end
    return rhs!
end


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

NVAR   = 4    # total number of variables
Omega0 = 60.    # starting angular frequency
IDX_C  = 2    # 1-based index of c in the state vector
IDX_S  = 4    # 1-based index of s in the state vector
beta_forcing = 3.0




rhs! = make_rhs(
    #real(F.coefficients),
    #F.multiindex_set.exponents,
    IDX_C,
    IDX_S,
)


# Initial condition:
#   - all other variables: your best guess (zeros, steady state, linearisation)
#   - c(0) = 1, s(0) = 0  →  c(t) = cos(Ω·t),  s(t) = sin(Ω·t)
u0 = zeros(Float64, NVAR)
# u0[...] = ...    # fill non-c,s variables with your initial guess
u0[IDX_C] = beta_forcing   # c(0) = 1
u0[IDX_S] = 0.0   # s(0) = 0

T0 = 2π / Omega0  # exact period at starting Ω

# ---------------------------------------------------------------------------
# Integrate to approach the periodic orbit
# ---------------------------------------------------------------------------

println("Integrating to approach periodic orbit …")
ode_prob = ODEProblem(rhs!, u0, (0.0, 200 * T0), Omega0)
sol = odesolve(ode_prob, Rodas5P(), reltol=1e-10, abstol=1e-12)
ode_prob = ODEProblem(rhs!, sol.u[end], (0.0, T0), Omega0)
sol = odesolve(ode_prob, Rodas5P(), reltol=1e-10, abstol=1e-12)
# x1=Vector{Float64}(undef,length(sol));
# for i=1:length(sol);
# x1[i]=sol.u[i][1];end
# plot(sol.t[29300:end]/T0,x1[29300:end])



# ---------------------------------------------------------------------------
# BifurcationProblem
# ---------------------------------------------------------------------------

bif_prob = BifurcationProblem(
    rhs!,
    u0,
    Omega0,        # plain scalar — no struct, no field access
    (@optic _),    # continue in p directly since p IS Omega
)


# ---------------------------------------------------------------------------
# Collocation periodic orbit problem
# ---------------------------------------------------------------------------

n_time = 40    # number of mesh intervals (increase for sharper solutions)
n_coll = 4     # Gauss-Legendre collocation degree

coll, ig, = BifurcationKit.generate_ci_problem(
    PeriodicOrbitOCollProblem(n_time, n_coll;
        update_section_every_step = 0,
    ),
    bif_prob,
    sol,
    T0;
)


# ---------------------------------------------------------------------------
# Newton solve onto the periodic orbit
# ---------------------------------------------------------------------------

newton_opts = NewtonPar(
    tol            = 1e-10,
    verbose        = true,
    max_iterations = 30,
)

println("Solving for periodic orbit …")
po_sol = newton(coll, ig, newton_opts)

# ---------------------------------------------------------------------------
# Continuation — p_min/p_max are now Omega bounds
# ---------------------------------------------------------------------------

cont_opts = ContinuationPar(
    p_min          = 60.,
    p_max          = 90.,
    ds             = 0.025,
    dsmin          = 1e-6,
    dsmax          = 0.025,
    max_steps      = 1500,
    newton_options = NewtonPar(tol=1e-10, max_iterations=20),
)

branch = continuation(
    coll, po_sol.u, PALC(tangent=Bordered()), cont_opts;
    verbosity            = 2,
    plot                 = false,
    record_from_solution = (u, p; k...) -> begin
    T     = BifurcationKit.getperiod(coll, u)
    Omega = p.p    # p is a NamedTuple{prob, p} — extract the scalar here
    nodes = BifurcationKit.get_time_slices(coll, u)
    c0    = nodes[IDX_C, 1]
    s0    = nodes[IDX_S, 1]
    (
        T            = T,
        T_exact      = 2π / Omega,
        period_error = abs(T - 2π / Omega),
        c0           = c0,
        s0           = s0,
        cs_norm      = c0^2 + s0^2,
    )
    end,
)

A1=Vector{Float64}(undef,length(branch));
Ω=Vector{Float64}(undef,length(branch));
for i=1:length(branch)
    X1 = branch.sol[i].x[1:NVAR:end-1]
    A1[i] = maximum(X1)
    Ω[i] = branch.sol[i].p
end

plt1=plot(xlims  = (60, 90), ylims = (0,0.2))
plot!(plt1,Ω,A1)