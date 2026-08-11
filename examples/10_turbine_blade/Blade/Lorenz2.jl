Blade_dir = "/Users/alessandravizzaccaro/Documents/julia/MORFE.jl/demo/Blade"

using Pkg: Pkg
Pkg.activate(Blade_dir)
if !isfile(joinpath(Blade_dir, "Manifest.toml"))
	Pkg.develop(Pkg.PackageSpec(path = joinpath(Blade_dir, "../..")))
	Pkg.add(["Ferrite", "LinearMaps", "StaticArrays", "WriteVTK"])
	Pkg.add(["BifurcationKit", "Plots", "StaticArrays", "OrdinaryDiffEq"])
	Pkg.add("Arpack", "FFTW")
    Pkg.add("DataInterpolations")
end
Pkg.instantiate()

using Ferrite, WriteVTK, FFTW
using MORFE, FerriteGmsh, SparseArrays, LinearAlgebra, Arpack, LinearMaps, Serialization, StaticArrays, Printf
#using DataFrames, CSV
using StaticArrays: SVector
using MORFE.Polynomials: DensePolynomial, evaluate, extract_component,
	each_term, similar_poly
using MORFE.Realification: realify
using BifurcationKit
using Plots
import OrdinaryDiffEq: ODEProblem, Rodas5P, Midpoint, RK4
import OrdinaryDiffEq: solve as odesolve
using DataInterpolations
ENV["GKSwstype"] = "nul"

include(joinpath(Blade_dir, "setup/mesh.jl"))
include(joinpath(Blade_dir, "setup/assembly.jl"))
include(joinpath(Blade_dir, "setup/logging.jl"))
#include(joinpath(Blade_dir, "setup/write_vtk.jl"))


# not used here but this is the one coming from the MainLorenz.jl script
R = deserialize(joinpath(Blade_dir, "R_Lorenz.jls"))


# --- ODE system ---
# Lorenz rewritten around one nontrivial equilibrium and scaled to have linear freq=72.5864
# only used for time integration of the Lorenz system, not for the ROM
# no need to run the time integration as the results are already saved in the jls files:
# x_Lorenz.jls, y_Lorenz.jls, z_Lorenz.jls, t_Lorenz.jls
function Lorenz!(du, u, scale_time_val, t)
    du[1] = scale_time_val*(σ*(u[2]-u[1]))
    du[2] = scale_time_val*(u[1]-u[2]-βρ*u[3]-u[1]*u[3])
    du[3] = scale_time_val*(u[1]*u[2]- β*u[3] + βρ*(u[1]+u[2]))
end


# Parameters 
scale_time0 = 72.5864/10.1945
σ=10.0  #ρ=28.0
β=8.0/3
βρ = 6*sqrt(2.0) # = sqrt(β*(ρ-1))


# Initial condition:
u0 = zeros(Float64, 3)
u0[1] = (1.1 - βρ)   #
#u0[2] = 1.0


tspan = (0.0/scale_time0, 16000.0/scale_time0)

# --- Solve ---
prob = ODEProblem(Lorenz!, u0, tspan, scale_time0)
if 1==0
    # no need to run the time integration as the results are already saved in the jls files:
    # x_Lorenz.jls, y_Lorenz.jls, z_Lorenz.jls, t_Lorenz.jls
    println("Integrating Lorenz system …")
    sol  = odesolve(prob, Rodas5P(), saveat=0.0001/scale_time0)


    t=Vector{Float64}(undef,length(sol));
    x=Vector{Float64}(undef,length(sol));
    y=Vector{Float64}(undef,length(sol));
    z=Vector{Float64}(undef,length(sol));

    for i=1:length(sol);
        t[i]=sol.t[i];
        x[i]=sol.u[i][1];
        y[i]=sol.u[i][2];
        z[i]=sol.u[i][3];
    end
else
    x = deserialize(joinpath(Blade_dir, "x_Lorenz.jls"))
    y = deserialize(joinpath(Blade_dir, "y_Lorenz.jls"))
    z = deserialize(joinpath(Blade_dir, "z_Lorenz.jls"))
    t = deserialize(joinpath(Blade_dir, "t_Lorenz.jls"))
end




if 1==1


# this is the inverse transformation from the modal coordinates to the original coordinates
InvEigVecLorenz = [-0.336995 0.174698 0.132497;
            0.168497+0.185624im -0.0873492+0.535684im 0.433751-0.0447537im;
            0.168497-0.185624im -0.0873492-0.535684im 0.433751+0.0447537im]
m1=[x y z]*InvEigVecLorenz[1,:]
m2=[x y z]*InvEigVecLorenz[2,:]
m3=[x y z]*InvEigVecLorenz[3,:]
# --- Interpolate forcing into a continuous function ---
forcing_interp_m1 = LinearInterpolation(m1, t)
forcing_interp_m2 = LinearInterpolation(m2, t)
forcing_interp_m3 = LinearInterpolation(m3, t)

# --- ODE system ---
function system!(du, u, forcing, t)
    f1,f2,f3 = forcing
    # the explicit equation should be equivalent to the following:
    # du12=evaluate(extract_component(R.poly,1),[u[1]+u[2]*1im, u[1]-u[2]*1im, 3*f1(t), 3*f2(t), 3*f3(t)])
    # however, I used this explicit one (from R_Lorenz.jls) for some reason I cannot remember (performance?)
    # not that f1 does not appear in the reduced dynamics because its associated eigenvalue is not resonant with the blade's ones
    du12 =  (-0.362932 + 72.58551589911019im)*(u[1]+u[2]*1im) + 
            (6.15757652821076e-14 - 0.020665279870899812im)*(3*f3(t)) + 
            (0.1574288484668464 - 123.70910348028646im)*(u[1]+u[2]*1im)^2*(u[1]-u[2]*1im) + 
            (-0.00011786334667083817 - 0.017609158554239425im)*(u[1]+u[2]*1im)^2*(3*f2(t)) +
            (0.00021705640178306806 - 0.017290725279845338im)*(u[1]+u[2]*1im)*(u[1]-u[2]*1im)*(3*f3(t)) +
            (9.101921043884228e-9 + 0.00014123564804243932im)*(u[1]+u[2]*1im)*(3*f2(t))*(3*f3(t)) +
            (-1.7757600151754315e-5 + 1.3199383247351357e-5im)*(u[1]-u[2]*1im)*(3*f3(t))^2 +
            (7.965970699375778e-6 - 1.733188548943378e-5im)*(3*f2(t))*(3*f3(t))^2
    # to generate the data for the linear blade case, simply comment the lines after the second one
    du[1] = real(du12)
    du[2] = imag(du12)
end


u0 = [0.0, 0.0]

tspan = (0.0/scale_time0, 12000.0/scale_time0)
# --- Solve ---
prob = ODEProblem(system!, u0, tspan, (forcing_interp_m1,forcing_interp_m2,forcing_interp_m3))
println("Integrating ROM with Lorenz forcing …")
sol_ROM  = odesolve(prob, Rodas5P() ,dt=0.0001/scale_time0, saveat=0.0001/scale_time0)

U=Vector{Float64}(undef,length(sol_ROM));
V=Vector{Float64}(undef,length(sol_ROM));
for i=1:length(sol_ROM);
    #t[i]=sol_ROM.t[i];
    U[i]=sol_ROM.u[i][1];
    V[i]=sol_ROM.u[i][2];
end

# plot(U[1:100:end])

end


t_ROM = sol_ROM.t
U_interp = U    #CubicSpline(U, t)
# the force is the real part of the modal coordinates forcing variables 2 and 3 (the resonant ones)
Force = rfft(real.(forcing_interp_m3(t_ROM)))
# the displacement is the real part of the modal coordinates solution variables 1 and 2 
Displ = rfft(U_interp)
N = length(U_interp)
dt = 0.0001/scale_time0
freq = 2*pi*(0:div(N,2)) ./ (N*dt)




plt1=plot(xlims  = (60, 90), ylims = (0,0.2))
#i1=argmin(abs.(freq_OSC .-60));
#i2=argmin(abs.(freq_OSC .-90));
#plot!(plt1,freq_OSC[i1:i2],(abs.(Displ_OSC[i1:i2]))./(abs.(Force_OSC[i1:i2]))./sc_OSC,lw=1)
i1=argmin(abs.(freq .-60));
i2=argmin(abs.(freq .-90));
plot!(plt1,freq[i1:i2],(abs.(Displ[i1:i2]))./(abs.(Force[i1:i2])),lw=0, marker = (:circle, 1, 1, stroke(0, 0.2)))

#plt2=plot(t_OSC,U_OSC,lw=0);plot!(plt2,t,U,lw=0)
