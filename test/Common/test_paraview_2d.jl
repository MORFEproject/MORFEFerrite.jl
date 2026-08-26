using Ferrite
using MORFEFerrite
using Test
using WriteVTK

@testset "2-D ParaView export" begin
    nodes = [
        Node((0.0, 0.0)),
        Node((1.0, 0.0)),
        Node((0.0, 1.0)),
        Node((1.0, 1.0)),
    ]
    grid = Grid([
        Triangle((1, 2, 3)),
        Triangle((2, 4, 3)),
    ], nodes)

    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 2}()^2)
    add!(dh, :p, Lagrange{RefTriangle, 1}())
    close!(dh)
    range_u = dof_range(dh, :u)
    range_p = dof_range(dh, :p)

    state = collect(Float64, 1:ndofs(dh))
    mode = ComplexF64.(state, reverse(state))

    mktempdir() do dir
        mesh_base = joinpath(dir, "mesh2d")
        mesh_file = write_paraview_mesh(mesh_base, grid)
        @test mesh_file == mesh_base * ".vtu"
        @test isfile(mesh_file)

        fields_base = joinpath(dir, "fields2d")
        info = write_paraview_p2p1(fields_base, grid, dh, range_u, range_p;
            state, mode, mode_dofs=collect(1:ndofs(dh)),
            reynolds=125.0, eigenvalue=0.1 + 2.0im,
            diagnostics=(; residual=1e-12))

        # Two P2 triangles share one edge: 6 + 6 - 3 shared P2 nodes = 9.
        @test info.n_points == 9
        @test info.n_cells == 2
        @test info.raw_velocity_scale > 0
        @test isfile(info.file)

        fom = (; grid, dh, dof_range_u=range_u, dof_range_p=range_p,
            free=collect(1:ndofs(dh)), free_dpim=collect(1:ndofs(dh)))
        convenience = write_paraview_p2p1(joinpath(dir, "convenience"), fom;
            state, mode)
        @test convenience.n_points == 9
        @test isfile(convenience.file)
        @test MORFEFerrite.FluidNavierStokes.write_paraview_p2p1 ===
            MORFEFerrite.write_paraview_p2p1

        raw_phase = write_paraview_p2p1(joinpath(dir, "raw_phase"), fom;
            state, mode, normalize_mode=true, align_mode_phase=false)
        @test raw_phase.n_points == 9
        @test isfile(raw_phase.file)

        xml = String(read(info.file))
        for name in ("base_velocity", "base_pressure", "mode_velocity_real",
                     "mode_velocity_imag", "mode_velocity_amplitude",
                     "mode_pressure_real", "mode_pressure_imag",
                     "mode_pressure_amplitude", "base_plus_mode_real",
                     "base_plus_mode_imag", "base_vorticity",
                     "mode_vorticity_real", "mode_vorticity_imag",
                     "mode_vorticity_amplitude", "total_vorticity_real",
                     "total_vorticity_imag", "visual_amplitude", "Reynolds",
                     "eigenvalue_real", "eigenvalue_imag", "residual")
            @test occursin("Name=\"$name\"", xml)
        end

        # Solid-body rotation u=(-y,x) has exact scalar vorticity +2. This
        # checks gradients, P2 node ordering, and shared-node recovery.
        solid = zeros(ndofs(dh))
        refs = Ferrite.reference_coordinates(Lagrange{RefTriangle, 2}())
        for cell in CellIterator(dh)
            gdofs = celldofs(cell)
            vdofs = gdofs[range_u]
            X = getcoordinates(cell)
            for a in 1:6
                ξ = refs[a]
                x = ξ[1]*X[1] + ξ[2]*X[2] + (1-ξ[1]-ξ[2])*X[3]
                solid[vdofs[2a-1]] = -x[2]
                solid[vdofs[2a]] = x[1]
            end
        end
        vort = write_paraview_p2p1(joinpath(dir, "vorticity_exact"), fom;
            state=solid, mode=im .* ComplexF64.(solid),
            normalize_mode=false, align_mode_phase=false)
        @test collect(vort.base_vorticity_extrema) ≈ [2.0, 2.0] atol=1e-12
        @test vort.mode_vorticity_max ≈ 2.0 atol=1e-12

        phased = write_paraview_p2p1(joinpath(dir, "vorticity_phase"), fom;
            state=solid, mode=im .* ComplexF64.(solid), phase=π/2,
            visual_amplitude=0.1, normalize_mode=false,
            align_mode_phase=false)
        @test phased.total_vorticity ≈ fill(1.8, phased.n_points) atol=1e-12
        phased_xml = String(read(phased.file))
        @test occursin("Name=\"total_velocity\"", phased_xml)
        @test occursin("Name=\"total_vorticity\"", phased_xml)
        @test occursin("Name=\"visualization_phase\"", phased_xml)
        @test occursin("Name=\"mode_raw_velocity_scale\"", phased_xml)

        animation = write_paraview_p2p1_phase_animation(
            joinpath(dir, "vorticity_animation"), fom;
            state=solid, mode=im .* ComplexF64.(solid), phase_frames=4,
            visual_amplitude=0.1, normalize_mode=false,
            align_mode_phase=false)
        @test animation.frames == 4
        @test isfile(animation.pvd)
        @test all(isfile, animation.files)
        @test count(line -> occursin("<DataSet", line),
            readlines(animation.pvd)) == 4

        @test_throws DimensionMismatch write_paraview_p2p1(
            joinpath(dir, "bad"), grid, dh, range_u, range_p;
            state=zeros(ndofs(dh) - 1))
    end
end
