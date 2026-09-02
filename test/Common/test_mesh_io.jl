using Ferrite
using Gmsh
using MORFEFerrite

const _mesh_examples = normpath(joinpath(@__DIR__, "..", "..", "examples", "mesh_import"))

function _with_gmsh(f::Function, path::AbstractString)
    gmsh.initialize()
    try
        gmsh.open(path)
        return f()
    finally
        gmsh.finalize()
    end
end

function _element_data(path::AbstractString, element_type::Integer)
    return _with_gmsh(path) do
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes(-1, -1)
        element_tags, connectivity = gmsh.model.mesh.getElementsByType(element_type)
        return (
            node_tags = Int64.(node_tags),
            coordinates = Float64.(coordinates),
            element_tags = Int64.(element_tags),
            connectivity = Int64.(connectivity),
        )
    end
end

@testset "MeshIO public API" begin
    for name in (
        :load_comsol_grid,
        :abaqus_to_gmsh, :abaqus_to_gmsh_linear,
        :comsol_to_gmsh, :comsol_to_gmsh_linear,
        :gmsh_to_comsol,
    )
        @test isdefined(MORFEFerrite, name)
        @test getfield(MORFEFerrite, name) === getfield(MORFEFerrite.Common.MeshIO, name)
    end
end

@testset "Abaqus to Gmsh" begin
    mktempdir() do directory
        c3d8 = joinpath(_mesh_examples, "Abaqus", "cube_c3d8.inp")
        c3d20 = joinpath(_mesh_examples, "Abaqus", "single_c3d20.inp")

        c3d8_out = joinpath(directory, "c3d8.msh")
        MORFEFerrite.abaqus_to_gmsh(c3d8, c3d8_out)
        data8 = _element_data(c3d8_out, 5)
        @test length(data8.node_tags) == 48
        @test length(data8.element_tags) == 18
        @test length(data8.connectivity) == 18 * 8

        c3d20_out = joinpath(directory, "c3d20.msh")
        MORFEFerrite.abaqus_to_gmsh(c3d20, c3d20_out)
        data20 = _element_data(c3d20_out, 17)
        @test length(data20.element_tags) == 1
        @test length(data20.connectivity) == 20
        coords = Dict(
            data20.node_tags[i] => Tuple(data20.coordinates[(3i - 2):3i])
            for i in eachindex(data20.node_tags)
        )
        @test coords[data20.connectivity[10]] == (0.0, 0.5, 0.0)
        @test _with_gmsh(c3d20_out) do
            tags, _ = gmsh.model.mesh.getElementsByType(17)
            _, determinants, _ = gmsh.model.mesh.getJacobian(Int(tags[1]), [0.0, 0.0, 0.0])
            determinants[1] > 0
        end

        linear_out = joinpath(directory, "c3d20-linear.msh")
        MORFEFerrite.abaqus_to_gmsh_linear(c3d20, linear_out)
        linear = _element_data(linear_out, 5)
        @test length(linear.connectivity) == 8

        # Abaqus node labels need not be contiguous; the output is compactly retagged.
        sparse_ids = joinpath(directory, "sparse-node-ids.inp")
        open(sparse_ids, "w") do io
            write(io, "*Node\n10,0,0,0\n20,1,0,0\n30,0,1,0\n40,0,0,1\n")
            write(io, "*Element, type=C3D4, elset=VOL\n7,10,20,30,40\n")
        end
        sparse_out = joinpath(directory, "sparse-node-ids.msh")
        MORFEFerrite.abaqus_to_gmsh(sparse_ids, sparse_out)
        sparse_data = _element_data(sparse_out, 4)
        @test sparse_data.node_tags == collect(1:4)
        @test sparse_data.connectivity == collect(1:4)
        @test sparse_data.element_tags == [7]

        # Exercise the same remapping for a quadratic solid, including its
        # Abaqus-to-Gmsh midside-node permutation.
        sparse_quadratic = joinpath(directory, "sparse-c3d20.inp")
        labels = Dict(i => 100 + 7i for i in 1:20)
        mode = :none
        open(sparse_quadratic, "w") do io
            for source_line in eachline(c3d20)
                if startswith(source_line, "*Node")
                    mode = :nodes
                    println(io, source_line)
                elseif startswith(source_line, "*Element")
                    mode = :elements
                    println(io, source_line)
                elseif startswith(source_line, "*")
                    mode = :none
                    println(io, source_line)
                elseif mode == :nodes && !isempty(strip(source_line))
                    fields = split(source_line, ',')
                    fields[1] = string(labels[parse(Int, strip(fields[1]))])
                    println(io, join(fields, ','))
                elseif mode == :elements && !isempty(strip(source_line))
                    fields = split(source_line, ',')
                    fields[1] = "41"
                    for i in 2:length(fields)
                        fields[i] = string(labels[parse(Int, strip(fields[i]))])
                    end
                    println(io, join(fields, ','))
                else
                    println(io, source_line)
                end
            end
        end
        sparse_quadratic_out = joinpath(directory, "sparse-c3d20.msh")
        MORFEFerrite.abaqus_to_gmsh(sparse_quadratic, sparse_quadratic_out)
        sparse_quadratic_data = _element_data(sparse_quadratic_out, 17)
        @test sparse_quadratic_data.node_tags == collect(1:20)
        @test sparse_quadratic_data.connectivity == data20.connectivity
        @test sparse_quadratic_data.element_tags == [41]
    end
end

@testset "COMSOL and Gmsh round trips" begin
    mktempdir() do directory
        comsol_dir = joinpath(_mesh_examples, "Comsol")

        t10_out = joinpath(directory, "t10.msh")
        MORFEFerrite.comsol_to_gmsh(joinpath(comsol_dir, "single_t10.mphtxt"), t10_out)
        t10 = _element_data(t10_out, 11)
        @test t10.connectivity == Int64[1, 2, 4, 3, 5, 7, 6, 8, 9, 10]
        t10_linear_out = joinpath(directory, "t10-linear.msh")
        MORFEFerrite.comsol_to_gmsh_linear(joinpath(comsol_dir, "single_t10.mphtxt"), t10_linear_out)
        @test _element_data(t10_linear_out, 4).connectivity == Int64[1, 2, 3, 4]

        h27_out = joinpath(directory, "h27.msh")
        h27_source = joinpath(comsol_dir, "single_h27.mphtxt")
        MORFEFerrite.comsol_to_gmsh(h27_source, h27_out)
        h27 = _element_data(h27_out, 12)
        @test length(h27.connectivity) == 27
        h27_linear_out = joinpath(directory, "h27-linear.msh")
        MORFEFerrite.comsol_to_gmsh_linear(h27_source, h27_linear_out)
        @test length(_element_data(h27_linear_out, 5).connectivity) == 8

        roundtrip_comsol = joinpath(directory, "roundtrip.mphtxt")
        roundtrip_gmsh = joinpath(directory, "roundtrip.msh")
        MORFEFerrite.gmsh_to_comsol(h27_out, roundtrip_comsol)
        MORFEFerrite.comsol_to_gmsh(roundtrip_comsol, roundtrip_gmsh)
        h27_roundtrip = _element_data(roundtrip_gmsh, 12)
        @test h27_roundtrip.node_tags == h27.node_tags
        @test h27_roundtrip.coordinates ≈ h27.coordinates atol = 1e-12 rtol = 1e-14
        @test h27_roundtrip.element_tags == h27.element_tags
        @test h27_roundtrip.connectivity == h27.connectivity

        t10_roundtrip_comsol = joinpath(directory, "t10-roundtrip.mphtxt")
        t10_roundtrip_gmsh = joinpath(directory, "t10-roundtrip.msh")
        MORFEFerrite.gmsh_to_comsol(t10_out, t10_roundtrip_comsol)
        MORFEFerrite.comsol_to_gmsh(t10_roundtrip_comsol, t10_roundtrip_gmsh)
        t10_roundtrip = _element_data(t10_roundtrip_gmsh, 11)
        @test t10_roundtrip.node_tags == t10.node_tags
        @test t10_roundtrip.coordinates ≈ t10.coordinates atol = 1e-12 rtol = 1e-14
        @test t10_roundtrip.element_tags == t10.element_tags
        @test t10_roundtrip.connectivity == t10.connectivity

        # The production arch contains every legacy surface/prism permutation.
        arch = normpath(joinpath(
            @__DIR__, "..", "..", "examples", "03_arch_comsol_wedge", "arch_2_force.mphtxt"))
        arch_out = joinpath(directory, "arch.msh")
        MORFEFerrite.comsol_to_gmsh(arch, arch_out)
        @test !isempty(_element_data(arch_out, 9).connectivity)   # T6
        @test !isempty(_element_data(arch_out, 10).connectivity)  # Q9
        @test !isempty(_element_data(arch_out, 13).connectivity)  # P18

        @test _with_gmsh(t10_out) do
            tags, _ = gmsh.model.mesh.getElementsByType(11)
            all(tags) do tag
                _, determinants, _ = gmsh.model.mesh.getJacobian(tag, [0.25, 0.25, 0.25])
                determinants[1] > 0
            end
        end
        @test _with_gmsh(h27_out) do
            tags, _ = gmsh.model.mesh.getElementsByType(12)
            all(tags) do tag
                _, determinants, _ = gmsh.model.mesh.getJacobian(tag, [0.0, 0.0, 0.0])
                determinants[1] > 0
            end
        end
        @test _with_gmsh(arch_out) do
            tags, _ = gmsh.model.mesh.getElementsByType(13)
            all(tags) do tag
                _, determinants, _ = gmsh.model.mesh.getJacobian(tag, [0.2, 0.2, 0.0])
                determinants[1] > 0
            end
        end

        arch_linear_out = joinpath(directory, "arch-linear.msh")
        MORFEFerrite.comsol_to_gmsh_linear(arch, arch_linear_out)
        @test !isempty(_element_data(arch_linear_out, 2).connectivity)
        @test !isempty(_element_data(arch_linear_out, 3).connectivity)
        @test !isempty(_element_data(arch_linear_out, 6).connectivity)

        # One mixed-mesh round trip covers every supported permutation and
        # verifies coordinates, element types/connectivity, and entity IDs.
        arch_roundtrip_comsol = joinpath(directory, "arch-roundtrip.mphtxt")
        arch_roundtrip_gmsh = joinpath(directory, "arch-roundtrip.msh")
        MORFEFerrite.gmsh_to_comsol(arch_out, arch_roundtrip_comsol)
        MORFEFerrite.comsol_to_gmsh(arch_roundtrip_comsol, arch_roundtrip_gmsh)
        for element_type in (9, 10, 11, 12, 13)
            before = _element_data(arch_out, element_type)
            isempty(before.element_tags) && continue
            after = _element_data(arch_roundtrip_gmsh, element_type)
            @test after.node_tags == before.node_tags
            @test after.coordinates ≈ before.coordinates atol = 1e-12 rtol = 1e-14
            @test after.element_tags == before.element_tags
            @test after.connectivity == before.connectivity
        end

        grid, constrained = load_comsol_grid(arch, Set([1]); scale = 1e-3)
        @test Ferrite.getncells(grid) > 0
        @test !isempty(Ferrite.getnodes(grid))
        @test constrained isa Set{Int}
    end
end
