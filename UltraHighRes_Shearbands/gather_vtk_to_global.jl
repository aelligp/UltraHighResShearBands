# gather_vtk_to_global.jl — assembles the per-rank VTK output of one step of a multi-GPU run into
# one global field of the plastic strain rate ε̇ᵖˡ_II, which the post-processing reads:
#   <dir>/gathered/global_fields_timestep_<step>.jld2   (key plastic_strain_rate_II)
# Run it with the same number of MPI ranks as the model run, so that the rank layout is the same.
#
#   srun -n 64 julia --project=. UltraHighRes_Shearbands/gather_vtk_to_global.jl dir=Resolution_Study/sharp_40946 step=20
#
# keys: dir (run folder, required)  step (required)

using JustRelax, ReadVTK, JLD2

"key=value command-line arguments as a Dict"
parse_args(args) = Dict(String(kv[1]) => String(kv[2]) for kv in split.(args, "="; limit = 2))

vti_file(dir, step, rank) = joinpath(dir, "vtk", "timestep_$(step)", "vtk_" * lpad("$(step)_$(rank)", 6, "0") * ".vti")

"local grid size (points per direction, including the halo) from the VTI extent"
function local_size(file)
    e = ReadVTK.get_whole_extent(VTKFile(file))
    return e[2] - e[1] + 1, e[4] - e[3] + 1
end

read_field(file, name, nx, ny) = reshape(Float32.(get_data(get_point_data(VTKFile(file))[name])), nx, ny)

function main()
    a    = parse_args(ARGS)
    dir  = a["dir"]
    step = parse(Int, a["step"])

    nx, ny = local_size(vti_file(dir, step, 0))   # all ranks have the same local size
    igg = IGG(init_global_grid(nx, ny, 1; init_MPI = true, select_device = false)...)

    # strip the one-cell halo of every rank and gather the inner arrays on rank 0
    field  = read_field(vti_file(dir, step, igg.me), "plastic_strain_rate_II", nx, ny)[2:end-1, 2:end-1]
    global_field = igg.me == 0 ? zeros(Float32, (nx - 2) * igg.dims[1], (ny - 2) * igg.dims[2]) : nothing
    gather!(field, global_field)

    if igg.me == 0
        file = joinpath(dir, "gathered", "global_fields_timestep_$(step).jld2")
        mkpath(dirname(file))
        jldsave(file; plastic_strain_rate_II = global_field)
        println("saved $(size(global_field)) → $file")
    end
    finalize_global_grid()
end

main()
