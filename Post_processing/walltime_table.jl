# Rows of the appendix table tab:sim_times: total simulation time per resolution.
# Single-GPU times are summed from its/walltime.jld2 of the runs in Refinements/mesh/.
#
#   julia --project=. Post_processing/walltime_table.jl

using JLD2

function fmt(s)
    s < 90   && return "$(round(s, digits=1))~s"
    s < 5400 && return "$(round(s/60, digits=1))~min"
    return "$(round(s/3600, digits=1))~h"
end

function main()
    rows = [(N, 1, sum(load(joinpath(@__DIR__, "..", "Refinements", "mesh", "$N", "its", "walltime.jld2"), "t_stokes")))
            for N in [64, 128, 256, 512, 640, 1024, 1280, 2560, 5120, 6400]]
    # these runs have no its/walltime.jld2; their reported wall times are used
    append!(rows, [(10240, 1, 60.3 * 3600), (20474, 16, 18.1 * 3600), (40946, 64, 20.1 * 3600)])
    for (N, g, t) in rows
        println("    $N & $g & $(fmt(t)) \\\\")
    end
end

main()
