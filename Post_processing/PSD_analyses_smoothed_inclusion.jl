# PSD_analyses_smoothed_inclusion.jl — radial PSD and P(θ) of the smoothed-inclusion runs,
# with the pipeline of PSD_analyses.jl. Writes data/smoothed_inclusion_PSD_results.jld2,
# which convergence_sharp_vs_smoothed.jl reads.
#
# The smoothed runs replace the sharp (binary phase-ratio) inclusion edge with a tanh phase
# transition one cell wide, to test whether the fine-scale initiation structure is an artefact
# of the sharp edge on the Cartesian grid.
#
#   julia --project=. Post_processing/PSD_analyses_smoothed_inclusion.jl   (GPU)

include(joinpath(@__DIR__, "PSD_analyses.jl"))   # PSDConfig, run_analysis, stage_files

smoothed_dir(n) = normpath(joinpath(@__DIR__, "..", "Resolution_Study", "smoothed_$n"))

function main(; resolutions = [64, 128, 256, 512, 640, 1024, 1280, 2560, 5120], stages = stage_times(),
                outfile = joinpath(@__DIR__, "data", "smoothed_inclusion_PSD_results.jld2"))
    files = Dict{Any,String}()
    for n in resolutions, (stage, f) in stage_files(smoothed_dir(n), stages)
        files[(n, stage)] = f
    end
    run_analysis(files, PSDConfig(); outfile = outfile)
    println("done → $outfile")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
