# run_it_multiple.jl — SLURM job scripts for the model runs of the paper, for either solver.
# Set the solver ("APT" or "DYREL") in main(); every run uses it.
#
#   julia run_it_multiple.jl            # writes slurm_jobs/<name>.sh
#   julia run_it_multiple.jl submit     # also submits them with sbatch
#
# Edit slurm_header() for your cluster (partition, account, modules). The 20474² and 40946² runs
# use 16 and 64 GPUs; gather steps 9 and 20 of their per-rank output with
# UltraHighRes_Shearbands/gather_vtk_to_global.jl before the post-processing.
#
# keep=last deletes each checkpoint once the next one is written, so only the final checkpoint
# remains; VTK output and its/ are kept for every step. The resolution study keeps all checkpoints
# (keep=all, the default) because the PSD analysis reads them at t = 1.5 and t = 3.33.

using Printf

solver_file(solver) = solver == "APT" ? "UltraHighRes_Shearbands/ShearBand2D_FFT_UHR.jl" :
                                        "UltraHighRes_Shearbands/ShearBand2D_FFT_UHR_DYREL.jl"

"(job name, MPI ranks, key=value arguments) of every run"
function paper_runs(solver)
    runs = Tuple{String,Int,String}[]
    # resolution study, sharp inclusion
    for n in [64, 128, 256, 512, 640, 1024, 1280, 2560, 5120, 6400, 10240]
        push!(runs, ("$(solver)_sharp_$n", 1, "n=$n out=Resolution_Study/sharp_$n"))
    end
    # multi-GPU runs: n is the local size per rank; global n = √ranks·(n - 2) + 2
    push!(runs, ("$(solver)_sharp_20474", 16, "n=5120 out=Resolution_Study/sharp_20474"))
    push!(runs, ("$(solver)_sharp_40946", 64, "n=5120 out=Resolution_Study/sharp_40946"))
    # resolution study, smoothed inclusion
    for n in [64, 128, 256, 512, 640, 1024, 1280, 2560, 5120]
        push!(runs, ("$(solver)_smoothed_$n", 1, "n=$n inclusion=smoothed out=Resolution_Study/smoothed_$n"))
    end
    # mesh convergence of the stress and wall times
    for n in [64, 128, 256, 512, 640, 1024, 1280, 2560, 5120, 6400, 10240]
        push!(runs, ("$(solver)_mesh_$n", 1, "n=$n keep=last out=Refinements/mesh/$n"))
    end
    # time refinement at 2560²
    for nt in [20, 40, 50, 60, 80, 100, 200]
        push!(runs, ("$(solver)_time_$nt", 1, "n=2560 nt=$nt keep=last out=Refinements/time/nt$nt"))
    end
    # regularisation viscosity × timestep at 2560²
    for η_reg in [8e-3, 8e-2, 8e-4, 0.2e-3, 0.96e-3, 2.4e-3, 1.96e-3], nt in [20, 40, 60, 80, 100]
        push!(runs, ("$(solver)_eta_$(η_reg)_$nt", 1, "n=2560 nt=$nt eta=$η_reg keep=last out=Reg_Viscosity_Systematics/eta$(η_reg)_nt$nt"))
    end
    # nonlinear-solver tolerance at 2560²
    for tol in [1e-2, 2e-2, 5e-2, 1e-3, 2e-3, 5e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        push!(runs, ("$(solver)_tol_$tol", 1, "n=2560 tol=$tol keep=last out=Tolerance_Study/tol$tol"))
    end
    return runs
end

function slurm_header(name, ranks)
    return """
    #!/bin/bash -l
    #SBATCH --job-name=$name
    #SBATCH --output=slurm_jobs/$name.o
    #SBATCH --error=slurm_jobs/$name.e
    #SBATCH --time=24:00:00
    #SBATCH --ntasks=$ranks
    #SBATCH --gpus-per-task=1
    #SBATCH --cpus-per-task=8
    """
end

function main()
    solver = "DYREL"   # "APT" or "DYREL"
    mkpath("slurm_jobs")   # existing job files of the same name are overwritten
    for (name, ranks, args) in paper_runs(solver)
        file = joinpath("slurm_jobs", "$name.sh")
        write(file, slurm_header(name, ranks) * "\nsrun julia --project=. $(solver_file(solver)) $args\n")
        "submit" in ARGS && run(`sbatch $file`)
        @printf "%-28s %s\n" name file
    end
end

main()
