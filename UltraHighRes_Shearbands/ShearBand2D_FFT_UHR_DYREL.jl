# 2D visco-elasto-viscoplastic shear-band model (weak circular inclusion, pure shear; Duretz et al.)
# solved with JustRelax.jl. This file uses the DYREL solver; ShearBand2D_FFT_UHR.jl is the same model
# with the accelerated pseudo-transient (APT) solver. The two files differ only in the solver lines.
#
#   julia --project=. UltraHighRes_Shearbands/ShearBand2D_FFT_UHR_DYREL.jl n=2560 [key=value ...]
#
# keys (default):  n (required, cells per direction and rank)  nt (20)  tmax (3.33)  eta (0.008, η_reg)
#                  inclusion (sharp | smoothed)  tol (1e-6, DYREL ϵ)  keep (all | last checkpoints)
#                  out (FFT_ShearBands2D_DYREL_<inclusion>_<nt>_<eta>_<n>_<n>)
# Multi-GPU: run with MPI (e.g. srun -n 16); n is then the local size per rank.

const isCUDA = true

@static if isCUDA
    using CUDA
end
using GeoParams
using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO, CairoMakie

const backend_JR = @static if isCUDA
    JustRelax.CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC
import JustPIC.GridGeometryUtils as GGU

const backend_JP = @static if isCUDA
    CUDA.CUDABackend # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
else
    JustPIC.CPU # Options: JustPIC.CPU, CUDA.CUDABackend, AMDGPU.ROCBackend
end

using JLD2
using Statistics

# HELPER FUNCTIONS ---------------------------------------------------------------

# analytical visco-elastic loading curve (reference line in the per-step figure)
solution(ε, t, G, η) = 2 * ε * η * (1 - exp(-G * t / η))

# Initialize phases. smooth_width = 0 gives the sharp inclusion (cell inside/outside the circle).
# Otherwise the inclusion fraction follows f(d) = (1 - tanh(2d/w))/2 in the signed distance d
# to the circle: 0.5 on the boundary, 0.98...0.02 across d = -w...w. `smooth_width` (w) is a
# length; the default of one cell keeps the boundary one cell wide at every resolution.
# The two fractions sum to 1 everywhere.
function init_phases!(phase_ratios, xci, xvi, circle; smooth_width = xci[1][2] - xci[1][1])
    ni = size(phase_ratios.center)

    @parallel_indices (i, j) function _init_phases!(phases, xc, yc, circle, w)
        d = GGU.distance(GGU.Point(xc[i], yc[j]), circle.center) - circle.radius
        f = iszero(w) ? (GGU.inside(GGU.Point(xc[i], yc[j]), circle) ? 1.0 : 0.0) :
                        0.5 * (1 - tanh(2 * d / w))  # 1 well inside the circle, 0 well outside
        @index phases[1, i, j] = 1 - f
        @index phases[2, i, j] = f
        return nothing
    end

    @parallel (@idx ni) _init_phases!(phase_ratios.center, xci..., circle, smooth_width)
    @parallel (@idx ni .+ 1) _init_phases!(phase_ratios.vertex, xvi..., circle, smooth_width)
    return nothing
end

"key=value command-line arguments as a Dict"
parse_args(args) = Dict(String(kv[1]) => String(kv[2]) for kv in split.(args, "="; limit = 2))

# MODEL --------------------------------------------------------------------------
# inclusion = "sharp" or "smoothed" (one-cell tanh edge)
# keep = "last" keeps only the newest checkpoint (2.9 GB each at 2560²) instead of one per step
function shearband(igg; nx = 64, ny = 64, nt = 20, tmax = 3.33, η_reg = 8e-3, inclusion = "sharp",
                   tol = 1.0e-6, keep = "all", figdir = "model_figs")

    # Physical domain ------------------------------------
    ly = 1.0e0          # domain length in y
    lx = ly             # domain length in x
    ni = nx, ny         # number of cells
    li = lx, ly         # domain length in x- and y-
    di = @. li / (nx_g(), ny_g()) # grid step in x- and -y
    origin = 0.0, 0.0   # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells

    # Physical properties using GeoParams ----------------
    τ_y = 1.6           # yield stress c·cos(ϕ)
    ϕ = 30              # friction angle
    C = τ_y             # Cohesion
    η0 = 1.0            # viscosity
    G0 = 1.0            # elastic shear modulus
    Gi = G0 / (6.0 - 4.0)  # elastic shear modulus of the (weak) inclusion
    εbg = 1.0           # background strain-rate
    dt = tmax / nt
    el_bg = ConstantElasticity(; G = G0, Kb = 4)
    el_inc = ConstantElasticity(; G = Gi, Kb = 4)
    visc = LinearViscous(; η = η0)
    pl = DruckerPrager_regularised(;
        C = C / cosd(ϕ),
        ϕ = ϕ,
        η_vp = η_reg,
        Ψ = 0,
    )

    rheology = (
        # background
        SetMaterialParams(;
            Phase = 1,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_bg, pl)),
            Elasticity = el_bg,
        ),
        # inclusion
        SetMaterialParams(;
            Phase = 2,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((visc, el_inc, pl)),
            Elasticity = el_inc,
        ),
    )

    # Initialize phase ratios -------------------------------
    phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    radius = 0.1
    circle = GGU.Circle((0.5, 0.5), radius)
    inclusion == "sharp" ? init_phases!(phase_ratios, xci, xvi, circle; smooth_width = 0.0) :
                           init_phases!(phase_ratios, xci, xvi, circle)

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)

    # Buoyancy forces
    ρg = @zeros(ni...), @zeros(ni...)
    args = (; T = @zeros(ni .+ 2...), P = stokes.P, dt = dt)

    # Rheology
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )

    # Boundary conditions
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
    )
    stokes.V.Vx .= PTArray(backend_JR)([ x * εbg for x in xvi[1], _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend_JR)([-y * εbg for _ in 1:(nx + 2), y in xvi[2]])
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
    take(figdir)
    vtkdir = joinpath(figdir, "vtk")
    take(vtkdir)
    checkpoint = joinpath(figdir, "checkpoint")
    take(checkpoint)
    its_dir = joinpath(figdir, "its")
    take(its_dir)

    # Time loop
    t, it = 0.0, 0
    t_stokes = zeros(nt+1)
    τII = Float64[]
    εII = Float64[]
    sol = Float64[]
    ttot = Float64[]
    var_tau_II = Float64[] # variance of the τII
    var_EII_pl = Float64[] # variance of the EII_pl
    var_epsII = Float64[] # variance of the εII
    var_eps_pl_II = Float64[] # variance of the ε_pl_II

    min_P = Float64[]
    max_P = Float64[]
    mean_P = Float64[]

    min_τII = Float64[]
    max_τII = Float64[]
    mean_τII = Float64[]

    min_EII_pl = Float64[]
    max_EII_pl = Float64[]
    mean_EII_pl = Float64[]

    min_εII = Float64[]
    max_εII = Float64[]
    mean_εII = Float64[]

    min_ε_pl_II = Float64[]
    max_ε_pl_II = Float64[]
    mean_ε_pl_II = Float64[]

    P_init = zeros(nx, ny)

    # SOLVER (the only part that differs from ShearBand2D_FFT_UHR.jl) ---------------
    dyrel = DYREL(backend_JR, stokes, rheology, phase_ratios, grid.di, dt; ϵ = tol)

    while it < nt

        # Stokes solver ----------------
        t_stokes[it+1] = @elapsed begin
            iters = solve_DYREL!(
                stokes,
                ρg,
                dyrel,
                flow_bcs,
                phase_ratios,
                rheology,
                args,
                grid,
                dt,
                igg;
                kwargs = (;
                    verbose_DR = false,
                    iterMax = 350.0e3,
                    total_iterMax = 350.0e3,
                    nout = 100,
                    rel_drop = 1e-2,
                    λ_relaxation_PH = 1,
                    λ_relaxation_DR = 1,
                    viscosity_relaxation = 1,
                    linear_viscosity = true,
                    viscosity_cutoff = (-Inf, Inf),
                )
            )
        end
        # END SOLVER ------------------------------------------------------------------

        if igg.me == 0
            println("Stokes solver time             ")
            println("   Total time:      $t_stokes s")
        end

        tensor_invariant!(stokes.ε)
        tensor_invariant!(stokes.ε_pl)

        # tracking arrays
        # Variance of the fields
        push!(var_tau_II, var(Array(stokes.τ.II)))
        push!(var_EII_pl, var(Array(stokes.EII_pl)))
        push!(var_epsII, var(Array(stokes.ε.II)))
        push!(var_eps_pl_II, var(Array(stokes.ε_pl.II)))

        # Min, max and mean of the fields
        push!(min_P, minimum(Array(stokes.P)))
        push!(max_P, maximum(Array(stokes.P)))
        push!(mean_P, mean(Array(stokes.P)))
        push!(min_τII, minimum(Array(stokes.τ.II)))
        push!(max_τII, maximum(Array(stokes.τ.II)))
        push!(mean_τII, mean(Array(stokes.τ.II)))
        push!(min_EII_pl, minimum(Array(stokes.EII_pl)))
        push!(max_EII_pl, maximum(Array(stokes.EII_pl)))
        push!(mean_EII_pl, mean(Array(stokes.EII_pl)))
        push!(min_εII, minimum(Array(stokes.ε.II)))
        push!(max_εII, maximum(Array(stokes.ε.II)))
        push!(mean_εII, mean(Array(stokes.ε.II)))
        push!(min_ε_pl_II, minimum(Array(stokes.ε_pl.II)))
        push!(max_ε_pl_II, maximum(Array(stokes.ε_pl.II)))
        push!(mean_ε_pl_II, mean(Array(stokes.ε_pl.II)))

        # loading curve: maximum of τxx and εxx (the post-processing reads these)
        push!(τII, maximum(stokes.τ.xx))
        push!(εII, maximum(stokes.ε.xx))

        it += 1
        t += dt

        if it == 1 && igg.me == 0
            P_init .= Array(stokes.P)
        end

        if igg.me == 0
            push!(sol, solution(εbg, t, G0, η0))
            push!(ttot, t)

            jldsave(joinpath(its_dir, "$(it).jld2"); iters)
            jldsave(joinpath(its_dir, "walltime.jld2"); t_stokes, t)
            jldsave(joinpath(its_dir, "τII_εII_loading.jld2"); τII, εII, t = ttot)   # t: time of each sample
            jldsave(joinpath(its_dir, "solution.jld2"); sol)
            ## Tracking arrays
            jldsave(joinpath(its_dir, "tracking_arrays.jld2"); var_tau_II, var_EII_pl, var_epsII, var_eps_pl_II, min_P, max_P, mean_P, min_τII, max_τII, mean_τII, min_EII_pl, max_EII_pl, mean_EII_pl, min_εII, max_εII, mean_εII, min_ε_pl_II, max_ε_pl_II, mean_ε_pl_II)
        end

        igg.me == 0 && println("it = $it; t = $t \n")

        # checkpoint and VTK output
        if igg.me == 0 metadata(pwd(), checkpoint, @__FILE__) end
        checkpoint_timestep = joinpath(checkpoint, "checkpoint_$(it)")
        take(checkpoint_timestep)
        checkpoint_rank = joinpath(checkpoint_timestep, "checkpoint_rank_$(igg.me)")
        checkpointing_jld2(checkpoint_rank, stokes, t, dt, igg; it = it, ttot = ttot, τII = τII)
        keep == "last" && it > 1 && rm(joinpath(checkpoint, "checkpoint_$(it - 1)"); recursive = true, force = true)
        data_c = (;
            strain_rate_II = Array(stokes.ε.II),
            plastic_strain_rate_II = Array(stokes.ε_pl.II),
            EII_pl = Array(stokes.EII_pl),
            pressure = Array(stokes.P),
            init_pressure = P_init,
            stress_II = Array(stokes.τ.II),
        )
        vtkdir_timestep = joinpath(vtkdir, "timestep_$(it)")
        take(vtkdir_timestep)
        save_vtk(
            joinpath(vtkdir_timestep, "vtk_" * lpad("$(it)_$(igg.me)", 6, "0")),
            xci,
            data_c,
            t = t,
            pvd = joinpath(vtkdir, "time_series")
        )

        # per-step overview figure
        th = 0:(pi / 50):(3 * pi)
        xunit = @. radius * cos(th) + 0.5
        yunit = @. radius * sin(th) + 0.5
        fig = Figure(size = (1600, 1600), title = "t = $t")
        ax1 = Axis(fig[1, 1], aspect = 1, title = L"\tau_{II}", titlesize = 35)
        ax2 = Axis(fig[2, 1], aspect = 1, title = L"E_{II}", titlesize = 35)
        ax3 = Axis(fig[1, 2], aspect = 1, title = L"\log_{10}(\varepsilon_{II})", titlesize = 35)
        ax4 = Axis(fig[2, 2], aspect = 1)
        ax5 = Axis(fig[3, 1], aspect = 1, title = L"Points at yield", titlesize = 35)
        ax6 = Axis(fig[3, 2], aspect = 1, title = L"Variance \ \varepsilon_{II}", titlesize = 35, xlabel = L"Time", ylabel = L"Variance \ \varepsilon_{II}")
        heatmap!(ax1, xci..., Array(stokes.τ.II), colormap = :batlow)
        heatmap!(ax2, xci..., Array(stokes.EII_pl), colormap = :batlow)
        heatmap!(ax3, xci..., Array(log10.(stokes.ε.II)), colormap = :batlow)
        lines!(ax2, xunit, yunit, color = :black, linewidth = 5)
        lines!(ax4, ttot, τII, color = :black)
        lines!(ax4, ttot, sol, color = :red)
        heatmap!(ax5, xci..., Array(stokes.P), colormap = :batlow)
        scatterlines!(ax6, ttot, var_eps_pl_II, color = :black)
        hidexdecorations!(ax1)
        hidexdecorations!(ax3)
        save(joinpath(figdir, "$(it).png"), fig)
    end

    return nothing
end

function main()
    a = parse_args(ARGS)
    n         = parse(Int, a["n"])
    nt        = parse(Int, get(a, "nt", "20"))
    tmax      = parse(Float64, get(a, "tmax", "3.33"))
    eta       = get(a, "eta", "0.008")
    inclusion = get(a, "inclusion", "sharp")
    inclusion in ("sharp", "smoothed") || error("inclusion must be sharp or smoothed, got $inclusion")
    tol       = parse(Float64, get(a, "tol", "1e-6"))
    keep      = get(a, "keep", "all")
    out       = get(a, "out", "FFT_ShearBands2D_DYREL_$(inclusion)_$(nt)_$(eta)_$(n)_$(n)")
    igg = IGG(init_global_grid(n, n, 1; init_MPI = true, select_device = false)...)
    @time shearband(igg; nx = n, ny = n, nt = nt, tmax = tmax, η_reg = parse(Float64, eta),
                    inclusion = inclusion, tol = tol, keep = keep, figdir = out)
end

main()
