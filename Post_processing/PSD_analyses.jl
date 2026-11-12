# PSD_analyses.jl — radial PSD P(k) and high-k orientation power P(θ) of ε̇ᵖˡ_II for the
# resolution study. Writes data/UHR_analysis_results_<aniso_nbins>bins.jld2, which
# convergence_figures_UHR.jl, convergence_sharp_vs_smoothed.jl and table_spectral_dimensions.jl read.
# Also included by band_width.jl and PSD_analyses_smoothed_inclusion.jl for its functions.
#
# Results are keyed (run, stage), e.g. (2560, :mature). Each stage is a physical time; per run the
# saved step closest to it is used, so any nt works (also the time-refinement runs).
# Gathered multi-GPU fields carry no time, so their steps are given explicitly.
#
#   julia --project=. Post_processing/PSD_analyses.jl   (GPU; the 40946² field needs ~13 GB per copy)

using FFTW, CUDA, LinearAlgebra, Statistics, Printf, JLD2

# --- config ---
Base.@kwdef struct PSDConfig
    use_gpu::Bool        = true
    nbins::Int           = 200      # radial spectrum bins
    aniso_k1::Float64    = 2000.0   # P(θ) annulus lower bound
    aniso_cap::Float64   = 0.8      # annulus upper bound = aniso_cap·k_nyq
    aniso_nbins::Int     = 360
end

# --- field files ---
stage_times() = (initiation = 1.5, mature = 3.33)
resolution_dir(n) = normpath(joinpath(@__DIR__, "..", "Resolution_Study", "sharp_$n"))
checkpoint_file(dir, step) = joinpath(dir, "checkpoint", "checkpoint_$(step)", "checkpoint_rank_0", "checkpoint0000.jld2")

"saved step of a run whose checkpoint time is closest to t"
function nearest_step(dir, t)
    steps = [parse(Int, d[12:end]) for d in readdir(joinpath(dir, "checkpoint")) if startswith(d, "checkpoint_")]
    times = [load(checkpoint_file(dir, s), "time") for s in steps]
    i = argmin(abs.(times .- t))
    @printf "   %s: t = %.3f → step %d (t = %.4f)\n" basename(dir) t steps[i] times[i]
    return steps[i]
end

"checkpoint file per stage of a run"
stage_files(dir, stages) = Dict(stage => checkpoint_file(dir, nearest_step(dir, t)) for (stage, t) in pairs(stages))

"|ε̇ᵖˡ_II| from a checkpoint or a gathered global field"
function load_field(file)::Matrix{Float64}
    d = load(file)
    return abs.(haskey(d, "stokes") ? d["stokes"].ε_pl.II : d["plastic_strain_rate_II"])
end

"gathered global field of a multi-GPU run (UltraHighRes_Shearbands/gather_vtk_to_global.jl)"
gathered_file(n, step) = joinpath(resolution_dir(n), "gathered", "global_fields_timestep_$(step).jld2")

# --- type-stable kernels ---
@inline function hann2d(i::Int, j::Int, a::Float64, b::Float64)
    m = i - 1.0; n = j - 1.0
    θ  = atan(n - b, m - a)
    r0 = sqrt((a*b)^2 / (b^2*cos(θ)^2 + a^2*sin(θ)^2))
    r  = sqrt((m - a)^2 + (n - b)^2)
    return r <= r0 ? 0.5*(1 + cos(π*r/r0)) : 0.0
end

"remove the least-squares plane in place (one accumulation pass + one subtract pass)"
function detrend_plane!(f::AbstractMatrix{Float64})
    nx, ny = size(f); x = range(0.0,1.0; length=nx); y = range(0.0,1.0; length=ny)
    Sx = ny*sum(x); Sy = nx*sum(y); Sxx = ny*sum(abs2,x); Syy = nx*sum(abs2,y)
    Sxy = sum(x)*sum(y); S1 = Float64(nx*ny); Sfx = 0.0; Sfy = 0.0; Sf = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        v = f[i,j]; Sf += v; Sfx += v*x[i]; Sfy += v*y[j]
    end
    abc = [Sxx Sxy Sx; Sxy Syy Sy; Sx Sy S1] \ [Sfx, Sfy, Sf]   # 3×3, negligible
    a, b, c = abc[1], abc[2], abc[3]
    @inbounds for j in 1:ny, i in 1:nx; f[i,j] -= a*x[i] + b*y[j] + c; end
    return nothing
end

# explicit GC only pays off for huge fields (free ~13 GB between stages at 40946²);
# skip it for the smaller systematics where it would just add full-collection cost
@inline _gc_big(npix::Int) = (npix > 50_000_000) && GC.gc()

frequency_wavenumber(n::Int)::Vector{Float64} = fftshift(fftfreq(n, Float64(n))) .* 2π

function radial_average(power::Matrix{Float64}, kx::Vector{Float64}, ky::Vector{Float64},
                        edges::Vector{Float64}, nbins::Int)
    Pk = zeros(nbins); cnt = zeros(Int, nbins)
    @inbounds for j in eachindex(ky)
        ky2 = ky[j]^2
        for i in eachindex(kx)
            idx = searchsortedlast(edges, sqrt(kx[i]^2 + ky2))
            if 1 < idx <= nbins; Pk[idx] += power[i,j]; cnt[idx] += 1; end
        end
    end
    @inbounds for b in 1:nbins; Pk[b] /= max(cnt[b], 1); end
    return Pk
end

"angular power in [k1,k2], folded to band orientation = Fourier+90°, sorted"
function angular_power(power::Matrix{Float64}, kx::Vector{Float64}, ky::Vector{Float64},
                       k1::Float64, k2::Float64, nb::Int)
    Pθ = zeros(nb); cnt = zeros(Int, nb)
    @inbounds for j in eachindex(ky)
        ky2 = ky[j]^2
        for i in eachindex(kx)
            kk = sqrt(kx[i]^2 + ky2); (kk < k1 || kk > k2) && continue
            θ = mod(atan(ky[j], kx[i]), π)
            b = min(nb, unsafe_trunc(Int, θ/π*nb) + 1); Pθ[b] += power[i,j]; cnt[b] += 1
        end
    end
    bandθ = Vector{Float64}(undef, nb); Pm = Vector{Float64}(undef, nb)
    @inbounds for b in 1:nb
        bandθ[b] = mod((b - 0.5)*(180/nb) + 90, 180); Pm[b] = Pθ[b]/max(cnt[b], 1)
    end
    o = sortperm(bandθ); return bandθ[o], Pm[o]
end

# --- PSD (GPU/CPU) ---
"radial PSD + 2D power (fftshifted, Float64) for anisotropy; raw `field` is untouched"
function compute_psd(field::Matrix{Float64}, cfg::PSDConfig)
    nx, ny = size(field); kx = frequency_wavenumber(nx); ky = frequency_wavenumber(ny)
    fw = copy(field); detrend_plane!(fw); σ = std(fw)
    a = (nx-1)/2.0; b = (ny-1)/2.0; sumW2 = 0.0           # window in one pass, no W array
    @inbounds for j in 1:ny, i in 1:nx
        w = hann2d(i, j, a, b); sumW2 += w*w; fw[i,j] = (fw[i,j]/σ)*w
    end
    C_w = (nx*ny)/sumW2
    power::Matrix{Float64} = if cfg.use_gpu
        d = CuArray(ComplexF32.(fw)); fw = Matrix{Float64}(undef,0,0); _gc_big(nx*ny)
        fft!(d); d .*= Float32(inv(nx*ny)); pc = Array(abs2.(d)); d = nothing; CUDA.reclaim()
        fftshift(Float64.(pc)) .* C_w
    else
        abs2.(fftshift(fft(fw)) .* inv(nx*ny)) .* C_w
    end
    _gc_big(nx*ny)
    parseval = sum(power); k_nyq = π*nx
    kmin = Inf; @inbounds for v in kx; (v > 0 && v < kmin) && (kmin = v); end
    kmax = sqrt(maximum(abs2, kx) + maximum(abs2, ky))
    edges = collect(10 .^ range(log10(kmin), log10(kmax); length=cfg.nbins+1))
    kc = [sqrt(edges[j]*edges[j+1]) for j in 1:cfg.nbins]
    Pk = radial_average(power, kx, ky, edges, cfg.nbins)
    return kc, Pk, power, kx, ky, parseval, k_nyq
end

# --- per-field analysis ---
function analyze_field(field::Matrix{Float64}, cfg::PSDConfig)
    nx = size(field, 1)
    # empty/constant field (e.g. ts1, no yielding yet): no measure, no variance → skip
    if !(sum(abs, field) > 0) || std(field) == 0
        return (k_centers=Float64[], Pk_rad=Float64[], k_nyq=π*nx, bandθ=Float64[], Pθ=Float64[])
    end
    kc, Pk, power, kx, ky, parseval, k_nyq = compute_psd(field, cfg)
    bandθ, Pθ = angular_power(power, kx, ky, cfg.aniso_k1, cfg.aniso_cap*k_nyq, cfg.aniso_nbins)
    power = Matrix{Float64}(undef,0,0); _gc_big(length(field))
    @printf "   Parseval=%.3f\n" parseval; flush(stdout)   # Parseval sanity check (≈var)
    return (k_centers=kc, Pk_rad=Pk, k_nyq=k_nyq, bandθ=bandθ, Pθ=Pθ)
end

"drive the pipeline over `files` (key => field file). Saves jld2 if `outfile`."
function run_analysis(files, cfg::PSDConfig; outfile::Union{Nothing,String}=nothing)
    results = Dict{Any,Any}()
    for (key, file) in files
        @printf "=== %s ===\n" string(key); flush(stdout)
        field = load_field(file)::Matrix{Float64}
        results[key] = analyze_field(field, cfg)
        _gc_big(length(field)); field = Matrix{Float64}(undef,0,0)
    end
    outfile !== nothing && (mkpath(dirname(outfile)); jldsave(outfile; results=results, cfg=cfg))
    return results
end

# --- main (resolution study) ---
function main(; resolutions = [256, 512, 1024, 2560, 5120, 10240],
                gathered = Dict(40946 => (initiation = 9, mature = 20)),
                stages = stage_times(), cfg = PSDConfig(),
                outfile = joinpath(@__DIR__, "data", "UHR_analysis_results_$(cfg.aniso_nbins)bins.jld2"))
    files = Dict{Any,String}()
    for n in resolutions, (stage, f) in stage_files(resolution_dir(n), stages)
        files[(n, stage)] = f
    end
    for (n, steps) in gathered, stage in keys(stages)
        files[(n, stage)] = gathered_file(n, steps[stage])
    end
    results = run_analysis(files, cfg; outfile=outfile)
    println("\ndone → $outfile")
    return results
end

abspath(PROGRAM_FILE) == (@__FILE__) && main()
