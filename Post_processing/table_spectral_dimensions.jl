# table_spectral_dimensions.jl — rows of the appendix table tab:spectral_dimensions.
# α, H and D_f per resolution over its own window (as convergence_figures_UHR.jl), plus H_m:
# H of the same spectrum fitted over the window of the m² model.
# Fit rules and helpers are copied from convergence_figures_UHR.jl.
#
#   julia --project=. Post_processing/table_spectral_dimensions.jl

using JLD2, Statistics, Printf

# Least-squares P(k) ∝ k^slope in log-log over krange = (k_lo, k_hi); nothing if < 2 points.
function fit_powerlaw(kc, Pk, krange)
    m = (kc .>= krange[1]) .& (kc .<= krange[2]) .& (Pk .> 0)
    count(m) < 2 && return nothing
    x = log10.(kc[m]); y = log10.(Pk[m])
    x_bar = mean(x); y_bar = mean(y)
    slope = sum((x .- x_bar) .* (y .- y_bar)) / sum((x .- x_bar) .^ 2)
    return (slope, y_bar - slope * x_bar)
end

# β = -slope, H = (β - d)/2, D_f = d + 1 - H for a d-D field.
function spectral_dimension(slope; d = 2)
    β = -slope; H = (β - d) / 2
    return (β, H, d + 1 - H)
end

# Centered sliding local log-log slope (window w; NaN where < 3 points).
function local_slopes(logk, logP; w)
    n = length(logk); s = fill(NaN, n); h = w ÷ 2
    for i in 1:n
        lo = max(1, i-h); hi = min(n, i+h)
        hi - lo + 1 < 3 && continue
        x = @view logk[lo:hi]; y = @view logP[lo:hi]
        x_bar = mean(x); y_bar = mean(y)
        s[i] = sum((x .- x_bar) .* (y .- y_bar)) / sum((x .- x_bar) .^ 2)
    end
    return s
end

# Fit window: mature (flatten=false) up to nyq_frac·k_nyq; initiation (flatten=true) up to the
# first k where the local slope rises above flat_slope.
function descent_window(r; flatten::Bool, k_lo, nyq_frac, slope_win, flat_slope)
    k_cap = nyq_frac * r.k_nyq
    flatten || return (k_lo, k_cap)
    m = (r.k_centers .>= k_lo) .& (r.k_centers .<= k_cap) .& (r.Pk_rad .> 0)
    count(m) < slope_win && return (k_lo, k_cap)
    kk = r.k_centers[m]
    s  = local_slopes(log10.(kk), log10.(r.Pk_rad[m]); w = slope_win)
    iflat = findfirst(i -> i > 2 && !isnan(s[i]) && s[i] > flat_slope, eachindex(s))
    return (k_lo, iflat === nothing ? k_cap : kk[iflat])
end

function main(; spectra_jld2 = joinpath(@__DIR__, "data", "UHR_analysis_results_360bins.jld2"),
                spec_res     = [1024, 2560, 5120, 10240, 40946],
                ref_res      = [1024, 2560, 5120, 10240],   # windows of the H_m columns
                stage_init      = :initiation,
                stage_mature    = :mature,
                nyq_frac     = 0.8,
                k_lo         = 200.0,
                slope_win    = 7,
                flat_slope   = -1.5)
    spectra = load(spectra_jld2)["results"]

    for (stage, name) in ((stage_init, "Initiation"), (stage_mature, "Mature"))
        win = Dict(n => descent_window(spectra[(n, stage)]; flatten = (stage == stage_init),
                                       k_lo, nyq_frac, slope_win, flat_slope) for n in spec_res)
        for n in spec_res
            r = spectra[(n, stage)]
            α, H, Df = spectral_dimension(fit_powerlaw(r.k_centers, r.Pk_rad, win[n])[1])
            Hm = [m < n ? @sprintf("%.3f", spectral_dimension(fit_powerlaw(r.k_centers, r.Pk_rad, win[m])[1])[2]) : "--   " for m in ref_res]
            @printf "%s & %-5d & %-6.0f & %.2f & %.3f & %.3f & %.3f & %s \\\\\n" name n win[n][2] log10(win[n][2] / k_lo) α H Df join(Hm, " & ")
        end
        println("\\hline")
    end
end

main()
