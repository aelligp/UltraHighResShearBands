# Appendix comparison figure: radial PSD of the sharp (left) and smoothed (right)
# inclusion runs at the initiation (top) and mature (bottom) stages. Panels, fit
# windows and the finest-run fit line are as in convergence_figures_UHR.jl (a,b).
#
#   julia --project=. Post_processing/convergence_sharp_vs_smoothed.jl

using JLD2, Statistics, Printf, CairoMakie, MathTeXEngine
CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# --- helpers (as in convergence_figures_UHR.jl) ---
function fit_powerlaw(kc, Pk, krange)
    m = (kc .>= krange[1]) .& (kc .<= krange[2]) .& (Pk .> 0)
    count(m) < 2 && return nothing
    x = log10.(kc[m]); y = log10.(Pk[m]); x̄ = mean(x); ȳ = mean(y)
    s = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    return (s, ȳ - s * x̄)
end

function local_slopes(logk, logP; w)
    n = length(logk); s = fill(NaN, n); h = w ÷ 2
    for i in 1:n
        lo = max(1, i-h); hi = min(n, i+h)
        hi - lo + 1 < 3 && continue
        x = @view logk[lo:hi]; y = @view logP[lo:hi]
        s[i] = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
    end
    return s
end

function descent_window(r; flatten::Bool, k_lo = 200.0, nyq_frac = 0.8, slope_win = 7, flat_slope = -1.5)
    k_cap = nyq_frac * r.k_nyq
    flatten || return (k_lo, k_cap)
    m = (r.k_centers .>= k_lo) .& (r.k_centers .<= k_cap) .& (r.Pk_rad .> 0)
    count(m) < slope_win && return (k_lo, k_cap)
    kk = r.k_centers[m]
    s  = local_slopes(log10.(kk), log10.(r.Pk_rad[m]); w = slope_win)
    iflat = findfirst(i -> i > 2 && !isnan(s[i]) && s[i] > flat_slope, eachindex(s))
    return (k_lo, iflat === nothing ? k_cap : kk[iflat])
end

function spectra_panel!(gpos, spectra, res, stage, title, res_cols, flatten; nyq_frac = 0.8)
    ax = Axis(gpos, xscale = log10, yscale = log10, title = title,
              xlabel = L"$|\mathbf{k}|$ [rad/L]", ylabel = L"$P(|\mathbf{k}|)$")
    for n in res
        r = spectra[(n, stage)]
        m = (r.k_centers .<= nyq_frac * r.k_nyq) .& (r.Pk_rad .> 0)
        lines!(ax, r.k_centers[m], r.Pk_rad[m], color = res_cols[n], label = "n = $n", linewidth = 1.5)
    end
    # reference fit on the finest run
    r = spectra[(res[end], stage)]
    krange = descent_window(r; flatten, nyq_frac)
    s, b = fit_powerlaw(r.k_centers, r.Pk_rad, krange)
    xl = range(krange..., length = 100)
    lines!(ax, xl, 10.0 .^ (b .+ s .* log10.(xl)); color = :gray, linestyle = :dash, linewidth = 2.5,
           label = (al = @sprintf("%.2f", -s); L"$$fit, $\alpha=%$al$"))
    H  = (-s - 2) / 2; Df = 3 - H   # 2-D field
    @printf "%-18s %-10s n=%-6d k_hi=%6.0f  α=%.3f  H=%.3f  D_f=%.3f\n" title string(stage) res[end] krange[2] -s H Df
    al, Hs, Ds = @sprintf("%.2f", -s), @sprintf("%.2f", H), @sprintf("%.2f", Df)
    text!(ax, 0.97, 0.96; space = :relative, align = (:right, :top), fontsize = 19,
          text = L"$P(|\mathbf{k}|)\propto|\mathbf{k}|^{-%$al}$, $H=%$Hs$, $D_\mathrm{f}=%$Ds$")
    ylims!(ax, 1e-15, 5e0)
    return ax
end

function main()
    out_dir = mkpath(joinpath(@__DIR__, "Figures"))
    stage_init, stage_mature = :initiation, :mature

    # resolutions per inclusion
    variants = [
        (name = "Sharp inclusion",    file = joinpath(@__DIR__, "data", "UHR_analysis_results_360bins.jld2"),
         res = [1024, 2560, 5120, 10240, 40946]),
        (name = "Smoothed inclusion", file = joinpath(@__DIR__, "data", "smoothed_inclusion_PSD_results.jld2"),
         res = [1024, 2560, 5120]),
    ]
    all_res  = sort(union((v.res for v in variants)...))
    # same colour per n in all panels; roma without its pale centre so every curve stands out on white
    res_cols = Dict(n => get(cgrad(:roma), x) for (n, x) in zip(all_res, [0.0, 0.2, 0.72, 0.86, 1.0]))

    # --- figure: rows = stage, columns = sharp | smoothed ---
    fig = Figure(size = (1200, 950), fontsize = 20)
    axs = Axis[]
    for (j, v) in enumerate(variants)
        spectra = load(v.file)["results"]
        for (i, stage) in enumerate((stage_init, stage_mature))
            ax = spectra_panel!(fig[i, j], spectra, v.res, stage, L"$$%$(v.name), %$(string(stage)) stage", res_cols, stage == stage_init)
            i == 2 && axislegend(ax, position = :lb, nbanks = 2, labelsize = 16)
            push!(axs, ax)
        end
    end
    linkxaxes!(axs...)   # same k-range in all panels for a direct comparison

    # panel labels (grey box), as in convergence_figures_UHR.jl
    for i in 1:2, j in 1:2
        inset = Axis(fig[i, j], width = Relative(0.07), height = Relative(0.10), halign = :left, valign = :top,
                     backgroundcolor = :gray95, spinewidth = 1.0)
        hidedecorations!(inset); translate!(inset.blockscene, 0, 0, 1000)
        text!(inset, 0.5, 0.5; text = string(Char('a' + 2(i - 1) + (j - 1))), align = (:center, :center), font = :bold, fontsize = 18)
    end

    for ext in ("png", "pdf")
        out = joinpath(out_dir, "convergence_sharp_vs_smoothed.$ext")
        save(out, fig); println("Saved → $out")
    end
end

main()
