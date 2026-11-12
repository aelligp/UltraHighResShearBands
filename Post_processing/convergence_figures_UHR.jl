# Convergence figure (radial PSD at initiation and mature stage, stress convergence upon mesh and
# time refinement, mature band width and profile) and band-orientation figure of the sharp-inclusion
# resolution study.
#
# D_f here is the self-affine roughness dimension of the strain-rate field, from the azimuthally
# averaged PSD slope. It is not the box-counting dimension of the shear-band pattern.
#
# Reads data/UHR_analysis_results_360bins.jld2 (PSD_analyses.jl), data/band_width_tmax.jld2
# (band_width.jl) and the stress curves of the runs in Refinements/mesh/ and Refinements/time/.
#
#   julia --project=. Post_processing/convergence_figures_UHR.jl

using JLD2, Statistics, Printf, CairoMakie, MathTeXEngine
CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# --- spectral fit helpers ---

# Least-squares P(k) ∝ k^slope in log-log, restricted to krange = (k_lo, k_hi).
# Returns nothing if fewer than two valid points fall in the window.
function fit_powerlaw(kc, Pk, krange)
    m = (kc .>= krange[1]) .& (kc .<= krange[2]) .& (Pk .> 0)
    count(m) < 2 && return nothing
    x = log10.(kc[m]); y = log10.(Pk[m])
    x_bar = mean(x); y_bar = mean(y)
    slope = sum((x .- x_bar) .* (y .- y_bar)) / sum((x .- x_bar) .^ 2)
    intercept = y_bar - slope * x_bar
    return (slope, intercept)
end

# Self-affine roughness from PSD slope. For an isotropic d-D field with
# P(k) ∝ k^{-β}: β = -slope, H = (β - d)/2, D_f = d + 1 - H.
# Meaningful only for d < β < d+2 (0 < H < 1, d < D_f < d+1).
function spectral_dimension(slope; d = 2)
    β  = -slope
    H  = (β - d) / 2
    Df = d + 1 - H
    return (β, H, Df)
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

# Per-curve fit window, lower bound k_lo (below it is the band-spacing plateau).
#   flatten=false (mature): entire descent up to nyq_frac·k_nyq.
#   flatten=true (initiation): up to the first k where the local slope rises above
#   flat_slope (onset of the high-k flattening). Falls back to the full descent.
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

# Draw a power-law reference line over its own k-range.
function draw_fit!(ax, kc, Pk, krange; color, label, width = 1.5)
    f = fit_powerlaw(kc, Pk, krange)
    f === nothing && return nothing
    slope, intercept = f
    xl = range(krange[1], krange[2], length=100)
    yl = 10.0 .^ (intercept .+ slope .* log10.(xl))
    lines!(ax, xl, yl; color, linestyle=:dash, linewidth=width,
        label = (al = @sprintf("%.2f", -slope); L"$$%$label, $\alpha=%$al$"))
    return slope
end

# --- stress-convergence helpers ---

# Curves sampled at the common times t = k/g (g = gcd of the lengths): exact samples, no interpolation.
function on_common_times(curves)
    g = gcd(collect(length.(values(curves))))
    return Dict(key => c[(1:g) .* (length(c) ÷ g)] for (key, c) in curves)
end

# Residual of the observed-order equation (h3^p - h2^p) / (h2^p - h1^p) = target.
order_residual(p, h1, h2, h3, target) = (h3^p - h2^p) / (h2^p - h1^p) - target

# Observed order from the three finest runs (h1 < h2 < h3): solve
# mean|u3-u2| / mean|u2-u1| = (h3^p - h2^p) / (h2^p - h1^p) for p by bisection; NaN if no root.
function observed_order(u1, u2, u3, h1, h2, h3)
    target = mean(abs.(u3 .- u2)) / mean(abs.(u2 .- u1))
    a, b = 0.05, 6.0
    order_residual(a, h1, h2, h3, target) * order_residual(b, h1, h2, h3, target) > 0 && return NaN
    for _ in 1:100
        m = (a + b) / 2
        order_residual(a, h1, h2, h3, target) * order_residual(m, h1, h2, h3, target) <= 0 ? (b = m) : (a = m)
    end
    return (a + b) / 2
end

# Mean absolute error of every run against the Richardson extrapolation of the two finest runs in
# `use`, u_ext = u1 + (u1 - u2) / ((h2/h1)^p - 1), pointwise in time; h = 1/key.
# p is observed from the three finest runs in `use`.
# Returns all keys (ascending), their errors and the observed order.
function richardson_errors(curves, use)
    c  = on_common_times(curves)
    ks = sort(collect(use); rev = true)   # extrapolation runs, finest first
    h  = 1 ./ ks
    p  = observed_order(c[ks[1]], c[ks[2]], c[ks[3]], h[1], h[2], h[3])
    isfinite(p) || error("no observed Richardson order from the three finest runs")
    u_ext = c[ks[1]] .+ (c[ks[1]] .- c[ks[2]]) ./ ((h[2] / h[1])^p - 1)
    all_ks = sort(collect(keys(c)))
    return all_ks, [mean(abs.(c[k] .- u_ext)) for k in all_ks], p
end

# Least-squares fit u_k(t) = u_ext(t) + c(t) h_k^p for one p; returns (coefficients, residual).
function lsq_fit(p, h, U)
    A = hcat(ones(length(h)), h .^ p); X = A \ U
    return (X, sum(abs2, U .- A * X))
end

# Least-squares Richardson over all runs (Eça & Hoekstra 2014, J. Comput. Phys. 262, 104–130):
# u_k(t) = u_ext(t) + c(t) h_k^p at every common time, one p shared by all times, p by grid search.
# Returns all keys (ascending), their errors vs u_ext and the fitted order.
function lsq_richardson_errors(curves)
    c  = on_common_times(curves)
    ks = sort(collect(keys(c)))
    h  = 1 ./ ks
    U  = reduce(hcat, (c[k] for k in ks))'              # runs × times
    ps = 0.2:0.001:3.0
    p  = ps[argmin([lsq_fit(q, h, U)[2] for q in ps])]
    u_ext = lsq_fit(p, h, U)[1][1, :]
    return ks, [mean(abs.(c[k] .- u_ext)) for k in ks], p
end

# Stress-time curve from a run's its/ directory. The "τII" key holds maximum(τxx) per step.
load_stress_curve(path) = Float64.(load(path)["τII"])

# Fit a power law to (xs, ys) in log-log, drawn over its own x-range.
function fit_loglog!(ax, xs, ys; color, label)
    x = log10.(xs); y = log10.(ys)
    x_bar = mean(x); y_bar = mean(y)
    slope = sum((x .- x_bar) .* (y .- y_bar)) / sum((x .- x_bar) .^ 2)
    intercept = y_bar - slope * x_bar
    xl = range(minimum(xs), maximum(xs), length=100)
    yl = 10.0 .^ (intercept .+ slope .* log10.(xl))
    lines!(ax, xl, yl; color, linestyle=:dash, linewidth=2.5,
        label = (sl = @sprintf("%.2f", slope); L"$$%$label, slope $%$sl$"))
    return slope
end

# Half-max crossings of a profile around its peak (linear interpolation), as in band_width.jl.
function halfmax_crossings(prof, x)
    ip = argmax(prof); h = prof[ip] / 2
    l = ip; while l > 1 && prof[l] > h; l -= 1; end
    r = ip; while r < length(prof) && prof[r] > h; r += 1; end
    xl = x[l] + (h - prof[l]) / (prof[l+1] - prof[l]) * (x[l+1] - x[l])
    xr = x[r-1] + (h - prof[r-1]) / (prof[r] - prof[r-1]) * (x[r] - x[r-1])
    return xl, xr
end

# --- panels ---

# Grey panel-label box in the top-left corner of a grid cell.
function panel_label!(gpos, label)
    inset_ax = Axis(gpos, width = Relative(0.07), height = Relative(0.10),
        halign = :left, valign = :top, backgroundcolor = :gray95,
        spinewidth = 1.0, leftspinecolor = :black, topspinecolor = :black,
        rightspinecolor = :black, bottomspinecolor = :black)
    hidedecorations!(inset_ax)
    translate!(inset_ax.blockscene, 0, 0, 1000)
    text!(inset_ax, 0.5, 0.5; text = label, align = (:center, :center),
        font = :bold, fontsize = 18)
end

function spectra_panel!(gpos, spectra, spec_res, spec_cols, stage, title, flatten; nyq_frac = 0.8)
    ax = Axis(gpos, xscale=log10, yscale=log10,
        title=title, xlabel=L"$|\mathbf{k}|$ [rad/L]", ylabel=L"$P(|\mathbf{k}|)$")
    for (i, n) in enumerate(spec_res)
        r = spectra[(n, stage)]
        m = (r.k_centers .<= nyq_frac * r.k_nyq) .& (r.Pk_rad .> 0)
        lines!(ax, r.k_centers[m], r.Pk_rad[m], color=spec_cols[i], label="n = $n", linewidth=1.5)
    end
    # one reference line + annotation from the finest resolution
    rfit = spectra[(spec_res[end], stage)]
    slope = draw_fit!(ax, rfit.k_centers, rfit.Pk_rad, descent_window(rfit; flatten, nyq_frac);
                      color=:gray, label="fit", width = 2.5)
    _, H, Df = spectral_dimension(slope)
    al, Hs, Ds = @sprintf("%.2f", -slope), @sprintf("%.2f", H), @sprintf("%.2f", Df)
    text!(ax, 0.97, 0.96; space=:relative, align=(:right, :top),
        text=L"$P(|\mathbf{k}|)\propto|\mathbf{k}|^{-%$al}$, $H=%$Hs$, $D_\mathrm{f}=%$Ds$",
        fontsize=19)
    ylims!(ax, 1e-15, 5e0)
    return ax
end

# P_θ(θ_b) is symmetric about 90° → clip to [0,90] for a single band-orientation peak.
function orientation_panel!(gpos, spectra, spec_res, spec_cols, stage, title, roscoe, arthur, mohr)
    ax = Axis(gpos, title=title, ylabel=L"$P_\theta(\theta_b)/\langle P_\theta\rangle$",
        xticks=0:15:90)
    for (i, n) in enumerate(spec_res)
        r = spectra[(n, stage)]
        lines!(ax, r.bandθ, r.Pθ ./ mean(r.Pθ), color=spec_cols[i], label="n = $n", linewidth=1.5)
        println("n=$n, stage=$stage: band orientation = $(r.bandθ[argmax(r.Pθ)])°")
    end
    vlines!(ax, [roscoe], color=:purple,     linestyle=:dash, label="Roscoe $(roscoe)°")
    vlines!(ax, [arthur], color=:green,      linestyle=:dash, label="Arthur $(arthur)°")
    vlines!(ax, [mohr],   color=:darkorange, linestyle=:dash, label="Mohr–Coulomb $(mohr)°")
    xlims!(ax, 0, 90)
    return ax
end

function main()
    repo        = normpath(joinpath(@__DIR__, ".."))
    out_dir     = mkpath(joinpath(@__DIR__, "Figures"))
    spectra     = load(joinpath(@__DIR__, "data", "UHR_analysis_results_360bins.jld2"))["results"]
    bw          = load(joinpath(@__DIR__, "data", "band_width_tmax.jld2"))
    spec_res    = [1024, 2560, 5120, 10240, 40946]
    stage_init     = :initiation
    stage_mature   = :mature
    # mesh study: DYREL runs (the APT 10240² run is iteration-capped after yielding)
    resolutions = [64, 128, 256, 512, 640, 1024, 1280, 2560, 5120, 6400, 10240]
    rich_mesh   = [2560, 5120, 10240]   # Richardson extrapolation runs: three finest of the 2× ladder
    # time study at 2560², η_reg = 0.008
    time_nts    = [20, 40, 50, 60, 80, 100, 200]
    err_ticks   = ([0.003, 0.01, 0.03, 0.1], [L"0.003", L"0.01", L"0.03", L"0.1"])

    mesh_curves = Dict(n => load_stress_curve(joinpath(repo, "Refinements", "mesh", "$n", "its", "τII_εII_loading.jld2"))
                       for n in resolutions)
    time_curves = Dict(nt => load_stress_curve(joinpath(repo, "Refinements", "time", "nt$nt", "its", "τII_εII_loading.jld2"))
                       for nt in time_nts)
    spec_cols = cgrad(:roma, length(spec_res); categorical=true)

    # --- figure (3×2): (a,b) spectra+fit; (c) mesh and (d) time stress convergence;
    #                   (e) mature band width, (f) band profile ---
    fig = Figure(size = (1200, 1400), fontsize = 20)
    spectra_panel!(fig[1, 1], spectra, spec_res, spec_cols, stage_init, L"$$Initiation stage", true)
    ax_b = spectra_panel!(fig[1, 2], spectra, spec_res, spec_cols, stage_mature, L"$$Mature stage", false)
    axislegend(ax_b, position=:lb, nbanks=2, labelsize=16)

    # --- (c) convergence upon mesh refinement ---
    ax_c = Axis(fig[2, 1], xscale=log10, yscale=log10,
        title = L"$$Convergence upon mesh refinement",
        xlabel = L"$$Resolution $n^2$", ylabel = L"$$MAE of max$(\tau_{xx})$ [-]",
        yticks = err_ticks, yminorticksvisible = true, yminorticks = IntervalsBetween(9))
    mesh_x, mesh_y, p_mesh = richardson_errors(mesh_curves, rich_mesh)
    @printf "mesh Richardson: extrapolation runs %s, observed order %.3f\n" string(rich_mesh) p_mesh
    for (n, e) in zip(mesh_x, mesh_y); @printf "  mesh n=%-6d MAE %.4e\n" n e; end
    text!(ax_c, 0.03, 0.03; space=:relative, align=(:left, :bottom), fontsize=16,
          text=L"$$vs. Richardson extrapolation, $p=%$(round(p_mesh, digits=2))$")
    # one fit line; open symbols = the runs the Richardson extrapolation is built from
    ext = in(rich_mesh).(mesh_x)
    fit_loglog!(ax_c, mesh_x, mesh_y; color=:darkorange, label="fit")
    scatter!(ax_c, mesh_x[.!ext], mesh_y[.!ext], color=:black, markersize=11, label=L"$$runs")
    scatter!(ax_c, mesh_x[ext], mesh_y[ext], color=:white, strokecolor=:black, strokewidth=1.5, markersize=11,
             label=L"$$extrapolation runs")
    # labels on the 2× ladder; off-ladder runs (640, 1280, 6400) as unlabelled minor ticks
    lad_ns = filter(in((64, 128, 256, 512, 1024, 2560, 5120, 10240)), resolutions)
    ax_c.xticks = (Float64.(lad_ns), [L"%$n" for n in lad_ns])
    ax_c.xminorticks = Float64.(setdiff(resolutions, lad_ns)); ax_c.xminorticksvisible = true; ax_c.xminorticksize = 6
    ax_c.xticklabelrotation = π/4
    ylims!(ax_c, 10^(-2.65), 10^(-0.85))
    xlims!(ax_c, minimum(mesh_x) / 2.2, maximum(mesh_x) * 1.5)
    axislegend(ax_c, position=:rt, labelsize=17, rowgap=0)

    # --- (d) convergence upon time refinement: least-squares Richardson over all runs ---
    ax_d = Axis(fig[2, 2], xscale=log10, yscale=log10,
        title = L"$$Convergence upon time refinement",
        xlabel = L"$$Number of timesteps $n_t$", yticks = err_ticks,
        yminorticksvisible = true, yminorticks = IntervalsBetween(9))
    time_x, time_y, p_time = lsq_richardson_errors(time_curves)
    @printf "time Richardson: runs nt = %s, order %.3f\n" string(time_x) p_time
    for (nt, e) in zip(time_x, time_y); @printf "  time nt=%-4d MAE %.4e\n" nt e; end
    text!(ax_d, 0.03, 0.03; space=:relative, align=(:left, :bottom), fontsize=16,
          text=L"$$vs. Richardson extrapolation, $p=%$(round(p_time, digits=2))$")
    fit_loglog!(ax_d, time_x, time_y; color=:darkorange, label="fit")
    scatter!(ax_d, time_x, time_y, color=:black, markersize=11, label=L"$$runs")
    ax_d.xticks = (Float64.(time_nts), [L"%$nt" for nt in time_nts])
    ax_d.xticklabelrotation = π/4
    linkyaxes!(ax_c, ax_d)
    ylims!(ax_d, 10^(-2.65), 10^(-0.85))
    xlims!(ax_d, minimum(time_x) / 1.5, maximum(time_x) * 1.3)
    axislegend(ax_d, position=:rt, labelsize=17, rowgap=0)

    # --- (e) mature band width vs resolution, (f) normalised band profiles ---
    # mean FWHM over the cross-band profiles of the four dominant bands (one per quadrant); bars = min–max
    wall = [reduce(vcat, wq) for wq in bw["w_quad"]]
    wn = Float64.(bw["ns"]); wmed = mean.(wall); wlo = minimum.(wall); whi = maximum.(wall)
    @printf "band width: mean of means %.4f L, range %.4f–%.4f L\n" mean(wmed) minimum(wmed) maximum(wmed)
    # broken y-axis: band width on a linear scale (top), grid spacing on a log scale (bottom)
    ge = fig[3, 1] = GridLayout()
    ax_e  = Axis(ge[1, 1], xscale=log10, title = L"$$Shear band width (mature stage)",
                 yticks = ([0.012, 0.013], [L"0.012", L"0.013"]), ylabel = L"$$Width $[L]$")
    ax_e2 = Axis(ge[2, 1], xscale=log10, yscale=log10, xlabel = L"$$Resolution $n^2$",
                 yticks = ([1e-4, 1e-3, 1e-2], [L"10^{-4}", L"10^{-3}", L"10^{-2}"]), ylabel = L"$\Delta x$ $[L]$")
    rowsize!(ge, 1, Relative(0.45)); rowgap!(ge, 22)
    linkxaxes!(ax_e, ax_e2)
    hidexdecorations!(ax_e; grid = false); ax_e.bottomspinevisible = false; ax_e2.topspinevisible = false
    hlines!(ax_e, [mean(wmed)]; color=:darkorange, linestyle=:dash, linewidth=2,
            label = (wm = @sprintf("%.4f", mean(wmed)); L"$$mean, $%$wm\,L$"))
    errorbars!(ax_e, wn, wmed, wmed .- wlo, whi .- wmed; color=:black, whiskerwidth=6)
    scatter!(ax_e, wn, wmed; color=:black, markersize=11, label=L"$$band width (FWHM), min–max")
    ylims!(ax_e, 0.0112, 0.0142)
    scatterlines!(ax_e2, wn, 1 ./ wn; color=:gray35, linestyle=:dot, linewidth=2, marker=:rect, markersize=9,
                  label=L"$$grid spacing $\Delta x = L/n$")
    # Cells across the measured band-normal FWHM, using the same mean widths as above.
    for (n, cells) in zip(wn, wmed .* wn)
        # Stagger nearby off-ladder resolutions so their labels do not overlap.
        text!(ax_e2, n, 1 / n; text=@sprintf("%.0f", cells), fontsize=14,
              color=:gray35, align=(:center, :bottom), offset=(0, n in (640, 1280, 6400) ? 24 : 8))
        @printf "  band n=%-6d cells across FWHM %.1f\n" n cells
    end
    ylims!(ax_e2, 1.2e-5, 1.5e-2)
    w_ns = Int.(wn); w_lad = filter(in((256, 512, 1024, 2560, 5120, 10240, 40946)), w_ns)
    ax_e2.xticks = (Float64.(w_lad), [L"%$n" for n in w_lad])
    ax_e2.xminorticks = Float64.(setdiff(w_ns, w_lad)); ax_e2.xminorticksvisible = true; ax_e2.xminorticksize = 6
    ax_e2.xticklabelrotation = π/4
    xlims!(ax_e2, minimum(wn) / 1.5, maximum(wn) * 1.5)
    # break marks on both spines, in figure pixels (axis scenes clip at their edges)
    marks = lift(ax_e.scene.viewport, ax_e2.scene.viewport) do a, b
        pts = Point2f[]
        for (o, w, y) in ((a.origin[1], a.widths[1], a.origin[2]), (b.origin[1], b.widths[1], b.origin[2] + b.widths[2])),
            x in (o, o + w)
            push!(pts, Point2f(x - 6, y - 6), Point2f(x + 6, y + 6))
        end
        pts
    end
    translate!(linesegments!(fig.scene, marks; color = :black, linewidth = 1.5, space = :pixel), 0, 0, 1000)
    axislegend(ax_e,  position=:rb, labelsize=16, rowgap=0, framevisible=false)
    axislegend(ax_e2, position=:lb, labelsize=16, framevisible=false)

    ax_f = Axis(fig[3, 2], title = L"$$Band profile (mature stage)",
        xlabel = L"$x - x_\mathrm{peak}$ $[L]$", ylabel = L"$\dot{\varepsilon}^{\mathrm{pl}}_{\mathrm{II}} / \max$")
    w_cols = cgrad(:roma, length(w_ns); categorical=true)
    for (i, n) in enumerate(w_ns)
        lines!(ax_f, bw["xc"][i], bw["prof"][i]; color=w_cols[i], linewidth=1.5, label=L"%$n")
    end
    # half maximum and the FWHM of the 10240² run (same crossings as band_width.jl)
    hlines!(ax_f, [0.5]; color=:gray45, linestyle=:dot, linewidth=1.5)
    i_ref = findfirst(==(10240), w_ns)
    xl, xh = halfmax_crossings(bw["prof"][i_ref], bw["xc"][i_ref])
    linesegments!(ax_f, [Point2f(xl, 0.5), Point2f(xh, 0.5)]; color=:black, linewidth=3)
    text!(ax_f, 0.0, 0.53; text=L"$$FWHM", align=(:center, :bottom), fontsize=17)
    xlims!(ax_f, -0.025, 0.025)
    axislegend(ax_f, position=:rt, labelsize=14, nbanks=2, rowgap=0, patchsize=(18, 10))

    for i in 1:3, j in 1:2
        panel_label!(fig[i, j], string(Char('a' + (i - 1) * 2 + (j - 1))))
    end
    for ext in ("png", "pdf")
        out = joinpath(out_dir, "convergence_figures_UHR.$ext")
        save(out, fig); println("Saved → $out")
    end

    # --- band-orientation figure: failure-angle references from φ = 30°, ψ = 0° ---
    phi_deg, psi_deg = 30.0, 0.0
    roscoe = 45 + psi_deg/2              # 45.0°
    arthur = 45 + (phi_deg + psi_deg)/4  # 52.5°
    mohr   = 45 + phi_deg/2              # 60.0°
    fig2 = Figure(size = (670, 990), fontsize = 20)   # half text width, as in the manuscript
    orientation_panel!(fig2[1, 1], spectra, spec_res, spec_cols, stage_init, L"$$Initiation stage", roscoe, arthur, mohr)
    ax2b = orientation_panel!(fig2[2, 1], spectra, spec_res, spec_cols, stage_mature, L"$$Mature stage", roscoe, arthur, mohr)
    CairoMakie.Legend(fig2[3,1], ax2b, nbanks=2, labelsize=18, orientation=:horizontal, framevisible=true, halign=:center,  tellheight = true, tellwidth = true)
    for j in 1:2
        panel_label!(fig2[j, 1], string(Char('a' + (j - 1))))
    end
    ax2b.xlabel=L"$$Band orientation $\theta_b$ [deg]"
    for ext in ("png", "pdf")
        out = joinpath(out_dir, "band_orientation_UHR.$ext")
        save(out, fig2); println("Saved → $out")
    end
end

main()
