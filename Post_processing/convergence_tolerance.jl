using JLD2
using CairoMakie, Statistics
using MathTeXEngine
CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# Appendix figure: nonlinear-solver tolerance convergence of τ_xx at 2560² (runs in Tolerance_Study/).
#
#   julia --project=. Post_processing/convergence_tolerance.jl

# tolerance as a LaTeX label, e.g. 10^{-3} or 2×10^{-3}
function sci(x)
    e = floor(Int, round(log10(x); digits = 6))
    m = round(Int, x / 10.0^e)
    m == 1 ? L"10^{%$e}" : L"%$m\times10^{%$e}"
end

folder(tol) = joinpath(@__DIR__, "..", "Tolerance_Study", "tol$tol")

# --- τ_xx vs time (left, with peak inset) + MAE vs tolerance (right) ---
function main()
    tols   = sort([1e-2, 2e-2, 5e-2, 1e-3, 2e-3, 5e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]; rev = true)
    labels = sci.(tols)
    fig = Figure(size = (1200, 600), fontsize = 20)   # font sizes match convergence_figures_UHR.jl
    colors = cgrad(:roma, length(tols); categorical = true)

    # left panel: τ_xx vs normalised time
    ax1 = Axis(fig[1, 1], xlabel = L"$$Normalised time [-]", ylabel = L"$$\tau_{xx}\ [-]")
    # fixed main-axis limits so the zoom rectangle has a known data span
    xlo, xhi, ylo, yhi = -0.02, 1.02, -0.03, 1.9
    limits!(ax1, xlo, xhi, ylo, yhi)
    # zoom window on the peak stresses around t = 0.5
    zx0, zx1, zy0, zy1 = 0.45, 0.55, 1.70, 1.83
    inset_h = 0.4
    inset_aspect = ((zx1 - zx0) * (yhi - ylo)) / ((zy1 - zy0) * (xhi - xlo))
    axin = Axis(fig[1, 1], width = Relative(inset_aspect * inset_h), height = Relative(inset_h),
        halign = 0.90, valign = 0.125, xticklabelsize = 16, yticklabelsize = 16,
        xticks = 0.45:0.05:0.55, backgroundcolor = (:white, 1.0))
    limits!(axin, zx0, zx1, zy0, zy1)
    # rectangle marking the zoom region on the main axis
    poly!(ax1, Rect(zx0, zy0, zx1 - zx0, zy1 - zy0),
        color = :transparent, strokecolor = :black, strokewidth = 1.5)

    τref = Float64[]
    for (k, tol) in enumerate(tols)
        p = joinpath(folder(tol), "its", "τII_εII_loading.jld2")
        isfile(p) || (@warn "missing τII" p; continue)
        τII = load(p)["τII"]
        tol == 1e-8 && (τref = τII)
        t = range(0, 1, length = length(τII) + 1)
        y = vcat(0.0, τII)
        lines!(ax1, t, y, color = colors[k], linewidth = 2.5, label = labels[k])
        lines!(axin, t, y, color = colors[k], linewidth = 2.5)
    end
    translate!(axin.blockscene, 0, 0, 100)

    # right panel: mean absolute error of τ_xx vs tolerance (reference = 1e-8)
    ax2 = Axis(fig[1, 2], xlabel = L"$$Tolerance [-]",
        ylabel = L"$$Error [-]",
        xscale = log10,
        yscale = log10)
    tolvals = tols[1:end-1]                           # drops the 1e-8 reference
    mae = map(tolvals) do tol
        τII = load(joinpath(folder(tol), "its", "τII_εII_loading.jld2"))["τII"]
        mean(abs.(τII .- τref))
    end
    lines!(ax2, tolvals, mae, color = :black, linewidth = 2.5)
    for k in eachindex(tolvals)
        scatter!(ax2, [tolvals[k]], [mae[k]], color = colors[k], markersize = 14,
            strokecolor = :black, strokewidth = 1)
    end
    xlims!(ax2, 5e-1, 4e-8)

    # single shared legend, inside the MAE panel (top right)
    elems = [LineElement(color = colors[k], linewidth = 2.5) for k in eachindex(tols)]
    axislegend(ax2, elems, labels, L"$$Tolerance", position = :rt,
        framevisible = true, labelsize = 16, titlesize = 18)

    for (j, lbl) in enumerate(("a", "b"))
        box = Axis(fig[1, j], width = Relative(0.07), height = Relative(0.10),
            halign = :left, valign = :top, backgroundcolor = :gray95, spinewidth = 1.0,
            leftspinecolor = :black, topspinecolor = :black,
            rightspinecolor = :black, bottomspinecolor = :black)
        hidedecorations!(box)
        translate!(box.blockscene, 0, 0, 1000)
        text!(box, 0.5, 0.5; text = lbl, align = (:center, :center), font = :bold, fontsize = 18)
    end

    Label(fig[0, :], L"$$Tolerance convergence of the shear stress $\tau_{xx}$ (2560$^2$)",
        fontsize = 22, font = :bold)

    out = joinpath(mkpath(joinpath(@__DIR__, "Figures")), "Tolerance_convergence")
    for ext in ("png", "pdf")
        save("$out.$ext", fig)
    end
    println("Saved $out.{png,pdf}")
end

main()
