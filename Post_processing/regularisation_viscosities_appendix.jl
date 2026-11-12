using JLD2
using Statistics
using CairoMakie
using MathTeXEngine
CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# Appendix figure: effect of the regularisation viscosity η_reg.
#   Row 1 (a,b,c): plastic strain-rate field (stokes.ε_pl.II) for the
#                  unregularised / regularised / over-regularised runs.
#   Row 2 (d,e,f): τII stress vs normalised time, one curve per timestep count nt.

# Grey panel-label box in the top-left corner of the grid cell `gpos` (e.g. gl[1,1] for the heatmap axis only).
function panel_label!(gpos, label)
    inset = Axis(gpos, width = Relative(0.12), height = Relative(0.12),
        halign = :left, valign = :top, backgroundcolor = :gray95,
        spinewidth = 1.0, leftspinecolor = :black, topspinecolor = :black,
        rightspinecolor = :black, bottomspinecolor = :black)
    hidedecorations!(inset)
    translate!(inset.blockscene, 0, 0, 1000)
    text!(inset, 0.5, 0.5; text = label, align = (:center, :center),
        font = :bold, fontsize = 18, color = :black)
end

folder(η_reg, nt) = joinpath(@__DIR__, "..", "Reg_Viscosity_Systematics", "eta$(η_reg)_nt$nt")

function main()
    li       = 1.0                       # domain length (used for field axes)
    nt_field = 100                       # timestep count used for the row-1 field maps
    nts      = [20, 40, 60, 80, 100]     # timestep sweep for the row-2 stress curves

    # columns ordered unregularised -> over-regularised (2560² runs)
    etas       = [8e-4, 8e-3, 8e-2]
    col_titles = [L"$$Under-regularised \eta_{reg} = 0.0008", L"$$Regularised \eta_{reg} = 0.008", L"$$Over-regularised \eta_{reg} = 0.08"]


    fig = Figure(size = (1300, 850))

    # --- column header boxes ---
    for (j, title) in enumerate(col_titles)
        Box(fig[0, j], color = :gray90)
        Label(fig[0, j], title, tellwidth = false, fontsize = 22, color = :black)
    end

    # --- Row 1: plastic strain-rate field maps ---
    fields = map(etas) do η_reg
        cp = joinpath(folder(η_reg, nt_field), "checkpoint", "checkpoint_$(nt_field)",
                    "checkpoint_rank_0", "checkpoint0000.jld2")
        isfile(cp) || (@warn "missing checkpoint" cp; return nothing)
        load(cp)["stokes"].ε_pl.II
    end
    for (j, field) in enumerate(fields)
        isnothing(field) && continue
        nx, ny = size(field)
        x = range(0, li, length = nx); y = range(0, li, length = ny)

        gl = fig[1, j] = GridLayout()
        ax = Axis(gl[1, 1], aspect = DataAspect(),
            xlabel = j == 2 ? L"$$x [-]" : "", ylabel = j == 1 ? L"$$y [-]" : "",
            xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16)
        # per-panel range, capped at 5.0 so the over-regularised panel keeps its own ~0.2 scale
        crange = (0.0, min(5.0, maximum(field)))
        hm = heatmap!(ax, x, y, field, colormap = :batlow, colorrange = crange, rasterize = true)
        # thin colorbar, height tracking the square domain; label on the last panel only
        Colorbar(gl[1, 2], hm, label = j == 3 ? L"$$\dot{\varepsilon}_{II}^{pl}\ [-]" : "",
            labelsize = 18, ticklabelsize = 14, width = 12, height = Relative(0.92))
        colgap!(gl, 8)
        # panel label drawn in data-space so it stays inside the domain regardless of DataAspect padding
        let w = 0.12li, h = 0.12li
            poly!(ax, Rect2f(0.0, li - h, w, h); color = :gray95, strokecolor = :black, strokewidth = 1.0)
            text!(ax, w/2, li - h/2; text = string('a' + (j - 1)),
                  align = (:center, :center), font = :bold, fontsize = 18)
        end
    end

    # --- Row 2: τII vs normalised time, one curve per nt ---
    colors = cgrad(:roma, length(nts); categorical = true)
    ax2_ref = nothing
    for (j, η_reg) in enumerate(etas)
        ax = Axis(fig[2, j],
            xlabel = j == 2 ? L"$$Normalised time [-]" : "",
            ylabel = j == 1 ? L"$$\tau_{xx}\ [-]" : "",
            xlabelsize = 22, ylabelsize = 22, xticklabelsize = 16, yticklabelsize = 16)
        ax2_ref = ax
        for (k, nt) in enumerate(nts)
            p = joinpath(folder(η_reg, nt), "its", "τII_εII_loading.jld2")
            isfile(p) || (@warn "missing τII" p; continue)
            τII = load(p)["τII"]
            @assert length(τII) == nt "τII length $(length(τII)) ≠ nt=$nt for η_reg = $η_reg"
            t = range(0, 1, length = length(τII) + 1)
            lines!(ax, t, vcat(0.0, τII), color = colors[k], linewidth = 2.5, label = "nt = $nt")
        end
        panel_label!(fig[2, j], string('d' + (j - 1)))
    end

    # nt legend inside the last stress-time panel (f)
    isnothing(ax2_ref) || axislegend(ax2_ref, L"$$Timesteps", position = :rb,
        framevisible = true, labelsize = 16, titlesize = 18)

    display(fig)
    out = joinpath(mkpath(joinpath(@__DIR__, "Figures")), "regularisation_viscosities_appendix")
    for ext in ("png", "pdf")
        save("$out.$ext", fig)
    end
    println("Saved $out.{png,pdf}")
end

main()
