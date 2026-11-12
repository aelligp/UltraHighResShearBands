# Appendix figure: temporal evolution of the plastic strain rate over the full domain (2×3 panels).
#
#   julia --project=. Post_processing/time_evolution_figure.jl
#
# Source: the nt = 20 run of the time refinement (2560², η_reg = 0.008), which reaches
# t_max = 3.333; its fields are read from the per-step VTK output, times from the .pvd file.
# Default steps: onset at the inclusion (7), initiation (9, t = 1.5), onset of the large-scale bands (10),
# crossover (11), intermediate (15) and t_max (20). CPU only.

using ReadVTK, CairoMakie, MathTeXEngine, Printf
CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# step → (field, time) from the per-step VTK output and the .pvd file
function load_step(src, n, ts)
    vti = joinpath(src, "timestep_$ts", "vtk_" * lpad("$(ts)_0", 6, "0") * ".vti")
    e   = abs.(Float32.(reshape(get_data(get_point_data(VTKFile(vti))["plastic_strain_rate_II"]), n, n)))
    pvd = read(joinpath(src, "time_series.pvd"), String)
    t   = parse(Float64, match(Regex("timestep=\"([^\"]+)\"[^>]*file=\"timestep_$(ts)/"), pvd)[1])
    println("ts = $ts, t = $t, max = $(maximum(e))"); flush(stdout)
    return (e = e, t = t)
end
function make_figure(fields, steps, stages, vmax, out)
    length(steps) == 6 || error("the 2×3 layout needs six steps")
    fig = Figure(size = (1150, 820), fontsize = 20)
    hm = nothing
    for (k, ts) in enumerate(steps)
        e, t = fields[ts]
        r, c = (k - 1) ÷ 3 + 1, (k - 1) % 3 + 1
        ts_str, stage = @sprintf("%.2f", t), get(stages, ts, "")
        ax = Axis(fig[r, c]; aspect = DataAspect(),
                  title = ts == 20 ? L"$t = %$ts_str$, mature ($t_\mathrm{max}$)" : L"$t = %$ts_str$, %$stage",
                  xticks = 0.25:0.25:0.75, yticks = 0.25:0.25:0.75, xlabel = r == 2 ? L"$x$" : "", ylabel = c == 1 ? L"$y$" : "")
        r == 1 && hidexdecorations!(ax; ticks = false, grid = false)
        c > 1  && hideydecorations!(ax; ticks = false, grid = false)
        m = size(e, 1)
        hm = heatmap!(ax, range(0.5/m, 1 - 0.5/m; length = m), range(0.5/m, 1 - 0.5/m; length = m), e;
                      colormap = :vik, colorrange = (0.0, vmax), rasterize = 3)
        θ = range(0, 2π; length = 721)
        lines!(ax, 0.5 .+ 0.1cos.(θ), 0.5 .+ 0.1sin.(θ); color = :white, linewidth = 0.8)
        limits!(ax, 0, 1, 0, 1)
        # panel label: grey box, as in the convergence and orientation figures
        poly!(ax, Rect2f(0, 0.9, 0.1, 0.1); color = :gray95, strokecolor = :black, strokewidth = 1.0)
        text!(ax, 0.05, 0.95; text = string('a' + k - 1), align = (:center, :center), font = :bold, fontsize = 18)
    end
    Colorbar(fig[1:2, 4], hm; label = L"\dot{\varepsilon}^{\mathrm{pl}}_{\mathrm{II}}", labelsize = 26, height = Relative(0.8))
    colgap!(fig.layout, 12); rowgap!(fig.layout, 8)
    for ext in ("png", "pdf"); save("$out.$ext", fig; px_per_unit = 3); println("Saved → $out.$ext"); end
    return fig
end

function main()
    n      = 2560
    steps  = [7, 9, 10, 11, 15, 20]
    stages = Dict(7 => "onset", 9 => "initiation", 10 => "crossover", 11 => "crossover", 13 => "crossover",
                  15 => "intermediate", 20 => "mature")
    vmax   = 3.5           # upper limit of the colour range shared by all panels
    src    = joinpath(@__DIR__, "..", "Refinements", "time", "nt20", "vtk")
    fields = Dict(ts => load_step(src, n, ts) for ts in steps)
    return make_figure(fields, steps, stages, vmax, joinpath(mkpath(joinpath(@__DIR__, "Figures")), "time_evolution_$(n)"))
end

main()
