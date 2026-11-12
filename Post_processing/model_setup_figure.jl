using CairoMakie
using MathTeXEngine
CairoMakie.update_theme!(fonts = (regular = texfont(), bold = texfont(:bold), italic = texfont(:italic)))

# Schematic of the model setup; the values are those of the model-setup table.

function main()
    fig = Figure(size = (720, 640), fontsize = 20)
    ax = Axis(fig[1, 1], aspect = DataAspect(),
        xlabel = L"$$x [-]", ylabel = L"$$y [-]",
        limits = (-0.0, 1.0, -0.0, 1.0),
        xticks = 0:0.5:1, yticks = 0:0.5:1,
        title = L"$$Model setup and material parameters", titlesize = 28)

    cmap = cgrad(:lapaz)

    # domain: background phase X_0 (no rim)
    poly!(ax, Rect(0, 0, 1, 1), color = cmap[0.15])

    # inclusion
    xc, yc, r = 0.5, 0.5, 0.1
    poly!(ax, Circle(Point2f(xc, yc), r), color = cmap[0.85], strokecolor = :black, strokewidth = 2)

    # radius arrow (centre -> edge, 45°), label inside the circle, tilted along the arrow
    arrows!(ax, [xc], [yc], [r * cosd(45)], [r * sind(45)], color = :black, linewidth = 3, arrowsize = 14)
    text!(ax, xc + 0.5 * r * cosd(45) - 0.02 * sind(45), yc + 0.5 * r * sind(45) + 0.014 * cosd(45),
        text = L"r", rotation = deg2rad(45), align = (:center, :center), color = :black, fontsize = 20)

    # pure-shear loading (extension in x, shortening in y) — stated once below
    al = 0.14
    arrows!(ax, [0.16, 0.84], [0.5, 0.5], [-al, al], [0.0, 0.0], color = :white, linewidth = 3, arrowsize = 16)
    arrows!(ax, [0.5, 0.5], [0.98, 0.02], [0.0, 0.0], [-al, al], color = :white, linewidth = 3, arrowsize = 16)

    # general parameters (common to both phases)
    text!(ax, 0.03, 0.97, text = L"$$General", font = :bold, align = (:left, :top), color = :white, fontsize = 24)
    gen = [
        L"K = 4.0",
        L"\eta_0 = 1.0",
        L"\phi = 30^{\circ},\ \psi = 0^{\circ}",
        L"C = 1.848",
        L"\eta_\mathrm{reg} = 8\times10^{-3}",
        L"\dot{\varepsilon}_\mathrm{bg} = 1",
        L"t_\mathrm{max} = 3.33",
    ]
    for (k, ln) in enumerate(gen)
        text!(ax, 0.03, 0.91 - 0.055 * (k - 1), text = ln, align = (:left, :top), color = :white, fontsize = 20)
    end

    # background: label
    text!(ax, 0.97, 0.95, text = L"$$Background", font = :bold, align = (:right, :top), color = :white, fontsize = 24)
    text!(ax, 0.97, 0.895, text = L"G_0 = 1.0", align = (:right, :top), color = :white, fontsize = 20)

    # inclusion: label placed just below the circle (no arrow needed)
    text!(ax, xc, yc - r - 0.03, text = L"$$Inclusion", font = :bold, align = (:center, :top), color = :white, fontsize = 24)
    incl = [L"G_i = 0.5", L"r = 0.1"]
    for (k, ln) in enumerate(incl)
        text!(ax, xc, yc - r - 0.03 - 0.055 * k, text = ln, align = (:center, :top), color = :white, fontsize = 20)
    end

    display(fig)
    out = joinpath(mkpath(joinpath(@__DIR__, "Figures")), "model_setup_figure")
    for ext in ("png", "pdf")
        save("$out.$ext", fig)
    end
    println("Saved $out.{png,pdf}")
end

main()
