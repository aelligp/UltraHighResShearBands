# band_width.jl — width of the four dominant shear bands in the mature stage vs resolution.
# One isolated band per quadrant (inclusion → corner); the top-right window (no crossings for
# y ∈ [0.72,0.86]) is mirrored about x = 0.5 and y = 0.5 for the other three:
# per row, FWHM of |ε̇ᵖˡ_II| along x; band-normal width = row FWHM · sinβ, with the
# band angle β from a line fit of the peak positions.
#
#   julia --project=. Post_processing/band_width.jl

include(joinpath(@__DIR__, "PSD_analyses.jl"))   # stage_files, gathered_file, load_field

mirror(w) = (1 - w[2], 1 - w[1])

"(x, y) window of each quadrant band; tr is measured, the others are its mirror images"
function band_windows()
    ywin = (0.72, 0.86)   # rows used (above 0.86 a secondary band tip enters xwin)
    xwin = (0.62, 0.86)   # contains only the target band (neighbour band reaches x≈0.91 at y=0.72)
    return (tr = (xwin, ywin), tl = (mirror(xwin), ywin), br = (xwin, mirror(ywin)), bl = (mirror(xwin), mirror(ywin)))
end

"Half-max crossings (xl, xr) of the peak at ip in prof (linear interpolation)"
function halfmax_crossings(prof, x, ip)
    h = prof[ip] / 2
    l = ip; while l > 1 && prof[l] > h; l -= 1; end
    r = ip; while r < length(prof) && prof[r] > h; r += 1; end
    xl = x[l] + (h - prof[l]) / (prof[l+1] - prof[l]) * (x[l+1] - x[l])
    xr = x[r-1] + (h - prof[r-1]) / (prof[r] - prof[r-1]) * (x[r] - x[r-1])
    return xl, xr
end

"FWHM of the peak at ip in prof"
fwhm(prof, x, ip) = -(reverse(halfmax_crossings(prof, x, ip))...)

function measure(f::Matrix{Float64}, xwin, ywin)
    n = size(f, 1); x = range(0.0, 1.0; length=n)
    ilo, ihi = searchsortedfirst(x, xwin[1]), searchsortedlast(x, xwin[2])
    js = searchsortedfirst(x, ywin[1]):max(1, n ÷ 256):searchsortedlast(x, ywin[2])
    xp = Float64[]; yp = Float64[]; w = Float64[]
    for j in js
        prof = abs.(f[ilo:ihi, j]); ip = argmax(prof)
        push!(xp, x[ilo+ip-1]); push!(yp, x[j]); push!(w, fwhm(prof, x[ilo:ihi], ip))
    end
    s = ([yp ones(length(yp))] \ xp)[1]          # dx/dy of the band centreline
    sinβ = 1 / sqrt(1 + s^2)
    jm = js[length(js) ÷ 2]; prof = abs.(f[ilo:ihi, jm]); ip = argmax(prof)
    return (w=w .* sinβ, β=atand(1, abs(s)), xc=collect(x[ilo:ihi]) .- x[ilo+ip-1],
            prof=prof ./ prof[ip])
end

function main(; resolutions = [256, 512, 640, 1024, 1280, 2560, 5120, 6400, 10240],
                gathered = Dict(20474 => 20, 40946 => 20), t = stage_times().mature)
    windows = band_windows()
    quad = Dict{Int,Any}()
    @printf "%7s %4s %6s %10s %10s %10s %8s\n" "n" "band" "rows" "w_mean" "w_min" "w_max" "β[°]"
    files = Dict(n => stage_files(resolution_dir(n), (mature = t,))[:mature] for n in resolutions)
    for (n, step) in gathered; files[n] = gathered_file(n, step); end
    for n in sort(collect(keys(files)))
        f = load_field(files[n])
        quad[n] = Dict(q => measure(f, windows[q]...) for q in keys(windows)); f = nothing; GC.gc()
        for q in keys(windows)
            r = quad[n][q]
            @printf "%7d %4s %6d %10.5f %10.5f %10.5f %8.2f\n" n q length(r.w) mean(r.w) minimum(r.w) maximum(r.w) r.β
        end
    end
    ns = sort(collect(keys(quad)))
    # cache for convergence_figures_UHR.jl: widths of all four bands, top-right row profile
    mkpath(joinpath(@__DIR__, "data"))
    jldsave(joinpath(@__DIR__, "data", "band_width_tmax.jld2");
            ns, xc = [quad[n][:tr].xc for n in ns], prof = [quad[n][:tr].prof for n in ns],
            w_quad = [[quad[n][q].w for q in keys(windows)] for n in ns])
end

main()
