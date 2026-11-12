# Ultra-High-Resolution Modelling Reveals the Complexity of Evolving Shear Band Networks

This repository contains the scripts that reproduce the results of

> Aellig, P. S., de Montserrat, A., Duretz, T., Räss, L., Kaus, B. J. P., and Utkin, I.: Ultra-High-Resolution Modelling Reveals the Complexity of Evolving Shear Band Networks, *Geodynamica*.

The model is a 2D visco-elasto-viscoplastic pure-shear box with a weak circular inclusion, solved on GPUs with [JustRelax.jl](https://github.com/PTsolvers/JustRelax.jl). The scripts run every model of the publication, from 64² up to 40946² cells, and compute every figure and table from the model output. If you use this code, please cite the publication and this repository (see [CITATION.cff](CITATION.cff)).

## Requirements

- Julia 1.13. The `Manifest.toml` pins the exact package versions used.
- NVIDIA GPUs. The 20474² and 40946² runs use 16 and 64 GPUs; all other runs use one GPU.
- A SLURM cluster for `run_it_multiple.jl`.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Contents

- `UltraHighRes_Shearbands/ShearBand2D_FFT_UHR_DYREL.jl` and `ShearBand2D_FFT_UHR.jl`: the model with the DYREL and the accelerated pseudo-transient (APT) solver. The two files differ only in the solver.
- `UltraHighRes_Shearbands/gather_vtk_to_global.jl`: assembles the per-rank output of a multi-GPU run into one global field.
- `run_it_multiple.jl`: writes a SLURM job script for every run of the publication. The solver is set in its `main()`.
- `Post_processing/`: the analysis, figure and table scripts.

## Reproducing the results

Run every command from the repository root.

1. Run the models. This writes the runs to `Resolution_Study/`, `Refinements/mesh/`, `Refinements/time/`, `Reg_Viscosity_Systematics/` and `Tolerance_Study/`.

   ```bash
   julia --project=. run_it_multiple.jl submit
   ```

2. Gather steps 9 and 20 of the two multi-GPU runs, with as many MPI ranks as the run used:

   ```bash
   srun -n 16 julia --project=. UltraHighRes_Shearbands/gather_vtk_to_global.jl dir=Resolution_Study/sharp_20474 step=9
   srun -n 16 julia --project=. UltraHighRes_Shearbands/gather_vtk_to_global.jl dir=Resolution_Study/sharp_20474 step=20
   srun -n 64 julia --project=. UltraHighRes_Shearbands/gather_vtk_to_global.jl dir=Resolution_Study/sharp_40946 step=9
   srun -n 64 julia --project=. UltraHighRes_Shearbands/gather_vtk_to_global.jl dir=Resolution_Study/sharp_40946 step=20
   ```

3. Compute the spectra and band widths, which are written to `Post_processing/data/`:

   ```bash
   julia --project=. Post_processing/PSD_analyses.jl                     # GPU
   julia --project=. Post_processing/PSD_analyses_smoothed_inclusion.jl  # GPU
   julia --project=. Post_processing/band_width.jl
   ```

4. Make the figures and tables. Figures are written to `Post_processing/Figures/`, and tables are printed as LaTeX rows.

   | Script | Output |
   |---|---|
   | `model_setup_figure.jl` | Model setup |
   | `time_evolution_figure.jl` | Time evolution of the plastic strain rate at 2560² |
   | `convergence_figures_UHR.jl` | Spectra, stress convergence upon mesh and time refinement, band width, and band orientation |
   | `convergence_sharp_vs_smoothed.jl` | Spectra of the sharp vs. the smoothed inclusion |
   | `table_spectral_dimensions.jl` | Table of spectral slopes and dimensions |
   | `regularisation_viscosities_appendix.jl` | Effect of the regularization viscosity (appendix) |
   | `convergence_tolerance.jl` | Convergence with the nonlinear-solver tolerance (appendix) |
   | `walltime_table.jl` | Table of wall times (appendix) |

## License

MIT, see [LICENSE](LICENSE).
