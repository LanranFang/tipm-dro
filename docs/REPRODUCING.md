# Running the paper experiments

Run commands from the repository root. Python 3.13 and the versions in `requirements.txt` were available for the local checks. Python 3.12 is also compatible with the source syntax and declared dependencies.

## Installation and saved figures

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/check_project.py
python tools/plot_reference_results.py
```

The last command reads archived CSVs and produces PDF and PNG versions of Figures 1–5 in `results/reference_plots/`. It redraws the same statistics in a portable layout; byte-for-byte copies of the original plotted PDFs are in `reference_results/`. No optimization solver or system LaTeX installation is needed to redraw the reference data.

## Small execution checks

These commands reduce the dimensions, trials, auxiliary samples, or tuning grids as appropriate for each experiment. Quick results have their own output directories and do not replace the archived results or the full-run outputs.

```bash
python convergence/suboptimal_dual_gap_vs_M_dimension.py --quick --workers 1
python convergence/suboptimal_dual_gap_vs_M_dimension_student_t_xi.py --quick --workers 1
python newsvendor/run_gaussian_3d.py --quick --solver CLARABEL --workers 1
python newsvendor/run_chisquare_3d.py --quick --solver CLARABEL --workers 1
python newsvendor/run_lognormal_3d.py --quick --solver CLARABEL --workers 1
python newsvendor/run_pareto_3d.py --quick --solver CLARABEL --workers 1
python regression/run_regression.py --quick --experiment all --jobs 1
```

Quick runs verify execution of the optimization and validation paths. Their output should not be interpreted as a reproduction of the paper's 50-trial figures.

## Full experiments

Monte Carlo convergence, Figures 1 and 5:

```bash
python convergence/suboptimal_dual_gap_vs_M_dimension.py --workers 4
python convergence/suboptimal_dual_gap_vs_M_dimension_student_t_xi.py --workers 4
```

Each full run solves `3 dimensions × 50 replications × 8 M values = 1200` sampled problems. Outputs are under `results/convergence/`. The original conic solver order is ECOS, CLARABEL, SCS.

Three-item newsvendor, Figure 2:

```bash
python newsvendor/run_gaussian_3d.py --solver CLARABEL --workers 1
python newsvendor/run_chisquare_3d.py --solver CLARABEL --workers 1
python newsvendor/run_lognormal_3d.py --solver CLARABEL --workers 1
python newsvendor/run_pareto_3d.py --solver CLARABEL --workers 1
```

Each distribution uses 7 sample sizes, 50 trials, 800 auxiliary decisions plus the SAA decision, and a 30-value radius grid. The open solver is the portable default; `--solver GUROBI` or `--solver MOSEK` uses a separately installed and licensed solver. `IPM_DRO_SOLVER` and `IPM_DRO_WORKERS` also configure the newsvendor defaults. Outputs are under `results/newsvendor/<distribution>/`.

Outlier-corrupted regression, Figures 3 and 4:

```bash
python regression/run_regression.py --experiment dimension --jobs 4
python regression/run_regression.py --experiment contamination --jobs 4
```

The dimension sweep uses dimensions `[1,5,10,50]`, five sample sizes, and 50 trials. The contamination sweep uses scales 5 through 20 and 50 trials. Both use 800 sampled decisions in the full configuration. Outputs are under `results/regression/`. See [the regression notes](regression_notes.md) for solver fallback, validation grids, resume behavior, and the preserved second-moment setting.

## Notebooks

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

Open a notebook under one of the experiment directories and execute its cells in order. Paths resolve relative to the repository rather than the original machine. Use the Python entry points for explicit quick modes and command-line controls. The regression notebook preserves the original additional exploratory dimension `d=3`; the paper runner uses the four dimensions shown in Figure 3.

## Convergence settings and moment bounds

The full convergence experiments preserve dimensions `[5,25,100]`, Monte Carlo sample sizes `[2,5,10,20,50,100,250,500]`, 50 replications, 100 symmetric historical observations, `epsilon=0`, and seed `20260519`. The decision set is the unit ball, with the known empirical optimizer `x_star=0.1*ones(d)`. Four Gaussian unit directions and weights from 0.5 to 1.5 define eight affine loss pieces. Each replication uses nested prefixes of one pool of 500 auxiliary decisions.

For Figure 1, the original bounded-support implementation and archived CSVs use `Omega=1.0`. Appendix E.1 instead describes the empirical training second moment. The implementation setting is preserved; historical samples in the unit ball satisfy its moment bound. For Figure 5, both the implementation and Appendix E.2 use `Omega=d/(d+2)`. Historical coordinates are independent Student-t samples with five degrees of freedom, scaled to variance `1/(d+2)`. The sampled dual is enforced on all of `R^d` without a Section 4 tail constraint.

For Student-t observations, the empirical second moment can exceed the population value `Omega`, even with symmetric sample pairs. The identity `V_0(x)=E_P_hat[loss(x,xi)]` at `epsilon=0` requires that `P_hat` satisfy the ambiguity set's moment bound; that feasibility condition is not automatic in this experiment. Symmetry and convexity still make `x_star` an empirical optimizer. The archived curves should therefore be interpreted as empirical objective gaps of the sampled-dual solutions. They show mean percentage gaps, interquartile bands, and pooled fixed-slope `C/sqrt(M)` fits, whose constants are 26.97917269 and 30.59275634.

The prepared programs preserve the original model coefficients, moment bounds, sampling rules and solver tolerances. They use repository-relative output paths, default to four workers, check solver status and finite solutions, and render with MathText instead of requiring LaTeX. Their source and reference-output hashes are in [source_manifest.json](source_manifest.json).

## Interpretation and provenance

Archived CSVs are unchanged copies of original runs. Numerical settings with differences from the supplied manuscript are explained above and in [regression_notes.md](regression_notes.md). Operational repairs and the selected newsvendor cells are documented in [newsvendor_notes.md](newsvendor_notes.md). No full experiment was rerun merely to organize this repository.

Checks performed on the prepared copy are recorded in [VALIDATION.md](VALIDATION.md). [SOURCE_MAP.md](SOURCE_MAP.md) and [source_manifest.json](source_manifest.json) record which original files supplied the notebooks, scripts, modules, and reference results.
