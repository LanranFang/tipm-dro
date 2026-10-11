# TIPM-DRO

Code organized from the existing experiments for **Distributionally Robust Optimization via Targeted Integral Probability Metric for General Data Processes**, by Lanran Fang, Jianqiang Cheng, Grani A. Hanasusanto, and Yijie Wang.

The supplied manuscript contains mathematical formulations and experiment descriptions. This repository collects their matching Python and Jupyter implementations from the original project, together with the saved results used for the figures.

| Paper figure | Experiment | Code |
| --- | --- | --- |
| Figure 1; Appendix E.1 | Monte Carlo approximation, bounded support | `convergence/suboptimal_dual_gap_vs_M_dimension.ipynb` |
| Figure 2; Section 5.1; Appendix E.3 | Three-item newsvendor: Gaussian, Chi-square, Lognormal, Pareto | `newsvendor/` |
| Figures 3–4; Section 5.2; Appendix E.4 | Regression with outlier corruption | `regression/` |
| Figure 5; Appendix E.2 | Monte Carlo approximation, Student-t uncertainty | `convergence/suboptimal_dual_gap_vs_M_dimension_student_t_xi.ipynb` |

Each experiment notebook has a corresponding Python entry point. `reference_results/` contains selected original CSV tables and figures. New experiment output is written under `results/`, which Git ignores.

## Installation

Use Python 3.12 or 3.13; checks for this copy used Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the notebook interface:

```bash
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

The original experiments use several conic solvers, including Gurobi and MOSEK. The included open solvers support small verification runs. To use Gurobi, install `requirements-gurobi.txt` and configure its license. Solver choice and numerical tolerances can affect regenerated results.

## Run and inspect results

Run the commands from this repository's root. Detailed commands and experiment settings are in [the reproduction guide](docs/REPRODUCING.md). Saved reference results can be plotted without solving the optimization problems:

```bash
python tools/plot_reference_results.py
```

This generates Figures 1–5 under `results/reference_plots/`. Run the structural check with:

```bash
python tools/check_project.py
```

Full experiments use 50 trials, large Monte Carlo sets, and many validation candidates. Small verification runs confirm that the code executes; they do not reproduce the paper's numerical results.

## Reproduction notes

The organization preserves the original model formulations and selected figure settings. It removes notebook output, exploratory newsvendor cells, machine-specific paths, and stale launch scripts. Runtime repairs and differences between manuscript descriptions and existing implementations are documented in [convergence notes](docs/convergence_notes.md), [regression notes](docs/regression_notes.md), and [newsvendor notes](docs/newsvendor_notes.md).

In particular, the bounded-support implementation uses `Omega = 1.0`, while Appendix E.1 describes an empirical second moment. The regression implementation uses the empirical training second moment, while Appendix E.4 states twice that value. These original settings are preserved and documented. Saved reference results come from the original runs; deterministic seeding repairs and solver changes can make new results differ.

The original regression worker evaluated a zero predictor when a fit failed. The prepared worker reports failed methods explicitly and records missing risk values instead. Archived regression CSVs retain the original behavior and should be read with that limitation; see the regression notes for details.

Source provenance is recorded in [the source map](docs/SOURCE_MAP.md) and [the checksum manifest](docs/source_manifest.json). The supplied PDF and unrelated exploratory experiments are not included in this upload folder.

## Upload to GitHub

Create an empty GitHub repository, then run:

```bash
git init -b main
git add .
git commit -m "Add TIPM-DRO paper experiments"
git remote add origin https://github.com/YOUR_USERNAME/tipm-dro.git
git push -u origin main
```

Replace the example repository URL with your own. This folder contains no configured GitHub remote.

中文说明：[README.zh-CN.md](README.zh-CN.md).
