# Regression code and experiment provenance

This directory corresponds to Section 5.2, Appendix E.4, and Figures 3 and 4 of
the [paper on Optimization Online](https://optimization-online.org/?p=34965). The original files have been left untouched.

| Original source | Repository copy | Role |
| --- | --- | --- |
| `exp1_oos.ipynb` | `regression/exp1_oos.ipynb` | Dimension and contamination sweeps and plots |
| `exp1_oos_worker.py` | `regression/exp1_oos_worker.py` | LAD, Std WDRO, OR-WDRO with two TV radii, and MoM TIPM-DRO |
| Notebook sweep cells | `regression/run_regression.py` | Import-safe command-line entry point with a `__main__` guard |
| `results/dim_sweep/dim_sweep_d{1,5,10,50}.csv` | `reference_results/regression/dim_sweep/` | Original Figure 3 trial results |
| `results/cont_scale/cont_scale_cs{5,...,20}.csv` | `reference_results/regression/cont_scale/` | Original Figure 4 trial results |

The four dimension CSVs each contain 250 rows (50 trials at each of five sample
sizes). The sixteen contamination CSVs each contain 50 rows. The extra original
dimension `d=3` is retained in the notebook's sweep settings, but its CSV and plot
are not included in the paper reference set. The CLI uses the paper's four dimensions.

`figure_3.ipynb` is a portfolio experiment despite its filename;
`dro_comparison.ipynb` is a Markov decision experiment; `Untitled-1` is an older
single-item newsvendor notebook. These files do not implement this regression
experiment. The original `run_exp1.sh` refers to the absent `run_exp1_offline.py`
and has been replaced by the runnable CLI.

## Running

Install the repository's requirements and use Python 3. NumPy, CVXPY, pandas, and
Matplotlib are required for the CLI; Jupyter is needed only for the notebook.
The worker tries installed GUROBI, MOSEK, CLARABEL, and SCS solvers, in that order.
CLARABEL and SCS provide open-source alternatives to commercial solver licenses.

From the repository root, replot the archived results without optimization:

```bash
python regression/run_regression.py --reference
```

This reads the archived CSVs and writes five new PDF plots under
`results/regression/`; the reference files are unchanged. The same command can be
run from `regression/` as `python run_regression.py --reference`.

For a genuine small optimization check:

```bash
python regression/run_regression.py --experiment all --quick --jobs 1
```

Quick mode runs one dimension task and one contamination task, each fitting all
five models. It uses `d=1`, `N=20`, `N_test=40`, `M=8`, one WDRO radius `rW=1`,
`K=2`, and one TIPM radius scale `s=1`. The dimension task uses `C=10`; the
contamination task uses `C=5`. Results are written under
`results/regression/quick/` and are smoke checks, not paper experiments.

Full sweeps retain the source's model formulations and tuning settings and can
take substantial time:

```bash
python regression/run_regression.py --experiment dimension --jobs 4
python regression/run_regression.py --experiment contamination --jobs 4
python regression/run_regression.py --experiment all --jobs 4
```

`--plot-only` replots previously generated results; `--output-dir PATH` overrides
the output location; `--overwrite` recomputes existing files. The runner reuses an
existing CSV only when its `(trial, n)` coverage matches the requested tasks.
That coverage check does not establish that optimization succeeded or that a
CSV was generated with matching model settings. Inspect `failed_methods` and
finite risk values when assessing newly generated results.

The notebook can be opened with Jupyter launched from either the repository
root or `regression/`. Outputs use the same repository-relative results directory.
Its first diagnostic cells contain the original separate lognormal setup and
do not produce the paper results. Its two full sweep cells are expensive.
`IPM_DRO_WORKERS` controls notebook concurrency (default at most four workers).

## Exact source settings and paper differences

Each trial draws a coefficient uniformly from the unit sphere, Gaussian clean
features, and responses without added noise. It contaminates `floor(0.2*N)` pairs
with `(u, v) -> (C*u, -C**2*v + 0.1)`, shuffles, and uses an 80/20 training/validation
split. Validation discards the largest `ceil(0.2*N_val)` absolute residuals.
The independent clean test risk is the mean absolute residual; the clean oracle
has zero loss in this setup, so this is also excess risk.

| Setting | Figure 3 source sweep | Figure 4 source sweep |
| --- | --- | --- |
| Feature dimensions | `1, 3, 5, 10, 50` (paper and CLI omit `3`) | `10` |
| Sample sizes | `10, 20, 50, 75, 100` | `50` |
| Contamination scale | `10` | `5, 6, ..., 20` |
| Trials / clean test size | `50 / 1000` | `50 / 1000` |
| Contamination / validation trim fraction | `0.2 / 0.2` | `0.2 / 0.2` |
| Response shift / confidence delta | `0.1 / 0.1` | `0.1 / 0.1` |
| WDRO radii | `0.01, 0.1, 0.2, 0.5, 1.0` | `0.01, 0.05, 0.1, 0.5, 1.0, 2.0` |
| OR-WDRO moment scale `sigma` | `(sqrt(d)+1)*e` | `(sqrt(10)+1)*e` |
| MoM sampled decisions `M` | `800` | `800` |
| MoM decision radius and coefficient bound | `sqrt(d)` | `sqrt(10)` |
| Second-moment multiplier in the worker | `1.0` | `1.0` |

With `N_tr=floor(0.8*N)` and `omega=0.2`, the worker computes
`K_theory=ceil(4*(1+2*omega)/(1-2*omega)**2*log(1/delta))`, then chooses from
`{K_theory, min(N_tr//3, 2*K_theory), N_tr//2}` after removing invalid or duplicate
counts. It keeps the original fallback if that set is empty. The TIPM radius
scales are `0.01, 0.05, 0.1, 0.2, 0.5, 1, 5, 10`, multiplied by
`4*sqrt(e)*sigma_loss/(sqrt(N_tr)*(1-2*omega_tv))` and floored at `1e-6`.
`sigma_loss` is the empirical standard deviation of absolute projections onto
the augmented direction of all ones normalized to length one. MoM block
averages use the first `K*(N_tr//K)` observations, discarding the remaining tail.
The same sampled decision points are reused across a trial's CV candidates.
The non-MoM solvers retain their source coefficient bound of `10`.

Appendix E.4 states `Omega=2*Omega_tr`, whereas the available worker uses
`omega_scale=[1.0]` and all archived regression CSVs have `best_omega=1.0`.
This repository preserves that available implementation; it does not silently
change the objective to match the written multiplier. The original notebook
also contains the extra dimension `3`. These differences prevent describing
the available code and saved results as an exact reproduction of every written
paper setting.

## Portability and failure reporting changes

- Notebook outputs, execution counts, and machine-specific runtime metadata have
  been removed. Imports and output paths now work from the repository root and
  the regression directory, with bounded default process concurrency.
- The original inline diagnostic solvers now use the shared solver fallback;
  their diagnostic objectives and bounds are preserved.
- The worker skips unavailable solvers and accepts optimal or optimal-inaccurate
  solutions. If every available solver fails, `_solve` raises an explicit error.
  Output suppression supports notebook streams that do not expose `fileno()`.
- Invalid final fitted coefficients produce a `RuntimeWarning`, `NaN` risk, and
  an entry in the semicolon-separated `failed_methods` column. CV can continue
  past failed candidates and other methods can still be evaluated. Unexpected
  trial errors and failure of the direct LAD solve raise with trial context.
  Plots handle groups with no valid risk values without calling percentiles on
  empty arrays.
- The original `None` fallback evaluated the zero predictor as if it were the
  fitted model, and the original outer exception handler returned zero-predictor
  risks for every model. Original archived CSVs lack solver-status columns and
  retain those historical semantics. Their finite values do not prove that
  every model solved successfully. They are included for provenance and replotting.
- The original sampled-decision seed used `hash((n, t, 'Z'))`, whose string hash
  changes with Python's hash randomization. The copy uses
  `SeedSequence([seed, n, t, d, 1701])`; this is stable across worker processes
  and repeated runs. Clean data and training-data seeds retain the source's
  behavior. Changing `cfg['seed']` now controls the sampled decisions; it does
  not control all generated clean/training data.
- Explicit Fortran reshape order preserves CVXPY's original behavior while
  avoiding warnings about a future default change. Reduced `K`/epsilon grids
  are optional and used only by quick mode; full defaults are unchanged.

The stable seed and explicit failure handling can change newly generated
results relative to the archived CSVs. Those CSVs and five archived PDFs were
copied byte for byte, not regenerated.

## Source fingerprints and checks

SHA-256 of the original, unmodified source files:

```text
exp1_oos.ipynb
058d67cfdb8ae65adac1226d733df5df66f5d04f0701d119986be5901825f12d

exp1_oos_worker.py
6c18958d995ba7c0022e19df2713afb1ebec3c56b592d479243f6643ee295d7a
```

Checks performed during packaging: Python and notebook-cell parsing; cleared
notebook outputs/counts; real quick-mode solves for both sweeps; all five model
fits using CLARABEL; solver output with a stream lacking `fileno()`; explicit
missing-solver failure; warnings/NaNs for a failed OR-WDRO fit; empty-risk
plot groups; root/subdirectory notebook imports; process-pool CLI smoke mode;
and replotting all archived regression CSVs from the
regression subdirectory. Full optimization sweeps have not been rerun.
