# Newsvendor experiment: Figure 2

The four experiment notebooks implement Section 5.1 and Appendix E.3 of the paper. Each notebook retains the final selected-trial experiment and the OOS-versus-N plot; exploratory epsilon sweeps, single-item examples, Gamma demand and the K=1/3/5 extension are excluded.

| Figure 2 panel | Notebook in `newsvendor/` | Python entry point | Original retained cells |
| --- | --- | --- | --- |
| (a) Truncated Gaussian | `IPM_DRO_multi_Newsvendor_Gaussian_3D.ipynb` | `run_gaussian_3d.py` | 0, 3, 4, 5, 11, 12 |
| (b) Rescaled chi-square | `IPM_DRO_multi_Newsvendor_ChiSquare_3D.ipynb` | `run_chisquare_3d.py` | 0, 3, 4, 5, 11, 12 |
| (c) Lognormal | `IPM_DRO_multi_Newsvendor_Lognormal_3D.ipynb` | `run_lognormal_3d.py` | 0, 3, 4, 5, 11, 12 |
| (d) Pareto | `IPM_DRO_multi_Newsvendor_Pareto_3D.ipynb` | `run_pareto_3d.py` | 0, 3, 4, 5, 10, 11 |

Cell indices are zero-based in the original notebooks. `ipm_3d_explicit_models.py` supplies the finite conic TIPM-DRO formulations, including the empirical second-moment model in Equation (41). Notebook model cells keep the original SAA and 2-Wasserstein implementations and import the TIPM functions from this helper.

## Run the experiments

Run commands from the repository root. The four notebook kernels can also start in the `newsvendor/` directory; repository paths are discovered automatically.

```bash
python newsvendor/run_gaussian_3d.py --quick --solver CLARABEL --workers 1
python newsvendor/run_chisquare_3d.py --quick --solver CLARABEL --workers 1
python newsvendor/run_lognormal_3d.py --quick --solver CLARABEL --workers 1
python newsvendor/run_pareto_3d.py --quick --solver CLARABEL --workers 1
```

`--quick` checks the complete generate/split/solve/validate/OOS/CSV/plot path with one trial at N=10, 8 uniform decision points plus the SAA solution, 100 OOS points and epsilon=0.01. These settings verify execution and do not reproduce the paper. Quick outputs go to `results/newsvendor/<distribution>/quick/eps_tables/`.

Omit `--quick` for the paper-size experiment:

```bash
python newsvendor/run_gaussian_3d.py --solver CLARABEL --workers 1
```

Choose the other entry point for its corresponding demand distribution. All scripts accept `--workers`, `--solver`, `--output-dir` and `--quick`; `--help` lists their descriptions. `--output-dir` overrides the distribution output directory and stores its tables and plot in an `eps_tables/` child. Python entry points use a main guard and initialize spawned workers with the chosen parameters and solver.

The default solver is CLARABEL so the repository can run without a commercial solver license. `IPM_DRO_SOLVER=MOSEK` before launching Python or the notebook kernel selects the original solver; Python entry points also accept `--solver MOSEK`. MOSEK must be installed and licensed. Solver changes can cause numerical differences even though the equations, coefficients and experimental defaults are unchanged. Worker count defaults to 1 and can also be set with `IPM_DRO_WORKERS`. Use the Python entry points for parallel runs.

Successful trials are resumed from the output CSV; incomplete rows remain eligible for retry. Use a new `--output-dir` when changing the solver or experiment settings to avoid reusing results from another run. Solver errors are recorded in the generated trial table's `errors` column. A sample size with no successful result for a method produces an explicit error after saving diagnostic tables.

## Paper settings retained

- Three products; holding costs h=(0.1, 0.2, 0.3), shortage costs b=(1, 1, 1), CVaR tail probability rho=0.05 and order upper bounds U=(50, 50, 50).
- Demand parameters mu=(5, 6, 6), sigma=(5, 6, 8), with independent coordinate generation.
- N in {10, 20, 30, 50, 75, 100, 150}; 50 trials for each N; 80/20 training/validation split; 10,000 independent OOS observations.
- Both robust methods tune epsilon on {a*10^b: a in {1, 3, 5, 7, 9}, b in {-3, -2, -1, 0, 1, 2}} using validation cost.
- TIPM draws 800 uniform decision points in [0, 50]^3 and appends the same trial's SAA solution, resulting in 801 points in a successful trial. Its second-moment constraint uses the training mean of squared demand-vector norms.
- Seed formula BASE=42; training seed=BASE+100000*N+1000*trial; OOS seed=training seed+1.

For Gaussian demand, mu and sigma parameterize the untruncated Normal draws. Rejection of negative draws changes the realized mean and standard deviation. This follows the original notebooks and Appendix E.3, even though the main text describes common demand moments. The other three distributions use the original moment-matching formulas.

OOS CVaR preserves the original evaluation rule: find the empirical (1-rho) quantile using the `higher` convention and average all losses at or above that threshold. This can differ from a fractional-tail empirical CVaR at small validation sample sizes or in the presence of ties. The optimization objective and the published evaluation rule have been retained.

## Archived paper results

`reference_results/newsvendor/<distribution>/` contains byte-identical copies of:

- `NV_3D_selected_trials.csv`;
- `NV_3D_selected_summary.csv`;
- `NV_3D_oos_summary_by_N_comparison.csv`;
- `NV_3D_OOS_plot_by_N_<distribution>.pdf`.

Each archived trial table contains 350 distinct trials: 50 at each of the seven sample sizes, with complete SAA, 2-WDRO and TIPM-DRO OOS values. The archived mean and 10th/90th percentile summaries were checked against these trial tables. These files preserve the existing full experiment results; the complete optimization sweep was not rerun during repository preparation. The original Pareto output directory was named `pareto_3D`; the prepared repository consistently uses `pareto`.

`newsvendor/SOURCE_MANIFEST.json` records the original source files, SHA-256 hashes, retained cells, edits and archived output copies. Preparation removed stored notebook execution state and a malformed exploratory Gaussian cell containing an incomplete `n_trials_fig5 =` assignment. No original source file was modified.

Validation included syntax/import checks for all files, four successful CLARABEL quick runs with all three methods, and a successful Gaussian run with two spawned workers. The serial and parallel quick results agree.
