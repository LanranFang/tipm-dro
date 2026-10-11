# Checks on the prepared repository

Preparation date: 2026-10-11. Checks used Python 3.13.0, NumPy 2.3.4, Pandas 2.3.3, Matplotlib 3.10.7, CVXPY 1.7.3, ECOS 2.0.14 and CLARABEL 0.11.1. These are execution checks and audits of saved outputs; the full optimization experiments were not rerun.

| Check | Result |
| --- | --- |
| Python source and every notebook code cell parse | Passed; 11 Python files and 7 notebooks |
| Notebook outputs, execution counts and machine-specific code paths | Cleared / removed |
| Original source files and supplied PDF | Unmodified; original checksum baseline retained during preparation |
| Archived outputs | 49 byte-identical copies: 38 CSVs and 11 PDFs |
| Reference raw-data coverage | Both convergence experiments: 1200 rows each; all four newsvendor distributions: 350 trials each; regression: 4 dimensions with 250 trials each and 16 contamination scales with 50 trials each |
| Figures 1–5 redrawn from reference CSVs | Passed; PDF and PNG output without solver calls or LaTeX |
| Pooled convergence fit constants | 26.97917269 and 30.59275634; match the paper's 26.98 and 30.59 |
| Two convergence quick runs | Passed; two optimal ECOS solves and generated plots per experiment |
| Four newsvendor quick runs | Passed with CLARABEL; SAA, 2-WDRO and TIPM-DRO all produced finite values |
| Newsvendor multiprocessing | Gaussian quick run passed with two spawned workers; agreed with the serial run |
| Newsvendor archived summaries | Means and 10th/90th percentiles match recomputation from the 350-trial tables |
| Regression quick runs | Passed for both quick experiments; all five methods returned finite risks |
| Regression portable execution | Passed from the root and subdirectory, with one and two workers |
| Regression open-solver fallback | Passed with available solvers restricted to CLARABEL |
| Regression failure handling | Missing solver raises; unsupported stdout capture works; unavailable model fit reports warning, NaN risk and `failed_methods`; empty risk groups can be plotted |
| Original regression CSVs redrawn | Passed; four dimension plots and one contamination plot |

Quick modes use reduced settings and isolated output paths. They do not establish agreement with the full paper curves. Archived regression CSVs preserve the original zero-predictor fallback and do not contain sufficient status information to verify that every original fit succeeded. Second-moment and evaluation differences are documented in the experiment notes.

Recheck the prepared sources and reference coverage with `python tools/check_project.py`. Redraw the saved figure statistics with `python tools/plot_reference_results.py`.

## Code-only GitHub distribution

The manuscript and exported figure files are excluded from Git and the release
ZIP. The README and citation metadata link to
https://optimization-online.org/?p=34965. Numerical CSV records remain available.
After this change, `tools/check_project.py` passed syntax, clean-notebook,
54-file provenance, and reference coverage checks. `tools/plot_reference_results.py`
regenerated all five figures locally from CSV data.
