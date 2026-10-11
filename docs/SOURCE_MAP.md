# Source map

Paths in the original-source column are relative to the original `/home/lanran/projects/` directory. Original files were read and copied; all preparation edits are confined to the new `tipm-dro/` directory.

| Original source | Prepared source or entry point | Paper use |
| --- | --- | --- |
| `suboptimal_dual_gap_vs_M_dimension.ipynb` | Same basename under `convergence/`, plus a `.py` entry point | Figure 1; Appendix E.1 |
| `suboptimal_dual_gap_vs_M_dimension_student_t_xi.ipynb` | Same basename under `convergence/`, plus a `.py` entry point | Figure 5; Appendix E.2 |
| `newsvendor/IPM_DRO_multi_Newsvendor_Gaussian_3D.ipynb` | Same notebook under `newsvendor/`; `run_gaussian_3d.py` | Figure 2(a) |
| `newsvendor/IPM_DRO_multi_Newsvendor_ChiSquare_3D.ipynb` | Same notebook under `newsvendor/`; `run_chisquare_3d.py` | Figure 2(b) |
| `newsvendor/IPM_DRO_multi_Newsvendor_Lognormal_3D.ipynb` | Same notebook under `newsvendor/`; `run_lognormal_3d.py` | Figure 2(c) |
| `newsvendor/IPM_DRO_multi_Newsvendor_Pareto_3D.ipynb` | Same notebook under `newsvendor/`; `run_pareto_3d.py` | Figure 2(d) |
| `newsvendor/ipm_3d_explicit_models.py` | Same helper under `newsvendor/` | Equation (41) |
| `exp1_oos.ipynb` | `regression/exp1_oos.ipynb`; command-line adaptation `run_regression.py` | Figures 3–4 |
| `exp1_oos_worker.py` | `regression/exp1_oos_worker.py` | Equation (42) and regression baselines |

The newsvendor notebooks retain the final selected-trial experiment and plotting cells; their original zero-based cell indices and individual edits are in [newsvendor/SOURCE_MANIFEST.json](../newsvendor/SOURCE_MANIFEST.json). The reproduction guide and regression notes document the other edits and parameter differences.

## Reference outputs

| Original outputs | Prepared destination |
| --- | --- |
| `suboptimal_known_opt_gap_results.csv`, `suboptimal_known_opt_gap_summary.csv`, selected files in `suboptimal_gap_figures/` | `reference_results/convergence/bounded/` |
| Student-t raw/summary CSVs and selected files in `suboptimal_gap_figures_student_t_xi/` | `reference_results/convergence/student_t/` |
| Final selected-trial CSVs, summaries and OOS-versus-N PDFs in `newsvendor/results/{gaussian_3d,chisquare_3d,lognormal_3d,pareto_3D}/eps_tables/` | `reference_results/newsvendor/{gaussian,chisquare,lognormal,pareto}/` |
| `results/dim_sweep/dim_sweep_d{1,5,10,50}.csv`, `results/cont_scale/cont_scale_cs{5..20}.csv`, matching plotted PDFs | `reference_results/regression/` |

Reference outputs are byte-identical copies of the original files. Their original and prepared SHA-256 hashes are recorded in [source_manifest.json](source_manifest.json). Newly drawn figures from `tools/plot_reference_results.py` are generated under the ignored `results/` directory.

## Excluded material

The supplied `IPM_DRO.pdf` is used to identify the experiments and is not part of the upload folder. The original `figure_3.ipynb` implements portfolio experiments, despite its filename, and `dro_comparison.ipynb` implements a Markov-data comparison; neither supplies the current manuscript's Figures 3–4. Early temporary newsvendor versions, single-item/Gamma/K=1/3/5 extensions, theory-proxy convergence notebooks, duplicate notebooks, empty portfolio notebooks, stale shell/Slurm launchers, environments, checkpoints and unrelated outputs are excluded.

`tools/check_project.py`, `tools/plot_reference_results.py`, repository documentation, dependency lists and ignore rules are new preparation utilities. They are not represented as original manuscript source code.
