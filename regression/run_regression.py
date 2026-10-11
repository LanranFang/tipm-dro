#!/usr/bin/env python3
"""Run or plot the MAD regression experiments for Figures 3 and 4."""

import argparse
import os
from pathlib import Path

# Limit numerical threads before importing NumPy or the worker module.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

from exp1_oos_worker import run_one_trial


ROOT = Path(__file__).resolve().parents[1]
METHODS = ["risk_std", "risk_or_2eps", "risk_or_eps", "risk_mom"]
LABELS = ["Std WDRO", "OR-WDRO (2ω)", "OR-WDRO (ω)", "MoM TIPM-DRO"]
COLORS = ["red", "green", "limegreen", "orange"]


def config(d, contamination_scale, quick=False, seed=42):
    cfg = dict(
        d=d, n_test=1000, eps=0.2, contamination_scale=contamination_scale,
        rho_shift=0.1, sigma=float(np.sqrt(d) + 1) * np.e,
        eps_tv=0.2, rho_list=[0.01, 0.1, 0.2, 0.5, 1.0],
        M_samples=800, seed=seed, delta=0.1, split_rate=0.8,
    )
    if quick:
        cfg.update(n_test=40, M_samples=8, rho_list=[1.0],
                   K_list=[2], epsilon_scales=[1.0])
    return cfg


def run_tasks(tasks, destination, jobs, overwrite):
    if destination.exists() and not overwrite:
        existing = pd.read_csv(destination)
        expected = {(int(t), int(n)) for t, n, _ in tasks}
        actual = set(zip(existing["trial"].astype(int), existing["n"].astype(int)))
        if actual != expected or len(existing) != len(tasks):
            raise RuntimeError(f"Incomplete or incompatible result file: {destination}; use --overwrite")
        print(f"Using {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if jobs == 1:
        rows = [run_one_trial(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(run_one_trial, task) for task in tasks]
            rows = [future.result() for future in as_completed(futures)]
    pd.DataFrame(rows).sort_values(["n", "trial"]).to_csv(destination, index=False)
    print(f"Saved {destination}")


def plot_dimension(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted((source / "dim_sweep").glob("dim_sweep_d*.csv"),
                   key=lambda path: int(path.stem.split("_d")[-1]))
    if not files:
        raise FileNotFoundError(f"No dimension results under {source / 'dim_sweep'}")
    for path in files:
        d = int(path.stem.split("_d")[-1])
        frame = pd.read_csv(path)
        sizes = sorted(frame["n"].unique())
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, label, color in zip(METHODS, LABELS, COLORS):
            values = [frame.loc[frame["n"] == n, method].dropna().to_numpy() for n in sizes]
            mean = [np.mean(value) if value.size else np.nan for value in values]
            lo = [np.percentile(value, 10) if value.size else np.nan for value in values]
            hi = [np.percentile(value, 90) if value.size else np.nan for value in values]
            ax.plot(sizes, mean, color=color, marker="o", label=label)
            ax.fill_between(sizes, lo, hi, color=color, alpha=0.15)
        ax.set_xlabel("Training Sample Size $N$", fontsize=16)
        ax.set_ylabel("OOS excess risk (MAD)", fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=14)
        fig.tight_layout()
        target = output / f"exp1_oos_dim_sweep_d{d}.pdf"
        fig.savefig(target)
        plt.close(fig)
        print(f"Saved {target}")


def plot_contamination(source, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted((source / "cont_scale").glob("cont_scale_cs*.csv"))
    if not files:
        raise FileNotFoundError(f"No contamination results under {source / 'cont_scale'}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    scales = sorted(frame["contamination_scale"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label, color in zip(METHODS[1:], LABELS[1:], COLORS[1:]):
        values = [frame.loc[frame["contamination_scale"] == scale, method].dropna().to_numpy()
                  for scale in scales]
        ax.plot(scales, [np.mean(value) if value.size else np.nan for value in values], color=color, marker="o", label=label)
        ax.fill_between(scales, [np.percentile(value, 10) if value.size else np.nan for value in values],
                        [np.percentile(value, 90) if value.size else np.nan for value in values], color=color, alpha=0.15)
    ax.set_xlabel("Contamination scale", fontsize=16)
    ax.set_ylabel("OOS excess risk (MAD)", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=16)
    ax.set_xticks(scales)
    fig.tight_layout()
    target = output / "exp1_contamination_scale.pdf"
    fig.savefig(target)
    plt.close(fig)
    print(f"Saved {target}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["dimension", "contamination", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="One trial with d=1, N=20, M=8 and reduced grids")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--plot-only", action="store_true", help="Plot existing CSVs without optimization")
    parser.add_argument("--reference", action="store_true", help="Plot the original archived CSVs; implies --plot-only")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing result files")
    parser.add_argument("--output-dir", type=Path, help="Override the repository-relative output directory")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.reference and args.quick:
        parser.error("--reference and --quick cannot be combined")
    output = args.output_dir or ROOT / "results" / "regression" / ("quick" if args.quick else "")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "reference_results" / "regression" if args.reference else output
    experiments = ["dimension", "contamination"] if args.experiment == "all" else [args.experiment]
    if not (args.plot_only or args.reference):
        trials = 1 if args.quick else 50
        sizes = [20] if args.quick else [10, 20, 50, 75, 100]
        dimensions = [1] if args.quick else [1, 5, 10, 50]
        scales = [5] if args.quick else list(range(5, 21))
        for experiment in experiments:
            if experiment == "dimension":
                for d in dimensions:
                    cfg = config(d, 10, args.quick)
                    tasks = [(t, n, cfg) for n in sizes for t in range(trials)]
                    run_tasks(tasks, output / "dim_sweep" / f"dim_sweep_d{d}.csv", args.jobs, args.overwrite)
            else:
                for index, scale in enumerate(scales):
                    cfg = config(1 if args.quick else 10, scale, args.quick, seed=42 + index)
                    if not args.quick:
                        cfg["rho_list"] = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
                    tasks = [(t, 20 if args.quick else 50, cfg) for t in range(trials)]
                    run_tasks(tasks, output / "cont_scale" / f"cont_scale_cs{scale}.csv", args.jobs, args.overwrite)
        metadata = {"experiment": args.experiment, "quick": args.quick, "trials": trials,
                    "dimensions": dimensions, "sample_sizes": sizes, "contamination_scales": scales,
                    "note": "Archived CSVs were generated by the original source; stable Z seeding is a portability change."}
        (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if "dimension" in experiments:
        plot_dimension(source, output)
    if "contamination" in experiments:
        plot_contamination(source, output)


if __name__ == "__main__":
    main()
