"""Redraw paper figures from archived CSVs, without optimization or LaTeX."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference_results"


def find_csv(folder: Path, name: str) -> Path:
    paths = sorted(folder.rglob(name))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {name} under {folder}, found {len(paths)}")
    return paths[0]


def save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    for extension in ("pdf", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.pdf and {name}.png")


def convergence(output: Path, student_t: bool) -> None:
    suffix = "_student_t_xi" if student_t else ""
    path = find_csv(REFERENCE / "convergence", f"suboptimal_known_opt_gap_summary{suffix}.csv")
    data = pd.read_csv(path).sort_values(["dimension", "M"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for dimension, group in data.groupby("dimension"):
        xs = group["M"].to_numpy()
        ax.plot(xs, group["mean_pct_gap"], marker="o", label=f"d = {dimension}")
        ax.fill_between(xs, group["pct_q25"], group["pct_q75"], alpha=0.16)
    positive = data[(data["mean_pct_gap"] > 0) & (data["M"] > 0)]
    constant = float(np.exp(np.mean(np.log(positive["mean_pct_gap"]) + 0.5 * np.log(positive["M"]))))
    xs = np.linspace(float(data["M"].min()), float(data["M"].max()), 400)
    ax.plot(xs, constant / np.sqrt(xs), "k--", label=f"{constant:.2f} / sqrt(M)")
    ax.set(xlabel="Monte Carlo sample size M", ylabel="Percentage optimality gap (%)", yscale="log")
    ax.grid(alpha=0.2)
    ax.legend()
    save(fig, output, "figure_5_student_t" if student_t else "figure_1_bounded")
    print(f"  Fitted constant: {constant:.8f}")


def newsvendor(output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    methods = [("SAA", "SAA"), ("Wass", "2-WDRO"), ("IPM", "TIPM-DRO")]
    for ax, distribution in zip(axes.flat, ("gaussian", "chisquare", "lognormal", "pareto")):
        path = find_csv(REFERENCE / "newsvendor" / distribution, "NV_3D_oos_summary_by_N_comparison.csv")
        data = pd.read_csv(path).sort_values("N")
        for key, label in methods:
            ax.plot(data["N"], data[f"mean_oos_{key}"], marker="o", label=label)
            ax.fill_between(data["N"], data[f"q10_oos_{key}"], data[f"q90_oos_{key}"], alpha=0.13)
        ax.set(title=distribution.capitalize(), xlabel="Training sample size N", ylabel="Mean OOS cost")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, output, "figure_2_newsvendor")


REGRESSION_METHODS = [
    ("risk_std", "Std WDRO"),
    ("risk_or_2eps", "OR-WDRO(2 omega)"),
    ("risk_or_eps", "OR-WDRO(omega)"),
    ("risk_mom", "MoM TIPM-DRO"),
]


def regression_panel(ax, data: pd.DataFrame, key: str, methods) -> None:
    for column, label in methods:
        grouped = data.groupby(key)[column]
        means = grouped.mean().sort_index()
        low = grouped.quantile(0.1).reindex(means.index)
        high = grouped.quantile(0.9).reindex(means.index)
        ax.plot(means.index, means, marker="o", label=label)
        ax.fill_between(means.index, low, high, alpha=0.13)
    ax.set_ylabel("OOS excess risk (MAD)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)


def regression(output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, dimension in zip(axes.flat, (1, 5, 10, 50)):
        path = find_csv(REFERENCE / "regression", f"dim_sweep_d{dimension}.csv")
        regression_panel(ax, pd.read_csv(path), "n", REGRESSION_METHODS)
        ax.set(title=f"d = {dimension}", xlabel="Training sample size N")
    save(fig, output, "figure_3_regression_dimensions")

    files = [find_csv(REFERENCE / "regression", f"cont_scale_cs{scale}.csv") for scale in range(5, 21)]
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    regression_panel(ax, data, "contamination_scale", REGRESSION_METHODS[1:])
    ax.set_xlabel("Contamination scale C")
    save(fig, output, "figure_4_regression_contamination")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "reference_plots")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
    convergence(args.output, student_t=False)
    newsvendor(args.output)
    regression(args.output)
    convergence(args.output, student_t=True)


if __name__ == "__main__":
    main()
