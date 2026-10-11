#!/usr/bin/env python3
"""Figure 2 (c) Lognormal experiment. Use --quick for a small execution check."""

import os
import sys
import json
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

if "__file__" in globals():
    REPO_ROOT = Path(__file__).resolve().parents[1]
else:
    _cwd = Path.cwd().resolve()
    REPO_ROOT = next(
        (p for p in (_cwd, *_cwd.parents)
         if (p / "newsvendor" / "ipm_3d_explicit_models.py").is_file()),
        None,
    )
    if REPO_ROOT is None:
        raise RuntimeError("Open the notebook from the repository or its newsvendor directory.")

NOTEBOOK_DIR = REPO_ROOT / "newsvendor"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))
from ipm_3d_explicit_models import DEFAULT_SOLVER, solve_and_check
RESULTS_ROOT = REPO_ROOT / "results" / "newsvendor"


# Data generation helpers

def _as_1d_param(value, d, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(d, float(arr))
    arr = arr.reshape(-1)
    if arr.size != d:
        raise ValueError(f"{name} must be a scalar or have length {d}.")
    return arr


def _as_2d_samples(value, d, name):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.size != d:
            raise ValueError(f"{name} must have shape (n, {d}) or be a single {d}-vector.")
        arr = arr.reshape(1, d)
    if arr.ndim != 2 or arr.shape[1] != d:
        raise ValueError(f"{name} must have shape (n, {d}).")
    return arr


def log_generate_data_nd(mu, std, size, seed=None):
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)

    if mu.ndim == 0:
        mu = mu.reshape(1)
    if std.ndim == 0:
        std = np.full_like(mu, float(std))
    if mu.shape != std.shape:
        raise ValueError('mu and std must have the same shape.')
    if np.any(mu <= 0):
        raise ValueError('All means must be > 0 for lognormal marginals.')
    if np.any(std < 0):
        raise ValueError('All std entries must be >= 0.')
    if size is None or size <= 0:
        raise ValueError('size must be a positive integer.')

    sigma2 = np.log(1.0 + (std ** 2) / (mu ** 2))
    sigma = np.sqrt(sigma2)
    mean = np.log(mu) - 0.5 * sigma2

    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=mean, sigma=sigma, size=(size, mu.size))


def enumerate_newsvendor_pieces_3d(h, b):
    h = _as_1d_param(h, 3, 'h')
    b = _as_1d_param(b, 3, 'b')

    pieces = []
    for pattern in product((0, 1), repeat=3):
        mask = np.array(pattern, dtype=int)
        name = ''.join('O' if bit == 0 else 'U' for bit in mask)
        p = np.where(mask == 0, h, -b)
        g = np.where(mask == 0, -h,  b)
        pieces.append({'name': name, 'p': p, 'q': 0.0, 'g': g})

    pieces.append({
        'name': 'T',
        'p': np.zeros(3, dtype=float),
        'q': 1.0,
        'g': np.zeros(3, dtype=float),
    })
    return pieces

def Wass2_DRO_3D(eps, U, rho, h, b, xi, solver=DEFAULT_SOLVER, verbose=True):
    h = _as_1d_param(h, 3, 'h')
    b = _as_1d_param(b, 3, 'b')
    U = _as_1d_param(U, 3, 'U')
    xi = _as_2d_samples(xi, 3, 'xi')

    N = xi.shape[0]
    base_pieces = enumerate_newsvendor_pieces_3d(h, b)[:-1]
    xi_sq = np.square(xi)

    x = cp.Variable(3, nonneg=True)
    t = cp.Variable()
    Lambda = cp.Variable(nonneg=True)
    s = cp.Variable(N)

    const = [x <= U, s >= t]
    for piece in base_pieces:
        scaled_p = piece['p'] / rho
        scaled_g = piece['g'] / rho
        quad_slack = cp.Variable(3, nonneg=True)
        w = cp.Variable((N, 3))
        Q = cp.vstack([quad_slack for _ in range(N)])

        for j in range(3):
            const.append(cp.SOC(quad_slack[j] + Lambda, cp.hstack([scaled_g[j], quad_slack[j] - Lambda])))

        const += [
            w >= cp.multiply(xi, scaled_g.reshape(1, 3)) + Q,
            w >= -cp.multiply(xi_sq, Lambda),
            s >= scaled_p @ x + (1 - 1 / rho) * t + cp.sum(w, axis=1),
        ]

    obj = cp.Minimize(Lambda * (eps ** 2) + (1 / N) * cp.sum(s))

    prob = cp.Problem(obj, const)
    solve_and_check(prob, solver=solver, verbose=verbose)

    return np.asarray(x.value).reshape(-1), float(prob.value)

def SAA_3D(rho, U, h, b, xi, solver=DEFAULT_SOLVER, verbose=False):
    h = _as_1d_param(h, 3, 'h')
    b = _as_1d_param(b, 3, 'b')
    U = _as_1d_param(U, 3, 'U')
    xi = _as_2d_samples(xi, 3, 'xi')

    N = xi.shape[0]
    Hmat = np.broadcast_to(h, (N, 3))
    Bmat = np.broadcast_to(b, (N, 3))

    x = cp.Variable(3, nonneg=True)
    t = cp.Variable()
    X = cp.vstack([x for _ in range(N)])

    loss = cp.sum(
        cp.multiply(Hmat, cp.pos(X - xi))
        + cp.multiply(Bmat, cp.pos(xi - X)),
        axis=1,
    )

    prob = cp.Problem(
        cp.Minimize(t + (1 / (rho * N)) * cp.sum(cp.pos(loss - t))),
        [x <= U],
    )
    solve_and_check(prob, solver=solver, verbose=verbose)
    return np.asarray(x.value).reshape(-1), float(prob.value)

from ipm_3d_explicit_models import (
    IPM_DRO_3D_explicit as IPM_DRO_3D,
    IPM_DRO_Sec_3D_explicit as IPM_DRO_Sec_3D,
)


# Evaluation helpers

def _quantile_higher(a, q):
    try:
        return np.quantile(a, q, method='higher')
    except TypeError:
        return np.quantile(a, q, interpolation='higher')


def _newsvendor_losses_3d(h, b, xi, x):
    h = _as_1d_param(h, 3, 'h')
    b = _as_1d_param(b, 3, 'b')
    xi = _as_2d_samples(xi, 3, 'xi')
    x = _as_1d_param(x, 3, 'x')

    diff = x.reshape(1, 3) - xi
    loss = np.maximum(h.reshape(1, 3) * diff, 0.0) + np.maximum(b.reshape(1, 3) * (-diff), 0.0)
    return loss.sum(axis=1)


def _cvar_from_losses(Lambda, losses):
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return np.nan, np.nan
    q = _quantile_higher(losses, 1 - Lambda)
    tail = losses[losses >= q]
    cvar = float(tail.mean()) if tail.size else float(q)
    return cvar, float(q)


def oos_valid_3d(Lambda, h, b, xi, x, return_var=False):
    losses = _newsvendor_losses_3d(h, b, xi, x)
    cvar, var = _cvar_from_losses(Lambda, losses)
    return (cvar, var) if return_var else cvar


def oos_performance_3d(Lambda, h, b, xi_oos, x, return_var=False):
    losses = _newsvendor_losses_3d(h, b, xi_oos, x)
    cvar, var = _cvar_from_losses(Lambda, losses)
    return (cvar, var) if return_var else cvar


def make_eps_grid(a_list, b_list):
    grid = [a * (10 ** b) for a in a_list for b in b_list]
    return np.array(sorted(grid), dtype=float)


# Save selected trial tables for 3D experiments: train -> validation select eps -> test evaluate
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import os

results_dir_3d = str(RESULTS_ROOT / "lognormal")
save_dir_eps_3d = os.path.join(results_dir_3d, 'eps_tables')
BASE = 42
h_fig5 = np.array([0.1, 0.2, 0.3])
b_fig5 = np.array([1.0, 1.0, 1.0])
mu_fig5 = np.array([5.0, 6.0, 6.0])
sigma_fig5 = np.array([5.0, 6.0, 8.0])
size_oos_fig5 = 10000
Lambda_fig5 = 0.05
U_fig5 = np.array([50.0, 50.0, 50.0])
split_rate_fig5 = 0.8
n_trials_fig5 = 50
N_samp_fig5 = 800
SOLVER = DEFAULT_SOLVER

a_fig5 = [1, 3, 5, 7, 9]
b_pow_ipm_fig5 = [-3, -2, -1, 0, 1, 2]
eps_grid_I_3d = np.array(sorted({a * (10.0 ** b) for a in a_fig5 for b in b_pow_ipm_fig5}), dtype=float)
N_list_I_3d = [10, 20, 30, 50, 75, 100, 150]


def run_one_selected_trial_3d(args):
    size_tr, trial_id = args

    seed_tr = BASE + 100_000 * size_tr + 1_000 * trial_id
    seed_oos = seed_tr + 1

    xi_insample = log_generate_data_nd(mu_fig5, sigma_fig5, size=size_tr, seed=seed_tr)
    n_tr = int(size_tr * split_rate_fig5)
    xi_tr, xi_val = xi_insample[:n_tr], xi_insample[n_tr:]
    Omega_fig5 = 1 * float(np.mean(np.sum(xi_tr**2, axis=1)))
    xi_oos = log_generate_data_nd(mu_fig5, sigma_fig5, size=size_oos_fig5, seed=seed_oos)

    rng = np.random.default_rng(seed_tr)
    samples = rng.uniform(0.0, U_fig5, size=(N_samp_fig5, 3))

    row = {
        'size': size_tr,
        'trial': trial_id,
        'x_SAA': None,
        'obj_SAA': None,
        'val_SAA': None,
        'oos_SAA': None,
        'x_Wass': None,
        'obj_Wass': None,
        'eps_Wass': None,
        'val_Wass': None,
        'oos_Wass': None,
        'x_IPM': None,
        'obj_IPM': None,
        'eps_IPM': None,
        'val_IPM': None,
        'oos_IPM': None,
        'errors': [],
    }

    try:
        x_saa, obj_saa = SAA_3D(Lambda_fig5, U_fig5, h_fig5, b_fig5, xi_tr, solver=SOLVER, verbose=False)
        row['x_SAA'] = x_saa.tolist()
        row['obj_SAA'] = float(obj_saa)
        row['val_SAA'] = float(oos_valid_3d(Lambda_fig5, h_fig5, b_fig5, xi_val, x_saa))
        row['oos_SAA'] = float(oos_performance_3d(Lambda_fig5, h_fig5, b_fig5, xi_oos, x_saa))
        samples = np.vstack([samples, x_saa.reshape(1, 3)])
    except Exception as exc:
        row["errors"].append(str(exc))

    best_wass = {'val': np.inf, 'eps': None, 'x': None, 'obj': None}
    for eps in eps_grid_I_3d:
        try:
            x_wass, obj_wass = Wass2_DRO_3D(eps, U_fig5, Lambda_fig5, h_fig5, b_fig5, xi_tr, solver=SOLVER, verbose=False)
            val_wass = oos_valid_3d(Lambda_fig5, h_fig5, b_fig5, xi_val, x_wass)
            if np.isfinite(val_wass) and val_wass < best_wass['val']:
                best_wass.update(val=float(val_wass), eps=float(eps), x=x_wass, obj=float(obj_wass))
        except Exception as exc:
            row["errors"].append(str(exc))
            continue

    if best_wass['x'] is not None:
        row['x_Wass'] = best_wass['x'].tolist()
        row['obj_Wass'] = best_wass['obj']
        row['eps_Wass'] = best_wass['eps']
        row['val_Wass'] = best_wass['val']
        try:
            row['oos_Wass'] = float(oos_performance_3d(Lambda_fig5, h_fig5, b_fig5, xi_oos, best_wass['x']))
        except Exception as exc:
            row["errors"].append(str(exc))

    best_ipm = {'val': np.inf, 'eps': None, 'x': None, 'obj': None}
    for eps in eps_grid_I_3d:
        try:
            x_ipm, obj_ipm = IPM_DRO_Sec_3D(eps, Lambda_fig5, Omega_fig5, U_fig5, h_fig5, b_fig5, xi_tr, samples, solver=SOLVER, verbose=False)
            val_ipm = oos_valid_3d(Lambda_fig5, h_fig5, b_fig5, xi_val, x_ipm)
            if np.isfinite(val_ipm) and val_ipm < best_ipm['val']:
                best_ipm.update(val=float(val_ipm), eps=float(eps), x=x_ipm, obj=float(obj_ipm))
        except Exception as exc:
            row["errors"].append(str(exc))
            continue

    if best_ipm['x'] is not None:
        row['x_IPM'] = best_ipm['x'].tolist()
        row['obj_IPM'] = best_ipm['obj']
        row['eps_IPM'] = best_ipm['eps']
        row['val_IPM'] = best_ipm['val']
        try:
            row['oos_IPM'] = float(oos_performance_3d(Lambda_fig5, h_fig5, b_fig5, xi_oos, best_ipm['x']))
        except Exception as exc:
            row["errors"].append(str(exc))

    row["errors"] = "; ".join(dict.fromkeys(row["errors"]))
    return row


def safe_mean(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else np.nan


def safe_pct(arr, q):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size > 0 else np.nan



def _configure_worker(settings):
    # Spawned workers receive the same parameters and solver as the parent.
    globals().update(settings)


def _worker_settings():
    names = ["BASE", "h_fig5", "b_fig5", "mu_fig5", "sigma_fig5", "size_oos_fig5",
             "Lambda_fig5", "U_fig5", "split_rate_fig5", "n_trials_fig5", "N_samp_fig5",
             "eps_grid_I_3d", "N_list_I_3d", "SOLVER"]
    return {name: globals()[name] for name in names}


def main(argv=None):
    global results_dir_3d, save_dir_eps_3d, max_workers_eps_3d
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the paper Figure 2 Lognormal newsvendor experiment.")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("IPM_DRO_WORKERS", "1")))
    parser.add_argument("--solver", default=DEFAULT_SOLVER,
                        help="CVXPY conic solver; default CLARABEL or IPM_DRO_SOLVER.")
    parser.add_argument("--output-dir", type=Path,
                        help="Override this distribution's output directory.")
    parser.add_argument("--quick", action="store_true",
                        help="Small execution check; does not reproduce the paper.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    solver_name = args.solver.upper()
    if solver_name not in cp.installed_solvers():
        parser.error(f"Solver {solver_name} is not installed: {cp.installed_solvers()}")
    globals()["SOLVER"] = solver_name
    globals()["max_workers_eps_3d"] = args.workers
    if args.output_dir:
        globals()["results_dir_3d"] = str(args.output_dir.resolve())
        globals()["save_dir_eps_3d"] = str(args.output_dir.resolve() / "eps_tables")
    if args.quick:
        globals().update(n_trials_fig5=1, N_list_I_3d=[10], N_samp_fig5=8,
                         size_oos_fig5=100,
                         eps_grid_I_3d=np.array([0.01], dtype=float))
        if args.output_dir is None:
            globals()["results_dir_3d"] = str(RESULTS_ROOT / "lognormal" / "quick")
            globals()["save_dir_eps_3d"] = str(Path(results_dir_3d) / "eps_tables")
    print(f"Solver={SOLVER}; workers={max_workers_eps_3d}; trials={n_trials_fig5}; "
          f"N={N_list_I_3d}; MC samples={N_samp_fig5}")

    os.makedirs(save_dir_eps_3d, exist_ok=True)

    selected_all_path_3d = os.path.join(save_dir_eps_3d, 'NV_3D_selected_trials.csv')
    selected_trial_failures_3d = 0
    selected_trial_error_examples_3d = []
    expected_selected_cols_3d = [
        'size', 'trial',
        'x_SAA', 'obj_SAA', 'val_SAA', 'oos_SAA',
        'x_Wass', 'obj_Wass', 'eps_Wass', 'val_Wass', 'oos_Wass',
        'x_IPM', 'obj_IPM', 'eps_IPM', 'val_IPM', 'oos_IPM', 'errors',
    ]

    if os.path.exists(selected_all_path_3d):
        existing_selected_df_3d = pd.read_csv(selected_all_path_3d)
        existing_selected_df_3d = existing_selected_df_3d.reindex(columns=expected_selected_cols_3d)
        existing_selected_df_3d = existing_selected_df_3d.drop_duplicates(subset=['size', 'trial'], keep='last')
        existing_selected_df_3d = existing_selected_df_3d.sort_values(['size', 'trial']).reset_index(drop=True)
        print(f'Loaded existing selected trial table: {selected_all_path_3d} ({len(existing_selected_df_3d)} rows)')
    else:
        existing_selected_df_3d = pd.DataFrame(columns=expected_selected_cols_3d)
        print('No existing selected trial table found; starting from scratch.')

    completed_tasks_selected_3d = {
        (int(size_tr), int(trial_id))
        for size_tr, trial_id in existing_selected_df_3d.dropna(
            subset=['oos_SAA', 'oos_Wass', 'oos_IPM'])[['size', 'trial']].itertuples(index=False, name=None)
    }

    all_tasks_selected_3d = [
        (size_tr, trial_id)
        for size_tr in N_list_I_3d
        for trial_id in range(n_trials_fig5)
        if (size_tr, trial_id) not in completed_tasks_selected_3d
    ]

    rows_selected_3d = existing_selected_df_3d.to_dict('records')
    max_workers_eps_3d = globals().get("max_workers_eps_3d", max(1, int(os.environ.get("IPM_DRO_WORKERS", "1"))))
    print(f'Submitting {len(all_tasks_selected_3d)} new selected-trial tasks with max_workers={max_workers_eps_3d} ...')

    if all_tasks_selected_3d:
        if max_workers_eps_3d == 1:
            for i, task in enumerate(all_tasks_selected_3d, 1):
                try:
                    rows_selected_3d.append(run_one_selected_trial_3d(task))
                except Exception as exc:
                    selected_trial_failures_3d += 1
                    if len(selected_trial_error_examples_3d) < 5:
                        selected_trial_error_examples_3d.append(repr(exc))
                if i % 50 == 0 or i == len(all_tasks_selected_3d):
                    print(f'  progress: {i}/{len(all_tasks_selected_3d)}')
        else:
            with ProcessPoolExecutor(max_workers=max_workers_eps_3d,
                                     mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_configure_worker,
                                     initargs=(_worker_settings(),)) as ex:
                futs = [ex.submit(run_one_selected_trial_3d, task) for task in all_tasks_selected_3d]
                for i, fut in enumerate(as_completed(futs), 1):
                    try:
                        rows_selected_3d.append(fut.result())
                    except Exception as exc:
                        selected_trial_failures_3d += 1
                        if len(selected_trial_error_examples_3d) < 5:
                            selected_trial_error_examples_3d.append(repr(exc))
                    if i % 50 == 0 or i == len(all_tasks_selected_3d):
                        print(f'  progress: {i}/{len(all_tasks_selected_3d)}')
    else:
        print('No new selected-trial tasks to run.')

    selected_trials_df_3d = pd.DataFrame(rows_selected_3d, columns=expected_selected_cols_3d)
    selected_trials_df_3d = selected_trials_df_3d.drop_duplicates(subset=['size', 'trial'], keep='last')
    selected_trials_df_3d = selected_trials_df_3d.sort_values(['size', 'trial']).reset_index(drop=True)
    if selected_trial_failures_3d:
        print(f'Selected-trial tasks failed: {selected_trial_failures_3d}/{len(all_tasks_selected_3d)}')
        for err in selected_trial_error_examples_3d:
            print(f'  example error: {err}')
    if selected_trials_df_3d.empty:
        print('No selected-trial rows were produced; saved tables will be empty until upstream task failures are fixed.')
    selected_trials_df_3d.to_csv(selected_all_path_3d, index=False)
    print(f'Saved selected trial table: {selected_all_path_3d}')

    summary_rows_eps_3d = []
    for size_tr in N_list_I_3d:
        dfN = selected_trials_df_3d[selected_trials_df_3d['size'] == size_tr].copy()
        per_n_path = os.path.join(save_dir_eps_3d, f'NV_3D_N{size_tr}_selected_trials.csv')
        dfN.to_csv(per_n_path, index=False)
        print(f'Saved: {per_n_path}')

        summary_rows_eps_3d.append({
            'size': int(size_tr),
            'n_success_SAA': int(dfN['oos_SAA'].notna().sum()),
            'n_success_Wass': int(dfN['oos_Wass'].notna().sum()),
            'n_success_IPM': int(dfN['oos_IPM'].notna().sum()),
            'mode_eps_Wass': dfN['eps_Wass'].dropna().mode().iloc[0] if dfN['eps_Wass'].dropna().size else np.nan,
            'mode_eps_IPM': dfN['eps_IPM'].dropna().mode().iloc[0] if dfN['eps_IPM'].dropna().size else np.nan,
            'mean_oos_SAA': safe_mean(dfN['oos_SAA'].values),
            'q10_oos_SAA': safe_pct(dfN['oos_SAA'].values, 10),
            'q90_oos_SAA': safe_pct(dfN['oos_SAA'].values, 90),
            'mean_oos_Wass': safe_mean(dfN['oos_Wass'].values),
            'q10_oos_Wass': safe_pct(dfN['oos_Wass'].values, 10),
            'q90_oos_Wass': safe_pct(dfN['oos_Wass'].values, 90),
            'mean_oos_IPM': safe_mean(dfN['oos_IPM'].values),
            'q10_oos_IPM': safe_pct(dfN['oos_IPM'].values, 10),
            'q90_oos_IPM': safe_pct(dfN['oos_IPM'].values, 90),
        })

    summary_eps_df_3d = pd.DataFrame(summary_rows_eps_3d).sort_values('size').reset_index(drop=True)
    summary_eps_path_3d = os.path.join(save_dir_eps_3d, 'NV_3D_selected_summary.csv')
    summary_eps_df_3d.to_csv(summary_eps_path_3d, index=False)
    print(f'Saved summary: {summary_eps_path_3d}')
    print(summary_eps_df_3d.to_string(index=False))

    if selected_trials_df_3d.empty:
        raise RuntimeError("No trials completed; inspect the reported task errors.")
    if any(summary_eps_df_3d[f"n_success_{m}"].eq(0).any() for m in ("SAA", "Wass", "IPM")):
        raise RuntimeError("A method has no successful trials for a sample size. Inspect the errors column.")


    save_dir_eps_3d = os.path.join(results_dir_3d, 'eps_tables')

    # Plot OOS by N from the saved 3D selected-summary table
    summary_eps_path_3d = os.path.join(results_dir_3d, 'eps_tables', 'NV_3D_selected_summary.csv')
    plot_df_3d = pd.read_csv(summary_eps_path_3d)

    summ = plot_df_3d.sort_values('size').rename(columns={'size': 'N'}).reset_index(drop=True)
    print('===== Mean OOS by N (validation-selected epsilon per trial) =====')
    print(summ.to_string(index=False))
    best_lines = [
        f"N={int(n)}: {str(c).replace('mean_oos_', '')} ({v:.6g})"
        for n, c, v in zip(
            summ['N'],
            summ.filter(like='mean_oos_').idxmin(axis=1),
            summ.filter(like='mean_oos_').min(axis=1),
        )
    ]
    print('\n'.join(best_lines))

    summary_path = os.path.join(results_dir_3d, 'eps_tables', 'NV_3D_oos_summary_by_N_comparison.csv')
    summ.to_csv(summary_path, index=False)
    print()
    print(f'Saved summary CSV to: {summary_path}')

    plot_path = os.path.join(results_dir_3d, 'eps_tables', 'NV_3D_OOS_plot_by_N_lognormal.pdf')

    plt.figure(figsize=(7, 5))
    legend_labels = {'SAA': 'SAA', 'Wass': '2-WDRO', 'IPM': 'TIPM-DRO'}
    colors = {'SAA': '#1f77b4', 'Wass': '#4EC65C', 'IPM': '#DAA520'}
    for method, marker in zip(['SAA', 'Wass', 'IPM'], ['o', 's', '^']):
        mean = summ[f'mean_oos_{method}']
        q10 = summ[f'q10_oos_{method}']
        q90 = summ[f'q90_oos_{method}']

        plt.plot(summ['N'], mean, marker=marker, label=legend_labels[method], color=colors[method])
        plt.fill_between(summ['N'], q10, q90, alpha=0.18, color=colors[method])

    plt.xlabel('Training Sample Size $N$', fontsize=16)
    plt.ylabel('OOS', fontsize=16)
    #plt.title('OOS vs Training Size with 10-90% Quantile', fontsize=16)
    plt.tick_params(axis='both', labelsize=13)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved plot to: {plot_path}')


if __name__ == "__main__":
    main()
