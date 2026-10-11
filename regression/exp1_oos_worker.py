"""
exp1_oos_worker.py
Solver functions and run_one_trial worker for exp1_oos.ipynb.
Regression with L1 loss; contamination = adversarial outlier samples + y-shift.
"""

import math
import os
import sys
import contextlib
import warnings
import numpy as np
import cvxpy as cp
from math import ceil, floor


@contextlib.contextmanager
def _suppress_stdout():
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        # Notebook kernels and captured output may not expose a file descriptor.
        yield
        return
    old = os.dup(fd)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, fd)
    os.close(null)
    try:
        yield
    finally:
        os.dup2(old, fd)
        os.close(old)


def _solve(prob, solver=cp.GUROBI):
    """Solve with an installed conic solver and report failure explicitly."""
    installed = set(cp.installed_solvers())
    errors = []
    for slv in dict.fromkeys([solver, cp.MOSEK, cp.CLARABEL, cp.SCS]):
        if slv not in installed:
            continue
        try:
            with _suppress_stdout():
                prob.solve(solver=slv, verbose=False)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return
            errors.append(f"{slv}: {prob.status}")
        except Exception as exc:
            errors.append(f"{slv}: {exc}")
    details = "; ".join(errors) or "no supported conic solver installed"
    raise RuntimeError(f"Regression optimization failed: {details}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def cheap_robust_mean(X, eps):
    """Coordinate-wise trimmed mean."""
    n = X.shape[0]
    X_sorted = np.sort(X, axis=0)
    start = ceil(eps * n)
    end   = floor((1 - eps) * n)
    if start >= end:
        return X.mean(axis=0)
    return X_sorted[start:end].mean(axis=0)


# ── Solvers ────────────────────────────────────────────────────────────────────

def lad_regression(X, y, theta_bound=10, solver=cp.GUROBI):
    """Plain LAD (no robustness, rho=0)."""
    n, d = X.shape
    Z     = np.hstack([X, y.reshape(-1, 1)])   # (n, d+1)
    theta = cp.Variable(d)
    s     = cp.Variable(n, nonneg=True)
    a     = cp.hstack([theta, np.array([-1.0])])
    prob  = cp.Problem(cp.Minimize(cp.mean(s)),
                       [s >= Z @ a, s >= -(Z @ a),
                        cp.norm(theta, 2) <= theta_bound])
    _solve(prob, solver)
    return theta.value if (theta.value is not None and not np.any(np.isnan(theta.value))) else None


def wdro_regression(X, y, rho, theta_bound=10, solver=cp.GUROBI):
    """Standard WDRO for regression (L1 loss, dual norm = 2)."""
    n, d = X.shape
    Z     = np.hstack([X, y.reshape(-1, 1)])
    theta = cp.Variable(d)
    s     = cp.Variable(n, nonneg=True)
    a     = cp.hstack([theta, np.array([-1.0])])
    prob  = cp.Problem(
        cp.Minimize(rho * cp.norm(a, 2) + cp.mean(s)),
        [s >= Z @ a, s >= -(Z @ a),
         cp.norm(theta, 2) <= theta_bound]
    )
    _solve(prob, solver)
    return theta.value if (theta.value is not None and not np.any(np.isnan(theta.value))) else None


def or_wdro_regression(X, y, sigma, rho, eps_tv, z_0, theta_bound=10, solver=cp.GUROBI):
    """
    OR-WDRO for regression (L1 loss, q=2, p=1).
    z_0 : robust center in augmented space R^{d+1} = [x; y].

    Two pieces (J=2):
      j=1 (s=+1): a_j = [theta; -1],  equality: [-theta; 1] + zG1 + zW1 = 0
      j=2 (s=-1): a_j = [-theta; 1],  equality: [theta; -1] + zG2 + zW2 = 0
    """
    n, d = X.shape
    D    = d + 1
    Z    = np.hstack([X, y.reshape(-1, 1)])   # (n, D)
    z_0  = z_0.ravel()                         # (D,)

    theta    = cp.Variable(d)
    lambda_1 = cp.Variable(nonneg=True)
    lambda_2 = cp.Variable(nonneg=True)
    alpha    = cp.Variable()
    s        = cp.Variable(n, nonneg=True)
    zG1      = cp.Variable((D, n))
    zW1      = cp.Variable((D, n))
    t1       = cp.Variable(n, nonneg=True)
    zG2      = cp.Variable((D, n))
    zW2      = cp.Variable((D, n))
    t2       = cp.Variable(n, nonneg=True)

    obj  = cp.Minimize(
        lambda_1 * sigma**2 + lambda_2 * rho
        + cp.sum(s) / (n * (1 - eps_tv)) + alpha
    )

    cons = []
    for i in range(n):
        g1i = zG1[:, i]; w1i = zW1[:, i]
        g2i = zG2[:, i]; w2i = zW2[:, i]

        # j=1 (s=+1): [-theta; 1] + zG1 + zW1 = 0
        cons.append(cp.hstack([-theta, np.array([1.0])]) + g1i + w1i == 0)
        cons.append(s[i] >= z_0 @ g1i + t1[i] + Z[i] @ w1i - alpha)
        cons.append(cp.SOC(
            lambda_1 + 0.5 * t1[i],
            cp.hstack([2 * g1i, cp.reshape(lambda_1 - 0.5 * t1[i], (1,), order='F')])
        ))
        cons.append(cp.norm(w1i, 2) <= lambda_2)

        # j=2 (s=-1): [theta; -1] + zG2 + zW2 = 0
        cons.append(cp.hstack([theta, np.array([-1.0])]) + g2i + w2i == 0)
        cons.append(s[i] >= z_0 @ g2i + t2[i] + Z[i] @ w2i - alpha)
        cons.append(cp.SOC(
            lambda_1 + 0.5 * t2[i],
            cp.hstack([2 * g2i, cp.reshape(lambda_1 - 0.5 * t2[i], (1,), order='F')])
        ))
        cons.append(cp.norm(w2i, 2) <= lambda_2)

    cons.append(cp.norm(theta, 2) <= theta_bound)
    prob = cp.Problem(obj, cons)
    _solve(prob, solver)
    return theta.value if (theta.value is not None and not np.any(np.isnan(theta.value))) else None


def mom_dro_regression(X, y, epsilon, K, M_samples, Omega, R=10, Z_fixed=None, theta_bound=10, solver=cp.GUROBI):
    """
    MoM IPM-DRO for scalar regression with L1 loss (k=1, J=2).

    Z_fixed : optional (M, d) array of pre-sampled test directions.
              If None, samples fresh from sphere of radius R.
              Pass a fixed Z so all CV evaluations with the same R use
              the same test functions, making CV comparisons stable.
    """
    n, d = X.shape
    M    = M_samples
    Xi   = np.hstack([X, y.reshape(-1, 1)])   # (n, d+1)

    if Z_fixed is not None:
        Z = Z_fixed                            # (M, d), already scaled to radius R
    else:
        rng   = np.random.default_rng()
        Z_raw = rng.standard_normal((M, d))
        Z_dir = Z_raw / np.linalg.norm(Z_raw, axis=1, keepdims=True)      # uniform on sphere
        r     = rng.uniform(0, 1, size=(M, 1)) ** (1 / d) * R             # radial CDF: P(r<t) = (t/R)^d
        Z     = Z_dir * r                                                  # (M, d), uniform in ball

    # MoM estimate: loss_i for z_m = |z_m @ x_i - y_i|
    am        = np.hstack([Z, -np.ones((M, 1))])   # (M, d+1): [z_m; -1]
    B         = n // K
    Xi_use    = Xi[:K * B]
    inner     = Xi_use @ am.T                        # (K*B, M)
    loss      = np.abs(inner)
    blk_means = loss.reshape(K, B, M).mean(axis=1)  # (K, M)
    mu_mom    = blk_means[0] if K == 1 else np.median(blk_means, axis=0)
    tau       = mu_mom + epsilon                     # (M,)

    theta = cp.Variable(d)
    alpha = cp.Variable()
    beta  = cp.Variable(nonneg=True)
    nu    = cp.Variable(M, nonneg=True)
    zeta1 = cp.Variable(nonneg=True)
    zeta2 = cp.Variable(nonneg=True)
    lam1p = cp.Variable(M, nonneg=True)
    lam1n = cp.Variable(M, nonneg=True)
    lam2p = cp.Variable(M, nonneg=True)
    lam2n = cp.Variable(M, nonneg=True)

    diff1 = lam1p - lam1n
    diff2 = lam2p - lam2n

    c1 = cp.hstack([
        theta - (1.0 / M) * (Z.T @ diff1),
        cp.reshape(-1.0 + (1.0 / M) * cp.sum(diff1), (1,), order='F')
    ])
    c2 = cp.hstack([
        -theta - (1.0 / M) * (Z.T @ diff2),
        cp.reshape(1.0 + (1.0 / M) * cp.sum(diff2), (1,), order='F')
    ])

    cons = [
        zeta1 <= alpha,
        zeta2 <= alpha,
        lam1p + lam1n == nu,
        lam2p + lam2n == nu,
        cp.SOC(zeta1 + beta, cp.hstack([c1, cp.reshape(zeta1 - beta, (1,), order='F')])),
        cp.SOC(zeta2 + beta, cp.hstack([c2, cp.reshape(zeta2 - beta, (1,), order='F')])),
        cp.norm(theta, 2) <= theta_bound,
    ]

    prob = cp.Problem(cp.Minimize(alpha + beta * Omega + (1.0 / M) * (nu @ tau)), cons)
    _solve(prob, solver)

    th = theta.value
    return th if (th is not None and not np.any(np.isnan(th))) else None


# ── Worker ─────────────────────────────────────────────────────────────────────

def run_one_trial(args):
    """
    Worker for ProcessPoolExecutor.
    args = (t, n, cfg)
    Returns dict with OOS excess risks.
    """
    os.environ.setdefault("OMP_NUM_THREADS",      "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS",      "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

    t, n, cfg = args

    d                   = cfg["d"]
    n_test              = cfg["n_test"]
    eps                 = cfg["eps"]
    contamination_scale = cfg["contamination_scale"]
    rho_shift           = cfg["rho_shift"]
    sigma               = cfg["sigma"]
    eps_tv              = cfg["eps_tv"]
    rho_list            = cfg["rho_list"]
    M_samples           = cfg["M_samples"]
    delta               = cfg["delta"]
    split_rate          = cfg["split_rate"]
    seed                = cfg.get("seed", 42)

    # Set deterministic seed for this trial so results are reproducible
    trial_seed = abs(hash((n, t, seed))) % (2**31)
    np.random.seed(trial_seed)

    try:
        # ── Generate theta_star and clean test set (seed = t) ──
        rng_t      = np.random.default_rng(int(t) % (2**31))
        theta_star = rng_t.standard_normal(d)
        theta_star /= np.linalg.norm(theta_star)
        X_test     = rng_t.standard_normal((n_test, d))
        y_test     = X_test @ theta_star

        # ── Generate contaminated training set (seed = hash(n,t)) ──
        rng_tr = np.random.default_rng(abs(hash((n, t))) % (2**31))
        X      = rng_tr.standard_normal((n, d))
        y      = X @ theta_star

        n_cont = floor(eps * n)
        X_cont = X.copy()
        y_cont = y.copy()
        X_cont[:n_cont] *= contamination_scale
        y_cont[:n_cont] *= -(contamination_scale**2)
        y_cont[:n_cont] += rho_shift

        perm   = rng_tr.permutation(n)
        X_cont = X_cont[perm]
        y_cont = y_cont[perm]

        # ── Holdout split ──
        n_tr  = floor(split_rate * n)
        perm2 = rng_tr.permutation(n)
        X_tr  = X_cont[perm2[:n_tr]]
        y_tr  = y_cont[perm2[:n_tr]]
        X_val = X_cont[perm2[n_tr:]]
        y_val = y_cont[perm2[n_tr:]]

        failed_methods = []

        def eval_test(th, method):
            if th is None or not np.all(np.isfinite(th)):
                failed_methods.append(method)
                warnings.warn(
                    f"{method} has no valid fit (n={n}, trial={t}, d={d}); risk recorded as NaN",
                    RuntimeWarning,
                )
                return float("nan")
            return float(np.mean(np.abs(X_test @ th - y_test)))

        # def eval_val(th):
        #     if th is None or np.any(np.isnan(th)):
        #         return np.inf
        #     return float(np.mean(np.abs(X_val @ th - y_val)))

        def eval_val_robust(th):
            """Trimmed-mean val loss: drop top eps fraction of worst residuals.
            This prevents contaminated val samples from dominating CV selection,
            so robust methods (large K) are not penalized by outliers in val set."""
            if th is None or np.any(np.isnan(th)):
                return np.inf
            losses = np.abs(X_val @ th - y_val)
            n_val  = len(losses)
            n_keep = max(1, n_val - ceil(eps_tv * n_val))
            return float(np.sort(losses)[:n_keep].mean())

        # ── Plain LAD ──
        theta_lad = lad_regression(X_tr, y_tr)
        risk_lad  = eval_test(theta_lad, "lad")

        # ── Standard WDRO CV ──
        best_val_std = np.inf
        theta_std    = None
        for rho in rho_list:
            try:
                th = wdro_regression(X_tr, y_tr, rho)
                if th is None:
                    continue
                v = eval_val_robust(th)
                if v < best_val_std:
                    best_val_std = v
                    theta_std    = th
            except Exception:
                continue
        risk_std = eval_test(theta_std, "std")

        # ── OR-WDRO CV (two contamination-parameter variants) ──
        Z_tr_aug     = np.hstack([X_tr, y_tr.reshape(-1, 1)])
        z_0_tr       = cheap_robust_mean(Z_tr_aug, 2 * eps_tv)

        # variant 1: eps_contam = 2*eps_tv  (more conservative)
        best_val_or_2eps = np.inf
        theta_or_2eps    = None
        for rho in rho_list:
            try:
                th = or_wdro_regression(X_tr, y_tr, sigma, rho, 2*eps_tv, z_0_tr)
                if th is None:
                    continue
                v = eval_val_robust(th)
                if v < best_val_or_2eps:
                    best_val_or_2eps = v
                    theta_or_2eps    = th
            except Exception:
                continue
        risk_or_2eps = eval_test(theta_or_2eps, "or_2eps")

        # variant 2: eps_contam = eps_tv  (less conservative)
        best_val_or_eps = np.inf
        theta_or_eps    = None
        for rho in rho_list:
            try:
                th = or_wdro_regression(X_tr, y_tr, sigma, rho, eps_tv, z_0_tr)
                if th is None:
                    continue
                v = eval_val_robust(th)
                if v < best_val_or_eps:
                    best_val_or_eps = v
                    theta_or_eps    = th
            except Exception:
                continue
        risk_or_eps = eval_test(theta_or_eps, "or_eps")

        # ── MoM IPM-DRO CV ──
        R_tb_list = [math.sqrt(d)]# R and theta_bound are always equal
        omega_k = eps
        K_theory = ceil(4 * (1 + 2*omega_k) / (1 - 2*omega_k)**2 * np.log(1.0 / delta))
        K_list   = sorted(set(filter(lambda k: 1 <= k <= n_tr, [
            K_theory,
            min(n_tr // 3, 2 * K_theory), n_tr // 2
        ])))
        if not K_list:
            K_list = [min(3, n_tr)]
        # Optional reduced grids are used only by the CLI smoke mode.
        K_list = cfg.get("K_list", K_list)

        z_rep       = np.ones(d + 1) / np.sqrt(d + 1)
        Xi_tr       = np.hstack([X_tr, y_tr.reshape(-1, 1)])
        sigma_loss  = np.std(np.abs(Xi_tr @ z_rep))
        Gamma_omega = 1.0 / (1.0 - 2.0 * eps_tv)
        eps_theory  = 4.0 * np.sqrt(np.e) * sigma_loss * Gamma_omega / np.sqrt(n_tr)
        eps_scales = cfg.get("epsilon_scales", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 10.0])
        eps_list = [max(eps_theory * scale, 1e-6) for scale in eps_scales]

        Omega_tr    = float(np.mean(np.sum(Xi_tr**2, axis=1)))
        omega_scale = [1.0]

        # Pre-generate one fixed Z per R so all (K, eps, omega) CV comparisons
        # with the same R use the same test functions → stable CV selection.
        # String hashes depend on PYTHONHASHSEED; SeedSequence is stable across runs.
        rng_z = np.random.default_rng(np.random.SeedSequence([seed, n, t, d, 1701]))
        Z_cache = {}
        for R in R_tb_list:
            Z_raw      = rng_z.standard_normal((M_samples, d))
            Z_dir      = Z_raw / np.linalg.norm(Z_raw, axis=1, keepdims=True)
            r          = rng_z.uniform(0, 1, size=(M_samples, 1)) ** (1 / d) * R
            Z_cache[R] = Z_dir * r

        best_val_mom         = np.inf
        theta_mom            = None
        best_K_val           = K_list[0]
        best_eps_val         = eps_list[0]
        best_omega_val       = omega_scale[0]
        best_R_val           = R_tb_list[0]
        best_theta_bound_val = R_tb_list[0]

        for R in R_tb_list:
            for K in K_list:
                for eps_m in eps_list:
                    for ws in omega_scale:
                        try:
                            th = mom_dro_regression(X_tr, y_tr, eps_m, K, M_samples,
                                                    ws * Omega_tr, R=R,
                                                    Z_fixed=Z_cache[R],
                                                    theta_bound=R)
                            if th is None or np.any(np.isnan(th)):
                                continue
                            v = eval_val_robust(th)
                            if v < best_val_mom:
                                best_val_mom         = v
                                theta_mom            = th
                                best_K_val           = K
                                best_eps_val         = eps_m
                                best_omega_val       = ws
                                best_R_val           = R
                                best_theta_bound_val = R
                        except Exception:
                            continue

        risk_mom = eval_test(theta_mom, "mom")

        print(
            f"  n={n} t={t}  lad={risk_lad:.4f}  std={risk_std:.4f}"
            f"  or(2eps)={risk_or_2eps:.4f}  or(eps)={risk_or_eps:.4f}  mom={risk_mom:.4f}"
            f"  (K={best_K_val} eps={best_eps_val:.3g} w={best_omega_val} R={best_R_val} tb={best_theta_bound_val})"
        )

        return {
            "n": n, "trial": t,
            "contamination_scale": contamination_scale,
            "risk_lad":     risk_lad,
            "risk_std":     risk_std,
            "risk_or_2eps": risk_or_2eps,
            "risk_or_eps":  risk_or_eps,
            "risk_mom":     risk_mom,
            "best_K":       best_K_val,
            "best_eps":     best_eps_val,
            "best_omega":   best_omega_val,
            "best_R":            best_R_val,
            "best_theta_bound":  best_theta_bound_val,
            "failed_methods": ";".join(failed_methods),
        }

    except Exception as exc:
        raise RuntimeError(f"Regression trial failed (n={n}, trial={t}, d={d}): {exc}") from exc
