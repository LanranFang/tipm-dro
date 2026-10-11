"""Reproduce the companion convergence notebook.

The complete experiment runs only when main() is called or this script is
executed directly. Use --quick for a small smoke experiment saved separately
in results/convergence/quick/. See docs/REPRODUCING.md.
"""


def main(*, quick=False, workers=None):
    if workers is not None:
        if workers < 1:
            raise ValueError("workers must be a positive integer.")
        import os
        os.environ["IPM_DRO_WORKERS"] = str(workers)

    # Notebook cell 3
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import cvxpy as cp
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd

    plt.rcParams.update({
        "figure.figsize": (7.4, 4.8),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    })

    cp.installed_solvers()
    from pathlib import Path

    def find_repository_root():
        """Find the repository when run from its root or a subdirectory."""
        start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        for candidate in (start, *start.parents):
            if (candidate / "convergence" / "suboptimal_dual_gap_vs_M_dimension.ipynb").is_file():
                return candidate
        raise RuntimeError("Run from the repository root or one of its subdirectories.")

    REPO_ROOT = find_repository_root()
    RESULTS_DIR = REPO_ROOT / "results" / "convergence"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Notebook cell 4
    # Experiment knobs.
    DIMENSIONS = [5, 25, 100]
    M_VALUES = np.array([2, 5, 10, 20, 50, 100, 250, 500])
    REPS = 50

    # Must be even because P_hat is built from symmetric pairs xi and -xi.
    N_HIST = 100

    EPSILON = 0.0
    UNIT_BALL_RADIUS = 1.0
    X_STAR_LEVEL = 0.1
    N_LOSS_DIRECTIONS = 4
    LOSS_WEIGHT_LOW = 0.5
    LOSS_WEIGHT_HIGH = 1.5
    RNG_SEED = 20260519

    # Parallelize over independent (dimension, replication) tasks.
    # The Section 3.2 SOCP has O(J^2 M) lambda variables, so avoid oversubscribing memory.
    PARALLEL = True
    MAX_WORKERS = int(os.environ.get("IPM_DRO_WORKERS", "4"))
    if MAX_WORKERS < 1:
        raise ValueError("IPM_DRO_WORKERS must be a positive integer.")

    # ECOS is fast for this LP/SOCP family; CLARABEL and SCS are fallbacks.
    SOLVER_SEQUENCE = [s for s in ["ECOS", "CLARABEL", "SCS"] if s in cp.installed_solvers()]
    if not SOLVER_SEQUENCE:
        raise RuntimeError("No supported conic solver is installed. Install ECOS, CLARABEL, or SCS.")
    SOLVER_SEQUENCE, MAX_WORKERS

    # Optional smoke experiment; full-run numerical settings above are preserved.
    if quick:
        DIMENSIONS = [5]
        M_VALUES = np.array([2, 5])
        REPS = 1
        N_HIST = 10
        RESULTS_DIR = RESULTS_DIR / "quick"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)


    # Notebook cell 5
    def make_x_star(d):
        x_star = X_STAR_LEVEL * np.ones(d)
        if np.linalg.norm(x_star) > UNIT_BALL_RADIUS + 1e-12:
            raise ValueError("x_star is outside the Euclidean unit ball; reduce X_STAR_LEVEL or dimensions.")
        return x_star

    def make_loss_model(d):
        rng = np.random.default_rng(RNG_SEED + 31_415 + d)
        directions = rng.normal(size=(N_LOSS_DIRECTIONS, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        weights = np.linspace(LOSS_WEIGHT_LOW, LOSS_WEIGHT_HIGH, N_LOSS_DIRECTIONS)
        weights = weights / weights.mean()
        signed_directions = np.vstack([weights[:, None] * directions, -weights[:, None] * directions])
        return {
            "directions": directions,
            "direction_weights": weights,
            "pieces": signed_directions,
        }

    def sample_unit_ball(n, d, rng):
        directions = rng.normal(size=(n, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng.random(size=(n, 1)) ** (1.0 / d)
        return UNIT_BALL_RADIUS * radii * directions

    def make_symmetric_phat(n_hist, d, rng):
        assert n_hist % 2 == 0
        half = sample_unit_ball(n_hist // 2, d, rng)
        return np.vstack([half, -half])

    def sample_decisions(n, d, rng):
        return sample_unit_ball(n, d, rng)

    def loss_matrix(Z, Xi, x_star, loss_model):
        """Matrix with entries loss(z_m, xi_k) for the finite max-affine loss."""
        pieces = loss_model["pieces"]
        centered_Z = Z - x_star[None, :]
        scores = np.einsum("mnd,jd->mnj", centered_Z[:, None, :] - Xi[None, :, :], pieces)
        return np.max(scores, axis=2)

    def empirical_objective(x, Xi, x_star, loss_model):
        x = np.asarray(x, dtype=float)
        pieces = loss_model["pieces"]
        scores = ((x - x_star)[None, :] - Xi) @ pieces.T
        return float(np.max(scores, axis=1).mean())

    def omega_for_unit_ball_support():
        # Every xi in the Euclidean unit ball satisfies ||xi||_2^2 <= 1.
        return float(UNIT_BALL_RADIUS ** 2)

    # Notebook cell 6
    def solve_problem(prob):
        last_error = None
        inaccurate_solver = None
        for solver in SOLVER_SEQUENCE:
            try:
                if solver == "ECOS":
                    prob.solve(solver=solver, verbose=False, abstol=1e-7, reltol=1e-7, feastol=1e-7, max_iters=2000)
                elif solver == "CLARABEL":
                    prob.solve(solver=solver, verbose=False, tol_gap_abs=1e-7, tol_feas=1e-7, max_iter=2000)
                elif solver == "SCS":
                    prob.solve(solver=solver, verbose=False, eps=2e-4, max_iters=12000)
                if prob.status == cp.OPTIMAL:
                    return solver
                if prob.status == cp.OPTIMAL_INACCURATE and inaccurate_solver is None:
                    inaccurate_solver = solver
            except Exception as exc:
                last_error = exc
        if inaccurate_solver is not None:
            solver = inaccurate_solver
            if solver == "ECOS":
                prob.solve(solver=solver, verbose=False, abstol=1e-7, reltol=1e-7, feastol=1e-7, max_iters=2000)
            elif solver == "CLARABEL":
                prob.solve(solver=solver, verbose=False, tol_gap_abs=1e-7, tol_feas=1e-7, max_iter=2000)
            elif solver == "SCS":
                prob.solve(solver=solver, verbose=False, eps=2e-4, max_iters=12000)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return solver
            raise RuntimeError(f"Fallback solver {solver} failed on retry: status={prob.status}")
        raise RuntimeError(f"No solver produced an optimal solution. status={prob.status}; last_error={last_error}")

    def solve_sampled_dual(Z, Xi, x_star, loss_model):
        """Solve the Section 3.2 sampled dual SOCP and return x_M."""
        M, d = Z.shape
        pieces = loss_model["pieces"]
        J = pieces.shape[0]
        a_piece = -pieces
        b_z = (Z - x_star[None, :]) @ pieces.T

        L_z_xi = loss_matrix(Z, Xi, x_star, loss_model)
        e_phat_z = L_z_xi.mean(axis=1)
        omega = omega_for_unit_ball_support()

        alpha = cp.Variable(name="alpha")
        moment_beta = cp.Variable(nonneg=True, name="moment_beta")
        nu = cp.Variable(M, nonneg=True, name="nu")
        x = cp.Variable(d, name="x")
        lam = cp.Variable((J * M, J), nonneg=True, name="lambda")
        theta = cp.Variable((J, d), name="theta")
        zeta = cp.Variable(J, nonneg=True, name="zeta")

        constraints = [cp.norm2(x) <= UNIT_BALL_RADIUS]
        b_x = pieces @ (x - x_star)

        for j in range(J):
            lam_j = lam[j * M : (j + 1) * M, :]
            constraints.append(cp.sum(lam_j, axis=1) == nu)

            avg_lambda_b = cp.sum(cp.multiply(lam_j, b_z)) / M
            avg_lambda_a = cp.sum(lam_j @ a_piece, axis=0) / M

            constraints.append(zeta[j] + UNIT_BALL_RADIUS * cp.norm2(theta[j, :]) + b_x[j] - avg_lambda_b <= alpha)
            soc_vector = cp.hstack([a_piece[j] - avg_lambda_a - theta[j, :], zeta[j] - moment_beta])
            constraints.append(cp.SOC(zeta[j] + moment_beta, soc_vector))

        objective = cp.Minimize(alpha + omega * moment_beta + ((e_phat_z + EPSILON) @ nu) / M)
        prob = cp.Problem(objective, constraints)
        solver = solve_problem(prob)
        if x.value is None or prob.value is None or not np.isfinite(prob.value):
            raise RuntimeError(f"Solver {solver} returned no finite solution: status={prob.status}")

        return {
            "value": float(prob.value),
            "status": prob.status,
            "solver": solver,
            "x": np.asarray(x.value, dtype=float),
            "alpha": float(alpha.value),
            "moment_beta": float(moment_beta.value),
            "omega": omega,
            "num_pieces": J,
        }

    # Notebook cell 8
    def run_one_replication(d, rep, seed=RNG_SEED):
        rng = np.random.default_rng(seed + 10_000 * rep + 997 * d)
        Xi = make_symmetric_phat(N_HIST, d, rng)
        x_star = make_x_star(d)
        loss_model = make_loss_model(d)
        Z_pool = sample_decisions(int(np.max(M_VALUES)), d, rng)

        obj_star = empirical_objective(x_star, Xi, x_star, loss_model)
        rows = []
        t0 = time.time()

        for M in M_VALUES:
            Z_M = Z_pool[: int(M)]
            sol = solve_sampled_dual(Z_M, Xi, x_star, loss_model)
            x_M = sol["x"]
            obj_xM = empirical_objective(x_M, Xi, x_star, loss_model)
            signed_gap = obj_xM - obj_star
            gap = max(signed_gap, 0.0)
            relative_gap = gap / max(obj_star, 1e-12)
            rows.append({
                "dimension": d,
                "rep": rep,
                "M": int(M),
                "epsilon": EPSILON,
                "gap_exact": gap,
                "signed_gap_exact": signed_gap,
                "relative_gap_exact": relative_gap,
                "percentage_gap_exact": 100.0 * relative_gap,
                "empirical_obj_xM": obj_xM,
                "empirical_obj_xstar": obj_star,
                "weight_min": float(loss_model["direction_weights"].min()),
                "weight_max": float(loss_model["direction_weights"].max()),
                "weight_mean": float(loss_model["direction_weights"].mean()),
                "num_pieces": int(sol["num_pieces"]),
                "sampled_dual_obj": sol["value"],
                "x_norm_M": float(np.linalg.norm(x_M)),
                "x_norm_star": float(np.linalg.norm(x_star)),
                "x_distance_to_star": float(np.linalg.norm(x_M - x_star)),
                "omega": sol["omega"],
                "solver": sol["solver"],
                "status": sol["status"],
                "elapsed_so_far_sec": time.time() - t0,
            })
        return rows

    tasks = [(d, rep) for d in DIMENSIONS for rep in range(REPS)]
    all_rows = []

    if PARALLEL and len(tasks) > 1:
        print(f"Running {len(tasks)} tasks with {MAX_WORKERS} workers ...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(run_one_replication, d, rep): (d, rep) for d, rep in tasks}
            for future in as_completed(future_to_task):
                d, rep = future_to_task[future]
                rows = future.result()
                all_rows.extend(rows)
                elapsed = rows[-1]["elapsed_so_far_sec"] if rows else float("nan")
                print(f"finished d={d}, rep={rep} in {elapsed:.1f}s")
    else:
        for d, rep in tasks:
            print(f"running d={d}, rep={rep} ...")
            all_rows.extend(run_one_replication(d, rep))

    results = pd.DataFrame(all_rows).sort_values(["dimension", "rep", "M"]).reset_index(drop=True)
    results.to_csv(RESULTS_DIR / "suboptimal_known_opt_gap_results.csv", index=False)
    results.head()

    # Notebook cell 9
    summary = (
        results.groupby(["dimension", "M"], as_index=False)
        .agg(
            mean_gap=("gap_exact", "mean"),
            mean_pct_gap=("percentage_gap_exact", "mean"),
            median_pct_gap=("percentage_gap_exact", "median"),
            pct_q25=("percentage_gap_exact", lambda x: np.quantile(x, 0.25)),
            pct_q75=("percentage_gap_exact", lambda x: np.quantile(x, 0.75)),
            median_gap=("gap_exact", "median"),
            q25=("gap_exact", lambda x: np.quantile(x, 0.25)),
            q75=("gap_exact", lambda x: np.quantile(x, 0.75)),
            mean_x_dist=("x_distance_to_star", "mean"),
            mean_obj_star=("empirical_obj_xstar", "mean"),
            mean_signed_gap=("signed_gap_exact", "mean"),
            weight_min=("weight_min", "first"),
            weight_max=("weight_max", "first"),
            weight_mean=("weight_mean", "first"),
            num_pieces=("num_pieces", "first"),
        )
    )
    summary.to_csv(RESULTS_DIR / "suboptimal_known_opt_gap_summary.csv", index=False)
    summary

    # Notebook cell 10
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import cvxpy as cp
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd

    results_dir = str(RESULTS_DIR)
    save_dir = os.path.join(results_dir, "suboptimal_gap_figures")
    os.makedirs(save_dir, exist_ok=True)

    summary_path = os.path.join(results_dir, "suboptimal_known_opt_gap_summary.csv")
    summ = pd.read_csv(summary_path).sort_values(["dimension", "M"]).reset_index(drop=True)

    print("===== Suboptimality gap summary by dimension and M =====")
    print(summ.to_string(index=False))

    out_summary_path = os.path.join(save_dir, "suboptimal_gap_summary_for_plot.csv")
    summ.to_csv(out_summary_path, index=False)
    print()
    print(f"Saved summary CSV to: {out_summary_path}")


    def fit_inverse_sqrt(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = (x > 0) & (y > 0)
        x_fit = x[valid]
        y_fit = y[valid]

        # log y = log C - 0.5 log M
        c = float(np.exp(np.mean(np.log(y_fit) + 0.5 * np.log(x_fit))))
        y_hat = c / np.sqrt(x)

        log_rmse = float(
            np.sqrt(np.mean((np.log(y_fit) - np.log(c / np.sqrt(x_fit))) ** 2))
        )
        return c, y_hat, log_rmse


    plot_dimensions = sorted(summ["dimension"].unique())
    base_colors = ["#1f77b4", "#4EC65C", "#DAA520"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(plot_dimensions))]
    markers = ["o", "s", "^", "D", "P", "X"]

    m_min = float(summ["M"].min())
    m_max = float(summ["M"].max())
    m_grid = np.linspace(m_min, m_max, 300)

    # Notebook cell 11
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import os
    import numpy as np

    mpl.rcParams.update({
        "text.usetex": False,
    })

    plot_path_pct = os.path.join(
        save_dir, "suboptimal_percentage_gap_vs_M_by_dim.pdf"
    )

    plt.figure(figsize=(7, 5))

    for color, marker, d in zip(colors, markers, plot_dimensions):
        sub = summ[summ["dimension"] == d].sort_values("M")

        mean = np.maximum(sub["mean_pct_gap"].to_numpy(dtype=float), 1e-8)
        q_low = np.maximum(sub["pct_q25"].to_numpy(dtype=float), 1e-8)
        q_high = np.maximum(sub["pct_q75"].to_numpy(dtype=float), 1e-8)

        plt.plot(
            sub["M"],
            mean,
            marker=marker,
            linewidth=1.9,
            label=rf"$D_{{\mathbf{{x}}}}={int(d)}$",
            color=color,
        )
        plt.fill_between(sub["M"], q_low, q_high, alpha=0.18, color=color)

    fit_x = summ["M"].to_numpy(dtype=float)
    fit_y = np.maximum(summ["mean_pct_gap"].to_numpy(dtype=float), 1e-8)
    c_pct, _, rmse_pct = fit_inverse_sqrt(fit_x, fit_y)

    plt.plot(
        m_grid,
        c_pct / np.sqrt(m_grid),
        "k--",
        linewidth=1.35,
        label=rf"${c_pct:.2f}/\sqrt{{M}}\ \mathrm{{fit}}$",
    )

    plt.yscale("log")
    plt.xlabel(r"Monte Carlo sample size $M$", fontsize=16)
    plt.ylabel(r"Percentage optimality gap $(\%)$", fontsize=16)

    def format_percent_tick(y, _):
        if y <= 0:
            return ""
        if y >= 1:
            return rf"${y:g}\%$"
        if y >= 0.1:
            return rf"${y:.1f}\%$"
        if y >= 0.01:
            return rf"${y:.2f}\%$"
        if y >= 0.001:
            return rf"${y:.3f}\%$"
        return rf"${y:.4f}\%$"

    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(format_percent_tick))
    plt.xlim(0, m_max * 1.02)
    plt.ylim(bottom=0.1)
    plt.tick_params(axis="both", labelsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=15, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path_pct, dpi=300, bbox_inches="tight")
    plt.show()

    # Notebook cell 12
    # Absolute optimality gap plot

    plot_path_abs = os.path.join(
        save_dir, "suboptimal_absolute_gap_vs_M_by_dim.pdf"
    )

    plt.figure(figsize=(7, 5))

    for color, marker, d in zip(colors, markers, plot_dimensions):
        sub = summ[summ["dimension"] == d].sort_values("M")

        mean = np.maximum(sub["mean_gap"].to_numpy(dtype=float), 1e-10)
        q_low = np.maximum(sub["q25"].to_numpy(dtype=float), 1e-10)
        q_high = np.maximum(sub["q75"].to_numpy(dtype=float), 1e-10)

        plt.plot(
            sub["M"],
            mean,
            marker=marker,
            linewidth=1.9,
            label=f"d={int(d)}",
            color=color,
        )
        plt.fill_between(
            sub["M"],
            q_low,
            q_high,
            alpha=0.18,
            color=color,
        )

    fit_x = summ["M"].to_numpy(dtype=float)
    fit_y = np.maximum(summ["mean_gap"].to_numpy(dtype=float), 1e-10)
    c_abs, _, rmse_abs = fit_inverse_sqrt(fit_x, fit_y)

    plt.plot(
        m_grid,
        c_abs / np.sqrt(m_grid),
        "k--",
        linewidth=1.35,
        label=rf"{c_abs:.3f}/$\sqrt{{M}}$ fit",
    )

    plt.yscale("log")
    plt.xlabel("Monte Carlo sample size M", fontsize=16)
    plt.ylabel(
        r"$E_{\hat P}[\ell(x_M,\xi)]-E_{\hat P}[\ell(x_\star,\xi)]$",
        fontsize=16,
    )
    plt.title("Absolute optimality gap, known nonzero x_star", fontsize=16)

    plt.xlim(0, m_max * 1.02)
    plt.ylim(bottom=1e-3)
    plt.tick_params(axis="both", labelsize=13)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=15, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path_abs, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved absolute gap plot to: {plot_path_abs}")
    print(f"Absolute gap pooled C/sqrt(M): C={c_abs:.6g}, log-RMSE={rmse_abs:.4f}")


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run d=5, M=[2,5], one replication with 10 historical points.")
    parser.add_argument("--workers", type=int, help="Override IPM_DRO_WORKERS (default: 4).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(quick=args.quick, workers=args.workers)
