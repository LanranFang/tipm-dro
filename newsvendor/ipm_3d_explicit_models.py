"""Finite conic TIPM-DRO newsvendor models (paper Equation 41)."""

import os
import numpy as np
import cvxpy as cp
from itertools import product


DEFAULT_SOLVER = os.environ.get("IPM_DRO_SOLVER", "CLARABEL").upper()


def solve_and_check(prob, solver=DEFAULT_SOLVER, verbose=False):
    """Solve with the requested conic solver and reject unusable solutions."""
    if solver not in cp.installed_solvers():
        raise RuntimeError(f"Solver {solver} is not installed. Available: {cp.installed_solvers()}")
    prob.solve(solver=solver, verbose=verbose)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Optimization failed with status {prob.status!r} using {solver}.")
    if prob.value is None or not np.isfinite(prob.value):
        raise RuntimeError("Optimization returned a non-finite objective.")


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


def Empirical_loss_3d(h, b, y, xi):
    h = _as_1d_param(h, 3, "h")
    b = _as_1d_param(b, 3, "b")
    y = _as_2d_samples(y, 3, "y")
    xi = _as_2d_samples(xi, 3, "xi")

    diff = y[:, None, :] - xi[None, :, :]
    loss = np.maximum(h.reshape(1, 1, 3) * diff, 0.0) + np.maximum(b.reshape(1, 1, 3) * (-diff), 0.0)
    return loss.sum(axis=2).mean(axis=1)


def _piece_specs_3d(h, b):
    h = _as_1d_param(h, 3, "h")
    b = _as_1d_param(b, 3, "b")

    pieces = []
    for pattern in product((0, 1), repeat=3):
        mask = np.array(pattern, dtype=int)
        name = "".join("O" if bit == 0 else "U" for bit in mask)
        p = np.where(mask == 0, h, -b)
        g = np.where(mask == 0, -h, b)
        pieces.append(
            {
                "name": name,
                "p1": float(p[0]),
                "p2": float(p[1]),
                "p3": float(p[2]),
                "g1": float(g[0]),
                "g2": float(g[1]),
                "g3": float(g[2]),
                "use_t": False,
            }
        )

    pieces.append(
        {
            "name": "T",
            "p1": 0.0,
            "p2": 0.0,
            "p3": 0.0,
            "g1": 0.0,
            "g2": 0.0,
            "g3": 0.0,
            "use_t": True,
        }
    )
    return pieces


def _business_piece_arrays_3d(h, b):
    h = _as_1d_param(h, 3, "h")
    b = _as_1d_param(b, 3, "b")

    names = []
    p_mat = []
    g_mat = []
    for pattern in product((0, 1), repeat=3):
        mask = np.array(pattern, dtype=int)
        names.append("".join("O" if bit == 0 else "U" for bit in mask))
        p_mat.append(np.where(mask == 0, h, -b))
        g_mat.append(np.where(mask == 0, -h, b))

    return names, np.asarray(p_mat, dtype=float), np.asarray(g_mat, dtype=float)


def IPM_DRO_3D_explicit(eps, U, rho, h, b, xi, samples, solver=DEFAULT_SOLVER, verbose=True):
    h = _as_1d_param(h, 3, "h")
    b = _as_1d_param(b, 3, "b")
    U = _as_1d_param(U, 3, "U")
    xi = _as_2d_samples(xi, 3, "xi")
    y = _as_2d_samples(samples, 3, "samples")

    M = y.shape[0]
    basis_names, basis_p, basis_g = _business_piece_arrays_3d(h, b)
    target_specs = {piece["name"]: piece for piece in _piece_specs_3d(h, b)}
    basis_b_vals = y @ basis_p.T

    x = cp.Variable(3, nonneg=True)
    t = cp.Variable()
    alpha = cp.Variable()
    beta = cp.Variable(M, nonneg=True)

    const = [x <= U]
    for target_name in list(basis_names) + ["T"]:
        target = target_specs[target_name]
        target_p = np.array([target["p1"], target["p2"], target["p3"]], dtype=float)
        target_g = np.array([target["g1"], target["g2"], target["g3"]], dtype=float)
        affine_piece = target_p @ x + (t if target["use_t"] else 0.0)

        lam = cp.Variable((M, len(basis_names)), nonneg=True)
        dual_mass = cp.sum(lam, axis=0)
        affine_dual = (1.0 / M) * cp.sum(cp.multiply(lam, basis_b_vals))
        gradient_dual = (1.0 / M) * (basis_g.T @ dual_mass)

        const += [
            affine_piece - affine_dual <= alpha,
            target_g - gradient_dual <= 0,
            cp.sum(lam, axis=1) == beta,
        ]

    e_loss = Empirical_loss_3d(h, b, y, xi)
    obj = cp.Minimize((1 - 1 / rho) * t + (1 / rho) * (alpha + (beta @ (e_loss + eps)) / M))

    prob = cp.Problem(obj, const)
    solve_and_check(prob, solver=solver, verbose=verbose)
    return np.asarray(x.value).reshape(-1), float(prob.value)


def IPM_DRO_Sec_3D_explicit(eps, rho, Omega, U, h, b, xi, samples, solver=DEFAULT_SOLVER, verbose=True):
    h = _as_1d_param(h, 3, "h")
    b = _as_1d_param(b, 3, "b")
    U = _as_1d_param(U, 3, "U")
    xi = _as_2d_samples(xi, 3, "xi")
    y = _as_2d_samples(samples, 3, "samples")

    M = y.shape[0]
    basis_names, basis_p, basis_g = _business_piece_arrays_3d(h, b)
    target_specs = {piece["name"]: piece for piece in _piece_specs_3d(h, b)}
    basis_b_vals = y @ basis_p.T

    x = cp.Variable(3, nonneg=True)
    t = cp.Variable()
    alpha = cp.Variable()
    beta = cp.Variable(M, nonneg=True)
    gamma = cp.Variable(nonneg=True)

    const = [x <= U]
    for target_name in list(basis_names) + ["T"]:
        target = target_specs[target_name]
        target_p = np.array([target["p1"], target["p2"], target["p3"]], dtype=float)
        target_g = np.array([target["g1"], target["g2"], target["g3"]], dtype=float)
        affine_piece = target_p @ x + (t if target["use_t"] else 0.0)

        lam = cp.Variable((M, len(basis_names)), nonneg=True)
        eta = cp.Variable(nonneg=True)
        u = cp.Variable(3)

        dual_mass = cp.sum(lam, axis=0)
        affine_dual = (1.0 / M) * cp.sum(cp.multiply(lam, basis_b_vals))
        gradient_dual = (1.0 / M) * (basis_g.T @ dual_mass)

        const += [
            eta + affine_piece - affine_dual <= alpha,
            u <= 0,
            cp.sum(lam, axis=1) == beta,
        ]
        const.append(cp.SOC(eta + gamma, cp.hstack([target_g - gradient_dual - u, eta - gamma])))

    e_loss = Empirical_loss_3d(h, b, y, xi)
    obj = cp.Minimize((1 - 1 / rho) * t + (1 / rho) * (alpha + (beta @ (e_loss + eps)) / M + gamma * Omega))

    prob = cp.Problem(obj, const)
    solve_and_check(prob, solver=solver, verbose=verbose)
    return np.asarray(x.value).reshape(-1), float(prob.value)
