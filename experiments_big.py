# experiments_big.py
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import time
import numpy as np
import csv
import os

from data_generation import (
    generate_graph,
    precision_from_adjacency_laplacian,
    sample_gaussian_from_precision,
    sample_stationary_signals,
    laplacian_from_adjacency,
)

from methods import (
    sample_cov_centered,
    estimate_theta_ridge_from_cov,
    estimate_theta_glasso_sklearn_solver_time,
    estimate_theta_glasso_cvxpy_from_cov,
    estimate_theta_pgd_from_cov,
)

from metrics import (
    fro_error_rel_offdiag,
    fro_error_rel_full,
    sparsity_offdiag,
    theta_to_laplacian,
)


def save_results_csv(rows: list[dict], filename: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_case(
    method: str,
    seed: int,
    N: int,
    M: int,
    avg_degree: int,
    signal_type: str,
    beta_ws: float = 0.1,
    # gaussian params
    alpha_lap: float = 1.0,
    eps: float = 0.1,
    # stationary filter params
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0,
    # inference params
    lam: float = 0.05,
    gamma_ridge: float = 1e-2,
    cvxpy_solver: str = "SCS",
    thr_theta_to_L: float = 1e-4,
):
    # 1) Graph: ER only
    graph_type = "erdos_renyi"
    A_true = generate_graph(
        N=N,
        avg_degree=avg_degree,
        graph_type=graph_type,
        seed=seed,
        beta_ws=beta_ws,   # not used for ER, but harmless
    )

    # 2) Generate data X + truth for metric
    if signal_type == "gaussian":
        Theta_true = precision_from_adjacency_laplacian(A_true, alpha_lap=alpha_lap, eps=eps)
        X = sample_gaussian_from_precision(Theta_true, M=M, seed=seed + 123)
        L_true = None
    elif signal_type == "stationary":
        X = sample_stationary_signals(A_true, M=M, seed=seed + 123, h0=h0, h1=h1, h2=h2)
        L_true = laplacian_from_adjacency(A_true)
        Theta_true = None
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    # 3) Precompute covariance (outside solver timing)
    S = sample_cov_centered(X)

    # 4) Solver-only timing + estimate Theta_hat
    t0 = time.perf_counter()

    if method == "ridge":
        Theta_hat = estimate_theta_ridge_from_cov(S, gamma=gamma_ridge)
    elif method == "glasso_skl":
        Theta_hat = estimate_theta_glasso_sklearn_solver_time(X, lam=lam)
    elif method == "glasso_cvx":
        Theta_hat = estimate_theta_glasso_cvxpy_from_cov(S, lam=lam, solver=cvxpy_solver)
    elif method == "pgd":
        Theta_hat = estimate_theta_pgd_from_cov(S, lam=lam)
    else:
        raise ValueError(f"Unknown method: {method}")

    dt = time.perf_counter() - t0

    # 5) Metric depends on signal type
    if signal_type == "gaussian":
        err = fro_error_rel_offdiag(Theta_hat, Theta_true)
        err_label = "Fro(Theta_offdiag)"
    else:
        L_hat = theta_to_laplacian(Theta_hat, thr=thr_theta_to_L)
        err = fro_error_rel_full(L_hat, L_true)
        err_label = "Fro(L_full)"

    return {
        "graph_type": graph_type,
        "signal_type": signal_type,
        "method": method,
        "N": int(N),
        "M": int(M),
        "avg_degree": int(avg_degree),
        "lam": float(lam),
        "err_label": err_label,
        "err": float(err),
        "sparsity_theta": float(sparsity_offdiag(Theta_hat)),
        "solver_time_ms": float(dt * 1000.0),
    }


def print_results(title: str, results: list[dict]):
    print(title)
    for r in results:
        print(
            f"{r['method']:10s} | "
            f"{r['err_label']}={r['err']:.3f} | "
            f"sparsity(Θ̂)={r['sparsity_theta']:.3f} | "
            f"solver_time={r['solver_time_ms']:.2f} ms"
        )


if __name__ == "__main__":
    # ----------------------------
    # BIG CASE requested
    # ----------------------------
    N = 100
    M = 500
    avg_degree = 6
    seed = 0

    lam = 0.05
    cvxpy_solver = "SCS"

    signal_types = ["gaussian", "stationary"]
    methods = ["glasso_skl", "pgd", "glasso_cvx"]  # añade "ridge" si quieres baseline

    all_rows = []

    for stype in signal_types:
        results = [
            run_case(
                method=meth,
                seed=seed,
                N=N,
                M=M,
                avg_degree=avg_degree,
                signal_type=stype,
                lam=lam,
                cvxpy_solver=cvxpy_solver,
            )
            for meth in methods
        ]

        print_results(
            f"\n=== BIG RUN | ER | N={N} | M={M} | avg_degree={avg_degree} | signal={stype} | lam={lam} ===",
            results,
        )

        all_rows.extend(results)

    save_results_csv(all_rows, "results_csv/big_ER_N100_M500.csv")
    print("\nSaved CSV -> results_csv/big_ER_N100_M500.csv")
