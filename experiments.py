# experiments.py
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import time
import numpy as np

import csv
import os

def save_rows_to_csv(rows: list[dict], filename: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


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


def run_single(
    method: str,
    seed: int = 0,
    N: int = 20,
    M: int = 50,
    avg_degree: int = 4,
    graph_type: str = "erdos_renyi",
    beta_ws: float = 0.1,
    signal_type: str = "gaussian",   # "gaussian" or "stationary"
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0,
    alpha_lap: float = 1.0,
    eps: float = 0.1,
    lam: float = 0.05,
    gamma_ridge: float = 1e-2,
    cvxpy_solver: str = "SCS",
):
    # 1) Graph
    A_true = generate_graph(
        N=N,
        avg_degree=avg_degree,
        graph_type=graph_type,
        seed=seed,
        beta_ws=beta_ws,
    )

    # 2) Generate X and "truth" for metric
    if signal_type == "gaussian":
        Theta_true = precision_from_adjacency_laplacian(A_true, alpha_lap=alpha_lap, eps=eps)
        X = sample_gaussian_from_precision(Theta_true, M=M, seed=seed + 123)
        L_true = None  # not used

    elif signal_type == "stationary":
        X = sample_stationary_signals(A_true, M=M, seed=seed + 123, h0=h0, h1=h1, h2=h2)
        L_true = laplacian_from_adjacency(A_true)
        Theta_true = None  # not used

    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    # 3) Precompute covariance OUTSIDE solver timing
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

    # 5) Metrics depend on signal_type
    if signal_type == "gaussian":
        err = fro_error_rel_offdiag(Theta_hat, Theta_true)  # precision error (offdiag)
        err_key = "fro_theta"
    else:
        L_hat = theta_to_laplacian(Theta_hat, thr=1e-4)
        err = fro_error_rel_full(L_hat, L_true)            # Laplacian error (full)
        err_key = "fro_L"

    out = {
        "method": method,
        "graph_type": graph_type,
        "signal_type": signal_type,
        "sparsity_theta": sparsity_offdiag(Theta_hat),
        "time_ms": dt * 1000.0,
    }
    out[err_key] = err
    return out


def _extract_err(row: dict):
    if "fro_theta" in row:
        return "Fro(Θ)", row["fro_theta"]
    return "Fro(L)", row["fro_L"]


def sweep_over_M(
    methods,
    M_list,
    n_seeds=10,
    N=20,
    avg_degree=4,
    graph_type="erdos_renyi",
    beta_ws=0.1,
    signal_type="gaussian",
    h0=1.0, h1=-0.25, h2=0.0,
    alpha_lap=1.0,
    eps=0.1,
    lam=0.05,
    gamma_ridge=1e-2,
    cvxpy_solver="SCS",
):
    rows = []
    for M in M_list:
        for method in methods:
            err_list, sp_list, t_list = [], [], []
            for seed in range(n_seeds):
                out = run_single(
                    method=method,
                    seed=seed,
                    N=N,
                    M=M,
                    avg_degree=avg_degree,
                    graph_type=graph_type,
                    beta_ws=beta_ws,
                    signal_type=signal_type,
                    h0=h0, h1=h1, h2=h2,
                    alpha_lap=alpha_lap,
                    eps=eps,
                    lam=lam,
                    gamma_ridge=gamma_ridge,
                    cvxpy_solver=cvxpy_solver,
                )
                _, err_val = _extract_err(out)
                err_list.append(err_val)
                sp_list.append(out["sparsity_theta"])
                t_list.append(out["time_ms"])

            rows.append({
                "graph_type": graph_type,
                "signal_type": signal_type,
                "method": method,
                "M": M,
                "err_mean": float(np.mean(err_list)),
                "err_std": float(np.std(err_list)),
                "sp_mean": float(np.mean(sp_list)),
                "sp_std": float(np.std(sp_list)),
                "t_mean": float(np.mean(t_list)),
                "t_std": float(np.std(t_list)),
            })
    return rows


def sweep_over_lam(
    methods,
    lam_list,
    n_seeds=10,
    N=20,
    M=100,
    avg_degree=4,
    graph_type="erdos_renyi",
    beta_ws=0.1,
    signal_type="gaussian",
    h0=1.0, h1=-0.25, h2=0.0,
    alpha_lap=1.0,
    eps=0.1,
    gamma_ridge=1e-2,
    cvxpy_solver="SCS",
):
    rows = []
    for lam in lam_list:
        for method in methods:
            err_list, sp_list, t_list = [], [], []
            for seed in range(n_seeds):
                out = run_single(
                    method=method,
                    seed=seed,
                    N=N,
                    M=M,
                    avg_degree=avg_degree,
                    graph_type=graph_type,
                    beta_ws=beta_ws,
                    signal_type=signal_type,
                    h0=h0, h1=h1, h2=h2,
                    alpha_lap=alpha_lap,
                    eps=eps,
                    lam=lam,
                    gamma_ridge=gamma_ridge,
                    cvxpy_solver=cvxpy_solver,
                )
                _, err_val = _extract_err(out)
                err_list.append(err_val)
                sp_list.append(out["sparsity_theta"])
                t_list.append(out["time_ms"])

            rows.append({
                "graph_type": graph_type,
                "signal_type": signal_type,
                "method": method,
                "lam": lam,
                "err_mean": float(np.mean(err_list)),
                "err_std": float(np.std(err_list)),
                "sp_mean": float(np.mean(sp_list)),
                "sp_std": float(np.std(sp_list)),
                "t_mean": float(np.mean(t_list)),
                "t_std": float(np.std(t_list)),
            })
    return rows


def print_single(title: str, results):
    print(title)
    for r in results:
        lbl, err = _extract_err(r)
        print(
            f"{r['method']:10s} | "
            f"{lbl}={err:.3f} | "
            f"sparsity={r['sparsity_theta']:.3f} | "
            f"solver_time={r['time_ms']:.2f} ms"
        )


def print_sweep_M(title: str, rows):
    print(title)
    current_M = None
    for r in rows:
        if current_M != r["M"]:
            current_M = r["M"]
            print(f"\n--- M = {current_M} ---")
        print(
            f"{r['method']:10s} | "
            f"Err={r['err_mean']:.3f} ± {r['err_std']:.3f} | "
            f"sparsity={r['sp_mean']:.3f} ± {r['sp_std']:.3f} | "
            f"solver_time={r['t_mean']:.2f} ± {r['t_std']:.2f} ms"
        )


def print_sweep_lam(title: str, rows):
    print(title)
    current_lam = None
    for r in rows:
        if current_lam != r["lam"]:
            current_lam = r["lam"]
            print(f"\n--- lam = {current_lam} ---")
        print(
            f"{r['method']:10s} | "
            f"Err={r['err_mean']:.3f} ± {r['err_std']:.3f} | "
            f"sparsity={r['sp_mean']:.3f} ± {r['sp_std']:.3f} | "
            f"solver_time={r['t_mean']:.2f} ± {r['t_std']:.2f} ms"
        )


if __name__ == "__main__":
    N = 20
    avg_degree = 4
    graph_types = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
    beta_ws = 0.1

    signal_types = ["gaussian", "stationary"]

    alpha_lap = 1.0
    eps = 0.1

    h0, h1, h2 = 1.0, -0.25, 0.0

    lam = 0.05
    gamma_ridge = 1e-2
    cvxpy_solver = "SCS"

    # ✅ Single run: incluye ridge para demostrar que es malo
    methods_single = ["ridge", "glasso_skl", "glasso_cvx", "pgd"]

    # ✅ Sweeps: quitamos ridge
    methods_sweep_M = ["glasso_skl", "glasso_cvx", "pgd"]
    methods_sweep_lam = ["glasso_skl", "glasso_cvx", "pgd"]

    # 1) Single runs (con ridge)
    for gtype in graph_types:
        for stype in signal_types:
            single = [
                run_single(
                    method=m,
                    seed=0,
                    N=N,
                    M=50,
                    avg_degree=avg_degree,
                    graph_type=gtype,
                    beta_ws=beta_ws,
                    signal_type=stype,
                    h0=h0, h1=h1, h2=h2,
                    alpha_lap=alpha_lap,
                    eps=eps,
                    lam=lam,
                    gamma_ridge=gamma_ridge,
                    cvxpy_solver=cvxpy_solver,
                )
                for m in methods_single
            ]
            print_single(f"\n=== {gtype.upper()} | {stype.upper()} | Single run ===", single)

    # 2) Sweep over M (sin ridge)
    M_list = [10, 20, 50, 100, 200]
    n_seeds = 10
    for gtype in graph_types:
        for stype in signal_types:
            rows_M = sweep_over_M(
                methods=methods_sweep_M,
                M_list=M_list,
                n_seeds=n_seeds,
                N=N,
                avg_degree=avg_degree,
                graph_type=gtype,
                beta_ws=beta_ws,
                signal_type=stype,
                h0=h0, h1=h1, h2=h2,
                alpha_lap=alpha_lap,
                eps=eps,
                lam=lam,
                gamma_ridge=gamma_ridge,      # da igual aquí, ridge no se usa
                cvxpy_solver=cvxpy_solver,
            )
            rows_M = sorted(rows_M, key=lambda d: (d["M"], d["method"]))
            print_sweep_M(f"\n=== {gtype.upper()} | {stype.upper()} | Sweep over M ===", rows_M)
            save_rows_to_csv(
                rows_M,
                f"results_csv/sweep_M_{gtype}_{stype}.csv"
            )


    # 3) Sweep over lam (sin ridge)
    lam_list = [0.01, 0.02, 0.05, 0.1, 0.2]
    M_fixed = 100
    for gtype in graph_types:
        for stype in signal_types:
            rows_lam = sweep_over_lam(
                methods=methods_sweep_lam,
                lam_list=lam_list,
                n_seeds=n_seeds,
                N=N,
                M=M_fixed,
                avg_degree=avg_degree,
                graph_type=gtype,
                beta_ws=beta_ws,
                signal_type=stype,
                h0=h0, h1=h1, h2=h2,
                alpha_lap=alpha_lap,
                eps=eps,
                gamma_ridge=gamma_ridge,      # da igual aquí, ridge no se usa
                cvxpy_solver=cvxpy_solver,
            )
            rows_lam = sorted(rows_lam, key=lambda d: (d["lam"], d["method"]))
            print_sweep_lam(f"\n=== {gtype.upper()} | {stype.upper()} | Sweep over lam ===", rows_lam)
            save_rows_to_csv(
                rows_lam,
                f"results_csv/sweep_lam_{gtype}_{stype}.csv"
            )       