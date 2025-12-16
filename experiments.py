# experiments.py
import warnings
from sklearn.exceptions import ConvergenceWarning

# (Opcional pero recomendado) para que no ensucie la salida:
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import time
import numpy as np

from data_generation import (
    generate_graph,
    precision_from_adjacency_laplacian,
    sample_gaussian_from_precision,
)
from methods import (
    estimate_theta_ridge_inv_cov,
    estimate_theta_glasso_sklearn,
    estimate_theta_glasso_cvxpy,
    estimate_theta_pgd,
)
from metrics import fro_error_rel, sparsity_offdiag


def run_single(
    method: str,
    seed: int = 0,
    N: int = 20,
    M: int = 50,
    avg_degree: int = 4,
    graph_type: str = "erdos_renyi",
    beta_ws: float = 0.1,  # solo se usa si graph_type == "watts_strogatz"
    alpha_lap: float = 1.0,
    eps: float = 0.1,
    lam: float = 0.05,
    gamma_ridge: float = 1e-2,
):
    # 1) Generate ground-truth graph and precision
    A_true = generate_graph(
        N=N,
        avg_degree=avg_degree,
        graph_type=graph_type,
        seed=seed,
        beta_ws=beta_ws,
    )
    Theta_true = precision_from_adjacency_laplacian(A_true, alpha_lap=alpha_lap, eps=eps)

    # 2) Sample signals
    X = sample_gaussian_from_precision(Theta_true, M=M, seed=seed + 123)

    # 3) Estimate precision
    t0 = time.time()
    if method == "ridge":
        Theta_hat = estimate_theta_ridge_inv_cov(X, gamma=gamma_ridge)
    elif method == "glasso_skl":
        Theta_hat = estimate_theta_glasso_sklearn(X, lam=lam)
    elif method == "glasso_cvx":
        Theta_hat = estimate_theta_glasso_cvxpy(X, lam=lam)
    elif method == "pgd":
        Theta_hat = estimate_theta_pgd(X, lam=lam)
    else:
        raise ValueError(f"Unknown method: {method}")
    dt = time.time() - t0

    # 4) Metrics on precision
    return {
        "method": method,
        "graph_type": graph_type,
        "fro_theta": fro_error_rel(Theta_hat, Theta_true),
        "sparsity_theta": sparsity_offdiag(Theta_hat),
        "time_ms": dt * 1000.0,
    }


def sweep_over_M(
    methods,
    M_list,
    n_seeds=10,
    N=20,
    avg_degree=4,
    graph_type="erdos_renyi",
    beta_ws=0.1,
    alpha_lap=1.0,
    eps=0.1,
    lam=0.05,
    gamma_ridge=1e-2,
):
    rows = []
    for M in M_list:
        for method in methods:
            fro_list, sp_list, t_list = [], [], []
            for seed in range(n_seeds):
                out = run_single(
                    method=method,
                    seed=seed,
                    N=N,
                    M=M,
                    avg_degree=avg_degree,
                    graph_type=graph_type,
                    beta_ws=beta_ws,
                    alpha_lap=alpha_lap,
                    eps=eps,
                    lam=lam,
                    gamma_ridge=gamma_ridge,
                )
                fro_list.append(out["fro_theta"])
                sp_list.append(out["sparsity_theta"])
                t_list.append(out["time_ms"])

            rows.append({
                "graph_type": graph_type,
                "method": method,
                "M": M,
                "fro_mean": float(np.mean(fro_list)),
                "fro_std": float(np.std(fro_list)),
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
    alpha_lap=1.0,
    eps=0.1,
    gamma_ridge=1e-2,
):
    rows = []
    for lam in lam_list:
        for method in methods:
            fro_list, sp_list, t_list = [], [], []
            for seed in range(n_seeds):
                out = run_single(
                    method=method,
                    seed=seed,
                    N=N,
                    M=M,
                    avg_degree=avg_degree,
                    graph_type=graph_type,
                    beta_ws=beta_ws,
                    alpha_lap=alpha_lap,
                    eps=eps,
                    lam=lam,
                    gamma_ridge=gamma_ridge,
                )
                fro_list.append(out["fro_theta"])
                sp_list.append(out["sparsity_theta"])
                t_list.append(out["time_ms"])

            rows.append({
                "graph_type": graph_type,
                "method": method,
                "lam": lam,
                "fro_mean": float(np.mean(fro_list)),
                "fro_std": float(np.std(fro_list)),
                "sp_mean": float(np.mean(sp_list)),
                "sp_std": float(np.std(sp_list)),
                "t_mean": float(np.mean(t_list)),
                "t_std": float(np.std(t_list)),
            })
    return rows


def print_single(title: str, results):
    print(title)
    for r in results:
        print(
            f"{r['method']:10s} | "
            f"Fro(Θ)={r['fro_theta']:.3f} | "
            f"sparsity={r['sparsity_theta']:.3f} | "
            f"time={r['time_ms']:.2f} ms"
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
            f"Fro(Θ)={r['fro_mean']:.3f} ± {r['fro_std']:.3f} | "
            f"sparsity={r['sp_mean']:.3f} ± {r['sp_std']:.3f} | "
            f"time={r['t_mean']:.2f} ± {r['t_std']:.2f} ms"
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
            f"Fro(Θ)={r['fro_mean']:.3f} ± {r['fro_std']:.3f} | "
            f"sparsity={r['sp_mean']:.3f} ± {r['sp_std']:.3f} | "
            f"time={r['t_mean']:.2f} ± {r['t_std']:.2f} ms"
        )


if __name__ == "__main__":
    # ---------------------------
    # Global experiment settings
    # ---------------------------
    N = 20
    avg_degree = 4

    # Graph types (as you requested)
    graph_types = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]

    # Small-world parameter (only affects watts_strogatz)
    beta_ws = 0.1

    # Generator (truth) params
    alpha_lap = 1.0
    eps = 0.1

    # Inference params
    lam = 0.05
    gamma_ridge = 1e-2

    methods_all = ["ridge", "glasso_skl", "glasso_cvx", "pgd"]
    methods_lam = ["glasso_skl", "glasso_cvx", "pgd"]  # ridge doesn't use lam

    # ---------------------------
    # 1) Single runs per graph type
    # ---------------------------
    for gtype in graph_types:
        single = [
            run_single(
                m,
                seed=0,
                N=N,
                M=50,
                avg_degree=avg_degree,
                graph_type=gtype,
                beta_ws=beta_ws,
                alpha_lap=alpha_lap,
                eps=eps,
                lam=lam,
                gamma_ridge=gamma_ridge,
            )
            for m in methods_all
        ]
        print_single(
            f"\n=== {gtype.upper()} | Single run (N={N}, M=50, avg_degree={avg_degree}, alpha_lap={alpha_lap}, eps={eps}, lam={lam}) ===",
            single,
        )

    # ---------------------------
    # 2) Sweep over M per graph type
    # ---------------------------
    M_list = [10, 20, 50, 100, 200]
    n_seeds = 10

    for gtype in graph_types:
        rows_M = sweep_over_M(
            methods=methods_all,
            M_list=M_list,
            n_seeds=n_seeds,
            N=N,
            avg_degree=avg_degree,
            graph_type=gtype,
            beta_ws=beta_ws,
            alpha_lap=alpha_lap,
            eps=eps,
            lam=lam,
            gamma_ridge=gamma_ridge,
        )
        rows_M = sorted(rows_M, key=lambda d: (d["M"], d["method"]))
        print_sweep_M(
            f"\n=== {gtype.upper()} | Sweep over M (mean ± std, n_seeds={n_seeds}) | alpha_lap={alpha_lap}, eps={eps}, lam={lam} ===",
            rows_M,
        )

    # ---------------------------
    # 3) Sweep over lam per graph type (M fixed)
    # ---------------------------
    lam_list = [0.01, 0.02, 0.05, 0.1, 0.2]
    M_fixed = 100

    for gtype in graph_types:
        rows_lam = sweep_over_lam(
            methods=methods_lam,
            lam_list=lam_list,
            n_seeds=n_seeds,
            N=N,
            M=M_fixed,
            avg_degree=avg_degree,
            graph_type=gtype,
            beta_ws=beta_ws,
            alpha_lap=alpha_lap,
            eps=eps,
            gamma_ridge=gamma_ridge,
        )
        rows_lam = sorted(rows_lam, key=lambda d: (d["lam"], d["method"]))
        print_sweep_lam(
            f"\n=== {gtype.upper()} | Sweep over lam (M={M_fixed}, mean ± std, n_seeds={n_seeds}) | alpha_lap={alpha_lap}, eps={eps} ===",
            rows_lam,
        )
