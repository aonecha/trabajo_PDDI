# experiments.py
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

def plot_three_graphs_once(N: int, avg_degree: int, beta_ws: float, seed_graph: int = 0):
    """
    Genera ER / WS / BA UNA sola vez y guarda 3 figuras.
    Devuelve un diccionario con las matrices de adyacencia para reutilizarlas.
    """
    graph_types = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
    pretty = {
        "erdos_renyi": "ER",
        "watts_strogatz": "SW",
        "barabasi_albert": "BA",
    }

    cache = {}
    for gtype in graph_types:
        A = generate_graph(
            N=N,
            avg_degree=avg_degree,
            graph_type=gtype,
            seed=seed_graph,
            beta_ws=beta_ws,
        )
        cache[gtype] = A

        if gtype == "erdos_renyi":
            title = f"ER (N={N}, avg_degree≈{avg_degree}, seed={seed_graph})"
        elif gtype == "watts_strogatz":
            title = f"SW (N={N}, k≈{avg_degree}, beta={beta_ws}, seed={seed_graph})"
        else:
            title = f"BA (N={N}, avg_degree≈{avg_degree}, seed={seed_graph})"

        outpath = f"figures/graph_{pretty[gtype]}_N{N}_seed{seed_graph}.png"
        print("Saved graph figure:", outpath)

    return cache


def run_single(
    method: str,
    seed: int = 0,
    N: int = 20,
    M: int = 50,
    avg_degree: int = 4,
    graph_type: str = "erdos_renyi",
    beta_ws: float = 0.1,
    signal_type: str = "gaussian",   
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0,
    alpha_lap: float = 1.0,
    eps: float = 0.1,
    lam: float = 0.05,
    gamma_ridge: float = 1e-2,
    cvxpy_solver: str = "SCS",
    A_true_override: np.ndarray | None = None,
):
    """
    Ejecuta una única corrida del experimento para un método concreto.
    Genera datos, estima la matriz de precisión y devuelve error, sparsity y tiempo.
    """
    # 1) Graph
    if A_true_override is not None:
        A_true = A_true_override
    else:
        A_true = generate_graph(
            N=N,
            avg_degree=avg_degree,
            graph_type=graph_type,
            seed=seed,
            beta_ws=beta_ws,
        )

    # 2) Generate X and truth for metric
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

    # 3) Covariance outside solver timing
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
        err_label = "Fro(Θ)"
    else:
        L_hat = theta_to_laplacian(Theta_hat, thr=1e-4)
        err = fro_error_rel_full(L_hat, L_true)
        err_label = "Fro(L)"

    return {
        "method": method,
        "graph_type": graph_type,
        "signal_type": signal_type,
        "err_label": err_label,
        "err": float(err),
        "sparsity_theta": sparsity_offdiag(Theta_hat),
        "time_ms": dt * 1000.0,
    }


def print_single(title: str, results):
    """
    Imprime por pantalla los resultados de varias corridas individuales.
    Muestra error, sparsity y tiempo de cada método de forma legible.
    """
    print(title)
    for r in results:
        print(
            f"{r['method']:10s} | "
            f"{r['err_label']}={r['err']:.3f} | "
            f"sparsity={r['sparsity_theta']:.3f} | "
            f"solver_time={r['time_ms']:.2f} ms"
        )


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
    """
    Realiza un barrido sobre distintos valores del número de muestras M.
    Promedia resultados sobre varias semillas para cada método.
    """
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
                err_list.append(out["err"])
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
    """
    Realiza un barrido sobre distintos valores del parámetro de regularización λ.
    Evalúa el compromiso entre error, sparsity y tiempo de cómputo.
    """   
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
                err_list.append(out["err"])
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


def print_sweep_M(title: str, rows):
    """
    Imprime los resultados del barrido sobre M agrupados por tamaño de muestra.
    Muestra medias y desviaciones típicas para cada método.
    """
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
    """
    Imprime los resultados del barrido sobre λ agrupados por valor de regularización.
    Resume error, sparsity y tiempo para cada método.
    """
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
    """
    1) Define parámetros globales.
    2) Genera y guarda figuras de grafos (ER/SW/BA) una sola vez y cachea sus adyacencias.
    3) Ejecuta corridas individuales comparando métodos sobre el MISMO grafo (comparación justa).
    4) Ejecuta barridos sobre M (nº de muestras) promediando sobre varias semillas.
    5) Ejecuta barridos sobre λ (regularización) con M fijo para estudiar el trade-off error/sparsity.
    """
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


    A_cache = plot_three_graphs_once(N=N, avg_degree=avg_degree, beta_ws=beta_ws, seed_graph=0)

    methods_single = ["ridge", "glasso_skl", "glasso_cvx", "pgd"]
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
                    A_true_override=A_cache[gtype],  
                )
                for m in methods_single
            ]
            print_single(f"\n=== {gtype.upper()} | {stype.upper()} | Single run (same plotted graph) ===", single)

    methods_sweep = ["glasso_skl", "glasso_cvx", "pgd"]

    M_list = [10, 20, 50, 100, 200]
    n_seeds = 10
    for gtype in graph_types:
        for stype in signal_types:
            rows_M = sweep_over_M(
                methods=methods_sweep,
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
                gamma_ridge=gamma_ridge,
                cvxpy_solver=cvxpy_solver,
            )
            rows_M = sorted(rows_M, key=lambda d: (d["M"], d["method"]))
            print_sweep_M(f"\n=== {gtype.upper()} | {stype.upper()} | Sweep over M ===", rows_M)

    lam_list = [0.01, 0.02, 0.05, 0.1, 0.2]
    M_fixed = 100
    for gtype in graph_types:
        for stype in signal_types:
            rows_lam = sweep_over_lam(
                methods=methods_sweep,
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
                gamma_ridge=gamma_ridge,
                cvxpy_solver=cvxpy_solver,
            )
            rows_lam = sorted(rows_lam, key=lambda d: (d["lam"], d["method"]))
            print_sweep_lam(f"\n=== {gtype.upper()} | {stype.upper()} | Sweep over lam ===", rows_lam)
