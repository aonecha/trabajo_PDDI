# experiments_big.py
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


def save_graph_figure_from_adjacency(A: np.ndarray, outpath: str, title: str, layout_seed: int = 0):
    """
    Dibuja y guarda un grafo a partir de su matriz de adyacencia.
    Usa un spring layout reproducible y exporta la figura a un archivo PNG.
    """  
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    G = nx.from_numpy_array(A)  # undirected
    plt.figure()
    pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw(G, pos=pos, node_size=40, with_labels=False)
    plt.title(title)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()

def save_adjacency_matrix_figure(A: np.ndarray, outpath: str, title: str):
    """
    Guarda un "heatmap" de la matriz de adyacencia para inspección visual.
    Reordena los nodos por grado (descendente) para que la estructura se vea más clara.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    deg = A.sum(axis=1)
    order = np.argsort(-deg)  # grados descendentes
    A_ord = A[np.ix_(order, order)]

    plt.figure(figsize=(6, 6))
    plt.imshow(A_ord, interpolation="nearest", aspect="equal")
    plt.title(title)
    plt.xlabel("nodes (sorted by degree)")
    plt.ylabel("nodes (sorted by degree)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(outpath, dpi=250, bbox_inches="tight")
    plt.close()


def run_case(
    method: str,
    seed: int,
    A_true: np.ndarray,         
    M: int,
    avg_degree: int,
    signal_type: str,
    alpha_lap: float = 1.0,
    eps: float = 0.1,
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0,
    lam: float = 0.05,
    gamma_ridge: float = 1e-2,
    cvxpy_solver: str = "SCS",
    thr_theta_to_L: float = 1e-4,
):
    """
    Ejecuta una corrida (un seed) para un método usando un grafo YA fijado (A_true).
    1) Genera datos X según el tipo de señal (gaussian o stationary).
    2) Calcula la covarianza muestral S (fuera del timing).
    3) Estima Θ_hat con el método elegido midiendo SOLO el tiempo del solver.
    4) Calcula el error respecto a la verdad-terreno:
       - gaussian: compara Θ_hat con Θ_true (Frobenius relativo off-diagonal)
       - stationary: convierte Θ_hat -> L_hat y compara con L_true (Frobenius relativo full)
    Devuelve métricas: error, sparsity de Θ_hat y tiempo del solver en ms.
    """
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
        err_label = "Fro(Θ)"
    else:
        L_hat = theta_to_laplacian(Theta_hat, thr=thr_theta_to_L)
        err = fro_error_rel_full(L_hat, L_true)
        err_label = "Fro(L)"

    return {
        "method": method,
        "signal_type": signal_type,
        "err_label": err_label,
        "err": float(err),
        "sparsity_theta": sparsity_offdiag(Theta_hat),
        "solver_time_ms": dt * 1000.0,
    }


def print_results(title: str, results: list[dict]):
    """
    Imprime un resumen legible de resultados (lista de diccionarios devueltos por run_case).
    Muestra por método: error, sparsity de Θ_hat y tiempo de solver en milisegundos.
    """
    print(title)
    for r in results:
        print(
            f"{r['method']:10s} | "
            f"{r['err_label']}={r['err']:.3f} | "
            f"sparsity(Θ̂)={r['sparsity_theta']:.3f} | "
            f"solver_time={r['solver_time_ms']:.2f} ms"
        )


if __name__ == "__main__":
    """
    Ejecución “BIG CASE” para probar escalabilidad en un grafo más grande.
    1) Fija N, M, avg_degree, seed y parámetros del solver.
    2) Genera un único grafo ER (A_true) y guarda su dibujo + heatmap de adyacencia ordenada por grado.
    3) Para cada tipo de señal (gaussian y stationary), ejecuta varios métodos sobre EXACTAMENTE el mismo A_true
       y muestra métricas comparables (error/sparsity/tiempo).
    """
    N = 100
    M = 500
    avg_degree = 6
    seed = 0

    lam = 0.05
    cvxpy_solver = "SCS"

    signal_types = ["gaussian", "stationary"]

    methods = ["glasso_skl", "pgd", "glasso_cvx"]

    # 0) Generar el grafo UNA vez (ER) y guardarlo
    A_true = generate_graph(
        N=N,
        avg_degree=avg_degree,
        graph_type="erdos_renyi",
        seed=seed,
        beta_ws=0.1,  # no afecta a ER
    )

    outpath = f"figures/BIG_ER_N{N}_avgdeg{avg_degree}_seed{seed}.png"
    title = f"BIG ER (N={N}, avg_degree≈{avg_degree}, seed={seed})"
    save_graph_figure_from_adjacency(A_true, outpath, title=title, layout_seed=seed)
    print("Saved graph figure:", outpath)

    out_adj = f"figures/BIG_ER_N{N}_avgdeg{avg_degree}_seed{seed}_adjacency.png"
    title_adj = f"BIG ER adjacency (sorted by degree) | N={N}, avg_degree≈{avg_degree}, seed={seed}"
    save_adjacency_matrix_figure(A_true, out_adj, title=title_adj)
    print("Saved adjacency matrix figure:", out_adj)

    for stype in signal_types:
        results = [
            run_case(
                method=meth,
                seed=seed,
                A_true=A_true,     
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
