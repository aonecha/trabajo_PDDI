import pandas as pd
import matplotlib.pyplot as plt

from experiments import sweep_over_M

# -----------------------------
# Parámetros globales
# -----------------------------
N = 20
avg_degree = 4
alpha_lap = 1.0
eps = 0.1
lam = 0.05
gamma_ridge = 1e-2
beta_ws = 0.1

M_list = [10, 20, 50, 100, 200]

# Métodos SIN ridge
methods = ["glasso_skl", "glasso_cvx", "pgd"]

# Tipos de grafo
graph_types = [
    "erdos_renyi",
    "watts_strogatz",
    "barabasi_albert",
]

markers = {
    "glasso_skl": "o",
    "glasso_cvx": "s",
    "pgd": "D",
}

# -----------------------------
# Loop sobre tipos de grafo
# -----------------------------
for graph_type in graph_types:

    print(f"\nGenerando gráficas para: {graph_type}")

    rows_M = sweep_over_M(
        methods=methods,
        M_list=M_list,
        n_seeds=10,
        N=N,
        avg_degree=avg_degree,
        graph_type=graph_type,
        beta_ws=beta_ws,
        alpha_lap=alpha_lap,
        eps=eps,
        lam=lam,
        gamma_ridge=gamma_ridge,
    )

    df = pd.DataFrame(rows_M)

    # -----------------------------
    # FIGURA: Error y Tiempo vs M
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # ---- (a) Error vs M ----
    ax = axes[0]
    for method in methods:
        df_m = df[df["method"] == method]
        ax.errorbar(
            df_m["M"],
            df_m["fro_mean"],
            yerr=df_m["fro_std"],
            marker=markers[method],
            capsize=4,
            label=method,
        )

    ax.set_title(f"(a) Error vs M — {graph_type}")
    ax.set_xlabel("Número de muestras (M)")
    ax.set_ylabel("Error relativo de Frobenius")
    ax.grid(True)
    ax.legend()

    # ---- (b) Tiempo vs M ----
    ax = axes[1]
    for method in methods:
        df_m = df[df["method"] == method]
        ax.plot(
            df_m["M"],
            df_m["t_mean"],
            marker=markers[method],
            label=method,
        )

    ax.set_title(f"(b) Tiempo vs M — {graph_type}")
    ax.set_xlabel("Número de muestras (M)")
    ax.set_ylabel("Tiempo medio (ms)")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()
