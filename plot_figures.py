import os
import pandas as pd
import matplotlib.pyplot as plt

from experiments import sweep_over_M, sweep_over_lam

# ======================================================
# ESTILO GLOBAL MATPLOTLIB (SIN SEABORN)
# ======================================================
plt.style.use("default")

plt.rcParams.update({
    "figure.figsize": (7, 4.5),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.1,
})

# ======================================================
# CONFIGURACIÓN GENERAL
# ======================================================
FIG_DIR = "figures_interpretation"
os.makedirs(FIG_DIR, exist_ok=True)

GRAPH_TYPES = ["erdos_renyi", "watts_strogatz", "barabasi_albert"]
SIGNAL_TYPES = ["gaussian", "stationary"]
METHODS = ["glasso_skl", "glasso_cvx", "pgd"]

COLORS = {
    "glasso_skl": "#1f77b4",   # azul
    "glasso_cvx": "#ff7f0e",   # naranja
    "pgd": "#2ca02c",          # verde
}

LINEWIDTH = 2.5
ALPHA_BAND = 0.25
MARKERSIZE = 6

# Parámetros del experimento
N = 20
AVG_DEGREE = 4
BETA_WS = 0.1

ALPHA_LAP = 1.0
EPS = 0.1

# Parámetros estacionarios
H0, H1, H2 = 1.0, -0.25, 0.0

LAM_DEFAULT = 0.05
M_LIST = [10, 20, 50, 100, 200]
LAM_LIST = [0.01, 0.02, 0.05, 0.1, 0.2]

N_SEEDS = 10
CVXPY_SOLVER = "SCS"
GAMMA_RIDGE = 1e-2

# ======================================================
# FUNCIÓN AUXILIAR: LÍNEA + BANDA
# ======================================================
def plot_mean_std(ax, x, mean, std, label, color):
    ax.plot(
        x, mean,
        label=label,
        color=color,
        linewidth=LINEWIDTH,
        marker="o",
        markersize=MARKERSIZE
    )
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=color,
        alpha=ALPHA_BAND
    )

# ======================================================
# FUNCIÓN AUXILIAR: CIERRE DE FIGURA 2-PANELES (SUptitle + Legend SIN SOLAPAR)
# ======================================================
def finalize_two_panel_figure(fig, axes, title, outpath):
    # Título arriba (y reservado)
    fig.suptitle(title, fontsize=14, y=0.98)

    # Leyenda basada en las líneas del primer subplot
    handles, labels = axes[0].get_legend_handles_labels()

    # Leyenda centrada arriba pero debajo del título
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(METHODS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.94)
    )

    # Reservar espacio superior para título + leyenda
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    fig.savefig(outpath, dpi=300)
    plt.close(fig)

# ======================================================
# ERROR Y TIEMPO VS M
# ======================================================
def plot_error_time_vs_M(df, graph, signal):
    d = df[
        (df.graph_type == graph) &
        (df.signal_type == signal) &
        (df.M > 10)   # ⬅️ elimina M = 10 SOLO en la gráfica
    ]

    if d.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Error vs M
    ax = axes[0]
    for m in METHODS:
        dm = d[d.method == m].sort_values("M")
        plot_mean_std(ax, dm.M, dm.err_mean, dm.err_std, m, COLORS[m])

    ax.set_xlabel("Número de muestras (M)")
    ax.set_ylabel("Error relativo")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Tiempo vs M
    ax = axes[1]
    for m in METHODS:
        dm = d[d.method == m].sort_values("M")
        plot_mean_std(ax, dm.M, dm.t_mean, dm.t_std, m, COLORS[m])

    ax.set_yscale("log")
    ax.set_xlabel("Número de muestras (M)")
    ax.set_ylabel("Tiempo de cómputo (ms, log)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    signal_pretty = {"gaussian": "Gaussiana", "stationary": "Estacionaria"}.get(signal, signal)
    title = f"{graph.replace('_',' ').title()} — Señal {signal_pretty}"
    outpath = f"{FIG_DIR}/{graph}_{signal}_error_time_vs_M.png"

    finalize_two_panel_figure(fig, axes, title, outpath)

# ======================================================
# ERROR Y SPARSITY VS LAMBDA
# ======================================================
def plot_error_sparsity_vs_lam(df, graph, signal):
    d = df[(df.graph_type == graph) & (df.signal_type == signal)]
    if d.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Error vs lambda
    ax = axes[0]
    for m in METHODS:
        dm = d[d.method == m].sort_values("lam")
        plot_mean_std(ax, dm.lam, dm.err_mean, dm.err_std, m, COLORS[m])

    ax.set_xlabel("Parámetro de regularización (λ)")
    ax.set_ylabel("Error relativo")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Sparsity vs lambda
    ax = axes[1]
    for m in METHODS:
        dm = d[d.method == m].sort_values("lam")
        plot_mean_std(ax, dm.lam, dm.sp_mean, dm.sp_std, m, COLORS[m])

    ax.set_xlabel("Parámetro de regularización (λ)")
    ax.set_ylabel("Esparsidad fuera de la diagonal")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    signal_pretty = {"gaussian": "Gaussiana", "stationary": "Estacionaria"}.get(signal, signal)
    title = f"{graph.replace('_',' ').title()} — Señal {signal_pretty}"
    outpath = f"{FIG_DIR}/{graph}_{signal}_error_sparsity_vs_lam.png"

    finalize_two_panel_figure(fig, axes, title, outpath)

# ======================================================
# PARETO PRECISIÓN–TIEMPO
# ======================================================
def plot_pareto(df, graph, signal):
    d = df[(df.graph_type == graph) & (df.signal_type == signal)]
    if d.empty:
        return

    plt.figure(figsize=(6.5, 4.8))

    for m in METHODS:
        dm = d[d.method == m]
        plt.scatter(
            dm.t_mean,
            dm.err_mean,
            s=80,
            color=COLORS[m],
            label=m,
            alpha=0.85
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Tiempo de cómputo (ms, log)")
    plt.ylabel("Error relativo (log)")

    signal_pretty = {"gaussian": "Gaussiana", "stationary": "Estacionaria"}.get(signal, signal)
    plt.title(
        f"Compromiso precisión–tiempo\n{graph.replace('_',' ').title()} — {signal_pretty}"
    )

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{graph}_{signal}_pareto.png", dpi=300)
    plt.close()

# ======================================================
# MAIN
# ======================================================
def main():
    all_rows_M = []
    all_rows_lam = []

    for graph in GRAPH_TYPES:
        for signal in SIGNAL_TYPES:
            print(f"▶ Ejecutando {graph} | {signal}")

            rows_M = sweep_over_M(
                methods=METHODS,
                M_list=M_LIST,
                n_seeds=N_SEEDS,
                N=N,
                avg_degree=AVG_DEGREE,
                graph_type=graph,
                beta_ws=BETA_WS,
                signal_type=signal,
                h0=H0,
                h1=H1,
                h2=H2,
                alpha_lap=ALPHA_LAP,
                eps=EPS,
                lam=LAM_DEFAULT,
                gamma_ridge=GAMMA_RIDGE,
                cvxpy_solver=CVXPY_SOLVER,
            )
            all_rows_M.extend(rows_M)

            rows_lam = sweep_over_lam(
                methods=METHODS,
                lam_list=LAM_LIST,
                n_seeds=N_SEEDS,
                N=N,
                avg_degree=AVG_DEGREE,
                graph_type=graph,
                beta_ws=BETA_WS,
                signal_type=signal,
                h0=H0,
                h1=H1,
                h2=H2,
                alpha_lap=ALPHA_LAP,
                eps=EPS,
                M=100,
                gamma_ridge=GAMMA_RIDGE,
                cvxpy_solver=CVXPY_SOLVER,
            )
            all_rows_lam.extend(rows_lam)

    df_M = pd.DataFrame(all_rows_M)
    df_lam = pd.DataFrame(all_rows_lam)

    for g in GRAPH_TYPES:
        for s in SIGNAL_TYPES:
            plot_error_time_vs_M(df_M, g, s)
            plot_error_sparsity_vs_lam(df_lam, g, s)
            plot_pareto(df_M, g, s)

    print("\n✔ Todas las figuras generadas en:", FIG_DIR)

if __name__ == "__main__":
    main()
