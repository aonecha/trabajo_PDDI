import numpy as np
import networkx as nx


# ---------- GRAPH GENERATORS ----------

def generate_graph_er(N: int, avg_degree: int, seed: int):
    p = avg_degree / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)


def generate_graph_ws(N: int, avg_degree: int, beta: float, seed: int):
    """
    Watts–Strogatz small-world graph.
    avg_degree must be even → we force it.
    """
    k = avg_degree if avg_degree % 2 == 0 else avg_degree + 1
    G = nx.watts_strogatz_graph(N, k, beta, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)


def generate_graph_ba(N: int, avg_degree: int, seed: int):
    """
    Barabási–Albert scale-free graph.
    m controls the number of edges per new node.
    """
    m = max(1, avg_degree // 2)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)


# ---------- ROUTER ----------

def generate_graph(
    N: int,
    avg_degree: int,
    graph_type: str,
    seed: int,
    beta_ws: float = 0.1,
):
    if graph_type == "erdos_renyi":
        return generate_graph_er(N, avg_degree, seed)

    elif graph_type == "watts_strogatz":
        return generate_graph_ws(N, avg_degree, beta_ws, seed)

    elif graph_type == "barabasi_albert":
        return generate_graph_ba(N, avg_degree, seed)

    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


# ---------- PRECISION + SAMPLING ----------

def precision_from_adjacency_laplacian(A: np.ndarray, alpha_lap: float, eps: float):
    d = A.sum(axis=1)
    L = np.diag(d) - A
    Theta = alpha_lap * L + eps * np.eye(A.shape[0])
    return Theta


def sample_gaussian_from_precision(Theta: np.ndarray, M: int, seed: int):
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(Theta)
    X = rng.multivariate_normal(
        mean=np.zeros(Theta.shape[0]),
        cov=Sigma,
        size=M
    ).T
    return X
