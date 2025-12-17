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


# ---------- GAUSSIAN MODEL (already used) ----------

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


# ---------- STATIONARY MODEL (new) ----------

def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    return np.diag(d) - A


def low_order_graph_filter(L: np.ndarray, h0: float = 1.0, h1: float = -0.25, h2: float = 0.0) -> np.ndarray:
    """
    Low-order graph filter:
      H = h0*I + h1*L + h2*L^2
    Keep it low-order (order 1 by default with h2=0).
    """
    N = L.shape[0]
    I = np.eye(N)
    H = h0 * I + h1 * L
    if abs(h2) > 0:
        H = H + h2 * (L @ L)
    return H


def sample_stationary_signals(
    A: np.ndarray,
    M: int,
    seed: int,
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0,
) -> np.ndarray:
    """
    Stationary graph signals:
      w ~ N(0, I)
      x = H w
    Returns X shape (N, M)
    """
    rng = np.random.default_rng(seed)
    L = laplacian_from_adjacency(A)
    H = low_order_graph_filter(L, h0=h0, h1=h1, h2=h2)

    W = rng.standard_normal(size=(A.shape[0], M))  # white noise
    X = H @ W
    return X


def precision_from_stationary_filter(
    A: np.ndarray,
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0,
    eps_pd: float = 1e-6,
) -> np.ndarray:
    """
    If x = H w and w~N(0,I) then:
      Sigma_true = H H^T
      Theta_true = Sigma_true^{-1}
    eps_pd adds tiny jitter for numerical stability.
    """
    L = laplacian_from_adjacency(A)
    H = low_order_graph_filter(L, h0=h0, h1=h1, h2=h2)

    Sigma = H @ H.T
    Sigma = 0.5 * (Sigma + Sigma.T) + eps_pd * np.eye(Sigma.shape[0])

    Theta = np.linalg.inv(Sigma)
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta
