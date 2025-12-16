import numpy as np
import networkx as nx


def generate_graph_er(N: int = 20, avg_degree: int = 4, seed: int = 0) -> np.ndarray:
    """
    Genera un grafo ER (Erdős–Rényi) NO dirigido.

    p ≈ avg_degree/(n-1) para que el grado medio esperado sea ~avg_degree.

    Returns:
      A: matriz de adyacencia (n,n), simétrica, diagonal 0
    """
    p = avg_degree / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype(float)
    return A


def precision_from_adjacency_laplacian(A: np.ndarray, alpha_lap: float = 1.0, eps: float = 0.1) -> np.ndarray:
    """
    Matriz de precisión SPD (invertible) a partir del Laplaciano:
        Theta* = alpha * L + eps * I
    """
    d = A.sum(axis=1)
    L = np.diag(d) - A
    Theta = alpha_lap * L + eps * np.eye(A.shape[0])
    return Theta


def sample_gaussian_from_precision(Theta: np.ndarray, M: int = 50, seed: int = 0) -> np.ndarray:
    """
    X ~ N(0, Sigma) con Sigma = inv(Theta)
    Returns:
      X: (m, n)
    """
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(Theta)
    X = rng.multivariate_normal(mean=np.zeros(Theta.shape[0]), cov=Sigma, size=M).T
    return X
