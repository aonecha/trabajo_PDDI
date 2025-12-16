import numpy as np
import networkx as nx


def generate_graph_er(N: int = 20, avg_degree: int = 4, seed: int = 0) -> np.ndarray:
    """
    Erdos-Renyi graph adjacency (0/1), symmetric, diag=0.
    """
    p = avg_degree / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    A = nx.to_numpy_array(G, dtype=float)
    np.fill_diagonal(A, 0.0)
    A = (A > 0).astype(float)
    return A


def precision_from_adjacency_laplacian(A: np.ndarray, alpha_lap: float = 1.0, eps: float = 0.1) -> np.ndarray:
    """
    Build a valid precision matrix from a graph:
        Theta_true = alpha_lap * L + eps * I
    where L is the unnormalized Laplacian D - A.

    eps > 0 ensures positive definiteness.
    """
    d = A.sum(axis=1)
    L = np.diag(d) - A
    Theta = alpha_lap * L + eps * np.eye(A.shape[0])
    return Theta


def sample_gaussian_from_precision(Theta: np.ndarray, M: int = 50, seed: int = 0) -> np.ndarray:
    """
    Sample X ~ N(0, Sigma) with Sigma = Theta^{-1}
    Returns X shape (N, M)
    """
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(Theta)
    X = rng.multivariate_normal(mean=np.zeros(Theta.shape[0]), cov=Sigma, size=M).T
    return X
