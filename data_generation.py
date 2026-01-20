import numpy as np
import networkx as nx


# ---------- GRAPH GENERATORS ----------

def generate_graph_er(N: int, avg_degree: int, seed: int):
    """
    Genera un grafo Erdős–Rényi (ER) con grado medio aproximado.
    Devuelve la matriz de adyacencia binaria sin autoconexiones.
    """
    p = avg_degree / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)


def generate_graph_ws(N: int, avg_degree: int, beta: float, seed: int):
    """
    Genera un grafo Watts–Strogatz (small-world).
    Ajusta el grado para que sea par y devuelve la matriz de adyacencia binaria.
    """
    k = avg_degree if avg_degree % 2 == 0 else avg_degree + 1
    G = nx.watts_strogatz_graph(N, k, beta, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)


def generate_graph_ba(N: int, avg_degree: int, seed: int):
    """
    Genera un grafo Barabási–Albert (scale-free).
    Usa m ≈ avg_degree / 2 enlaces por nuevo nodo.
    """
    m = max(1, avg_degree // 2)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)


def generate_graph(N: int, avg_degree: int, graph_type: str, seed: int, beta_ws: float = 0.1):
    """
    Genera un grafo según el tipo especificado (ER, WS o BA).
    Devuelve la matriz de adyacencia binaria correspondiente.
    """
    if graph_type == "erdos_renyi":
        return generate_graph_er(N, avg_degree, seed)
    elif graph_type == "watts_strogatz":
        return generate_graph_ws(N, avg_degree, beta_ws, seed)
    elif graph_type == "barabasi_albert":
        return generate_graph_ba(N, avg_degree, seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


# ---------- GAUSSIAN MODEL ----------

def precision_from_adjacency_laplacian(A: np.ndarray, alpha_lap: float, eps: float):
    """
    Construye la matriz de precisión a partir del Laplaciano del grafo.
    Theta = alpha * L + eps * I garantiza definición positiva.
    """
    d = A.sum(axis=1)
    L = np.diag(d) - A
    Theta = alpha_lap * L + eps * np.eye(A.shape[0])
    return Theta


def sample_gaussian_from_precision(Theta: np.ndarray, M: int, seed: int):
    """
    Genera M muestras de un modelo gaussiano con matriz de precisión Theta.
    Las muestras se devuelven en formato (N, M).
    """
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(Theta)
    X = rng.multivariate_normal(
        mean=np.zeros(Theta.shape[0]),
        cov=Sigma,
        size=M
    ).T
    return X


# ---------- STATIONARY MODEL ----------

def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Calcula el Laplaciano no normalizado a partir de la matriz de adyacencia.
    """
    d = A.sum(axis=1)
    return np.diag(d) - A


def low_order_graph_filter(
    L: np.ndarray,
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0
) -> np.ndarray:
    """
    Construye un filtro de grafo de orden bajo como polinomio del Laplaciano.
    H = h0 I + h1 L + h2 L^2 
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
    h2: float = 0.0
) -> np.ndarray:
    """
    Genera señales estacionarias sobre un grafo filtrando ruido blanco.
    Las señales se obtienen como X = HW, donde H es un filtro de grafo.
    """
    rng = np.random.default_rng(seed)
    L = laplacian_from_adjacency(A)
    H = low_order_graph_filter(L, h0=h0, h1=h1, h2=h2)
    W = rng.standard_normal(size=(A.shape[0], M))
    X = H @ W
    return X


def precision_from_stationary_filter(
    A: np.ndarray,
    h0: float = 1.0,
    h1: float = -0.25,
    h2: float = 0.0,
    eps_pd: float = 1e-6
) -> np.ndarray:
    """
    Calcula la matriz de precisión teórica del proceso estacionario inducido por el filtro.
    Se añade regularización para garantizar definición positiva. Se calcula la matriz de precisión
    para combrobar los resultados de los métodos de estimación.
    """
    L = laplacian_from_adjacency(A)
    H = low_order_graph_filter(L, h0=h0, h1=h1, h2=h2)

    Sigma = H @ H.T
    Sigma = 0.5 * (Sigma + Sigma.T) + eps_pd * np.eye(Sigma.shape[0])

    Theta = np.linalg.inv(Sigma)
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta
