import numpy as np
import networkx as nx


# ---------- GRAPH GENERATORS ----------

# Definimos un grapho ER G y su matriz de adyacencia A
def generate_graph_er(N: int, avg_degree: int, seed: int):
    p = avg_degree / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)

# Definimos un grafo Watts–Strogatz (small-world) y su matriz de adyacencia A
def generate_graph_ws(N: int, avg_degree: int, beta: float, seed: int):
    k = avg_degree if avg_degree % 2 == 0 else avg_degree + 1
    G = nx.watts_strogatz_graph(N, k, beta, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)

# Definimos un grafo Barabási–Albert (scale-free) y su matriz de adyacencia A
def generate_graph_ba(N: int, avg_degree: int, seed: int):
    m = max(1, avg_degree // 2)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    A = nx.to_numpy_array(G)
    np.fill_diagonal(A, 0.0)
    return (A > 0).astype(float)



# Funcion para generar un grafo dependiendo del tipo
def generate_graph(N: int, avg_degree: int, graph_type: str, seed: int, beta_ws: float = 0.1):
    if graph_type == "erdos_renyi":
        return generate_graph_er(N, avg_degree, seed)
    elif graph_type == "watts_strogatz":
        return generate_graph_ws(N, avg_degree, beta_ws, seed)
    elif graph_type == "barabasi_albert":
        return generate_graph_ba(N, avg_degree, seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


# ---------- GAUSSIAN MODEL ----------
# Calculamos la matriz de precisión
def precision_from_adjacency_laplacian(A: np.ndarray, alpha_lap: float, eps: float):
    d = A.sum(axis=1)
    L = np.diag(d) - A
    Theta = alpha_lap * L + eps * np.eye(A.shape[0])
    return Theta

# Generamos muestras de un modelo gausiano a partir de la matriz de precisión
def sample_gaussian_from_precision(Theta: np.ndarray, M: int, seed: int):
    rng = np.random.default_rng(seed)
    Sigma = np.linalg.inv(Theta)
    X = rng.multivariate_normal(mean=np.zeros(Theta.shape[0]), cov=Sigma, size=M).T
    return X


# ---------- STATIONARY MODEL ----------
# Definimos el Laplaciano
def laplacian_from_adjacency(A: np.ndarray) -> np.ndarray:
    d = A.sum(axis=1)
    return np.diag(d) - A

# Construimos un filtro de orden bajo definido como un polinomio del Laplaciano
def low_order_graph_filter(L: np.ndarray, h0: float = 1.0, h1: float = -0.25, h2: float = 0.0) -> np.ndarray:
    N = L.shape[0]
    I = np.eye(N)
    H = h0 * I + h1 * L
    if abs(h2) > 0:
        H = H + h2 * (L @ L)
    return H

# Genera señales estacionarias sobre un grafo aplicando un filtro de grafo a ruido blanco. x = H(filtro)W(ruido)
# Generamos datos sintéticos sobre un grafo  
def sample_stationary_signals(A: np.ndarray, M: int, seed: int, h0: float = 1.0, h1: float = -0.25, h2: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    L = laplacian_from_adjacency(A)
    H = low_order_graph_filter(L, h0=h0, h1=h1, h2=h2)
    W = rng.standard_normal(size=(A.shape[0], M))
    X = H @ W
    return X

# Calculamos la matriz de precisión del proceso estacionario incluido por el filtro H.
# Para conocer el modelo estadístico teórico que generan esas señales
def precision_from_stationary_filter(A: np.ndarray, h0: float = 1.0, h1: float = -0.25, h2: float = 0.0, eps_pd: float = 1e-6) -> np.ndarray:
    L = laplacian_from_adjacency(A)
    H = low_order_graph_filter(L, h0=h0, h1=h1, h2=h2)

    Sigma = H @ H.T
    Sigma = 0.5 * (Sigma + Sigma.T) + eps_pd * np.eye(Sigma.shape[0])

    Theta = np.linalg.inv(Sigma)
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta
