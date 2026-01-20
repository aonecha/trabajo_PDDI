import numpy as np


def fro_error_rel_offdiag(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    """
    Error de Frobenius relativo SOLO en las entradas fuera de la diagonal:
      ||(X_hat - X_true)_off||_F / ||(X_true)_off||_F

    Es coherente cuando se fuerza la diagonal de Theta a cero
    (matrices de precisión).
    """
    N = X_true.shape[0]
    mask = ~np.eye(N, dtype=bool)

    diff = X_hat[mask] - X_true[mask]
    num = np.linalg.norm(diff)
    den = np.linalg.norm(X_true[mask]) + 1e-12
    return float(num / den)


def fro_error_rel_full(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    """
    Error de Frobenius relativo sobre la matriz COMPLETA:
      ||X_hat - X_true||_F / ||X_true||_F

    Se utiliza cuando la diagonal es relevante,
    por ejemplo al comparar Laplacianos de grafo.
    """
    num = np.linalg.norm(X_hat - X_true, ord="fro")
    den = np.linalg.norm(X_true, ord="fro") + 1e-12
    return float(num / den)


def sparsity_offdiag(X: np.ndarray, tol: float = 1e-4) -> float:
    """
    Calcula la fracción de entradas fuera de la diagonal que son aproximadamente cero.
    Mide el nivel de esparsidad de la matriz ignorando la diagonal.
    """
    N = X.shape[0]
    mask = ~np.eye(N, dtype=bool)
    vals = np.abs(X[mask])
    return float(np.mean(vals < tol))


def theta_to_laplacian(Theta_hat: np.ndarray, thr: float = 1e-4) -> np.ndarray:
    """
    Convierte una matriz de precisión estimada Theta_hat
    en un Laplaciano de grafo estimado.

    Procedimiento:
      - W_ij = |Theta_ij| si |Theta_ij| > thr, y 0 en caso contrario
      - Se anula la diagonal y se fuerza simetría
      - L_hat = diag(sum_j W_ij) - W
    """
    W = np.abs(Theta_hat).copy()
    np.fill_diagonal(W, 0.0)
    W[W < thr] = 0.0
    W = 0.5 * (W + W.T)

    d = W.sum(axis=1)
    L_hat = np.diag(d) - W
    return L_hat
