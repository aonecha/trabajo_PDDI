import numpy as np


def fro_error_rel_offdiag(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    """
    Relative Frobenius error SOLO off-diagonal:
      ||(X_hat - X_true)_off||_F / ||(X_true)_off||_F
    (coherente cuando ponemos diagonal a 0 en Theta)
    """
    N = X_true.shape[0]
    mask = ~np.eye(N, dtype=bool)

    diff = X_hat[mask] - X_true[mask]
    num = np.linalg.norm(diff)
    den = np.linalg.norm(X_true[mask]) + 1e-12
    return float(num / den)


def fro_error_rel_full(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    """
    Relative Frobenius error sobre la matriz COMPLETA:
      ||X_hat - X_true||_F / ||X_true||_F
    (esto es lo que usamos para Laplaciano, porque su diagonal importa)
    """
    num = np.linalg.norm(X_hat - X_true, ord="fro")
    den = np.linalg.norm(X_true, ord="fro") + 1e-12
    return float(num / den)


def sparsity_offdiag(X: np.ndarray, tol: float = 1e-4) -> float:
    """
    Fraction of off-diagonal entries that are ~0.
    """
    N = X.shape[0]
    mask = ~np.eye(N, dtype=bool)
    vals = np.abs(X[mask])
    return float(np.mean(vals < tol))


def f1_support_offdiag(X_hat: np.ndarray, X_true: np.ndarray, tol: float = 1e-4) -> float:
    """
    OPTIONAL: F1 on support of off-diagonal entries (|X_ij| > tol).
    """
    N = X_true.shape[0]
    iu = np.triu_indices(N, k=1)
    y_true = (np.abs(X_true[iu]) > tol).astype(int)
    y_pred = (np.abs(X_hat[iu]) > tol).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return float((2 * tp) / (2 * tp + fp + fn + 1e-12))


def theta_to_laplacian(Theta_hat: np.ndarray, thr: float = 1e-4) -> np.ndarray:
    """
    Convierte Theta_hat (precisión estimada) a un Laplaciano estimado:
      W_ij = |Theta_ij| si |Theta_ij| > thr, si no 0
      L_hat = diag(sum_j W_ij) - W
    """
    W = np.abs(Theta_hat).copy()
    np.fill_diagonal(W, 0.0)
    W[W < thr] = 0.0
    W = 0.5 * (W + W.T)

    d = W.sum(axis=1)
    L_hat = np.diag(d) - W
    return L_hat
