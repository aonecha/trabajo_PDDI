import numpy as np


def fro_error_rel(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    """
    Relative Frobenius error SOLO off-diagonal:
      ||(X_hat - X_true)_off||_F / ||(X_true)_off||_F
    Esto es coherente si ponéis la diagonal a 0 en las estimaciones.
    """
    N = X_true.shape[0]
    mask = ~np.eye(N, dtype=bool)

    diff = X_hat[mask] - X_true[mask]
    num = np.linalg.norm(diff)
    den = np.linalg.norm(X_true[mask]) + 1e-12
    return float(num / den)



def sparsity_offdiag(X: np.ndarray, tol: float = 1e-4) -> float:
    """
    Fraction of off-diagonal entries that are ~0.
    tol=1e-4 works much better for CVXPY / numerical solvers than 1e-8.
    """
    N = X.shape[0]
    mask = ~np.eye(N, dtype=bool)
    vals = np.abs(X[mask])
    return float(np.mean(vals < tol))


def f1_support_offdiag(X_hat: np.ndarray, X_true: np.ndarray, tol: float = 1e-4) -> float:
    """
    OPTIONAL: F1 on support of off-diagonal entries (|X_ij| > tol).
    Not printed by default; use only if you define this tol in the report.
    """
    N = X_true.shape[0]
    iu = np.triu_indices(N, k=1)
    y_true = (np.abs(X_true[iu]) > tol).astype(int)
    y_pred = (np.abs(X_hat[iu]) > tol).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return float((2 * tp) / (2 * tp + fp + fn + 1e-12))
