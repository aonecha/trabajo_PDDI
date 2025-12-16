import numpy as np
import cvxpy as cp
from sklearn.covariance import GraphicalLasso


def _sample_cov(X: np.ndarray) -> np.ndarray:
    """
    Sample covariance S = (1/M) X X^T, assuming zero-mean signals.
    X shape (N, M)
    """
    N, M = X.shape
    return (X @ X.T) / max(M, 1)


# --------- BASELINE: Ridge inverse covariance (precision) ----------
def estimate_theta_ridge_inv_cov(X: np.ndarray, gamma: float = 1e-2) -> np.ndarray:
    """
    Baseline precision estimate:
        Theta_hat = (S + gamma I)^{-1}

    gamma stabilizes inversion when M is small / S is ill-conditioned.
    """
    S = _sample_cov(X)
    N = S.shape[0]
    Theta = np.linalg.inv(S + gamma * np.eye(N))
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta


# --------- GLASSO (Coordinate descent via sklearn) ----------
def estimate_theta_glasso_sklearn(X: np.ndarray, lam: float = 0.05, max_iter: int = 1000, tol: float = 1e-3) -> np.ndarray:
    """
    GraphicalLasso via sklearn (block coordinate descent).
    Relaxed tol and max_iter to avoid long runtimes / convergence warnings at small N.
    """
    model = GraphicalLasso(alpha=lam, max_iter=max_iter, tol=tol)
    model.fit(X.T)  # (M, N)
    Theta = model.precision_.copy()
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta



# --------- GLASSO (CVXPY) ----------
def estimate_theta_glasso_cvxpy(X: np.ndarray, lam: float = 0.05, solver: str = "SCS") -> np.ndarray:
    """
    Same objective as GLASSO, solved with a generic convex solver (slower).
    """
    S = _sample_cov(X)
    N = S.shape[0]

    Theta = cp.Variable((N, N), symmetric=True)
    obj = -cp.log_det(Theta) + cp.trace(S @ Theta) + lam * cp.norm1(Theta)
    constraints = [Theta >> 1e-6 * np.eye(N)]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=solver, verbose=False)

    Theta_hat = Theta.value
    Theta_hat = 0.5 * (Theta_hat + Theta_hat.T)
    np.fill_diagonal(Theta_hat, 0.0)
    return Theta_hat


# --------- Proximal Gradient (simple) ----------
def estimate_theta_pgd(
    X: np.ndarray,
    lam: float = 0.05,
    lr: float = 0.01,
    n_iter: int = 400,
    jitter: float = 1e-3,
) -> np.ndarray:
    """
    Simplified proximal-gradient-like scheme on the GLASSO objective.
    Not a production solver; intended for educational comparison.

    Update:
      grad = S - Theta^{-1}
      Theta <- Theta - lr * grad
      Theta <- SoftThreshold(Theta, lr*lam)   (prox for L1)
      Theta <- symmetrize + jitter*I
    """
    S = _sample_cov(X)
    N = S.shape[0]

    Theta = np.eye(N)
    for _ in range(n_iter):
        Theta_inv = np.linalg.inv(Theta + 1e-8 * np.eye(N))
        grad = S - Theta_inv
        Theta = Theta - lr * grad

        # soft-thresholding (prox L1)
        Theta = np.sign(Theta) * np.maximum(np.abs(Theta) - lr * lam, 0.0)

        # symmetrize and keep PD-ish
        Theta = 0.5 * (Theta + Theta.T)
        Theta = Theta + jitter * np.eye(N)

    np.fill_diagonal(Theta, 0.0)
    return Theta
