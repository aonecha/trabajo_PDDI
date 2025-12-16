import numpy as np
import cvxpy as cp
from sklearn.covariance import GraphicalLasso


# -----------------------------
# Covarianza (fuera del solver)
# -----------------------------
def sample_cov_centered(X: np.ndarray) -> np.ndarray:
    """
    X shape (N, M). Devuelve S=(1/M) Xc Xc^T con X centrado por nodo.
    Esto NO es solver, lo mediremos fuera.
    """
    N, M = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ Xc.T) / max(M, 1)


# ============================================================
# 1) RIDGE: solver = inversión de matriz
# ============================================================
def estimate_theta_ridge_from_cov(S: np.ndarray, gamma: float = 1e-2) -> np.ndarray:
    """
    Solver-only: Theta = (S + gamma I)^{-1}
    """
    N = S.shape[0]
    Theta = np.linalg.inv(S + gamma * np.eye(N))
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta


# ============================================================
# 2) GLASSO (sklearn): solver = model.fit
# ============================================================
def estimate_theta_glasso_sklearn_solver_time(X: np.ndarray, lam: float = 0.05, max_iter: int = 1000, tol: float = 1e-3) -> np.ndarray:
    """
    Aquí el solver real es model.fit(X.T). No se puede separar más fino
    (GraphicalLasso hace su pipeline interno).
    """
    model = GraphicalLasso(alpha=lam, max_iter=max_iter, tol=tol)
    model.fit(X.T)  # solver
    Theta = model.precision_.copy()
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta


# ============================================================
# 3) GLASSO (CVXPY): solver = prob.solve
#    -> cacheamos el problema para NO medir la construcción
# ============================================================
_CVXPY_CACHE = {}  # key: (N, solver) -> dict con prob, Theta_var, S_param, lam_param

def _get_cvxpy_problem(N: int, solver: str = "SCS"):
    key = (int(N), str(solver))
    if key in _CVXPY_CACHE:
        return _CVXPY_CACHE[key]

    Theta = cp.Variable((N, N), symmetric=True)
    S_param = cp.Parameter((N, N), symmetric=True)
    lam_param = cp.Parameter(nonneg=True)

    offdiag = Theta - cp.diag(cp.diag(Theta))
    obj = -cp.log_det(Theta) + cp.trace(S_param @ Theta) + lam_param * cp.norm1(offdiag)
    constraints = [Theta >> 1e-6 * np.eye(N)]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    _CVXPY_CACHE[key] = {"prob": prob, "Theta": Theta, "S": S_param, "lam": lam_param, "solver": solver}
    return _CVXPY_CACHE[key]


def estimate_theta_glasso_cvxpy_from_cov(S: np.ndarray, lam: float = 0.05, solver: str = "SCS") -> np.ndarray:
    """
    Solver-only: mediremos SOLO prob.solve. La construcción está cacheada.
    """
    N = S.shape[0]
    pack = _get_cvxpy_problem(N, solver=solver)
    pack["S"].value = 0.5 * (S + S.T)
    pack["lam"].value = float(lam)

    # El solve se hace fuera del timing en experiments.py para que sea puro.
    pack["prob"].solve(solver=solver, verbose=False)

    Theta_hat = pack["Theta"].value
    Theta_hat = 0.5 * (Theta_hat + Theta_hat.T)
    np.fill_diagonal(Theta_hat, 0.0)
    return Theta_hat


# ============================================================
# 4) PGD: solver = bucle iterativo (usando S ya computada)
# ============================================================
def estimate_theta_pgd_from_cov(
    S: np.ndarray,
    lam: float = 0.05,
    lr: float = 0.01,
    n_iter: int = 400,
    jitter: float = 1e-3,
) -> np.ndarray:
    """
    Solver-only: bucle PGD con soft-threshold SOLO off-diagonal.
    """
    N = S.shape[0]
    Theta = np.eye(N)

    for _ in range(n_iter):
        Theta_inv = np.linalg.inv(Theta + 1e-8 * np.eye(N))
        grad = S - Theta_inv
        Theta = Theta - lr * grad

        diag = np.diag(np.diag(Theta))
        off = Theta - diag
        off = np.sign(off) * np.maximum(np.abs(off) - lr * lam, 0.0)
        Theta = diag + off

        Theta = 0.5 * (Theta + Theta.T)
        Theta = Theta + jitter * np.eye(N)

    np.fill_diagonal(Theta, 0.0)
    return Theta
