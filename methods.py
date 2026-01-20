import numpy as np
import cvxpy as cp
from sklearn.covariance import GraphicalLasso


def sample_cov_centered(X: np.ndarray) -> np.ndarray:
    """
    Calcula la covarianza muestral centrada por nodo.
    Devuelve S = (1/M) * Xc * Xc^T.
    """
    N, M = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ Xc.T) / max(M, 1)


# --------- 1) RIDGE (solver = inv) ----------
def estimate_theta_ridge_from_cov(S: np.ndarray, gamma: float = 1e-2) -> np.ndarray:
    """
    Calculamos theta a traves de la formula de ridge que nos indica
    que la matriz de precision es la inversa de la covarianza
    """
    N = S.shape[0]
    Theta = np.linalg.inv(S + gamma * np.eye(N))
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta


# --------- 2) GLASSO sklearn (solver = fit) ----------
def estimate_theta_glasso_sklearn_solver_time(
    X: np.ndarray, lam: float = 0.05, max_iter: int = 1000, tol: float = 1e-3
) -> np.ndarray:
    """
    Estima la matriz de precisión con Graphical Lasso usando la implementación de scikit-learn.
    Entradas:
      - X con forma (N, M). scikit-learn espera (M, N), por eso se usa X.T.
      - lam: regularización L1 (alpha en sklearn) que promueve esparsidad en la precisión.
      - max_iter, tol: parámetros de convergencia del solver interno.

    Salida:
      - Theta estimada (precision_) simetrizada y con diagonal puesta a cero.
  
    """
    model = GraphicalLasso(alpha=lam, max_iter=max_iter, tol=tol)
    model.fit(X.T) 
    Theta = model.precision_.copy()
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta


# --------- 3) GLASSO CVXPY (solver = prob.solve; problema cacheado) ----------
_CVXPY_CACHE = {}

def _get_cvxpy_problem(N: int, solver: str = "SCS"):
    """
    Modelo (típico Glasso):
      min_Theta  -logdet(Theta) + trace(S Theta) + lam * ||Theta_offdiag||_1
      s.a.       Theta es SPD (se impone Theta >> 1e-6 I)

    Definimos el objetivo y las restricciones del problema de Graphical Lasso en CVXPY.
    Guardamos el problema en la caché para reutilizarlo en futuras llamadas.
    
    Devuelve un diccionario con:
      - prob: el problema CVXPY listo para resolver
      - Theta: variable de decisión optima
      - S, lam: parámetros que se actualizan antes de cada solve
      - solver: nombre del solver asociado al caché
    """
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
    Recuperamos el problema cacheado, introducimos los datos y resolvemos.
    Con esto obtenemos la estimación de la matriz de precisión.
    
    Estima la matriz de precisión con Graphical Lasso formulado en CVXPY, usando la covarianza S.
    """
    N = S.shape[0]
    pack = _get_cvxpy_problem(N, solver=solver)
    pack["S"].value = 0.5 * (S + S.T)
    pack["lam"].value = float(lam)

    pack["prob"].solve(solver=solver, verbose=False)

    if pack["Theta"].value is None:
        raise RuntimeError("CVXPY failed: Theta.value is None (solver did not converge / infeasible).")

    Theta_hat = pack["Theta"].value
    Theta_hat = 0.5 * (Theta_hat + Theta_hat.T)
    np.fill_diagonal(Theta_hat, 0.0)
    return Theta_hat


# --------- 4) PGD (solver = bucle; con S precomputada) ----------
def estimate_theta_pgd_from_cov(
    S: np.ndarray,
    lam: float = 0.05,
    lr: float = 0.01,
    n_iter: int = 400,
    jitter: float = 1e-3,
) -> np.ndarray:
    """
    Estima Theta mediante un esquema tipo Proximal Gradient Descent (PGD)

    Esquema:
      - Paso de gradiente sobre: -logdet(Theta) + trace(S Theta)
        grad = S - Theta^{-1}
      - Paso proximal soft-thresholding en las entradas off-diagonal para la norma L1 (esparsidad)
      - Se fuerza simetría y se añade jitter a la diagonal para mantener SPD numéricamente
    
    Salida:
      - Theta estimada, con diagonal puesta a cero (coherente con las métricas off-diagonal).
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
