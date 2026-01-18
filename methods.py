import numpy as np
import cvxpy as cp
from sklearn.covariance import GraphicalLasso


def sample_cov_centered(X: np.ndarray) -> np.ndarray:
    """
    Calcula la covarianza muestral centrada por nodo.

    Entrada:
      - X con forma (N, M): N nodos/variables, M muestras.
    Proceso:
      - Centra cada fila (nodo) restando su media.
      - Devuelve S = (1/M) * Xc * Xc^T.
    """
    N, M = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    return (Xc @ Xc.T) / max(M, 1)


# --------- 1) RIDGE (solver = inv) ----------
def estimate_theta_ridge_from_cov(S: np.ndarray, gamma: float = 1e-2) -> np.ndarray:
    """
    Estima una matriz de precisión mediante un estimador tipo ridge a partir de la covarianza S.

    Idea:
      - Regulariza la covarianza: (S + gamma I) para que sea invertible/estable.
      - Estima Theta = (S + gamma I)^{-1}.
      - Fuerza simetría y anula la diagonal (coherente con evaluar solo off-diagonal).
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
    Nota:
      - En tus experimentos, el tiempo de solver se mide alrededor de esta llamada.
    """
    model = GraphicalLasso(alpha=lam, max_iter=max_iter, tol=tol)
    model.fit(X.T)  # solver
    Theta = model.precision_.copy()
    Theta = 0.5 * (Theta + Theta.T)
    np.fill_diagonal(Theta, 0.0)
    return Theta


# --------- 3) GLASSO CVXPY (solver = prob.solve; problema cacheado) ----------
_CVXPY_CACHE = {}

def _get_cvxpy_problem(N: int, solver: str = "SCS"):
    """
    Crea (o reutiliza desde caché) el problema de Graphical Lasso en CVXPY para tamaño N.

    Motivación:
      - Construir el problema CVXPY (variables, parámetros, objetivo y restricciones) es costoso.
      - Como en los sweeps se resuelve muchas veces para el mismo N, se cachea y solo se cambian:
          * S_param (covarianza)
          * lam_param (regularización)

    Devuelve un diccionario con:
      - prob: el problema CVXPY listo para resolver
      - Theta: variable de decisión
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
    Estima la matriz de precisión con Graphical Lasso formulado en CVXPY, usando la covarianza S.

    Modelo (típico Glasso):
      min_Theta  -logdet(Theta) + trace(S Theta) + lam * ||Theta_offdiag||_1
      s.a.       Theta es SPD (se impone Theta >> 1e-6 I)

    Implementación:
      - Reutiliza un problema cacheado para el tamaño N.
      - Actualiza los parámetros S y lam y resuelve con el solver indicado.
      - Comprueba convergencia (Theta.value no None), simetriza y anula diagonal.

    Devuelve:
      - Theta_hat (N x N) estimada.
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
    Estima Theta mediante un esquema tipo Proximal Gradient Descent (PGD) sobre el objetivo Glasso.

    Esquema (aproximado):
      - Paso de gradiente sobre: -logdet(Theta) + trace(S Theta)
        grad = S - Theta^{-1}
      - Paso proximal soft-thresholding en las entradas off-diagonal para la norma L1 (esparsidad)
      - Se fuerza simetría y se añade jitter a la diagonal para mantener SPD numéricamente

    Entradas:
      - S: covarianza muestral (ya precomputada fuera del timing del solver).
      - lam: fuerza de esparsidad (umbral del shrinkage).
      - lr: learning rate del paso de gradiente.
      - n_iter: número de iteraciones del bucle.
      - jitter: pequeña suma en diagonal para estabilidad/invertibilidad.

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
