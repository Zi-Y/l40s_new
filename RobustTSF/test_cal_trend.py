import numpy as np
import cvxpy
from scipy.sparse import spdiags, coo_matrix, dia_matrix, dok_matrix
from itertools import chain

# -----------------------------------------------------------------------------
# Utility functions for finite differences and linear deviations
# -----------------------------------------------------------------------------

def first_derivative_matrix(n):
    """
    Construct an (n-1) x n sparse matrix D1 that computes backward differences:
        (D1 @ s)[i] = s[i+1] - s[i],  for i = 0 .. n-2
    Args:
        n (int): length of the signal s.
    Returns:
        scipy.sparse matrix of shape (n-1, n).
    """
    e = np.ones((1, n))
    # two diagonals: +1 on superdiagonal, -1 on main diagonal
    return spdiags(np.vstack((-e, e)), [0, 1], n - 1, n)


def second_derivative_matrix_nes(x, a_min=0.0, a_max=None, scale_free=False):
    """
    Construct an (n-2) x n matrix D2 for second differences on non-evenly spaced x:
        (D2 @ s)[k] ≈ s[k] - 2*s[k+1] + s[k+2], scaled by spacing.
    Args:
        x (array): increasing x-coordinates of length n.
        a_min (float): minimum allowed spacing.
        a_max (float): maximum allowed spacing (None = no clamp).
        scale_free (bool): if True, scale weights by half-sum of spacings.
    Returns:
        scipy.sparse.dia_matrix of shape (n-2, n).
    """
    n = len(x)
    m = n - 2
    values = []
    for i in range(1, n - 1):
        # compute local spacings
        a0 = max(x[i] - x[i - 1], a_min)
        a2 = max(x[i + 1] - x[i], a_min)
        if a_max is not None:
            a0 = min(a0, a_max)
            a2 = min(a2, a_max)
        a1 = a0 + a2
        scf = (a1 / 2.0) if scale_free else 1.0
        # weights for [s[i-1], s[i], s[i+1]]
        values.extend([2*scf/(a1*a2), -2*scf/(a0*a2), 2*scf/(a0*a1)])
    # build sparse indices
    row_idx = list(chain(*([[k]*3 for k in range(m)])))
    col_idx = list(chain(*([[k, k+1, k+2] for k in range(m)])))
    D2_coo = coo_matrix((values, (row_idx, col_idx)), shape=(m, n))
    return dia_matrix(D2_coo)


def first_derv_nes_cvxpy(x, y_var):
    """
    Return a CVXPY expression for first-order derivative of y_var wrt x:
        dy/dx at points between each consecutive x.
    Args:
        x (array): x-coordinates length n.
        y_var (cvxpy.Variable): variable of length n.
    Returns:
        cvxpy Expression of length n-1.
    """
    n = len(x)
    eps = 1e-9  # stability to avoid divide-by-zero
    inv_dx = 1.0 / (np.diff(x) + eps)
    # build backward difference matrix: -1 and +1
    M = spdiags(np.vstack((-np.ones((1, n)), np.ones((1, n)))), [0, 1], n - 1, n)
    return cvxpy.multiply(inv_dx, M @ y_var)


def get_model_deviation_matrix(x, mapping, n_deviates):
    """
    Build a binary mapping matrix that assigns each x to one of n_deviates bins.
    Args:
        x (array): values of length n.
        mapping (callable): function mapping each x[i] to integer in [0, n_deviates).
        n_deviates (int): number of deviation variables.
    Returns:
        scipy.sparse.dok_matrix of shape (n, n_deviates).
    """
    num_x = len(x)
    M = dok_matrix((num_x, n_deviates), dtype=np.float32)
    for i, xi in enumerate(x):
        j = mapping(xi)
        M[i, j] = 1.0
    return M


def complete_linear_deviations(linear_deviations, x):
    """
    Initialize linear_deviations: fill defaults and build model contributions.
    Each item in linear_deviations must provide:
      - 'n_vars': number of variables
      - 'mapping': function x->index
    Returns:
        List of dicts with keys:
          'name', 'alpha', 'variable', 'model_contribution'.
    """
    completed = []
    names = set()
    for idx, dev in enumerate(linear_deviations):
        d = dev.copy()
        name = d.get('name', f'linear_dev_{idx}')
        assert name not in names
        names.add(name)
        alpha = d.get('alpha', 1e-3)
        n_vars = d['n_vars']
        mapping = d['mapping']
        matrix = d.get('matrix', get_model_deviation_matrix(x, mapping, n_vars))
        var = d.get('variable', cvxpy.Variable(n_vars))
        completed.append({
            'name': name,
            'alpha': alpha,
            'variable': var,
            'model_contribution': matrix @ var
        })
    return completed

# -----------------------------------------------------------------------------
# Main API: compute_trend
# -----------------------------------------------------------------------------

def compute_trend(
    x,
    y,
    y_err=None,
    alpha_1=0.0,
    alpha_2=0.0,
    l_norm=2,
    constrain_zero=False,
    monotonic=False,
    positive=False,
    linear_deviations=None,
    loss='mse',
    solver='ECOS'
):
    """
    Trend filtering via convex optimization:
        minimize  sum(|y - s| or Huber) + alpha_1*||D1 s||_p + alpha_2*||D2 s||_p
    Supports optional constraints and linear deviations.

    Args:
        x (array): independent variable of length n.
        y (array): observations of length n.
        y_err (array): optional per-point error estimate; default ones.
        alpha_1 (float): weight for first-derivative penalty.
        alpha_2 (float): weight for second-derivative penalty.
        l_norm (int): 1 or 2, choose L1 or L2 norm for penalties.
        constrain_zero (bool): enforce s[0] == 0.
        monotonic (bool): enforce D1 s >= 0 (non-decreasing trend).
        positive (bool): enforce s >= 0.
        linear_deviations (list): list of dicts specifying additional linear terms.
        loss (str): 'mse' (uses Huber) or 'mae'.
        solver (str): CVXPY solver name.

    Returns:
        s (np.ndarray): trend estimate of length n.
    """
    n = len(x)
    # --- 1) observation weights ---
    if y_err is None:
        y_err = np.ones(n)
    buff = 0.01 * np.median(np.abs(y))
    isig = 1.0 / np.sqrt(buff**2 + y_err**2)

    # --- 2) setup linear deviations ---
    lin_devs = [] if linear_deviations is None else linear_deviations
    lin_devs = complete_linear_deviations(lin_devs, x)

    # --- 3) define CVXPY variables ---
    base = cvxpy.Variable(n, pos=positive)
    model = base
    for dev in lin_devs:
        model = model + dev['model_contribution']

    # --- 4) data fidelity term ---
    diff = cvxpy.multiply(isig, model - y)
    if loss == 'mse':
        data_term = cvxpy.sum(cvxpy.huber(diff))
    else:
        data_term = cvxpy.sum(cvxpy.abs(diff))

    # --- 5) derivative penalties ---
    d1 = first_derv_nes_cvxpy(x, base)           # (n-1,)
    D2 = second_derivative_matrix_nes(x, scale_free=True)  # (n-2, n)
    norm_fn = cvxpy.sum_squares if l_norm == 2 else cvxpy.norm1
    reg_terms = [alpha_1 * norm_fn(d1), alpha_2 * norm_fn(D2 @ base)]
    for dev in lin_devs:
        reg_terms.append(dev['alpha'] * norm_fn(dev['variable']))
    reg_sum = sum(reg_terms)

    # --- 6) solve convex problem ---
    obj = cvxpy.Minimize(data_term + reg_sum)
    constraints = []
    if constrain_zero:
        constraints.append(base[0] == 0)
    if monotonic:
        constraints.append(d1 >= 0)
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver=solver)

    # --- 7) return trend vector ---
    return base.value

# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Generate synthetic data
    x = np.linspace(0, 10, 200)
    trend_true = 0.2 * x + np.sin(x / 2)
    y = trend_true + 0.5 * np.random.randn(len(x))

    # Compute trend with L1 penalty on curvature
    s = compute_trend(
        x, y,
        alpha_1=0.0,
        alpha_2=10.0,
        l_norm=1,
        constrain_zero=False,
        monotonic=False,
        positive=False,
        loss='mae',
        solver='ECOS'
    )

    # Plot original vs estimated trend
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, '.', alpha=0.3, label='Noisy observations')
    plt.plot(x, trend_true, 'k--', lw=2, label='True trend')
    plt.plot(x, s, 'r-', lw=2, label='Estimated trend')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trend Filtering via Convex Optimization')
    plt.tight_layout()
    plt.show()