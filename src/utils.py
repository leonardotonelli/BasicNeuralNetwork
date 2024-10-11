import numpy as np


import numpy as np

def grad(f, x, delta=1e-5):
    n = len(x)
    g = np.zeros(n)                         
    for i in range(n):                      
        x_old = x[i]
        # change x[i] to x[i] + delta
        x[i] = x_old + delta
        fp = f(x)                           
        # change x[i] to x[i] - delta
        x[i] = x_old - delta
        fm = f(x)                           
        # restore x[i] to its original value
        x[i] = x_old

        ## compute the i-th component of the gradient and save it
        g[i] = (fp - fm) / (2 * delta)
    return g

def norm(x):
    return np.sqrt(np.sum(x**2))


class GDResults:
    def __init__(self, x, y, converged, iters, xs):
        self.x = x
        self.fun = y
        self.converged = converged
        self.iters = iters
        self.xs = xs

    def __repr__(self):
        s = ""
        if self.converged:
            s += "Optimization terminated successfully\n"
        else:
            s += "Optimization failed\n"
        s += f"  final x value: {self.x}\n"
        s += f"  final function value: {self.fun}\n"
        s += f"  iterations: {self.iters}\n"
        return s

def grad_desc(f, x0,
              grad_f = None,
              max_t = 1000,
              alpha = 0.01,
              beta = 0.0,
              epsilon = 1e-5,
              callback = None,
              verbosity = 0,
              keep_intermediate = False
              ):
    """
    Gradient Descent Algorithm: Minimizes `f` using gradient descent with step size `alpha`
    and optional Nesterov momentum `beta`. The gradient can be provided as `grad_f` or computed via finite differences.

    Parameters:
    - `f`: Function to minimize, takes a 1D NumPy array and returns a scalar.
    - `x0`: Initial guess, a 1D NumPy array.
    - `grad_f` (optional): Gradient of `f`, returns a 1D array of the same size. Defaults to finite-difference approximation.
    - `alpha`: Step size (default `1e-2`), must be positive.
    - `beta`: Nesterov momentum (default `0.0`), must be in [0, 1). If `0`, pure gradient descent.
    - `epsilon`: Convergence criterion for gradient norm (default `1e-5`).
    - `max_t`: Maximum iterations (default `1000`).
    - `verbosity`: Controls output (`0`: none, `1`: final result, `2+`: detailed output).
    - `keep_intermediate`: If `True`, stores intermediate results (default `False`).
    - `callback`: Function executed at each iteration, stops if it returns `True`.

    Returns: A `GDResults` object containing the final values, iteration count, and optionally, intermediate results.
    """

    if grad_f is None:
        grad_f = lambda xx: grad(f, xx)
    x = x0.copy()
    xs = []
    if keep_intermediate:
        xs.append(x0.copy())
    v = np.zeros(len(x))           
    converged = False
    for k in range(max_t):
        v *= beta
        p = grad_f(x + v)
        assert len(p) == len(x)
        v -= alpha * p
        x += v
        if verbosity >= 2 or keep_intermediate:
            y = f(x)
        if verbosity >= 2:
            print(f"step={k} x={x} f(x)={y} grad={p}")
        if keep_intermediate:
            xs.append(x.copy())
        if callback is not None:
            if callback(x):
                break
        if norm(p) < epsilon:
            converged = True
            break
    xs = np.array(xs)
    res = GDResults(x, f(x), converged, k+1, xs)
    if verbosity >= 1:
        print(res)
    return res

