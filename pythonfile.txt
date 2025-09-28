# %%
# Prerequisites
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from numba import njit, prange
import pandas as pd
import warnings
import linecache
import os
import time

from autograd import grad
import autograd.numpy as anp
from scipy.special import factorial

plt.style.use(["science", "high-vis", "grid"])
plt.rcParams.update(
    {
        "figure.figsize": (9, 9),
        "lines.markersize": 5,
        "lines.linewidth": 1.5,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "figure.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.constrained_layout.use": True,
        "legend.frameon": True,
        "axes.titlepad": 10,
    }
)


def warning_custom(message, category, filename, lineno, file=None, line=None):
    src = linecache.getline(filename, lineno).strip()
    short_file = os.path.basename(filename)
    print(f"{category.__name__} in {short_file}, line {lineno}: {message}\n    {src}")


warnings.showwarning = warning_custom


# %%
@njit(parallel=True)
def lagrange(x_data, y_data, x_eval):
    """Lagrange interpolation

    Args:
        x_data (array): x0,...,xn
        y_data (array): y0,...,yn
        x_eval (array): Evaluation points

    Returns:
        array: the value of the interpolation polynomial in x_eval
    """
    n = len(x_data) - 1
    eta = len(x_eval)
    y_eval = np.empty(eta)

    for k in prange(eta):
        p = 0.0  # p_n(x_eval[k])
        for i in range(n + 1):
            L_i = 1.0
            for j in range(n + 1):
                if i != j:
                    L_i *= (
                        (x_eval[k] - x_data[j]) / (x_data[i] - x_data[j])
                    )  # L_i(x_eval[k]) = prod_{j=0, j!=i} (x_eval[k]-x_data[j])/(x_data[i]-x_data[j])
            p += y_data[i] * L_i
        y_eval[k] = p

    return y_eval


@njit
def cheby_nodes(a, b, n):
    """Chebyshev nodes on [a,b]

    Args:
        a (float): left bound interval
        b (float): right bound interval
        n (int): number of nodes - 1

    Returns:
        array: nodes on the interval
    """
    # n + 1 cheby nodes in the interval [a, b]
    i = np.arange(n + 1)  # i = [0, 1, 2, 3,..., n]
    x = np.cos((2 * i + 1) * np.pi / (2 * (n + 1)))  # nodes over the interval [-1,1]
    return 0.5 * (b - a) * x + 0.5 * (b + a)  # nodes over the interval [a, b]


@njit
def runge_f(x):
    return 1 / (x**2 + 1)


def plotA(n=10, a=-5, b=5):
    x_equi = np.linspace(a, b, n + 1)
    x_cheby = cheby_nodes(a=a, b=b, n=n)

    y_equi = runge_f(x=x_equi)
    y_cheby = runge_f(x=x_cheby)
    x_eval = np.linspace(a, b, 1000)
    y_true = runge_f(x=x_eval)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

    ax1.plot(
        x_eval,
        lagrange(x_data=x_equi, y_data=y_equi, x_eval=x_eval),
        label="Equidistant",
    )
    ax1.scatter(x_equi, y_equi)

    ax2.plot(
        x_eval,
        lagrange(x_data=x_cheby, y_data=y_cheby, x_eval=x_eval),
        label="Chebyshev",
    )
    ax2.scatter(x_cheby, y_cheby)

    for ax in (ax1, ax2):
        ax.plot(x_eval, y_true, "--", label=r"$f$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.legend(loc="lower center")

    plt.suptitle(f"Runge function $(1+x^2)^{{-1}}$ on {n + 1} nodes")
    plt.show()


plotA()


# %%
# Define norm functions
def norm_inf(y_true, y_pred):
    """Returns the estimated infinity norm of two arrays of equal shape

    Args:
        y_true (NDArray)): True values of our function in some evaluation points
        y_pred (NDArray): Estimated values of our function in the same evaluation points

    Returns:
        NDArray: Estimated infinity norm of y_true and y_pred
    """
    return np.max(np.abs(y_true - y_pred))


def norm_2(y_true, y_pred, a, b, N):
    """Returns the estimated 2-norm of two arrays of equal shape

    Args:
        y_true (NDArray)): True values of our function in x
        y_pred (NDArray): Estimated values of our function in x

    Returns:
        NDArray: Estimated 2-norm of y_true and y_pred
    """
    return np.sqrt((b - a) / N * np.sum((y_true - y_pred) ** 2))


# Make sympy versions of the functions so we can easily display the functions in plots
x, k, n_ = sp.symbols(names="x k n")
f1 = sp.cos(2 * sp.pi * x)
f2 = sp.exp(3 * x) * sp.sin(2 * x)

# Same as above but for the derived error bounds
error_bound_equi_1 = (2 * sp.pi / k) ** (k + 1) / (4 * (k + 1))
error_bound_cheby_1 = (sp.pi) ** (k + 1) / (
    2 ** (k) * sp.factorial(k + 1)
)  # Theorem 8.1
error_bound_equi_2 = (
    sp.exp(3 * sp.pi / 4)
    * (sp.sqrt(13) * sp.pi) ** (k + 1)
    / (4 ** (k + 2) * k ** (k + 1) * (k + 1))
)
error_bound_cheby_2 = (
    sp.exp(3 * sp.pi / 4)
    * (sp.sqrt(13) * sp.pi) ** (k + 1)
    / (2 ** (4 * k + 3) * sp.factorial(k + 1))
)

# Lambdify to make python functions
bound_func_equi_1 = sp.lambdify(
    args=k, expr=error_bound_equi_1, modules=[{"factorial": factorial}, "numpy"]
)
bound_func_cheby_1 = sp.lambdify(
    args=k, expr=error_bound_cheby_1, modules=[{"factorial": factorial}, "numpy"]
)
bound_func_equi_2 = sp.lambdify(
    args=k, expr=error_bound_equi_2, modules=[{"factorial": factorial}, "numpy"]
)
bound_func_cheby_2 = sp.lambdify(
    args=k, expr=error_bound_cheby_2, modules=[{"factorial": factorial}, "numpy"]
)

# Make tuples to iterate through when computing errors for different functions
f_funcs = (
    sp.lambdify(args=x, expr=f1, modules="numpy"),
    sp.lambdify(args=x, expr=f2, modules="numpy"),
)
fs = (f1, f2)

# The parameter values we use in our plots
n_max = 30  # Highest degree of polynomial to consider
n_range = np.arange(1, n_max + 1, dtype=int)  # All degrees to be considered
domains = [(0, 1), (0, np.pi / 4)]  # The different domains for our two functions

fig, axs = plt.subplots(2, 1)

# Iterate through our two functions and their domains
for i, (f, (a, b)) in enumerate(zip(f_funcs, domains)):
    # Initialize arrays for which to place our computed errors in
    norm_inf_equi = np.zeros(n_max)
    norm_inf_cheby = np.zeros(n_max)
    norm_2_equi = np.zeros(n_max)
    norm_2_cheby = np.zeros(n_max)

    # Iterate through degree n of interpolation polynomial
    for n in n_range:
        idx = int(n) - 1
        N = 100 * int(n)

        # The nodes to consider
        x_eval = np.linspace(a, b, N + 1)
        x_equi = np.linspace(a, b, int(n) + 1)
        x_cheby = cheby_nodes(a=a, b=b, n=int(n))

        # Estimated and true values of f
        y_true = f(x_eval)
        y_equi = lagrange(x_data=x_equi, y_data=f(x_equi), x_eval=x_eval)
        y_cheby = lagrange(x_data=x_cheby, y_data=f(x_cheby), x_eval=x_eval)

        # Add norms to their respective array
        norm_inf_equi[idx] = norm_inf(y_true=y_true, y_pred=y_equi)
        norm_inf_cheby[idx] = norm_inf(y_true=y_true, y_pred=y_cheby)
        norm_2_equi[idx] = norm_2(y_true=y_true, y_pred=y_equi, a=a, b=b, N=N)
        norm_2_cheby[idx] = norm_2(y_pred=y_true, y_true=y_cheby, a=a, b=b, N=N)

    # Plot the estimated errors
    axs[i].semilogy(
        n_range,
        norm_inf_equi,
        "o-",
        label=r"Equidistant $\left\|  f(x) - p_n(x) \right\|_{{\infty}}$",
    )
    axs[i].semilogy(
        n_range,
        norm_inf_cheby,
        "o-",
        label=r"Chebyshev $\left\|  f(x) - p_n(x) \right\|_{{\infty}}$",
    )
    axs[i].semilogy(
        n_range,
        norm_2_equi,
        "o-",
        label=r"Equidistant $\left\|  f(x) - p_n(x) \right\|_2$",
    )
    axs[i].semilogy(
        n_range,
        norm_2_cheby,
        "o-",
        label=r"Chebyshev $\left\|  f(x) - p_n(x) \right\|_2$",
    )

    axs[i].set_xlabel("Degree of interpolation polynomial $[n]$")
    axs[i].set_ylabel("Estimated error")
    axs[i].set_title(rf"Estimated error $\left\| {sp.latex(fs[i])} - p_n(x) \right\|$")

# Plot the theoretical error bounds
axs[0].semilogy(
    n_range,
    bound_func_equi_1(n_range.astype(float)),
    "--",
    label=rf"Equidistant bound ${sp.latex(error_bound_equi_1.subs(k, n_))}$",
)
axs[0].semilogy(
    n_range,
    bound_func_cheby_1(n_range.astype(float)),
    "--",
    label=rf"Chebyshev bound ${sp.latex(error_bound_cheby_1.subs(k, n_))}$",
)
axs[1].semilogy(
    n_range,
    bound_func_equi_2(n_range.astype(float)),
    "--",
    label=rf"Equidistant bound ${sp.latex(error_bound_equi_2.subs(k, n_))}$",
)
axs[1].semilogy(
    n_range,
    bound_func_cheby_2(n_range.astype(float)),
    "--",
    label=rf"Chebyshev bound ${sp.latex(error_bound_cheby_2.subs(k, n_))}$",
)

for i in [0, 1]:
    axs[i].legend()
    axs[i].set_ylim(10 ** (-16), 10**2.5)

plt.show()
# %%


@njit
def f_cos(x):
    return np.cos(2 * np.pi * x)  # Define our function


# @njit
def taskC(a, b, K_max, n_max=10):
    """Piecewise-polynomial approximation

    Args:
        a (float): left bound interval
        b (float): right bound interval
        K_max (int): Maximum amount of subintervals
        n_max (int, optional): Maximum int. pol. degree on subinterval. Defaults to 10.

    Returns:
        tuple:  K_vals: 1*K_max array;
                results: n_max*K_max array, with error for each
        degree and each number of subintervals.
    """
    K_vals = np.arange(1, K_max + 1)
    results = np.zeros((n_max, K_max))
    times = np.zeros((n_max, K_max))

    for i in range(K_max):
        K = K_vals[i]  # Set K amount of subintervals
        v_a_arr = np.empty(K)
        v_b_arr = np.empty(K)

        for j in range(K):
            v_a_arr[j] = a + j * (b - a) / K  # Array with v_0,v_1,...,v_K-1
            v_b_arr[j] = a + (j + 1) * (b - a) / K  # Array with v_1,v_2,...,v_K

        for n in range(1, n_max + 1):  # Loop through max degree to be used
            start = time.perf_counter()  # start timer
            norm_inf_subints = np.zeros(K)

            for j in range(K):
                v_a = v_a_arr[j]  # Set subinterval left bound
                v_b = v_b_arr[j]  # Set subinterval right bound
                x_equi = np.linspace(v_a, v_b, n)  # Interpolation points on subinterval
                x_eval = np.linspace(
                    v_a, v_b, 100 * n
                )  # Evaluation points on subinterval
                y_equi = f_cos(x_equi)  # Interpolation y values on subinterval
                y_eval = lagrange(
                    x_equi, y_equi, x_eval
                )  # Evaluation y values on subinterval
                norm_inf_subints[j] = norm_inf(
                    y_true=f_cos(x_eval), y_pred=y_eval
                )  # Max error for current subinterval and degree n
            results[n - 1, i] = np.max(
                norm_inf_subints
            )  # Max error over every subinterval for current degree n
            times[n - 1, i] = time.perf_counter() - start  # elapsed time
    return K_vals, results, times


K_vals, results, times = taskC(a=0, b=1, K_max=200, n_max=10)

fig, axs = plt.subplots(3, 1, figsize=(9, 25))
fig.suptitle(r"Estimated error $\left\| f(x) - p_n(x) \right\|_{\infty}$")
for n in range(10):
    axs[0].semilogy(K_vals, results[n], label=rf"$n = {n + 1}$")
axs[0].set_title(r"Error vs. $K$")
axs[0].set_xlabel(r"Number of subintervals $[K]$")
axs[0].set_ylabel("Error")
axs[0].legend()

# n_max*K_max array with number of discretization points for each n and K
i = np.arange(10).reshape(-1, 1)
j = np.arange(200).reshape(1, -1)
arr = (i + 1) * (j + 1) + 1

for n in range(10):
    axs[1].loglog(
        arr[n], results[n], "o-", label=rf"$n={n + 1}$"
    )  # Plot the error for degree n and number of disc. points

# Compare to normal lagrange interpolation
n_max = 150
n_vals_lagrange = np.linspace(1, n_max, n_max)
norm_inf_equi = np.zeros(n_max)
norm_inf_cheby = np.zeros(n_max)
times_lagrange_equi = np.zeros(n_max)
times_lagrange_cheby = np.zeros(n_max)

for idx, n in enumerate(n_vals_lagrange):
    N = 100 * int(n)
    x_eval = np.linspace(a, b, N + 1)
    y_true = runge_f(x=x_eval)

    start = time.perf_counter()
    x_equi = np.linspace(a, b, int(n) + 1)
    y_equi = lagrange(x_data=x_equi, y_data=runge_f(x_equi), x_eval=x_eval)
    norm_inf_equi[idx] = norm_inf(y_true=y_true, y_pred=y_equi)
    times_lagrange_equi[idx] = time.perf_counter() - start

    start = time.perf_counter()
    x_cheby = cheby_nodes(a, b, n=int(n) + 1)
    y_cheby = lagrange(x_data=x_cheby, y_data=runge_f(x_cheby), x_eval=x_eval)
    norm_inf_cheby[idx] = norm_inf(y_true=y_true, y_pred=y_cheby)
    times_lagrange_cheby[idx] = time.perf_counter() - start


axs[1].loglog(
    n_vals_lagrange,
    norm_inf_equi,
    "o--",
    markerfacecolor="none",
    label=r"Equi. LI.",
)
axs[1].loglog(
    n_vals_lagrange,
    norm_inf_cheby,
    "o--",
    markerfacecolor="none",
    label=r"Cheby. LI.",
)

axs[1].set_title(r"Error vs. total number of discretization points $[nK+1]$")
axs[1].set_xlabel("Number of discretization points")
axs[1].set_ylabel("Error")
axs[1].set_ylim(10 ** (-15), 100)
axs[1].legend()

for n in range(10):
    axs[2].loglog(times[n], results[n], "o-", label=rf"$n={n + 1}$")
axs[2].loglog(
    times_lagrange_equi,
    norm_inf_equi,
    "o--",
    markerfacecolor="none",
    label=r"Cheby. LI.",
)
axs[2].loglog(
    times_lagrange_cheby,
    norm_inf_cheby,
    "o--",
    markerfacecolor="none",
    label=r"Cheby. LI.",
)
axs[2].set_title(r"Error vs. time")
axs[2].set_xlabel("Time [s]")
axs[2].set_ylabel("Error")
axs[2].legend()
axs[2].set_ylim(10 ** (-15), 100)
plt.show()


# %%
@njit
def phi(r, epsilon):
    return np.exp(-((epsilon * r) ** 2))


@njit
def RBF(x_data, y_data, x_eval, epsilon=1.26):
    """The radial basis function

    Args:
        x_data (NDArray): nodes of length n+1
        y_data (NDArray): f(x_data) for the function to approximate
        x_eval (NDArray): eta-values of length N+1
        epsilon (float, optional): Shape parameter. Defaults to 1.26.

    Returns:
        NDArray: The RBF in x_eval
    """
    xi_minus_xj = x_data[:, np.newaxis] - x_data  # Matrix with x_i-x_j in each cell
    M = phi(r=np.abs(xi_minus_xj), epsilon=epsilon)  # M as defined in the task
    w = np.linalg.solve(M, y_data)  # Solve for w and find weights

    x_minus_xi = (
        x_eval[:, np.newaxis] - x_data
    )  # (N+1)*(n+1), with x_eval being our eta-values. On the form
    # (
    # [eta_0 - x_0,..., eta_0-x_n],
    # ...
    # [eta_N - x_0,..., eta_N - x_n]
    # )
    phi_eval = phi(r=np.abs(x_minus_xi), epsilon=epsilon)
    # Mw = f, with phi_eval being M in this case
    y_eval = phi_eval @ w  # Multiply matrices to obtain estimated y-values

    return y_eval


def find_min_epsilon(search_a, search_b, tol, funcs):
    """Finds a minimal epsilon on some search interval.
        Not generalized outside the given problem, so the intervals are fixed.

    Args:
        search_a (float): start value of epsilon
        search_b (float): end value of epsilon
        tol (float): tolerance of minimum to be found
        funcs (tuple(function)): tuple of functions for which to find minimum epsilon

    Returns:
        Array1D: Array with optimal epsilons
    """
    epsilon_min = np.zeros(2)
    a = [-5, 0]
    b = [5, 1]
    fig, axs = plt.subplots(2, 1)
    for i in range(2):
        epsilons = np.arange(
            search_a, search_b, tol
        )  # Initialize array of possible epsilon values
        norms = np.ones_like(
            epsilons
        )  # Array to be used to compare norms between different values of epsilon
        x_data = np.linspace(a[i], b[i], n + 1)  # n is fixed to 10 for this problem
        x_eval = np.linspace(a[i], b[i], n_eval)  # 100*n
        y_data = funcs[i](x_data)  # Interpolation y-values
        y_true = funcs[i](x_eval)  # True value of function on x_eval

        # Find norms for different epsilon
        for j in prange(epsilons.shape[0]):
            y_eval = RBF(
                x_data=x_data, y_data=y_data, x_eval=x_eval, epsilon=epsilons[j]
            )
            norms[j] = norm_inf(y_true=y_true, y_pred=y_eval)

        # Find epsilon that minimizes norm
        epsilon_min[i] = epsilons[np.argmin(norms)]

        axs[i].semilogy(epsilons, norms, "o-", markersize=3)
        axs[i].axvline(
            x=epsilon_min[i],
            linestyle="--",
            color="black",
            label=rf"$\varepsilon = {epsilon_min[i]:.2f}$",
        )
        axs[i].set_xlabel(r"Shape parameter $[\varepsilon]$")
        axs[i].set_ylabel(r"$\left\| f - \tilde{{f}} \right\|_\infty$")
        axs[i].legend()
        axs[i].set_title(rf"$f(x) ={funcs_latex[i]}$")
    # plt.suptitle(
    #     r"$\left\| f - \tilde{{f}} \right\|_\infty \;\text{{vs.}}\; \varepsilon$"
    # )
    plt.show()

    return epsilon_min


funcs = [runge_f, f_cos]
funcs_latex = ["(1 + x^2)^{{-1}}", "\\cos(2\\pi x)"]
n = 10
n_eval = 100 * n
a = [-5, 0]
b = [5, 1]


epsilons = find_min_epsilon(search_a=0.01, search_b=2.0, tol=0.01, funcs=funcs)


fig, axs = plt.subplots(2, 1)
for i, func in enumerate(funcs):
    x_data = np.linspace(a[i], b[i], n + 1)
    x_eval = np.linspace(a[i], b[i], n_eval)
    y_data = func(x=x_data)
    y_eval = RBF(x_data=x_data, y_data=y_data, x_eval=x_eval, epsilon=epsilons[i])

    axs[i].plot(
        x_eval,
        y_eval,
        label=rf"RBF with $\varepsilon = {epsilons[i]}$",
    )
    axs[i].scatter(x_data, y_data, label=rf"${n}$ RBF nodes")
    axs[i].plot(x_eval, func(x=x_eval), "--", label=rf"$f(x) = {funcs_latex[i]}$")
    axs[i].legend()
    axs[i].set_title(
        rf"$f(x)={funcs_latex[i]} \text{{ on }}\left[{a[i]}, {b[i]}\right]$"
    )
    axs[i].set_xlabel(r"$x$")
    axs[i].set_ylabel(r"$y$")
plt.show()
# %%
# To use autograd we use a numpy wrapper


# The functions we use must use the numpy wrapper and can not use jit
def phi_ag(r, epsilon):
    return anp.exp(-((epsilon * r) ** 2))  # type: ignore[attr-defined]


def RBF_ag(x_data, y_data, x_eval, epsilon):
    x_data = anp.array(x_data)
    y_data = anp.array(y_data)
    x_eval = anp.array(x_eval)

    xi_minus_xj = x_data[:, None] - x_data
    M = phi_ag(r=anp.abs(xi_minus_xj), epsilon=epsilon)  # type: ignore[attr-defined]
    w = anp.linalg.solve(M, y_data)  # type: ignore[attr-defined]

    x_minus_xi = x_eval[:, None] - x_data
    phi_eval = phi_ag(r=anp.abs(x_minus_xi), epsilon=epsilon)  # type: ignore[attr-defined]
    y_eval = phi_eval @ w

    return y_eval


def runge_f_ag(x):
    x = anp.array(x)
    return 1 / (x**2 + 1)


# Initial parameters
epsilon = 1
a_interval, b_interval = -5, 5
n = 10
N = 1000
eta_vals = np.linspace(a_interval, b_interval, N + 1)
f_eta_vals = runge_f_ag(eta_vals)
x_data = np.linspace(a_interval, b_interval, n + 1)


# Define the cost function that takes one parameter, on the form [x0,...,xn, epsilon]
def cost_func(param):
    """Cost function

    Args:
        param (NDArray): Combined array of x_data and epsilon

    Returns:
        float: The cost in param
    """
    epsilon = param[-1]
    x_data = param[:-1]
    RBF_vals = RBF_ag(
        x_data=x_data,
        y_data=runge_f_ag(x_data),
        x_eval=eta_vals,
        epsilon=epsilon,
    )
    cost = (b_interval - a_interval) / N * anp.sum((f_eta_vals - RBF_vals) ** 2)  # type: ignore[attr-defined]
    return cost


def gradient_descent(step, tol, max_iter, param):
    """Gradient descent without backtracking

    Args:
        step (float): Step size [tau]
        tol (float): Tolerance
        max_iter (int): Maximum number of iterations
        param (NDArray): Input for function to minimize

    Returns:
        NDArray: param that minimizes the cost function and cost for each iteration
    """
    grad_func = grad(cost_func)  # type: ignore[arg-type]
    param_current = anp.array(param)
    gradient = grad_func(param_current)

    error = 1.0
    k = 0
    cost_vals = np.zeros(max_iter)

    # Start gradient descent like naive algorithm showed in problem sheet
    while error > tol:
        param_next = param_current - step * gradient

        error = float(np.max(np.abs(np.asarray(param_next - param_current))))
        cost_vals[k] = float(cost_func(param=param_current))
        param_current = param_next
        gradient = grad_func(param_current)
        k += 1

        if k >= max_iter - 1:
            # print("Reached max iterations")
            return (
                np.asarray(param_current[:-1]),
                float(param_current[-1]),
                cost_vals[: k + 1],
            )

    return np.asarray(param_current[:-1]), float(param_current[-1]), cost_vals[: k + 1]


def gradient_descent_bt(param, L, rho_up, rho_down, max_iter):
    """
    Gradient descent with backtracking as described in the problem sheet.
    """

    grad_func = grad(cost_func)  # type: ignore[arg-type]
    param = anp.array(param)
    phi = float(cost_func(param=param))
    cost_vals = np.zeros(max_iter)

    for k in range(max_iter):
        g = grad_func(param)
        cost_vals[k] = float(cost_func(param))

        for t in range(1000):
            param_tilde = param - 1 / L * g
            phi_tilde = float(cost_func(param=param_tilde))

            if phi_tilde <= phi + float(
                np.dot(np.asarray(g), np.asarray(param_tilde - param))
            ) + L / 2 * float(np.max(np.abs(np.asarray(param_tilde - param))) ** 2):
                param = param_tilde
                phi = phi_tilde
                L = rho_down * L
                break
            else:
                L = rho_up * L

        if k >= max_iter - 1:
            # print("Reached max iterations")
            return np.asarray(param[:-1]), float(param[-1]), cost_vals[: k + 1]

    return np.asarray(param[:-1]), float(param[-1]), cost_vals[: k + 1]


param_init = np.append(arr=x_data, values=epsilon)
fig, ax = plt.subplots()

for i, step in enumerate([0.1, 0.01]):
    x_optimal, epsilon_optimal, cost_vals = gradient_descent(
        step=step, max_iter=10000, tol=1e-6, param=param_init
    )

    ax.loglog(
        np.arange(cost_vals.shape[0]),
        cost_vals,
        "--",
        label=rf"W/o backtracking, $\tau={step},\, TOL = 10^{{-6}}$",
    )

for i, L in enumerate([1, 10, 100]):
    x_optimal, epsilon_optimal, cost_vals = gradient_descent_bt(
        param=param_init,
        L=L,
        rho_up=10,
        rho_down=0.1,
        max_iter=10000,
    )

    ax.loglog(
        np.arange(cost_vals.shape[0]),
        cost_vals,
        label=rf"W/ backtracking, $L={L},\, \overline{{\rho}}=10, \, \underline{{\rho}} = 0.1$",
    )
ax.set_xlabel(r"Number of iterations $[k]$")
ax.set_ylabel(r"$\mathcal{C}(\mathbf{x})$")
ax.legend(loc="lower left")
plt.suptitle(r"Cost function $\mathcal{C}(\mathbf{x})$ vs. number of iterations")
plt.show()
# %%
n_vals = [n for n in range(1, 60, 1)]  # Number of nodes we want to compare
a, b = -5, 5  # Function evaluation interval
# Initialize arrays to insert errors in
errors_optimal = np.zeros(len(n_vals))
errors_equi_RBF = np.zeros(len(n_vals))
errors_cheby_RBF = np.zeros(len(n_vals))
# errors_optimal_cheby_RBF = np.zeros(len(n_vals))

# Iterate through number of nodes
for i, n in enumerate(n_vals):
    epsilon_guess = 1
    N = 100 * n  # No. of evaluation points
    eta_vals = np.linspace(a, b, N + 1)
    f_eta_vals = runge_f_ag(eta_vals)
    x_equi = np.linspace(a, b, n + 1)
    x_cheby = cheby_nodes(a, b, n)
    param_init = np.append(x_equi, epsilon_guess)
    param_init_cheby = np.append(x_cheby, epsilon_guess)

    x_optimal, epsilon_optimal, _ = gradient_descent_bt(
        param=param_init,
        L=1,
        rho_up=10,
        rho_down=0.1,
        max_iter=500,
    )
    # x_optimal_cheby, epsilon_optimal_cheby, _ = gradient_descent_bt(
    #     param=param_init_cheby,
    #     L=L,
    #     rho_up=10,
    #     rho_down=0.1,
    #     max_iter=500,
    # )

    RBF_optimal = RBF(
        x_data=x_optimal,
        y_data=runge_f(x_optimal),
        x_eval=eta_vals,
        epsilon=epsilon_optimal,
    )
    RBF_equi = RBF(
        x_data=x_equi,
        y_data=runge_f(x=x_equi),
        x_eval=eta_vals,
        epsilon=epsilon_optimal,
    )
    RBF_cheby = RBF(
        x_data=x_cheby,
        y_data=runge_f(x_cheby),
        x_eval=eta_vals,
        epsilon=epsilon_optimal,
    )
    # RBF_optimal_cheby = RBF(
    #     x_data=x_optimal_cheby,
    #     y_data=runge_f(x_optimal_cheby),
    #     x_eval=eta_vals,
    #     epsilon=epsilon_optimal_cheby,
    # )

    errors_optimal[i] = norm_2(y_true=f_eta_vals, y_pred=RBF_optimal, a=a, b=b, N=N)
    errors_equi_RBF[i] = norm_2(y_true=f_eta_vals, y_pred=RBF_equi, a=a, b=b, N=N)
    errors_cheby_RBF[i] = norm_2(y_true=f_eta_vals, y_pred=RBF_cheby, a=a, b=b, N=N)
    # errors_optimal_cheby_RBF[i] = norm_2(f_eta_vals, RBF_optimal_cheby, a, b, N)


fig, ax = plt.subplots()
ax.semilogy(
    n_vals,
    errors_optimal,
    "o-",
    label=r"Optimal nodes RBF",
)
ax.semilogy(
    n_vals,
    errors_equi_RBF,
    "o-",
    label=r"Equidistant nodes RBF",
)
ax.semilogy(
    n_vals,
    errors_cheby_RBF,
    "o-",
    label=r"Chebyshev nodes RBF",
)
# ax.semilogy(
#     n_vals,
#     errors_optimal_cheby_RBF,
#     "o-",
#     label=r"Optimal nodes RBF starting with Chebyshev nodes",
# )

ax.set_title(rf"Estimated error for $f(x)={funcs_latex[0]}, \, x \in [-5,5]$")
ax.set_xlabel(r"Number of nodes - 1 $[n]$ ")
ax.set_ylabel(r"$\left\| f - \tilde{f}\right\|_2$")
ax.legend()
plt.show()

data = {
    "n": n_vals,
    "Error Optimal": errors_optimal,
    "Error Equidistant": errors_equi_RBF,
    "Error Chebyshev": errors_cheby_RBF,
}

df_errors = pd.DataFrame(data)
df_errors = df_errors.set_index("n")

pd.options.display.float_format = "{:.4e}".format
print(df_errors)
