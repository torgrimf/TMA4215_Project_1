import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time

# ----------------------------
# NumPy version
# ----------------------------
def lagrange_numpy(x_data, y_data, x_eval):
    n = len(x_data)
    L = np.ones((n, len(x_eval)))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i] *= (x_eval - x_data[j]) / (x_data[i] - x_data[j])
    return np.dot(y_data, L)

# ----------------------------
# Numba version
# ----------------------------
@njit(parallel=True)
def lagrange_numba(x_data, y_data, x_eval):
    n = len(x_data)
    m = len(x_eval)
    result = np.zeros(m)
    for k in prange(m):
        y_val = 0.0
        for i in range(n):
            Li = 1.0
            xi = x_data[i]
            for j in range(n):
                if i != j:
                    Li *= (x_eval[k] - x_data[j]) / (xi - x_data[j])
            y_val += y_data[i] * Li
        result[k] = y_val
    return result

# ----------------------------
# Benchmark
# ----------------------------
def benchmark(n=21, ms=[100, 1000, 5000, 20000]):
    x_data = np.linspace(-1, 1, n)
    y_data = 1 / (1 + 25 * x_data**2)

    numpy_times = []
    numba_times = []

    for m in ms:
        x_eval = np.linspace(-1, 1, m)

        # Warmup Numba
        if len(numba_times) == 0:
            lagrange_numba(x_data, y_data, x_eval)

        # NumPy timing
        t0 = time.perf_counter()
        lagrange_numpy(x_data, y_data, x_eval)
        t1 = time.perf_counter()
        numpy_times.append(t1 - t0)

        # Numba timing
        t0 = time.perf_counter()
        lagrange_numba(x_data, y_data, x_eval)
        t1 = time.perf_counter()
        numba_times.append(t1 - t0)

    return ms, numpy_times, numba_times


if __name__ == "__main__":
    ms, numpy_times, numba_times = benchmark(n=41)

    plt.loglog(ms, numpy_times, "o-", label="NumPy")
    plt.loglog(ms, numba_times, "o-", label="Numba (parallel)")
    plt.xlabel("Number of evaluation points m")
    plt.ylabel("Runtime (s)")
    plt.title("Lagrange interpolation performance (n=41 nodes)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
