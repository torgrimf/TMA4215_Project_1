import numpy as np
from numba import njit, prange
import time

N_small = 10**3   # 1k
N_medium = 10**5  # 100k
N_large = 10**6   # 1M

# test data
x = np.linspace(-5, 5, N_large, dtype=np.float64)
y = np.sin(x)

def cost_numpy(x, y):
    return np.sum((x - y) ** 2)

@njit
def cost_numpy_sum(x, y):
    return np.sum((x - y) ** 2)

@njit(parallel=True)
def cost_prange(x, y):
    total = 0.0
    for i in prange(x.shape[0]):
        diff = x[i] - y[i]
        total += diff * diff
    return total

def benchmark(N):
    xN, yN = x[:N], y[:N]

    # warm up JIT
    cost_numpy_sum(xN, yN)
    cost_prange(xN, yN)

    # pure NumPy (no JIT)
    t0 = time.perf_counter()
    r0 = cost_numpy(xN, yN)
    t1 = time.perf_counter()

    # JIT np.sum
    r1 = cost_numpy_sum(xN, yN)
    t2 = time.perf_counter()

    # JIT prange
    r2 = cost_prange(xN, yN)
    t3 = time.perf_counter()

    print(f"N={N:>8} | NumPy: {t1 - t0:.6f}s | njit+np.sum: {t2 - t1:.6f}s | prange: {t3 - t2:.6f}s")
    print(f"  diffs: NumPy-vs-njit={abs(r0-r1):.2e}, njit-vs-prange={abs(r1-r2):.2e}")

for N in [N_small, N_medium, N_large]:
    benchmark(N)
