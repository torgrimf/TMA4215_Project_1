import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def lagrange(x_data, y_data, x_evaluate):
    x = sp.symbols("x")
    n = len(x_data) - 1
    cardinals = []
    for i in range(n + 1):
        cardinal = 1

        for j in range(n + 1):
            if i != j:
                cardinal *= (x - x_data[j]) / (x_data[i] - x_data[j])

        cardinals.append(cardinal)

    poly = sp.lambdify(x, np.dot(cardinals, y_data))

    y_evaluate = poly(x_evaluate)
    return y_evaluate


x_data = np.linspace(0, 5, 5)
y_data = np.array([1, 2, 4, 5, 7])
x_evaluate = np.linspace(0, 5, 1000)

# print(lagrange(x_data, y_data, x_evaluate))
plt.plot(x_evaluate, lagrange(x_data, y_data, x_evaluate))
plt.scatter(x_data, y_data)
plt.show()
