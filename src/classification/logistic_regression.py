import numpy as np


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-a))


def model(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(np.dot(X, w) + b)


def loss(X: np.ndarray, y: int, w: np.ndarray, b: float) -> float:
    return y * np.log(model(X, w, b)) + (1 - y) * np.log(1 - model(X, w, b))


def cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    m = X.shape[0]
    sum = 0
    for i in range(m):
        sum += loss(X[i], y[i], w, b)

    return -1 / sum


def compute_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> np.ndarray:
    m, n = X.shape
    dj_dw = np.zeros(w)
    dj_db = 0

    for i in range(m):
        err = model((X[i], w, b)) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    return dj_dw / m, dj_db / m


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_in: np.ndarray,
    b_init: float,
    iteration: int,
    learning_rate: float,
):
    w = np.copy(w_in)
    b = b_init

    for i in range(iteration):
        dw_dj, dw_db = compute_gradient(X, y, w, b)
        w = w - learning_rate * dw_dj
        b = b - learning_rate * dw_db
    return w, b
