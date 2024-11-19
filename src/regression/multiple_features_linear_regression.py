import numpy as np

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])


def calculate_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    return sum([(np.dot(X[i], w) + b - y[i]) ** 2 for i in range(X.shape[0])]) / (
        2 * X.shape[0]
    )


def calculate_gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b):
    m, n = X.shape
    dj_dw = np.zeros(w.shape, dtype="int64")
    dj_db = 0

    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        dj_db += err
        for j in range(n):
            dj_dw[j] += err * X[i, j]

    return dj_dw / m, dj_db / m


def gradient_descend(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    b_init: float,
    learning_rate: float,
    iteration: int,
):
    w = np.copy(w_init)
    b = b_init

    for i in range(iteration):
        dj_dw, dj_db = calculate_gradients(X, y, w, b)
        w = w - learning_rate * dj_dw
        b = b - learning_rate * dj_db

    return w, b


w_new, b_new = gradient_descend(X_train, y_train, w_init, b_init, 0.01, 1000)
print(w_new, b_new)

for i in range(y_train.shape[0]):
    print(np.dot(X_train[i], w_new) + b_new)
    print(y_train[i])
    print("------")
