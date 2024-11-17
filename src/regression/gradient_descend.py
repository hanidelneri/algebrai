import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(
    description="plots the regression line using gradient descend"
)
parser.add_argument("--data-set-size", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=0.001)
parser.add_argument("--iteration", type=int, default=1000)

args = parser.parse_args()


np.random.seed(42)
x = np.random.rand(args.data_set_size) * 10
y = 2 * x + 3 + np.random.randn(args.data_set_size)


def gradient_descend(
    x: list,
    y: list,
    learning_rate: float,
    w: float,
    b: float,
    iteration: int,
    threshold: float,
):
    w_gradient_history = []
    b_gradient_history = []
    for i in range(iteration):
        w_gradient = w_derivative(x, y, w, b)
        b_gradient = b_derivative(x, y, w, b)

        w = w - learning_rate * w_gradient
        b = b - learning_rate * b_gradient

        w_gradient_history.append(w)
        b_gradient_history.append(b)
        if abs(w_gradient) < threshold and abs(b_gradient) < threshold:
            break

    return w, b, w_gradient_history, b_gradient_history


def w_derivative(x: list, y: list, w: float, b: float) -> float:
    return sum([x[i] * (w * x[i] + b - y[i]) for i in range(len(x))]) / len(x)


def b_derivative(x: list, y: list, w: float, b: float) -> float:
    return sum([w * x[i] + b - y[i] for i in range(len(x))]) / len(x)


w, b, w_gradient_history, b_gradient_history = gradient_descend(
    x, y, args.learning_rate, 0, 0, args.iteration, 0.0001
)


plt.scatter(x, y, label="Data points")
plt.plot(x, w * x + b, color="red", label="Regression line")
for i in range(0, len(w_gradient_history)):
    plt.plot(
        x,
        w_gradient_history[i] * x + b_gradient_history[i],
        "",
    )
plt.legend()
plt.show()
