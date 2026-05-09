"""CSC X65 Assignment 4, Problem I Question 5.

Runs batch gradient descent for the hinge-loss linear classifier and compares the
result with the one-pass SGD calculation from Problem I Question 3.
"""

DATA = [
    (-4.0, 0.0, -1),
    (-1.0, 1.0, 1),
    (0.0, -1.0, -1),
    (2.0, 1.0, 1),
    (3.0, 0.0, 1),
    (6.0, -1.0, -1),
]


def score(weights, x1, x2):
    w0, w1, w2 = weights
    return w0 + w1 * x1 + w2 * x2


def predict(weights, x1, x2):
    return 1 if score(weights, x1, x2) >= 0 else -1


def hinge_loss(weights):
    return sum(max(0.0, 1.0 - y * score(weights, x1, x2)) for x1, x2, y in DATA)


def misclassification_rate(weights):
    mistakes = sum(predict(weights, x1, x2) != y for x1, x2, y in DATA)
    return mistakes / len(DATA)


def example_subgradient(weights, example):
    x1, x2, y = example
    margin = y * score(weights, x1, x2)

    if margin >= 1.0:
        return (0.0, 0.0, 0.0)

    return (-y, -y * x1, -y * x2)


def sgd_one_pass(eta=0.1):
    weights = (0.0, 0.0, 0.0)
    history = []

    for example in DATA:
        gradient = example_subgradient(weights, example)
        weights = tuple(w - eta * g for w, g in zip(weights, gradient))
        history.append(weights)

    return weights, history


def batch_subgradient(weights):
    gradient = [0.0, 0.0, 0.0]

    for example in DATA:
        example_gradient = example_subgradient(weights, example)
        for i in range(3):
            gradient[i] += example_gradient[i]

    return tuple(gradient)


def batch_gradient_descent(eta=0.1, tolerance=1e-5, max_updates=100000):
    weights = (0.0, 0.0, 0.0)

    for updates in range(max_updates):
        gradient = batch_subgradient(weights)
        step = tuple(-eta * g for g in gradient)

        if max(abs(value) for value in step) < tolerance:
            return weights, updates

        weights = tuple(w + delta for w, delta in zip(weights, step))

    return weights, max_updates


def print_weight_table(history):
    print("SGD one-pass weights after each update:")
    print("iteration    w0         w1         w2")
    for i, weights in enumerate(history, start=1):
        w0, w1, w2 = weights
        print(f"{i:>9} {w0:>9.4f} {w1:>10.4f} {w2:>10.4f}")


def main():
    sgd_weights, sgd_history = sgd_one_pass()
    gd_weights, updates = batch_gradient_descent()

    print_weight_table(sgd_history)
    print()

    print("Final comparison:")
    print(
        "SGD final weights: "
        f"{sgd_weights}, training error = {misclassification_rate(sgd_weights):.6f}, "
        f"hinge loss = {hinge_loss(sgd_weights):.6f}"
    )
    print(
        "Batch GD weights: "
        f"{gd_weights}, updates = {updates}, "
        f"training error = {misclassification_rate(gd_weights):.6f}, "
        f"hinge loss = {hinge_loss(gd_weights):.6f}"
    )


if __name__ == "__main__":
    main()
