#------------------------------------
# Author: T. D. Devlin
# Completed for CSC X65 Assignment 4, Problem II
#-----------------------------------

from math import pi, sin
from random import random, seed


def f(x):
    return sin(pi * x)


def generate_training_examples(n=2):
    xs = [random() * 2 - 1 for _ in range(n)]
    return [(x, f(x)) for x in xs]


def fit_without_reg(examples):
    """Computes values of w0 and w1 that minimize the sum-of-squared-errors cost function.

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    """
    if len(examples) != 2:
        raise ValueError("fit_without_reg expects exactly two examples")

    (x1, y1), (x2, y2) = examples

    if x1 == x2:
        return (y1 + y2) / 2, 0

    w1 = (y2 - y1) / (x2 - x1)
    w0 = y1 - w1 * x1
    return w0, w1


def fit_with_reg(examples, lambda_hp):
    """Computes values of w0 and w1 that minimize the regularized SSE cost function.

    Args:
    - examples: a list of two (x, y) tuples, where x is the feature and y is the label
    - lambda_hp: a float representing the value of the lambda hyperparameter; a larger value means more regularization
    """
    w0 = 0
    w1 = 0
    eta = 0.05

    for _ in range(1000):
        grad_w0 = 2 * lambda_hp * w0
        grad_w1 = 2 * lambda_hp * w1

        for x, y in examples:
            prediction = w0 + w1 * x
            error = prediction - y
            grad_w0 += 2 * error
            grad_w1 += 2 * error * x

        w0 -= eta * grad_w0
        w1 -= eta * grad_w1

    return w0, w1


def test_error(w0, w1):
    n = 100
    xs = [i/n for i in range(-n, n + 1)]
    return sum((w0 + w1 * x - f(x)) ** 2 for x in xs) / len(xs)


def run_trials(num_trials=1000, lambda_hp=1, seed_value=0):
    if seed_value is not None:
        seed(seed_value)

    without_reg_params = []
    with_reg_params = []
    without_reg_error = 0
    with_reg_error = 0

    for _ in range(num_trials):
        examples = generate_training_examples()

        w0_without, w1_without = fit_without_reg(examples)
        w0_with, w1_with = fit_with_reg(examples, lambda_hp)

        without_reg_params.append((w0_without, w1_without))
        with_reg_params.append((w0_with, w1_with))

        without_reg_error += test_error(w0_without, w1_without)
        with_reg_error += test_error(w0_with, w1_with)

    return (
        without_reg_error / num_trials,
        with_reg_error / num_trials,
        without_reg_params,
        with_reg_params,
    )


def _svg_line(x1, y1, x2, y2, color, opacity, width):
    return (
        f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
        f'stroke="{color}" stroke-opacity="{opacity}" stroke-width="{width}" />'
    )


def _svg_polyline(points, color, width):
    point_text = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" />'
    )


def write_svg_plot(without_reg_params, with_reg_params, filename="regularization_plot.svg"):
    width = 1080
    height = 460
    margin = 55
    gap = 55
    panel_width = (width - 2 * margin - gap) / 2
    panel_height = height - 2 * margin
    y_min = -3
    y_max = 3

    def tx(x, panel_left):
        return panel_left + ((x + 1) / 2) * panel_width

    def ty(y):
        return margin + ((y_max - y) / (y_max - y_min)) * panel_height

    left_panel = margin
    right_panel = margin + panel_width + gap
    xs = [i / 100 for i in range(-100, 101)]
    target_left = [(tx(x, left_panel), ty(f(x))) for x in xs]
    target_right = [(tx(x, right_panel), ty(f(x))) for x in xs]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        f'<clipPath id="clip-without"><rect x="{left_panel}" y="{margin}" width="{panel_width}" height="{panel_height}" /></clipPath>',
        f'<clipPath id="clip-with"><rect x="{right_panel}" y="{margin}" width="{panel_width}" height="{panel_height}" /></clipPath>',
        "</defs>",
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{left_panel + panel_width / 2:.3f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">Without regularization</text>',
        f'<text x="{right_panel + panel_width / 2:.3f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">With L2 regularization</text>',
    ]

    for panel_left in (left_panel, right_panel):
        x_axis = ty(0)
        y_axis = tx(0, panel_left)
        lines.append(
            f'<rect x="{panel_left}" y="{margin}" width="{panel_width}" height="{panel_height}" '
            'fill="none" stroke="#222" stroke-width="1" />'
        )
        lines.append(_svg_line(panel_left, x_axis, panel_left + panel_width, x_axis, "#999", 0.7, 1))
        lines.append(_svg_line(y_axis, margin, y_axis, margin + panel_height, "#999", 0.7, 1))
        lines.append(
            f'<text x="{panel_left + panel_width / 2:.3f}" y="{height - 17}" text-anchor="middle" '
            'font-family="Arial" font-size="12">x</text>'
        )
        lines.append(
            f'<text x="{panel_left - 34}" y="{margin + panel_height / 2:.3f}" text-anchor="middle" '
            'font-family="Arial" font-size="12" transform="rotate(-90 '
            f'{panel_left - 34} {margin + panel_height / 2:.3f})">y</text>'
        )

    lines.append('<g clip-path="url(#clip-without)">')
    for w0, w1 in without_reg_params:
        lines.append(_svg_line(tx(-1, left_panel), ty(w0 - w1), tx(1, left_panel), ty(w0 + w1), "#2563eb", 0.045, 0.65))
    lines.append(_svg_polyline(target_left, "#c2410c", 2.4))
    lines.append("</g>")

    lines.append('<g clip-path="url(#clip-with)">')
    for w0, w1 in with_reg_params:
        lines.append(_svg_line(tx(-1, right_panel), ty(w0 - w1), tx(1, right_panel), ty(w0 + w1), "#047857", 0.055, 0.75))
    lines.append(_svg_polyline(target_right, "#c2410c", 2.4))
    lines.append("</g>")

    lines.extend(
        [
            f'<text x="{left_panel + 8}" y="{margin + 18}" font-family="Arial" font-size="12" fill="#c2410c">target f(x)</text>',
            f'<text x="{right_panel + 8}" y="{margin + 18}" font-family="Arial" font-size="12" fill="#c2410c">target f(x)</text>',
            "</svg>",
        ]
    )

    with open(filename, "w", encoding="utf-8") as output:
        output.write("\n".join(lines))

    return filename


if __name__ == "__main__":
    avg_without_reg, avg_with_reg, without_reg_params, with_reg_params = run_trials()

    print("Average test error over 1000 trials:")
    print(f"without regularization: {avg_without_reg:.6f}")
    print(f"with regularization:    {avg_with_reg:.6f}")

    plot_filename = write_svg_plot(without_reg_params, with_reg_params)
    print(f"Wrote plot to {plot_filename}")
