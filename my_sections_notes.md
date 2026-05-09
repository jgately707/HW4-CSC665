# Assignment 4 Notes for Assigned Sections

## Problem I, Q3

Using the hinge-loss subgradient with learning rate eta = 0.1:

If y * h(x1, x2) < 1, update

```text
w <- w + eta * y * (1, x1, x2)
```

If y * h(x1, x2) >= 1, the update is zero.

Weights after each SGD update:

| iteration | example | w0 | w1 | w2 |
|---:|---:|---:|---:|---:|
| 1 | (-4, 0, -1) | -0.1 | 0.4 | 0.0 |
| 2 | (-1, 1, +1) | 0.0 | 0.3 | 0.1 |
| 3 | (0, -1, -1) | -0.1 | 0.3 | 0.2 |
| 4 | (2, 1, +1) | 0.0 | 0.5 | 0.3 |
| 5 | (3, 0, +1) | 0.0 | 0.5 | 0.3 |
| 6 | (6, -1, -1) | -0.1 | -0.1 | 0.4 |

Final SGD weights:

```text
(w0, w1, w2) = (-0.1, -0.1, 0.4)
```

## Problem I, Q4

Using h(x1, x2) = -0.1 - 0.1*x1 + 0.4*x2 and predicting +1 when h >= 0:

| i | h(x1, x2) | prediction | y | correct? |
|---:|---:|---:|---:|---:|
| 1 | 0.3 | +1 | -1 | no |
| 2 | 0.4 | +1 | +1 | yes |
| 3 | -0.5 | -1 | -1 | yes |
| 4 | 0.1 | +1 | +1 | yes |
| 5 | -0.4 | -1 | +1 | no |
| 6 | -1.1 | -1 | -1 | yes |

There are 2 mistakes out of 6 examples.

```text
misclassification rate = 2/6 = 1/3 = 0.333333
```

## Problem I, Q5 EC

The file `problem1_gd.py` runs batch gradient descent using the full dataset on each update.

With eta = 0.1 and stopping threshold 1e-5:

```text
SGD final weights: (-0.1, -0.1, 0.4)
SGD training error: 0.333333
SGD hinge loss: 4.700000

Batch GD final weights: (-0.4, 0.5, 4.0)
Batch GD updates: 58
Batch GD training error: 0.000000
Batch GD hinge loss: 0.000000
```

## Problem II, Q3

For two points (x1, y1) and (x2, y2), the unregularized best-fit line that passes through both points is:

```text
w1 = (y2 - y1) / (x2 - x1)
w0 = y1 - w1*x1
```

This is implemented in `fit_without_reg`.

## Problem II, Q4

The regularized model is implemented in `fit_with_reg`. It uses 1000 gradient descent updates from w0 = w1 = 0, step size eta = 0.05, and lambda = 1 when called from the experiment.

## Problem II, Q5

The experiment in `regularization.py` uses 1000 trials with random seed 0 for reproducibility.

Average test errors:

```text
without regularization: 1.911505
with regularization:    0.436800
```

## Problem II, Q6 EC

The script writes the visualization to:

```text
regularization_plot.svg
```

I also generated a PNG copy for easier insertion into a document:

```text
regularization_plot.png
```

The left panel shows the 1000 unregularized fitted lines. The right panel shows the 1000 L2-regularized fitted lines. In both panels, the orange curve is the target function f(x) = sin(pi*x).

## GenAI Usage Note

GenAI was used to help implement and verify the coding portions of Problem I Q5 and Problem II Q3-Q6, and to draft/check the numeric notes for Problem I Q3-Q4 and Problem II Q5-Q6. The code was run locally, outputs were checked against the assignment formulas, and the generated notes were reviewed for consistency with the program output.
