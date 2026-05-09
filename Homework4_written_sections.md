# Team: Lasiru Weerasuriya, Jack Gately
# CSC 665
# Assignment 4

## Problem I, Question 1

The hinge loss for binary classification is

```math
\ell(h(x), y) = \max(0, 1 - yh(x)),
```

where the label is

```math
y \in \{-1, +1\}.
```

The value

```math
yh(x)
```

is the margin. If the margin is at least 1, then the point is not only classified correctly, but it is also far enough from the decision boundary, so the hinge loss is 0. If the margin is between 0 and 1, the point is technically classified correctly, but it is still too close to the boundary, so it gets penalized. If the margin is less than or equal to 0, then the point is misclassified and also gets penalized.

This makes hinge loss a good objective for large-margin binary classification because it cares about more than just getting the sign right. The usual 0-1 classification loss only checks whether the prediction is correct or incorrect, but it is hard to optimize directly. Hinge loss is a convex surrogate for that loss, which makes the optimization problem much easier to work with while still pushing the classifier toward a larger margin. This is the main idea behind support vector machines, where the learned classifier tries to separate the two classes while keeping the boundary as far as possible from the training points. Cortes and Vapnik introduced support-vector networks for this kind of two-class classification problem [1]. Shalev-Shwartz and Ben-David also explain why convex surrogate losses are useful when the original classification objective is difficult to optimize directly [2]. Other SVM references describe how the hinge-loss setup leads to a tractable convex optimization problem [3]. Hinge loss is also convenient for gradient-based and online learning methods because its subgradient is simple to compute [4].

## Problem I, Question 2

The model is

```math
h(x_1, x_2) = w_0 + w_1x_1 + w_2x_2.
```

The hinge loss is

```math
c(h(x_1, x_2), y) = \max(0, 1 - yh(x_1, x_2)).
```

Let the margin be

```math
m = yh(x_1, x_2).
```

Then the hinge loss can be written in two cases:

```math
c(h(x_1, x_2), y) =
\begin{cases}
0, & yh(x_1, x_2) \ge 1, \\
1 - yh(x_1, x_2), & yh(x_1, x_2) < 1.
\end{cases}
```

When

```math
yh(x_1, x_2) > 1,
```

the loss is flat, so all three partial derivatives are 0:

```math
\frac{\partial c}{\partial w_0} = 0,
\qquad
\frac{\partial c}{\partial w_1} = 0,
\qquad
\frac{\partial c}{\partial w_2} = 0.
```

When

```math
yh(x_1, x_2) < 1,
```

the loss is

```math
1 - y(w_0 + w_1x_1 + w_2x_2).
```

So the partial derivatives are:

```math
\frac{\partial c}{\partial w_0} = -y,
```

```math
\frac{\partial c}{\partial w_1} = -yx_1,
```

```math
\frac{\partial c}{\partial w_2} = -yx_2.
```

At the exact point where

```math
yh(x_1, x_2) = 1,
```

the hinge loss has a corner, so the regular derivative does not exist. We use a subgradient there. A valid subgradient can be any value between the zero side and the nonzero side. Using

```math
x_0 = 1,
```

we can write this as

```math
\frac{\partial c}{\partial w_j} \in \alpha(-yx_j),
\qquad
\alpha \in [0,1].
```

For example, at the boundary it is valid to choose the zero subgradient.

## Problem II, Question 1

The cost function is

```math
C(w) = \sum_{i=1}^{n} \left(y_i - (w_0 + w_1x_i)\right)^2.
```

Since this problem has

```math
n = 2,
```

we can write it as

```math
C(w) =
\left(y_1 - (w_0 + w_1x_1)\right)^2
+
\left(y_2 - (w_0 + w_1x_2)\right)^2.
```

First, take the derivative with respect to

```math
w_0.
```

For each term, the inside expression has derivative

```math
-1
```

with respect to

```math
w_0.
```

So

```math
\frac{dC}{dw_0}
=
-2\left(y_1 - (w_0 + w_1x_1)\right)
-2\left(y_2 - (w_0 + w_1x_2)\right).
```

An equivalent form is

```math
\frac{dC}{dw_0}
=
2(w_0 + w_1x_1 - y_1)
+
2(w_0 + w_1x_2 - y_2).
```

Now take the derivative with respect to

```math
w_1.
```

For each term, the inside expression has derivative

```math
-x_i
```

with respect to

```math
w_1.
```

Therefore,

```math
\frac{dC}{dw_1}
=
-2x_1\left(y_1 - (w_0 + w_1x_1)\right)
-2x_2\left(y_2 - (w_0 + w_1x_2)\right).
```

An equivalent form is

```math
\frac{dC}{dw_1}
=
2x_1(w_0 + w_1x_1 - y_1)
+
2x_2(w_0 + w_1x_2 - y_2).
```

## Problem II, Question 2

The regularized cost function is

```math
\tilde{C}(w)
=
\sum_{i=1}^{n} \left(y_i - (w_0 + w_1x_i)\right)^2
+
\lambda(w_0^2 + w_1^2).
```

Since

```math
n = 2,
```

this is

```math
\tilde{C}(w)
=
\left(y_1 - (w_0 + w_1x_1)\right)^2
+
\left(y_2 - (w_0 + w_1x_2)\right)^2
+
\lambda(w_0^2 + w_1^2).
```

The regularization term adds

```math
2\lambda w_0
```

to the derivative with respect to

```math
w_0
```

and

```math
2\lambda w_1
```

to the derivative with respect to

```math
w_1.
```

So the derivative with respect to

```math
w_0
```

is

```math
\frac{d\tilde{C}}{dw_0}
=
-2\left(y_1 - (w_0 + w_1x_1)\right)
-2\left(y_2 - (w_0 + w_1x_2)\right)
+
2\lambda w_0.
```

Equivalently,

```math
\frac{d\tilde{C}}{dw_0}
=
2(w_0 + w_1x_1 - y_1)
+
2(w_0 + w_1x_2 - y_2)
+
2\lambda w_0.
```

The derivative with respect to

```math
w_1
```

is

```math
\frac{d\tilde{C}}{dw_1}
=
-2x_1\left(y_1 - (w_0 + w_1x_1)\right)
-2x_2\left(y_2 - (w_0 + w_1x_2)\right)
+
2\lambda w_1.
```

Equivalently,

```math
\frac{d\tilde{C}}{dw_1}
=
2x_1(w_0 + w_1x_1 - y_1)
+
2x_2(w_0 + w_1x_2 - y_2)
+
2\lambda w_1.
```

In summation form, the same answer is

```math
\frac{d\tilde{C}}{dw_0}
=
-2\sum_{i=1}^{2}\left(y_i - (w_0 + w_1x_i)\right)
+
2\lambda w_0,
```

and

```math
\frac{d\tilde{C}}{dw_1}
=
-2\sum_{i=1}^{2}x_i\left(y_i - (w_0 + w_1x_i)\right)
+
2\lambda w_1.
```

## References

[1] Cortes, C., and Vapnik, V. (1995). [Support-vector networks](https://link.springer.com/article/10.1007/BF00994018). *Machine Learning, 20*, 273-297.

[2] Shalev-Shwartz, S., and Ben-David, S. (2014). [Convex Learning Problems](https://www.cambridge.org/core/books/understanding-machine-learning/convex-learning-problems/68712BE60A0D7640360C8305E86B3A57), in *Understanding Machine Learning*.

[3] Cristianini, N., and Shawe-Taylor, J. (2000). [Support Vector Machines](https://resolve.cambridge.org/core/books/abs/an-introduction-to-support-vector-machines-and-other-kernelbased-learning-methods/support-vector-machines/DD4EA48AA6C383944EA67BF8A7BEC6CC), in *An Introduction to Support Vector Machines*.

[4] Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., and Singer, Y. (2006). [Online Passive-Aggressive Algorithms](https://www.jmlr.org/beta/papers/v7/crammer06a.html). *Journal of Machine Learning Research, 7*, 551-585.
