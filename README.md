# Linear Regression from Scratch

A small Python program that fits a straight line to a set of data points using **gradient descent**, written without any machine learning libraries. It was made as a university assignment to show how linear regression works inside.

## Purpose

The program learns the parameters `w` (slope) and `b` (intercept) of a line from a small training dataset, then uses that line to predict the output for a new input value. Every step is written in plain Python so the math stays visible: the model, the cost function, the gradients and the parameter updates.

## The Model

Linear regression assumes the output `y` is a linear function of the input `x`:

```
y_hat = f(x) = w * x + b
```

- `w` is the slope of the line
- `b` is the intercept (where the line crosses the y-axis)
- `y_hat` is the model's prediction

The training data is six `(x, y)` pairs that lie roughly on a straight line:

| x | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 |
|---|-----|-----|-----|-----|-----|-----|
| y | 3.1 | 4.9 | 7.2 | 8.8 | 11.1 | 12.9 |

## Cost Function: Mean Squared Error (MSE)

To measure how well a line fits the data, the program squares the error of each prediction and averages those squared errors:

```
J(w, b) = (1/N) * Σ (f(x_i) - y_i)²
```

A lower MSE means a better fit. The goal is to find the `w` and `b` that make `J(w, b)` as small as possible.

## Gradient Descent

The MSE cost is a smooth, bowl-shaped function of `w` and `b`. Gradient descent finds the bottom of the bowl by repeatedly taking small steps downhill.

**1. Compute the gradients** (the slope of the cost with respect to each parameter):

```
dJ/dw = (2/N) * Σ (f(x_i) - y_i) * x_i
dJ/db = (2/N) * Σ (f(x_i) - y_i)
```

**2. Update the parameters** in the opposite direction of the gradient:

```
w = w - learning_rate * dJ/dw
b = b - learning_rate * dJ/db
```

The program starts from `w = 0`, `b = 0` and uses a learning rate of `0.01` for `10,000` iterations.

## Code Structure

All the code is in `linear_regression.py`:

| Function | Role |
|----------|------|
| `predict(x, w, b)` | Returns `w * x + b` |
| `compute_cost(x_data, y_data, w, b)` | Returns the MSE over the dataset |
| `compute_gradients(x_data, y_data, w, b)` | Returns `dJ/dw` and `dJ/db` |
| `gradient_descent(x_data, y_data, learning_rate, iterations)` | Trains the model and returns the learned `w` and `b` |

## How to Run

You need Python 3. No external packages are required.

```bash
python linear_regression.py
```

## Example Output

```
Training linear regression with gradient descent...

Iteration     0: cost = 34.951601, w = 0.6753, b = 0.1600
Iteration  1000: cost = 0.018520, w = 1.9807, b = 1.0649
Iteration  2000: cost = 0.018476, w = 1.9772, b = 1.0796
...
Iteration  9000: cost = 0.018476, w = 1.9771, b = 1.0800

----- Results -----
Learned w (slope):     1.9771
Learned b (intercept): 1.0800
Final cost (MSE):      0.018476
Model: y_hat = 1.9771 * x + 1.0800

Training data vs. predictions:
  x = 1.0   y = 3.10   y_hat = 3.06
  x = 2.0   y = 4.90   y_hat = 5.03
  x = 3.0   y = 7.20   y_hat = 7.01
  x = 4.0   y = 8.80   y_hat = 8.99
  x = 5.0   y = 11.10   y_hat = 10.97
  x = 6.0   y = 12.90   y_hat = 12.94

Prediction for new x = 7.0: y_hat = 14.9200
```

The cost drops quickly and then levels off. The final MSE is small but not zero, because the data points do not lie exactly on a single line. The learned values match the exact closed-form least-squares answer (`w ≈ 1.9771`, `b = 1.08`).

## Limitations

This is a learning exercise, not a general-purpose tool:

- It supports only **one input feature** (simple linear regression).
- The dataset, learning rate and number of iterations are **hard-coded** in the script.
- It does not include feature scaling, a train/test split or plotting.
