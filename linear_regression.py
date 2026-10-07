"""
Simple Linear Regression implemented from scratch.

Model:        f(x) = w*x + b
Loss:         (f(x_i) - y_i)^2                    (squared error for one example)
Cost (MSE):   J(w, b) = (1/N) * sum((f(x_i) - y_i)^2)
Goal:         find w and b that minimize J(w, b)

The parameters are found with gradient descent (no ML libraries are used).
"""

# ---------------------------------------------------------------
# 1. Training data (x_i, y_i)
#    Example: x = hours studied, y = exam score (out of 20).
#    The points lie roughly on a straight line, with a little noise.
# ---------------------------------------------------------------
x_train = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
y_train = [3.1, 4.9, 7.2, 8.8, 11.1, 12.9]


# ---------------------------------------------------------------
# 2. The model: y_hat = w*x + b
# ---------------------------------------------------------------
def predict(x, w, b):
    """Return the model's prediction for a single input x."""
    return w * x + b


# ---------------------------------------------------------------
# 3. Cost function: Mean Squared Error
#    J(w, b) = (1/N) * sum( (f(x_i) - y_i)^2 )
# ---------------------------------------------------------------
def compute_cost(x_data, y_data, w, b):
    """Return the average squared error of the model over the dataset."""
    n = len(x_data)
    total_error = 0.0
    for i in range(n):
        error = predict(x_data[i], w, b) - y_data[i]   # f(x_i) - y_i
        total_error += error ** 2                      # squared error (loss)
    return total_error / n                             # average -> MSE


# ---------------------------------------------------------------
# 4. Gradients of the cost with respect to w and b
#    dJ/dw = (2/N) * sum( (f(x_i) - y_i) * x_i )
#    dJ/db = (2/N) * sum( (f(x_i) - y_i) )
#    The gradient points in the direction where the cost increases,
#    so we will move in the OPPOSITE direction to decrease it.
# ---------------------------------------------------------------
def compute_gradients(x_data, y_data, w, b):
    n = len(x_data)
    dw = 0.0
    db = 0.0
    for i in range(n):
        error = predict(x_data[i], w, b) - y_data[i]
        dw += error * x_data[i]
        db += error
    dw = (2 / n) * dw
    db = (2 / n) * db
    return dw, db


# ---------------------------------------------------------------
# 5. Gradient descent: repeatedly update w and b to minimize the cost
#    w := w - learning_rate * dJ/dw
#    b := b - learning_rate * dJ/db
#    Because the MSE cost is a smooth "bowl" shape, taking small steps
#    downhill brings us to the minimum, i.e. the optimal w and b.
# ---------------------------------------------------------------
def gradient_descent(x_data, y_data, learning_rate, iterations):
    w = 0.0   # start from an arbitrary initial guess
    b = 0.0

    for step in range(iterations):
        dw, db = compute_gradients(x_data, y_data, w, b)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Print progress occasionally so we can see the cost going down
        if step % 1000 == 0:
            cost = compute_cost(x_data, y_data, w, b)
            print(f"Iteration {step:5d}: cost = {cost:.6f}, w = {w:.4f}, b = {b:.4f}")

    return w, b


# ---------------------------------------------------------------
# 6. Main program
# ---------------------------------------------------------------
if __name__ == "__main__":
    learning_rate = 0.01   # size of each step downhill
    iterations = 10000     # number of update steps

    print("Training linear regression with gradient descent...\n")
    w, b = gradient_descent(x_train, y_train, learning_rate, iterations)

    final_cost = compute_cost(x_train, y_train, w, b)

    print("\n----- Results -----")
    print(f"Learned w (slope):     {w:.4f}")
    print(f"Learned b (intercept): {b:.4f}")
    print(f"Final cost (MSE):      {final_cost:.6f}")
    print(f"Model: y_hat = {w:.4f} * x + {b:.4f}")

    # Show how the model fits each training example
    print("\nTraining data vs. predictions:")
    for i in range(len(x_train)):
        y_hat = predict(x_train[i], w, b)
        print(f"  x = {x_train[i]:.1f}   y = {y_train[i]:.2f}   y_hat = {y_hat:.2f}")

    # Use the trained model to predict a new, unseen value
    x_new = 7.0
    y_new = predict(x_new, w, b)
    print(f"\nPrediction for new x = {x_new}: y_hat = {y_new:.4f}")
