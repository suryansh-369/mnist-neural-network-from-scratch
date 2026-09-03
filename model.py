import numpy as np


def init_params(input_size=784, hidden_size=128, output_size=10):
    """
    Initialize weights and biases for a fully connected neural network.

    Architecture:
        input_size -> hidden_size -> output_size
    """

    # He initialization for ReLU hidden layer
    W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size)
    b1 = np.zeros((hidden_size, 1))

    # He initialization based on number of inputs to layer 2
    W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2 / hidden_size)
    b2 = np.zeros((output_size, 1))

    return W1, b1, W2, b2


def relu(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)


def deriv_relu(Z):
    """Derivative of ReLU."""
    return Z > 0


def softmax(Z):
    """
    Softmax activation for the output layer.

    Subtracting the maximum value improves numerical stability.
    """
    Z = Z - np.max(Z, axis=0, keepdims=True)

    exp_Z = np.exp(Z)

    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def one_hot(Y, num_classes=10):
    """Convert integer labels into one-hot encoded columns."""

    one_hot_Y = np.zeros((num_classes, Y.size))

    one_hot_Y[Y, np.arange(Y.size)] = 1

    return one_hot_Y


def forward_prop(W1, b1, W2, b2, X):
    """
    Perform forward propagation.

    Returns:
        Z1, A1, Z2, A2
    """

    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    """
    Perform manual backpropagation.

    Returns:
        dw1, db1, dw2, db2
    """

    m = Y.size

    one_hot_Y = one_hot(Y)

    # Output layer
    dz2 = A2 - one_hot_Y

    dw2 = (1 / m) * dz2.dot(A1.T)

    db2 = (1 / m) * np.sum(
        dz2,
        axis=1,
        keepdims=True
    )

    # Hidden layer
    dz1 = W2.T.dot(dz2) * deriv_relu(Z1)

    dw1 = (1 / m) * dz1.dot(X.T)

    db1 = (1 / m) * np.sum(
        dz1,
        axis=1,
        keepdims=True
    )

    return dw1, db1, dw2, db2


def update_params(
    W1, b1, W2, b2,
    dw1, db1, dw2, db2,
    alpha
):
    """Update parameters using gradient descent."""

    W1 = W1 - alpha * dw1
    b1 = b1 - alpha * db1

    W2 = W2 - alpha * dw2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


def get_pred(A):
    """Return predicted class for each example."""

    return np.argmax(A, axis=0)


def get_acc(pred, Y):
    """Calculate classification accuracy."""

    return np.mean(pred == Y)


def get_loss(A, Y):
    """
    Calculate multiclass cross-entropy loss.
    """

    m = Y.size

    epsilon = 1e-10

    correct_class_probs = A[Y, np.arange(m)]

    loss = -np.mean(
        np.log(correct_class_probs + epsilon)
    )

    return loss


def gradient_descent(X, Y, alpha, iterations, hidden_size=128):
    """
    Train the neural network using full-batch gradient descent.

    Args:
        X: Training data, shape (784, m)
        Y: Training labels, shape (m,)
        alpha: Learning rate
        iterations: Number of training iterations
        hidden_size: Number of neurons in hidden layer

    Returns:
        W1, b1, W2, b2
    """

    W1, b1, W2, b2 = init_params(
        input_size=X.shape[0],
        hidden_size=hidden_size,
        output_size=10
    )

    for i in range(iterations):

        # Forward propagation
        Z1, A1, Z2, A2 = forward_prop(
            W1, b1, W2, b2, X
        )

        # Backpropagation
        dw1, db1, dw2, db2 = back_prop(
            Z1, A1, Z2, A2,
            W2,
            X,
            Y
        )

        # Update parameters
        W1, b1, W2, b2 = update_params(
            W1, b1,
            W2, b2,
            dw1, db1,
            dw2, db2,
            alpha
        )

        if i % 10 == 0:

            predictions = get_pred(A2)

            accuracy = get_acc(
                predictions,
                Y
            )

            loss = get_loss(
                A2,
                Y
            )

            print(
                f"Iteration: {i:4d} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

    return W1, b1, W2, b2

