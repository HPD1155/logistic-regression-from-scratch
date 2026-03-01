"""
Binary Classifier from Scratch (NumPy Only)

This script implements a binary classification model using only NumPy.
It covers:

- Data generation
- Activation functions
- Binary cross entropy loss
- Forward pass
- Backward pass
- Gradient descent training loop
- Inference

The dataset is linearly separable with decision rule:

    y = 1 if 2*x1 < x2
    y = 0 otherwise
"""

import numpy as np


# =========================
# Data Generation
# =========================

def generate_data(n_samples=100):
    """
    Generates a linearly separable dataset.
    x1 ~ Uniform integer [0, 100)
    x2 ~ Uniform integer [0, 300)
    y = 1 if 2*x1 < x2 else 0
    """
    x1 = np.random.randint(100, size=(n_samples, 1))
    x2 = np.random.randint(300, size=(n_samples, 1))
    X = np.concatenate((x1, x2), axis=1)

    y = np.where(X[:, 0] * 2 < X[:, 1], 1, 0)
    y = y.reshape(n_samples, 1).astype(np.float64)

    return X, y


# =========================
# Activation Functions
# =========================

def linear(x):
    return x


def d_linear(x):
    return np.ones_like(x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


# =========================
# Loss Function
# =========================

def binary_cross_entropy(y, y_pred):
    """
    Binary Cross Entropy Loss
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(
        y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
    )


def d_binary_cross_entropy(y, y_pred):
    """
    Derivative of BCE wrt predictions
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y) / (y_pred * (1 - y_pred))


# =========================
# Model Training
# =========================

def train(X, y, epochs=20000, lr=0.01):
    """
    Trains logistic regression using gradient descent.
    """
    n_samples, n_features = X.shape

    # Initialize parameters
    w = np.random.rand(n_features, 1)
    b = np.ones((1,))

    for i in range(epochs):

        # Forward pass
        z = linear(np.dot(X, w) + b)
        y_pred = sigmoid(z)

        # Compute loss
        loss = binary_cross_entropy(y, y_pred)

        # Backward pass
        dBCE_dpred = d_binary_cross_entropy(y, y_pred)
        dsig_dz = d_sigmoid(z)
        dLoss_dz = dBCE_dpred * dsig_dz

        dw = X.T.dot(dLoss_dz) / n_samples
        db = np.sum(dLoss_dz) / n_samples

        # Update
        w -= lr * dw
        b -= lr * db

        if i % 1000 == 0:
            print(f"Epoch {i}: Loss = {loss:.6f}")

    print("\nTraining complete.")
    print(f"Final Loss: {loss:.6f}")
    print("Weights:", w.flatten())
    print("Bias:", b)

    return w, b


# =========================
# Inference
# =========================

def predict(X, w, b):
    """
    Returns probability predictions.
    """
    X = np.array(X)
    z = linear(np.dot(X, w) + b)
    return sigmoid(z)


def predict_class(X, w, b, threshold=0.5):
    """
    Returns binary class predictions.
    """
    probs = predict(X, w, b)
    return (probs >= threshold).astype(int)


# =========================
# Main Execution
# =========================

if __name__ == "__main__":

    # Generate data
    X, y = generate_data()

    # Train model
    w, b = train(X, y)

    # Example inference
    sample = np.array([[85, 159]])
    prob = predict(sample, w, b)
    pred_class = predict_class(sample, w, b)

    print("\nSample input:", sample)
    print("Predicted probability:", prob)
    print("Predicted class:", pred_class)