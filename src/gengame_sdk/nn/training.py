"""
Supervised training for NeuralNetwork via backpropagation + SGD.

Pure NumPy implementation — no external deep learning frameworks.
"""

from dataclasses import dataclass, field

import numpy as np

from .activations import activation_derivative, apply_activation
from .network import NeuralNetwork


@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32


@dataclass
class TrainingResult:
    losses: list[float] = field(default_factory=list)
    final_loss: float = 0.0


def _forward_with_cache(
    nn: NeuralNetwork, x: np.ndarray
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Forward pass storing intermediate values for backprop.

    Args:
        nn: Neural network
        x: Input batch of shape (batch_size, input_size)

    Returns:
        (output, cache) where cache is list of (z, a) per layer
    """
    cache: list[tuple[np.ndarray, np.ndarray]] = []
    current = x
    num_layers = len(nn.weights)

    for i, (w, b) in enumerate(zip(nn.weights, nn.biases, strict=True)):
        z = current @ w + b
        if i == num_layers - 1:
            a = apply_activation(z, nn.output_activation)
        else:
            a = apply_activation(z, nn.activation)
        cache.append((z, a))
        current = a

    return current, cache


def _cross_entropy_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Cross-entropy loss.

    Args:
        predictions: Softmax output, shape (batch_size, num_classes)
        targets: Class indices, shape (batch_size,)

    Returns:
        Scalar loss value
    """
    batch_size = predictions.shape[0]
    # Clip to avoid log(0)
    clipped = np.clip(predictions, 1e-12, 1.0)
    # Pick the predicted probability for the correct class
    correct_log_probs = -np.log(clipped[np.arange(batch_size), targets])
    return np.mean(correct_log_probs)


def _backward(
    nn: NeuralNetwork,
    x: np.ndarray,
    cache: list[tuple[np.ndarray, np.ndarray]],
    targets: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Backward pass computing gradients.

    Args:
        nn: Neural network
        x: Input batch, shape (batch_size, input_size)
        cache: List of (z, a) from forward pass
        targets: Class indices, shape (batch_size,)

    Returns:
        (dweights, dbiases) — gradients for each layer
    """
    batch_size = x.shape[0]
    num_layers = len(nn.weights)

    dweights = [None] * num_layers
    dbiases = [None] * num_layers

    # Output layer: softmax + cross-entropy derivative
    _z_last, a_last = cache[-1]
    # dz = softmax_output - y_onehot
    dz = a_last.copy()
    dz[np.arange(batch_size), targets] -= 1.0
    dz /= batch_size

    # Gradient for last layer
    if num_layers > 1:
        a_prev = cache[-2][1]
    else:
        a_prev = x

    dweights[-1] = a_prev.T @ dz
    dbiases[-1] = np.sum(dz, axis=0)

    # Hidden layers (back to front)
    for i in range(num_layers - 2, -1, -1):
        # Propagate gradient through next layer's weights
        dz = (dz @ nn.weights[i + 1].T) * activation_derivative(cache[i][0], nn.activation)

        if i > 0:
            a_prev = cache[i - 1][1]
        else:
            a_prev = x

        dweights[i] = a_prev.T @ dz
        dbiases[i] = np.sum(dz, axis=0)

    return dweights, dbiases


def train_backprop(
    nn: NeuralNetwork, X: np.ndarray, y: np.ndarray, config: TrainingConfig
) -> TrainingResult:
    """
    Train the neural network via backpropagation + SGD.

    Modifies nn.weights and nn.biases in-place.

    Args:
        nn: Initialized neural network (output_activation should be 'softmax')
        X: Training inputs, shape (n_samples, input_size)
        y: Target class indices, shape (n_samples,)
        config: Training hyperparameters

    Returns:
        TrainingResult with loss history
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_samples = X.shape[0]
    losses: list[float] = []

    for _epoch in range(config.epochs):
        # Shuffle data
        perm = np.random.permutation(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        epoch_losses = []

        # Mini-batches
        for start in range(0, n_samples, config.batch_size):
            end = min(start + config.batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # Forward
            output, cache = _forward_with_cache(nn, X_batch)

            # Loss
            loss = _cross_entropy_loss(output, y_batch)
            epoch_losses.append(loss)

            # Backward
            dweights, dbiases = _backward(nn, X_batch, cache, y_batch)

            # SGD update
            for i in range(len(nn.weights)):
                nn.weights[i] -= config.learning_rate * dweights[i]
                nn.biases[i] -= config.learning_rate * dbiases[i]

        avg_loss = np.mean(epoch_losses)
        losses.append(float(avg_loss))

    return TrainingResult(losses=losses, final_loss=losses[-1] if losses else 0.0)
