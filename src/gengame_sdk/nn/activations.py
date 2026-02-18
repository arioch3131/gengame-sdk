"""
Activation functions for neural networks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .network import Activation


def apply_activation(x: np.ndarray, activation: Activation) -> np.ndarray:
    """
    Apply activation function.

    Args:
        x: Input array
        activation: Activation function name

    Returns:
        Activated output
    """
    if activation == "tanh":
        return np.tanh(x)
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation == "leaky_relu":
        return np.where(x > 0, x, 0.01 * x)
    elif activation == "softmax":
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    elif activation == "linear":
        return x
    else:
        raise ValueError(f"Unknown activation: {activation}")


def activation_derivative(z: np.ndarray, activation: Activation) -> np.ndarray:
    """
    Derivative of activation function with respect to z.

    Args:
        z: Pre-activation values
        activation: Activation function name

    Returns:
        Derivative values
    """
    if activation == "tanh":
        t = np.tanh(z)
        return 1 - t * t
    elif activation == "relu":
        return (z > 0).astype(np.float64)
    elif activation == "sigmoid":
        s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return s * (1 - s)
    elif activation == "leaky_relu":
        return np.where(z > 0, 1.0, 0.01)
    elif activation == "linear":
        return np.ones_like(z)
    else:
        raise ValueError(f"Unknown activation for derivative: {activation}")
