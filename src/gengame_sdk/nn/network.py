"""
Neural Network module.
Simple feedforward network with configurable architecture.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .activations import apply_activation

# Type aliases
InitMethod = Literal["xavier", "he", "uniform", "normal"]
Activation = Literal["tanh", "relu", "sigmoid", "leaky_relu", "linear", "softmax"]


def _initialize_weights(
    in_size: int,
    out_size: int,
    method: InitMethod = "xavier",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Initialize weight matrix.

    Args:
        in_size: Input dimension
        out_size: Output dimension
        method: Initialization method
        rng: Random number generator (for reproducibility)

    Returns:
        Initialized weight matrix of shape (in_size, out_size)
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == "xavier":
        # Xavier/Glorot initialization - good for tanh/sigmoid
        limit = np.sqrt(6.0 / (in_size + out_size))
        return rng.uniform(-limit, limit, (in_size, out_size))

    elif method == "he":
        # He initialization - good for ReLU
        std = np.sqrt(2.0 / in_size)
        return rng.normal(0, std, (in_size, out_size))

    elif method == "uniform":
        # Simple uniform [-0.5, 0.5]
        return rng.uniform(-0.5, 0.5, (in_size, out_size))

    elif method == "normal":
        # Simple normal (0, 0.1)
        return rng.normal(0, 0.1, (in_size, out_size))

    else:
        raise ValueError(f"Unknown initialization method: {method}")


@dataclass
class NeuralNetwork:
    """
    Simple feedforward neural network.

    Attributes:
        architecture: List of layer sizes [input, hidden1, ..., output]
        activation: Activation function for hidden layers
        output_activation: Activation function for output layer (default: same as activation)
        init_method: Weight initialization method
        weights: List of weight matrices
        biases: List of bias vectors
    """

    architecture: list[int]
    activation: Activation = "tanh"
    output_activation: Activation | None = None
    init_method: InitMethod = "xavier"
    weights: list[np.ndarray] = field(default_factory=list)
    biases: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        # Default output_activation to activation if not specified
        if self.output_activation is None:
            self.output_activation = self.activation

    def initialize(self, seed: int | None = None) -> "NeuralNetwork":
        """
        Initialize weights and biases.

        Args:
            seed: Random seed for reproducibility

        Returns:
            self (for chaining)
        """
        rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []

        for i in range(len(self.architecture) - 1):
            in_size = self.architecture[i]
            out_size = self.architecture[i + 1]

            w = _initialize_weights(in_size, out_size, self.init_method, rng)
            b = np.zeros(out_size)

            self.weights.append(w)
            self.biases.append(b)

        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input array of shape (input_size,)

        Returns:
            Output array of shape (output_size,)
        """
        if not self.weights:
            raise RuntimeError("Network not initialized. Call initialize() first.")

        current = np.asarray(x, dtype=np.float64)
        num_layers = len(self.weights)

        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = current @ w + b

            # Use output_activation for last layer, activation for others
            if i == num_layers - 1:
                current = apply_activation(z, self.output_activation)
            else:
                current = apply_activation(z, self.activation)

        return current

    def to_flat(self) -> np.ndarray:
        """
        Flatten all weights and biases into a single array.

        Returns:
            1D array of all parameters
        """
        if not self.weights:
            raise RuntimeError("Network not initialized. Call initialize() first.")

        parts = []
        for w, b in zip(self.weights, self.biases, strict=True):
            parts.append(w.flatten())
            parts.append(b)
        return np.concatenate(parts)

    @classmethod
    def from_flat(
        cls,
        flat: np.ndarray,
        architecture: list[int],
        activation: Activation = "tanh",
        output_activation: Activation | None = None,
        init_method: InitMethod = "xavier",
    ) -> "NeuralNetwork":
        """
        Create network from flattened weights.

        Args:
            flat: Flattened parameter array
            architecture: Network architecture
            activation: Hidden layer activation
            output_activation: Output layer activation
            init_method: Initialization method (stored for reference)

        Returns:
            NeuralNetwork with loaded weights
        """
        nn = cls(
            architecture=architecture,
            activation=activation,
            output_activation=output_activation,
            init_method=init_method,
        )

        nn.weights = []
        nn.biases = []

        idx = 0
        for i in range(len(architecture) - 1):
            in_size = architecture[i]
            out_size = architecture[i + 1]

            w_size = in_size * out_size
            w = flat[idx : idx + w_size].reshape(in_size, out_size)
            idx += w_size

            b = flat[idx : idx + out_size].copy()
            idx += out_size

            nn.weights.append(w)
            nn.biases.append(b)

        return nn

    def copy(self) -> "NeuralNetwork":
        """Create a deep copy of the network."""
        return NeuralNetwork(
            architecture=self.architecture.copy(),
            activation=self.activation,
            output_activation=self.output_activation,
            init_method=self.init_method,
            weights=[w.copy() for w in self.weights],
            biases=[b.copy() for b in self.biases],
        )

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        if not self.weights:
            # Calculate without initializing
            total = 0
            for i in range(len(self.architecture) - 1):
                in_size = self.architecture[i]
                out_size = self.architecture[i + 1]
                total += in_size * out_size + out_size  # weights + biases
            return total
        return len(self.to_flat())

    @property
    def is_initialized(self) -> bool:
        """Check if network has been initialized."""
        return len(self.weights) > 0

    def __repr__(self) -> str:
        init_str = "initialized" if self.is_initialized else "not initialized"
        return (
            f"NeuralNetwork(arch={self.architecture}, "
            f"act={self.activation}, out_act={self.output_activation}, "
            f"init={self.init_method}, params={self.num_parameters}, {init_str})"
        )


def create_network(
    architecture: list[int],
    activation: Activation = "tanh",
    output_activation: Activation | None = None,
    init_method: InitMethod = "xavier",
    seed: int | None = None,
) -> NeuralNetwork:
    """
    Factory function to create and initialize a neural network.

    Args:
        architecture: List of layer sizes [input, hidden..., output]
        activation: Activation function for hidden layers
        output_activation: Activation for output layer (default: same as activation)
        init_method: Weight initialization method
        seed: Random seed for reproducibility

    Returns:
        Initialized NeuralNetwork

    Example:
        >>> nn = create_network([9, 32, 9], activation='tanh', output_activation='softmax', seed=42)
        >>> output = nn.forward(np.zeros(9))
    """
    nn = NeuralNetwork(
        architecture=architecture,
        activation=activation,
        output_activation=output_activation,
        init_method=init_method,
    )
    nn.initialize(seed=seed)
    return nn
