"""MLP model for font size regression."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class Parameter:
    """Simple parameter wrapper."""

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)


class FontSizeRegressorMLP:
    """
    Tiny MLP for font size regression.

    Architecture:
    - Input: 34 features (updated)
    - Hidden1: 64 units with ReLU
    - Hidden2: 32 units with ReLU
    - Output: 1 unit (font size in points)
    """

    def __init__(self, input_dim: int = 34, hidden1: int = 64, hidden2: int = 32):
        """
        Initialize MLP with random weights.

        Args:
            input_dim: Input feature dimension
            hidden1: First hidden layer size
            hidden2: Second hidden layer size
        """
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2

        # Initialize weights with He initialization
        scale1 = np.sqrt(2.0 / input_dim)
        self.W1 = Parameter(np.random.randn(input_dim, hidden1).astype(np.float32) * scale1)
        self.b1 = Parameter(np.zeros(hidden1, dtype=np.float32))

        scale2 = np.sqrt(2.0 / hidden1)
        self.W2 = Parameter(np.random.randn(hidden1, hidden2).astype(np.float32) * scale2)
        self.b2 = Parameter(np.zeros(hidden2, dtype=np.float32))

        scale3 = np.sqrt(2.0 / hidden2)
        self.W3 = Parameter(np.random.randn(hidden2, 1).astype(np.float32) * scale3)
        self.b3 = Parameter(np.zeros(1, dtype=np.float32))

        # Cache for backward pass
        self.cache: dict[str, Any] = {}

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        result: np.ndarray = np.maximum(0, x)
        return result

    def relu_backward(self, dout: np.ndarray, x: np.ndarray) -> np.ndarray:
        """ReLU gradient."""
        return dout * (x > 0)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim) or (input_dim,)
            training: Whether in training mode (saves cache for backward)

        Returns:
            Predictions (batch_size, 1) or (1,)
        """
        # Handle single sample
        single_sample = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_sample = True

        # Layer 1
        z1 = x @ self.W1.data + self.b1.data
        a1 = self.relu(z1)

        # Layer 2
        z2 = a1 @ self.W2.data + self.b2.data
        a2 = self.relu(z2)

        # Output layer
        out = a2 @ self.W3.data + self.b3.data

        if training:
            self.cache = {
                "x": x,
                "z1": z1,
                "a1": a1,
                "z2": z2,
                "a2": a2,
            }

        if single_sample:
            result: np.ndarray = out.flatten()
            return result

        output: np.ndarray = out
        return output

    def backward(self, dout: np.ndarray):
        """
        Backward pass.

        Args:
            dout: Gradient of loss w.r.t. output (batch_size, 1)
        """
        x = self.cache["x"]
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]
        a2 = self.cache["a2"]
        z2 = self.cache["z2"]

        batch_size = x.shape[0]

        # Output layer gradients
        self.W3.grad = a2.T @ dout / batch_size
        self.b3.grad = np.sum(dout, axis=0) / batch_size

        # Layer 2 gradients
        da2 = dout @ self.W3.data.T
        dz2 = self.relu_backward(da2, z2)
        self.W2.grad = a1.T @ dz2 / batch_size
        self.b2.grad = np.sum(dz2, axis=0) / batch_size

        # Layer 1 gradients
        da1 = dz2 @ self.W2.data.T
        dz1 = self.relu_backward(da1, z1)
        self.W1.grad = x.T @ dz1 / batch_size
        self.b1.grad = np.sum(dz1, axis=0) / batch_size

    def get_parameters(self):
        """Get all trainable parameters."""
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def save(self, path: str):
        """
        Save model weights to file.

        Args:
            path: Output file path
        """
        state = {
            "input_dim": self.input_dim,
            "hidden1": self.hidden1,
            "hidden2": self.hidden2,
            "W1": self.W1.data.tolist(),
            "b1": self.b1.data.tolist(),
            "W2": self.W2.data.tolist(),
            "b2": self.b2.data.tolist(),
            "W3": self.W3.data.tolist(),
            "b3": self.b3.data.tolist(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "FontSizeRegressorMLP":
        """
        Load model from file.

        Args:
            path: Input file path

        Returns:
            Loaded model
        """
        with open(path, "r") as f:
            state = json.load(f)

        model = cls(
            input_dim=state["input_dim"],
            hidden1=state["hidden1"],
            hidden2=state["hidden2"],
        )

        model.W1.data = np.array(state["W1"], dtype=np.float32)
        model.b1.data = np.array(state["b1"], dtype=np.float32)
        model.W2.data = np.array(state["W2"], dtype=np.float32)
        model.b2.data = np.array(state["b2"], dtype=np.float32)
        model.W3.data = np.array(state["W3"], dtype=np.float32)
        model.b3.data = np.array(state["b3"], dtype=np.float32)

        return model


class MSELoss:
    """Mean Squared Error loss."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compute MSE loss.

        Args:
            pred: Predictions (batch_size, 1)
            target: Ground truth (batch_size, 1)

        Returns:
            Loss value
        """
        self.pred = pred
        self.target = target
        loss: float = float(np.mean((pred - target) ** 2))
        return loss

    def backward(self) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. predictions.

        Returns:
            Gradient (batch_size, 1)
        """
        grad: np.ndarray = 2 * (self.pred - self.target)
        return grad


class AdamOptimizer:
    """Adam optimizer."""

    def __init__(self, parameters, lr: float = 0.001, betas=(0.9, 0.999), eps: float = 1e-8):
        """
        Initialize Adam optimizer.

        Args:
            parameters: List of Parameter objects
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Small constant for numerical stability
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        # Initialize moment estimates
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self):
        """Perform one optimization step."""
        self.t += 1

        for i, param in enumerate(self.parameters):
            # Update biased first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad**2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Zero out all gradients."""
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)
