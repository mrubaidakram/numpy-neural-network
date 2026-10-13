"""A small fully connected classifier with explicit NumPy backpropagation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _stable_softmax(logits: FloatArray) -> FloatArray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass
class DenseCache:
    inputs: FloatArray


@dataclass
class ReLUCache:
    inputs: FloatArray


class MLPClassifier:
    """Multilayer perceptron using dense/ReLU blocks and softmax cross entropy.

    Parameters use float64 for transparent numerical experiments. No autograd
    library is used: every gradient is computed in ``backward``.
    """

    def __init__(self, layer_sizes: tuple[int, ...], seed: int = 42):
        if len(layer_sizes) < 2 or any(size < 1 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive integers")
        self.layer_sizes = layer_sizes
        rng = np.random.default_rng(seed)
        self.weights: list[FloatArray] = []
        self.biases: list[FloatArray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # He initialization suits the ReLU hidden layers.
            self.weights.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

    def forward(self, inputs: FloatArray) -> tuple[FloatArray, list[DenseCache | ReLUCache]]:
        if inputs.ndim != 2 or inputs.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"expected input shape (batch, {self.layer_sizes[0]})")
        activations = inputs.astype(np.float64, copy=False)
        caches: list[DenseCache | ReLUCache] = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            dense_input = activations
            activations = dense_input @ weight + bias
            caches.append(DenseCache(dense_input))
            if index < len(self.weights) - 1:
                caches.append(ReLUCache(activations))
                activations = np.maximum(activations, 0.0)
        return activations, caches

    def loss_and_gradients(
        self, inputs: FloatArray, labels: NDArray[np.int64]
    ) -> tuple[float, list[FloatArray], list[FloatArray]]:
        logits, caches = self.forward(inputs)
        batch_size = inputs.shape[0]
        if labels.shape != (batch_size,):
            raise ValueError("labels must contain one class index per input row")
        if np.any(labels < 0) or np.any(labels >= self.layer_sizes[-1]):
            raise ValueError("label index is outside the output class range")

        probabilities = _stable_softmax(logits)
        loss = -np.log(np.maximum(probabilities[np.arange(batch_size), labels], 1e-15)).mean()
        grad = probabilities
        grad[np.arange(batch_size), labels] -= 1.0
        grad /= batch_size

        grad_weights: list[FloatArray] = [np.empty_like(w) for w in self.weights]
        grad_biases: list[FloatArray] = [np.empty_like(b) for b in self.biases]
        dense_cache_index = len(caches) - 1
        for layer_index in range(len(self.weights) - 1, -1, -1):
            dense_cache = caches[dense_cache_index]
            assert isinstance(dense_cache, DenseCache)
            grad_weights[layer_index] = dense_cache.inputs.T @ grad
            grad_biases[layer_index] = np.sum(grad, axis=0, keepdims=True)
            grad = grad @ self.weights[layer_index].T
            dense_cache_index -= 1
            if layer_index > 0:
                relu_cache = caches[dense_cache_index]
                assert isinstance(relu_cache, ReLUCache)
                grad *= relu_cache.inputs > 0.0
                dense_cache_index -= 1

        return float(loss), grad_weights, grad_biases

    def predict_proba(self, inputs: FloatArray) -> FloatArray:
        logits, _ = self.forward(inputs)
        return _stable_softmax(logits)

    def predict(self, inputs: FloatArray) -> NDArray[np.int64]:
        return np.argmax(self.predict_proba(inputs), axis=1)

    def train_batch(self, inputs: FloatArray, labels: NDArray[np.int64], learning_rate: float) -> float:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        loss, grad_weights, grad_biases = self.loss_and_gradients(inputs, labels)
        for index in range(len(self.weights)):
            self.weights[index] -= learning_rate * grad_weights[index]
            self.biases[index] -= learning_rate * grad_biases[index]
        return loss
