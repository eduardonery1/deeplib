from abc import ABC, abstractmethod
from nn import INeuralNet

import numpy as np


class IOptimizer(ABC):
    @abstractmethod
    def step(self, grad: np.ndarray) -> None:
        pass


class SGD(IOptimizer):
    def __init__(self, model: INeuralNet, lr = 1e-3):
        self.model = model
        self.lr = lr

    def step(self, grad: np.ndarray) -> None:
        for layer in reversed(self.model.layers):
            layer.backward(grad)
            grad = layer.params["x"]
            layer.params["w"] -= self.lr*layer.grads["w"]
            layer.params["b"] -= self.lr*layer.grads["b"]


