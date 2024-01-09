from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ILayer(ABC):
    params: Dict[str, np.ndarray]= field(default_factory=dict)
    grads: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        pass 

    @abstractmethod
    def backward(self, grad: np.ndarray) -> None:
        pass


class Linear(ILayer):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.params["w"] = np.random.rand(input_dim, output_dim) #W @ X 
        self.params["b"] = np.random.rand(1, output_dim) if bias is True else np.zeros((1, output_dim))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """ 
        X: (batch_size, input_size)
        Expects each data point to be a column vector.
        Computes: y = X @ W + b, (output_size, batch_size)
        """
        self.params["x"] = X
        return X @ self.params["w"] + self.params["b"]

    def backward(self, grad: np.ndarray) -> None:
        """
        y = f(x); x = a * b + c
        dy/da = f'(x) * b
        """
        self.grads["w"] = self.params["x"].T @ grad
        self.grads["x"] = grad @ self.params["w"].T
        self.grads["b"] = np.sum(grad, axis = 0, keepdims=True)
        


class Activation(ILayer):
    pass


class ReLU(Activation):
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.params["x"] = X
        self.zeros = np.zeros(X.shape)
        return np.where(X > 0, X, self.zeros)
    
    def backward(self, grad: np.ndarray) -> None:
        self.grads["x"] = np.where(grad > 0, grad, self.zeros) 

if __name__ == "__main__":
    layer = Linear(16, 4)
    output = layer.forward(np.random.rand(32, 16))
    layer.backward(output)
    print(layer.grads["x"].shape == (32, 16)) 