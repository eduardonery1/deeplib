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
        self.params["w"] = np.random.rand(output_dim, input_dim) #W @ X 
        self.params["b"] = np.random.rand(output_dim, 1) if bias is True else np.zeros((output_dim,1))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """ 
        X: (input_values, batch_size)
        Expects each data point to be a column vector.
        Computes: y = W @ X + b
        """
        self.params["x"] = X
        return self.params["w"] @ X + self.params["b"]

    def backward(self, grad: np.ndarray) -> None:
        """
        y = f(x); x = a * b + c
        dy/da = f'(x) * b
        """
        print(grad.shape, self.params["x"].T.shape)
        self.grads["w"] = grad @ self.params["x"].T
        self.grads["b"] = np.sum(grad, axis = 1, keepdims=True)
        self.grads["x"] = grad.T @ self.params["w"]


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