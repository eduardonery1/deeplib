from abc import ABC, abstractmethod
from typing import Dict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ILayer(ABC):
    params: Dict[str, np.array]= field(default_factory=dict)
    grads: Dict[str, np.array] = field(default_factory=dict)
    
    @abstractmethod
    def forward(self, input_data: np.array) -> np.array:
        pass 

    @abstractmethod
    def backward(self, grad: np.array) -> None:
        pass


class Linear(ILayer):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.params["w"] = np.random.rand(output_dim, input_dim) #W @ X.T 
        self.params["b"] = np.random.rand(output_dim, 1) if bias is True else np.zeros((output_dim,1))

    def forward(self, X: np.array) -> np.array:
        """
        Expects a row vector, or matrix with second dimension equals to input_dim.
        Computes: y = W @ X.T + b
        """
        self.params["x"] = X
        return self.params["w"] @ X.T + self.params["b"]

    def backward(self, grad: np.array) -> None:
        """
        y = f(x); x = a * b + c
        dy/da = f'(x) * b
        """
        self.grads["w"] = grad @ self.params["x"]
        self.grads["b"] = np.sum(grad, axis = 0)
        self.grads["x"] = grad.T @ self.params["w"]



if __name__ == "__main__":
    layer = Linear(16, 4)
    output = layer.forward(np.random.rand(32, 16))
    layer.backward(output)
    print(layer.grads["x"].shape == (32, 16)) 