import numpy as np
from abc import ABC, abstractmethod



class Layer(ABC):
    @abstractmethod
    def forward(self, input_data: array) -> array:
        raise AbstractMethodError()

    @abstractmethod
    def backward(self, grad: float) -> float:
        raise AbstractMethodError()


class Linear(Layer):
    def __init__(self, input_dim: int, output_dim: int, bias = True):
        self.weights =  np.random.rand((output_dim, input_dim)) #W @ X.T 
        self.bias = np.random.rand((output_dim, 1)) if bias is True else np.zeros((output_dim,1))

    def forward(self, X: np.array) -> np.array:
        """
        Expects a row vector, or matrix with second dimension equals to input_dim.
        Computes: y = W @ X.T + b
        """
        return self.weights @ X.T + self.bias

    def backward()
if __name__ == "__main__":
    pass 
