from abc import ABC, abstractmethod
from typing import List
from layer import ILayer
from dataclasses import dataclass, field

import numpy as np


@dataclass
class INeuralNet(ABC):
    layers: List[ILayer] = field(default_factory=list)
    
    @abstractmethod
    def forward(self, input_data: np.array) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> None:
        pass


class Sequential(INeuralNet):
    def __init__(self, *args: ILayer):
        super().__init__()
        for arg in args:
            try:
                assert(isinstance(arg, ILayer))
                self.layers.append(arg)
            except AssertionError:
                raise Exception("Tried creating a Sequential nn with invalid layer")
    
    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        return self.forward(input_data)
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        X = input_data.copy()
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            layer.backward(grad)


if __name__=="__main__":
    from layer import Linear
    from loss import MSELoss
    from optim import SGD


    batch_size  = 32
    input_size  = 8
    hidden      = 4
    output_size = 1
    epochs      = 10000

    model = Sequential(Linear(input_size, hidden),
                       Linear(hidden, output_size))
    criterion = MSELoss()
    optim = SGD(model)

    input_data  = np.random.rand(input_size, batch_size)
    dummy_label = np.random.rand(output_size, batch_size) 

    end = '\r'    
    max_loss = 0
    min_loss = 1000

    for e in range(epochs):
        output = model(input_data)
        criterion.loss(dummy_label, output)
        criterion.backward()
        optim.step(criterion.grads["L"])
        
        loss = criterion.param["loss"][0]
        if loss > max_loss: max_loss = loss
        if loss < min_loss: min_loss = loss

        if e == epochs-1:
            end = '\n' 
        print("LAST:", loss, "MAX:", max_loss, "MIN:", min_loss, end=end)


