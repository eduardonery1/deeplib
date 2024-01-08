from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np 


@dataclass
class ILoss(ABC):
    param: Dict[str, np.ndarray] = field(default_factory = dict)
    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward():
        pass


class MSELoss(ILoss):
    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        self.param["loss"]   = np.mean((y - y_pred) ** 2, axis = 1) 
        self.param["y"]      = y
        self.param["y_pred"] = y_pred 
        return self.loss()

    def loss(self):
        return self.param["loss"] 

    def backward(self):
        self.grads["L"] = -2/self.param["y"].shape[1]*(self.param["y"] - self.param["y_pred"])
    
if __name__=="__main__":
    criterion = MSELoss()
    y, y_pred = np.random.rand(32, 1), np.random(32, 1)
    loss = criterion(y, y_pred)
    print(loss)

