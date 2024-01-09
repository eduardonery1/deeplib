from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np 


@dataclass
class ILoss(ABC):
    param: Dict[str, np.ndarray] = field(default_factory = dict)
    grads: Dict[str, np.ndarray] = field(default_factory = dict)
   
    @abstractmethod
    def loss(self, y_pred: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def backward():
        pass


class MSELoss(ILoss):
    def loss(self, y: np.ndarray, y_pred: np.ndarray) -> None:
        self.param["loss"]   = np.mean((y - y_pred) ** 2, axis = 1) 
        self.param["y"]      = y
        self.param["y_pred"] = y_pred 

    def backward(self):
        self.grads["L"] = -2/self.param["y"].shape[1]*(self.param["y"] - self.param["y_pred"])
    
if __name__=="__main__":
    criterion = MSELoss()
    y, y_pred = np.random.rand(32, 1), np.random(32, 1)
    loss = criterion(y, y_pred)
    print(loss)

