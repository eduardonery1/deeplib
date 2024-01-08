from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np 


@dataclass
class ILoss(ABC):
    param: Dict[str, np.array] = field(default_factory = dict)
    grads: Dict[str, np.array] = field(default_factory = dict)
    @abstractmethod
    def __call__(self, y_pred: np.array, y: np.array) -> np.array:
        pass

    @abstractmethod
    def backward(self) -> None:
        pass


class MSELoss(ILoss):
    def __call__(self, y: np.array, y_pred: np.array) -> np.array:
        self.param["y_pred"] = y_pred
        self.param["y"] = y
        return np.mean((y - y_pred) ** 2, axis = 0)
    
    def backward(self) -> None:
        n = self.param["y_pred"].shape[0]
        self.grads["y_pred"] = -2/n*(self.param["y_pred"] - self.param["y"])
        
if __name__=="__main__":
    criterion = MSELoss()
    y, y_pred = np.random.rand(32, 1), np.random.rand(32, 1)
    loss = criterion(y, y_pred)
    print(loss)
    criterion.backward()
    print(criterion)


