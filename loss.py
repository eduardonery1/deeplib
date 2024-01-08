from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import numpy as np 


@dataclass
class ILoss(ABC):
    param: Dict[str, np.array] = field(default_factory = dict)
    @abstractmethod
    def __call__(self, y_pred: np.array, y: np.array) -> np.array:
        pass

    @abstractmethod
    def backward():
        pass


class MSELoss(ILoss):
    def __call__(self, y: np.array, y_pred: np.array) -> np.array:
        self.param["y_pred"] = y_pred
        return np.mean((y - y_pred) ** 2, axis = 0)
    

if __name__=="__main__":
    criterion = MSELoss()
    y, y_pred = np.random.rand(32, 1), np.random(32, 1)
    loss = criterion(y, y_pred)
    print(loss)

