## Deeplib is a deep learning library for quick testing AI ideas

### Installing
After cloning the repository, run:
```
conda create -n deeplib python==3.10 pip
pip install -r requirements.txt
```

### Try it!
In this example we create a simple model the learns celsius to fahrenheit conversion.
```
from layer import Linear
from nn import Sequential
from optim import SGD
from loss import MSELoss

import numpy as np


X = np.arange(1, 101).reshape((100, 1)) #100 different celsius temperatures
y = X.copy()*1.8 + 32 #its conversions 

model = Sequential(Linear(1, 10),
                   Linear(10, 1))

diff = np.mean(y - model(X), axis = 0) 
print("Difference before training:", diff) 

criterion = MSELoss()
optim = SGD(model, lr=0.0000001)

epochs = 100000
end = '\r'

for e in range(epochs):
    output = model(X)
    criterion.loss(y, output)    
    criterion.backward()
    optim.step(criterion.grads["L"])
    
    loss = criterion.param["loss"]
    if e == epochs -1:
        end = '\n'
    print(f"LOSS: {loss}", end = end)

diff2 = np.mean(y - model(X), axis = 0)
print("Difference after training:", diff2) 
```