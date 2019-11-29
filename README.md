# SparkTorch

Train and run Pytorch models on Apache Spark. 


## Install

`pip install sparktorch`

## Basic Example

```python
from sparktorch.util import serialize_torch_obj
from sparktorch.torch_async import SparkTorch
import torch
import torch.nn as nn

#Spark dataframe
dataframe = spark.createDataFrame(data, ["label", "features"])

# serialize object
model = serialize_torch_obj(
    nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ), 
    nn.MSELoss(), 
    torch.optim.Adam, 
    lr=0.001
)

# SparkTorch Model
stm = SparkTorch(
    inputCol='features',
    labelCol='label',
    predictionCol='predictions',
    torchObj=model,
    iters=5
).fit(dataframe)

results = stm.transform(dataframe)
```
