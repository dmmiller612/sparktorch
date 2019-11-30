# SparkTorch

This is an implementation of Pytorch on Spark. The goal of this library is to provide a simple, understandable interface 
in using Torch on Spark. With SparkTorch, you can easily integrate your deep learning model with a ML Spark Pipeline.
Underneath, SparkTorch uses a parameter server to train the Pytorch network in a distributed manner. Through the api,
the user can specify the style of training, whether that is Hogwild or async with locking.


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


## Literature and Inspiration

* HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent: https://arxiv.org/pdf/1106.5730.pdf
* Elephas: https://github.com/maxpumperla/elephas
* Scaling Distributed Machine Learning with the Parameter Server: https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf