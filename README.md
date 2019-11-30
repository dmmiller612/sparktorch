# SparkTorch

This is an implementation of Pytorch on Spark. The goal of this library is to provide a simple, understandable interface 
in using Torch on Spark. With SparkTorch, you can easily integrate your deep learning model with a ML Spark Pipeline.
Underneath, SparkTorch uses a parameter server to train the Pytorch network in a distributed manner. Through the api,
the user can specify the style of training, whether that is Hogwild or async with locking.

## Install

`pip install sparktorch`

## Full Basic Example

```python
from sparktorch import serialize_torch_obj, SparkTorch
import torch
import torch.nn as nn
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline

spark = SparkSession.builder.appName("examples").master('local[4]').getOrCreate()
df = spark.read.option("inferSchema", "true").csv('mnist_train.csv').coalesce(4)

network = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    nn.Softmax(dim=1)
)

# Build the pytorch object
torch_obj = serialize_torch_obj(
    model=network,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    lr=0.0001
)

# Setup features
vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')

# Create a SparkTorch Model with barrier execution
spark_model = SparkTorch(
    inputCol='features',
    labelCol='_c0',
    predictionCol='predictions',
    torchObj=torch_obj,
    iters=50,
    partitions=4,
    verbose=1,
    useBarrier=True
)

# Can be used in a pipeline and saved.
p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
p.save('simple_dnn')
```


## Literature and Inspiration

* HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent: https://arxiv.org/pdf/1106.5730.pdf
* Elephas: https://github.com/maxpumperla/elephas
* Scaling Distributed Machine Learning with the Parameter Server: https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf