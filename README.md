# SparkTorch

This is an implementation of Pytorch on Spark. The goal of this library is to provide a simple, understandable interface 
in using Torch on Spark. With SparkTorch, you can easily integrate your deep learning model with a ML Spark Pipeline.
Underneath, SparkTorch uses a parameter server to train the Pytorch network in a distributed manner. Through the api,
the user can specify the style of training, whether that is Hogwild or async with locking.

## Why should I use this?

Like SparkFlow, SparkTorch's main objective is to seamlessly work with Spark's ML Pipelines. This library provides three 
core components:

* Distributed training for large datasets. Multiple Pytorch models are ran in parallel with one central network that 
manages weights. This is useful for training very large datasets that do not fit into a single machine.
* Full integration with Spark's ML library. This ensures that you can save and load pipelines with your trained model.
* Inference. With SparkTorch, you can load your existing trained model and run inference on billions of records 
in parallel. 

On top of these features, SparkTorch can utilize barrier execution, ensuring that all executors run currently during 
training. 

## Install

Install SparkTorch via pip: `pip install sparktorch`

SparkTorch requires Apache Spark >= 2.4.4, and has only been tested on PyTorch versions >= 1.2.0.

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