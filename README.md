# SparkTorch

[![Build Status](https://travis-ci.com/dmmiller612/sparktorch.svg?branch=master)](https://travis-ci.org/dmmiller612/sparktorch)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/dmmiller612/sparktorch)

This is an implementation of Pytorch on Apache Spark. The goal of this library is to provide a simple, understandable interface 
in distributing the training of your Pytorch model on Spark. With SparkTorch, you can easily integrate your deep 
learning model with a ML Spark Pipeline. Underneath the hood, SparkTorch offers two distributed training approaches 
through tree reductions and a parameter server. Through the api, the user can specify the style of training, whether 
that is distributed synchronous or hogwild.

## Why should I use this?

Like SparkFlow, SparkTorch's main objective is to seamlessly work with Spark's ML Pipelines. This library provides three 
core components:

* Data parallel distributed training for large datasets. SparkTorch offers distributed synchronous and asynchronous training methodologies. 
This is useful for training very large datasets that do not fit into a single machine.
* Full integration with Spark's ML library. This ensures that you can save and load pipelines with your trained model.
* Inference. With SparkTorch, you can load your existing trained model and run inference on billions of records 
in parallel. 

On top of these features, SparkTorch can utilize barrier execution, ensuring that all executors run concurrently during 
training (This is required for synchronous training approaches). 

## Install

Install SparkTorch via pip: `pip install sparktorch`

SparkTorch requires Apache Spark >= 2.4.4, and has only been tested on PyTorch versions >= 1.3.0.

## Full Basic Example

```python
from sparktorch import serialize_torch_obj, SparkTorch
import torch
import torch.nn as nn
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline

spark = SparkSession.builder.appName("examples").master('local[2]').getOrCreate()
df = spark.read.option("inferSchema", "true").csv('mnist_train.csv').coalesce(2)

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

# Create a SparkTorch Model with torch distributed. Barrier execution is on by default for this mode.
spark_model = SparkTorch(
    inputCol='features',
    labelCol='_c0',
    predictionCol='predictions',
    torchObj=torch_obj,
    iters=50,
    verbose=1
)

# Can be used in a pipeline and saved.
p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
p.save('simple_dnn')
```

## Run the Examples

You can run the examples through docker by issuing the following commands at the root of the repository:
```bash
make docker-build
make docker-run-dnn
make docker-run-cnn
```

For the cnn example, you will need to give your docker container at least 8gb of memory.

## Documentation

This is a small documentation section on how to SparkTorch. Please look at the examples library for more details.

#### Creating a Torch Object

To create a Torch object for training, you will need to utilize the `serialize_torch_obj` from SparkTorch. To do so, 
simply add your network, loss criterion, the optimizer class, and any options as a dictionary to supply to the optimizer 
(such as learning rate). A simple example of this is:

```python
from sparktorch import serialize_torch_obj

torch_obj = serialize_torch_obj(
    model=network,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam,
    lr=0.0001
)
```

When training neural networks on Spark, one issue that many face is OOM errors. To avoid this issue on the driver, you 
can create a torch object that is only initialized on the worker nodes. To create this object, you can set up the 
following:

```python
from sparktorch import serialize_torch_obj_lazy

torch_obj = serialize_torch_obj_lazy(
    model=Network,
    criterion=nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    optimizer_params={'lr': 0.001},
    model_parameters={'some_model_param': 5}
)
``` 

The Network in the above example must be a nn.module pytorch class. If you need parameters passed into the constructor, 
you can use the `model_parameters` parameter. The item will be passed in as **kwargs to the constructor. 

NOTE: One thing to remember is that if your network is not a sequential, it will need to be saved in a separate file and
available in the python path. An example of this can be found in `simple_cnn.py`.

#### Training Options

There are two main training options with SparkTorch: `async` and `hogwild`. The async mode utilizes the torch distributed 
package, ensuring that the networks are in sync through each iteration. This is the most supported version. When using 
this option, you will need to be aware that barrier execution is enforced, meaning that the parallelism will need to match 
the partitions. 

The Hogwild approach utilizes a Flask Service underneath the hood. When using Hogwild, it is strongly recommended that you use the 
`useBarrier` option to force barrier execution. Below are a list of parameters to SparkTorch and their meaning.

```
inputCol: Standard Spark InputCol that must be a Vector.
labelCol: Standard Spark Label column. Can be null.
torchObj: The TorchObj which is described in the `Creating a Torch Object` section.
iters: Number of iterations to run per partition.
predictionCol: The standard spark prediction column for the dataframe.
partitions: Ability to repartition during training.
acquireLock: Used in Hogwild only. Forces locking on the server.
verbose: Describes whether you want real time logging
partitionShuffles: Only used in Hogwild. Will reshuffle data after completing training.
port: Only used in hogwild. Server port.
useBarrier: Only used in hogwild. Describes whether you want barrier execution. (Async mode uses barrier by default)
useVectorOut: Boolean to describe if you want the model output to be a vector (Defaults to float).
earlyStopPatience: If greater than 0, it will enforce early stopping based on validation.
miniBatch: Minibatch size for training per iteration. (Randomly shuffled)
validationPct: Percentage to use for validation.
mode: which training mode to use. `synchronous` leverages torch distributed. `hogwild` currently uses the flask service.
device: Supply 'cpu' or 'cuda'
```

#### Saving and Loading Pipelines

Since saving and loading custom ML Transformers in pure python has not been implemented in PySpark, an extension has been
added here to make that possible. In order to save a Pyspark Pipeline with Apache Spark, one will need to use the overwrite function:

```python
p = Pipeline(stages=[va, encoded, spark_model]).fit(df)
p.write().overwrite().save("location")
```

For loading, a Pipeline wrapper has been provided in the pipeline_utils file. An example is below:

```python
from sparktorch import PysparkPipelineWrapper
from pyspark.ml.pipeline import PipelineModel

p = PysparkPipelineWrapper.unwrap(PipelineModel.load('location'))
``` 
Then you can perform predictions, etc with:

```python
predictions = p.transform(df)
```

#### Getting the Pytorch model from the training session

If you just want to get the Pytorch model after training, you can execute the following code:

```python
stm = SparkTorch(
    inputCol='features',
    labelCol='label',
    predictionCol='predictions',
    torchObj=network_with_params,
    verbose=1,
    iters=5
).fit(data)

py_model = stm.getPytorchModel()
```


#### Using a pretrained Pytorch model for inference

If you already have a trained Pytorch model, you can attach it your existing pipeline by directly creating a SparkTorchModel. 
This can be done by running the following:

```python
from sparktorch import create_spark_torch_model

net = ... # Pretrained Network

spark_torch_model = create_spark_torch_model(
    net, 
    inputCol='features',
    predictionCol='predictions'
)
```

## Running

One big thing to remember is to add the `--executor cores 1` option to spark to ensure
each executor is only training one copy of the network. This will especially be needed for gpu training.

## Contributing

Contributions are always welcome. This could be fixing a bug, changing documentation, or adding a new feature. To test 
new changes against existing tests, we have provided a Docker container which takes in an argument of the python version. 
This allows the user to check their work before pushing to Github, where travis-ci will run.

For python 3.6
```
docker build -t local-test --build-arg PYTHON_VERSION=3.6 .
docker run --rm local-test:latest bash -i -c "pytest"
```

## Literature and Inspiration

* Distributed training: http://seba1511.net/dist_blog/
* HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent: https://arxiv.org/pdf/1106.5730.pdf
* Scaling Distributed Machine Learning with the Parameter Server: https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf