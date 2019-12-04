import pytest
from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.linalg import Vectors
import torch.nn as nn
import torch
from sparktorch.inference import create_spark_torch_model
from sparktorch.util import serialize_torch_obj, serialize_torch_obj_lazy
from sparktorch.torch_distributed import SparkTorch
from sparktorch.tests.simple_net import Net, AutoEncoder, ClassificationNet, NetworkWithParameters


@pytest.fixture()
def spark():
    return (SparkSession.builder
            .master('local[2]')
            .appName('sparktorch')
            .getOrCreate())


@pytest.fixture()
def data(spark):
    dat = [(1.0, Vectors.dense(np.random.normal(0,1,10))) for _ in range(0, 200)]
    dat2 = [(0.0, Vectors.dense(np.random.normal(2,1,10))) for _ in range(0, 200)]
    dat.extend(dat2)
    return spark.createDataFrame(dat, ["label", "features"]).repartition(2)


@pytest.fixture()
def sequential_model():
    model = torch.nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    return serialize_torch_obj(
        model, nn.MSELoss(), torch.optim.Adam, lr=0.001
    )


@pytest.fixture()
def lazy_model():
    model = serialize_torch_obj_lazy(
        Net, nn.MSELoss, torch.optim.Adam, optimizer_params={'lr': 0.001}
    )
    return model


@pytest.fixture()
def general_model():
    model = serialize_torch_obj(
        Net(), nn.MSELoss(), torch.optim.Adam, lr=0.001
    )
    return model


@pytest.fixture()
def network_with_params():
    return serialize_torch_obj_lazy(
        NetworkWithParameters,
        nn.MSELoss,
        torch.optim.Adam,
        optimizer_params={'lr': 0.001},
        model_parameters={'param': 40}
    )


def test_model_parameters(data, network_with_params):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=network_with_params,
        verbose=1,
        iters=5
    ).fit(data)

    py_model = stm.getPytorchModel()
    assert py_model.fc1 is not None
    assert py_model.fc2 is not None


def test_inference(lazy_model, data):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=lazy_model,
        verbose=1,
        iters=10
    ).fit(data)

    first_res = stm.transform(data).take(1)

    res = stm.getPytorchModel()
    spark_model = create_spark_torch_model(
        res,
        'features',
        'predictions'
    )

    res = spark_model.transform(data).take(1)
    assert first_res == res


def test_lazy(lazy_model, data):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=lazy_model,
        verbose=1,
        iters=5
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]
    assert type(res[0]['predictions']) is float


def test_simple_hogwild(data, sequential_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=sequential_model,
        verbose=1,
        mode='hogwild',
        iters=5
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]
    assert type(res[0]['predictions']) is float


def test_simple_sequential(data, sequential_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=sequential_model,
        verbose=1,
        iters=5
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]
    assert type(res[0]['predictions']) is float


def test_simple_torch_module(data, general_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=general_model,
        iters=5,
        verbose=1
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]
    assert type(res[0]['predictions']) is float


def test_barrier(data, general_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=general_model,
        iters=5,
        verbose=1,
        partitions=2,
        useBarrier=True
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]


def test_autoencoder(data):
    model = serialize_torch_obj(
        AutoEncoder(), nn.MSELoss(), torch.optim.Adam, lr=0.001
    )

    stm = SparkTorch(
        inputCol='features',
        predictionCol='predictions',
        torchObj=model,
        iters=5,
        verbose=1,
        partitions=2,
        useVectorOut=True
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]
    assert len(res[0]['predictions']) == 10


def test_classification(data):
    model = serialize_torch_obj(
        ClassificationNet(), nn.CrossEntropyLoss(), torch.optim.Adam, lr=0.001
    )

    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=model,
        iters=5,
        verbose=1,
        partitions=2
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]


def test_early_stopping_async(data, general_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=general_model,
        iters=25,
        verbose=1,
        earlyStopPatience=2
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]


def test_mini_batch(data, general_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=general_model,
        iters=10,
        verbose=1,
        partitions=2,
        miniBatch=5,
        acquireLock=True
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]


def test_cpu_device(data, general_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=general_model,
        iters=10,
        verbose=1,
        partitions=2,
        miniBatch=5,
        acquireLock=True,
        device='cpu'
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]


def test_validation_pct(data, general_model):
    stm = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=general_model,
        iters=10,
        verbose=1,
        partitions=2,
        validationPct=0.25
    ).fit(data)

    res = stm.transform(data).take(1)
    assert 'predictions' in res[0]

