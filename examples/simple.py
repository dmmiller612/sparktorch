from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
from pyspark.rdd import PipelinedRDD
import torch
import torch.nn as nn
from pyspark.ml.linalg import Vectors
import numpy as np
import torch.distributed as dist
from torch.multiprocessing import Process
import os

def mapPartitionsWithIndex(rdd, f, preservesPartitioning=False):
    return PipelinedRDD(rdd, f, preservesPartitioning, isFromBarrier=True)


def init_process():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    partitioner(-1, None)


def partitioner(index, data):
    torch.manual_seed(1234)
    print(index)
    model = torch.nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    if index < 0:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('gloo', rank=index+1, world_size=3)
        model.zero_grad()
        for i in range(1000):
            for param in model.parameters():
                param.sum().backward()
                param.grad.data = torch.zeros_like(param.grad.data)
                dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)

    else:
        dist.init_process_group('gloo', init_method='tcp://127.0.0.1:29500', rank=index + 1, world_size=3)

    if index < 0:
        return
    features = []
    labels = []
    for d, l in data:
        features.append(d)
        labels.append(l)

    features = np.stack(features)
    labels = np.asarray(labels)

    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()

        loss = criterion(model(features), labels)
        loss.backward()

        for param in model.parameters():
            ret = dist.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)

        optimizer.step()

    return [model.state_dict()]


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("examples") \
        .master('local[2]').config('spark.driver.memory', '2g') \
        .getOrCreate()

    dat = [(1.0, Vectors.dense(np.random.normal(0, 1, 10))) for _ in range(0, 200)]
    dat2 = [(0.0, Vectors.dense(np.random.normal(2, 1, 10))) for _ in range(0, 200)]
    dat.extend(dat2)
    df = spark.createDataFrame(dat, ["label", "features"]).repartition(2)
    p = Process(target=init_process)
    p.start()

    model = torch.nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    model.zero_grad()

    try:
        x = mapPartitionsWithIndex(df.rdd.map(lambda x: (x['features'].toArray(), x['label'])), partitioner).collect()
        print(x)
    finally:
        p.terminate()