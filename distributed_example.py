from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import torch
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
from pyspark.ml.linalg import Vectors
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("examples") \
        .master('spark://127.0.0.1:7077').config('spark.executor.cores', '1') \
        .getOrCreate()

    dat = [(1.0, Vectors.dense(np.random.normal(0,1,40))) for _ in range(0, 1000)]
    dat2 = [(0.0, Vectors.dense(np.random.normal(2,1,40))) for _ in range(0, 1000)]
    dat.extend(dat2)
    df = spark.createDataFrame(dat, ["label", "features"]).repartition(4)

    network = nn.Sequential(
        nn.Linear(40, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    # Build the pytorch object
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        lr=0.001
    )

    # Demonstration of some options. Not all are required
    # Note: This uses the barrier execution mode, which is sensitive to the number of partitions
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='label',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=1000,
        verbose=1,
        miniBatch=256,
        partitions=4,
        earlyStopPatience=40,
        validationPct=0.2
    ).fit(df)


    # Run predictions and evaluation
    predictions = spark_model.transform(df).persist()
    evaluator = MulticlassClassificationEvaluator(
        labelCol="_c0", predictionCol="predictions", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Train accuracy = %g" % accuracy)

