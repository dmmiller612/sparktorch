from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, OneHotEncoder
from sparktorch import SparkTorch, PysparkPipelineWrapper, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
import torch
import torch.nn as nn


if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("examples") \
        .master('local[4]').config('spark.driver.memory', '2g') \
        .getOrCreate()

    # Read in mnist_train.csv dataset
    df = spark.read.option("inferSchema", "true").csv('mnist_train.csv').orderBy(rand()).coalesce(4)

    network = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 96),
        nn.ReLU(),
        nn.Linear(96, 10),
        nn.Softmax(dim=1)
    )

    # Build the pytorch object
    torch_obj = serialize_torch_obj(
        model=network,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        lr=0.0001
    )

    # Setup features
    vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')
    encoder = OneHotEncoder(inputCol='_c0', outputCol='labels', dropLast=False)

    # Demonstration of some options. Not all are required
    # Note: This uses the barrier execution mode, which is sensitive to the number of partitions
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='labels',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=50,
        partitions=4,
        verbose=1,
        useBarrier=True
    )

    # Create and save the Pipeline
    p = Pipeline(stages=[vector_assembler, encoder, spark_model]).fit(df)
    p.save('simple_dnn')

    # Example of loading the pipeline
    loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_dnn'))
    demonstration = loaded_pipeline.transform(df).take(10)
    print(demonstration)
