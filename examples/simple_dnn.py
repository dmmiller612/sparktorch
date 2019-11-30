from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from sparktorch import SparkTorch, serialize_torch_obj
from pyspark.sql.functions import rand
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline, PipelineModel
from sparktorch import PysparkPipelineWrapper
import torch
import torch.nn as nn

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("examples") \
        .master('local[2]').config('spark.driver.memory', '2g') \
        .getOrCreate()

    # Read in mnist_train.csv dataset
    df = spark.read.option("inferSchema", "true").csv('mnist_train.csv').orderBy(rand()).repartition(2)

    network = nn.Sequential(
        nn.Linear(784, 256),
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
        lr=0.00001
    )

    # Setup features
    vector_assembler = VectorAssembler(inputCols=df.columns[1:785], outputCol='features')

    # Demonstration of some options. Not all are required
    # Note: This uses the barrier execution mode, which is sensitive to the number of partitions
    spark_model = SparkTorch(
        inputCol='features',
        labelCol='_c0',
        predictionCol='predictions',
        torchObj=torch_obj,
        iters=100,
        partitions=2,
        verbose=1,
        earlyStopPatience=10,
        useBarrier=True
    )

    # Create and save the Pipeline
    p = Pipeline(stages=[vector_assembler, spark_model]).fit(df)
    p.save('simple_dnn')

    # Example of loading the pipeline
    loaded_pipeline = PysparkPipelineWrapper.unwrap(PipelineModel.load('simple_dnn'))

    # Run predictions and evaluation
    predictions = loaded_pipeline.transform(df).persist()
    evaluator = MulticlassClassificationEvaluator(
        labelCol="_c0", predictionCol="predictions", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Train Error = %g" % (1.0 - accuracy))

