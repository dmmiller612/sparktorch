from sparktorch.torch_distributed import SparkTorchModel
import torch.nn as nn
import codecs
import dill
from pyspark.ml.pipeline import PipelineModel


def convert_to_serialized_torch(network: nn.Module) -> str:
    """
    Converts an existing torch network to a serialized string.

    :param network: a nn.Module that you want to serialize
    :return: Returns the serialized torch model.
    """
    return codecs.encode(dill.dumps(network), "base64").decode()


def create_spark_torch_model(
    network: nn.Module,
    inputCol: str = 'features',
    predictionCol: str = 'predicted',
    useVectorOut: bool = False
) -> SparkTorchModel:
    """
    Creates a spark SparkTorchModel from an already trained network. Useful for running inference on large datasets.

    :param network: an already trained network
    :param inputCol: The spark dataframe input column
    :param predictionCol: The spark dataframe prediction columns
    :param useVectorOut: Determines whether the output should return a spark vector
    :return: Returns a SparkTorchModel
    """

    return SparkTorchModel(
        inputCol=inputCol,
        predictionCol=predictionCol,
        modStr=convert_to_serialized_torch(network),
        useVectorOut=useVectorOut
    )


def attach_pytorch_model_to_pipeline(
    network: nn.Module,
    pipeline_model: PipelineModel,
    inputCol: str = 'features',
    predictionCol: str = 'predicted',
    useVectorOut: bool = False
) -> PipelineModel:
    """
    Attaches a pytorch model to an existing pyspark pipeline.

    :param network: Pytorch Network
    :param pipeline_model: An existing spark pipeline model (This is a fitted pipeline)
    :param inputCol: The input column to the dataframe for the pytorch network
    :param predictionCol: The prediction column.
    :param useVectorOut: option to use a vector output.
    :return: a spark PipelineModel
    """

    spark_model = create_spark_torch_model(network, inputCol, predictionCol, useVectorOut)
    return PipelineModel(stages=[pipeline_model, spark_model])
