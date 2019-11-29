from pyspark.sql import SparkSession
import unittest
import logging
import numpy as np
from pyspark.ml.linalg import Vectors
import torch.nn as nn
import torch
from sparktorch.util import serialize_torch_obj
from sparktorch.torch_async import SparkTorch
from tests.simple_net import Net


class PysparkTest(unittest.TestCase):

    @classmethod
    def suppress_logging(cls):
        logger = logging.getLogger('py4j')
        logger.setLevel(logging.WARN)

    @classmethod
    def create_testing_spark_session(cls):
        return (SparkSession.builder
                .master('local[2]')
                .appName('sparktorch')
                .getOrCreate())

    @classmethod
    def setUpClass(cls):
        cls.suppress_logging()
        cls.spark = cls.create_testing_spark_session()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


class SparkTorchTests(PysparkTest):

    def generate_random_data(self):
        dat = [(1.0, Vectors.dense(np.random.normal(0,1,10))) for _ in range(0, 200)]
        dat2 = [(0.0, Vectors.dense(np.random.normal(2,1,10))) for _ in range(0, 200)]
        dat.extend(dat2)
        return self.spark.createDataFrame(dat, ["label", "features"])

    @staticmethod
    def generate_sequential_model():
        model = torch.nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        return serialize_torch_obj(
            model, nn.MSELoss(), torch.optim.Adam, lr=0.001
        )

    def test_simple_sequential(self):
        model = self.generate_sequential_model()
        data = self.generate_random_data()

        stm = SparkTorch(
            inputCol='features',
            labelCol='label',
            predictionCol='predictions',
            torchObj=model,
            iters=5
        ).fit(data)

        res = stm.transform(data).take(1)
        self.assertTrue('predictions' in res[0])
        self.assertTrue(len(res[0]['predictions']) == 1)

    def test_simple_torch_module(self):
        model = serialize_torch_obj(
            Net(), nn.MSELoss(), torch.optim.Adam, lr=0.001
        )
        data = self.generate_random_data()

        stm = SparkTorch(
            inputCol='features',
            labelCol='label',
            predictionCol='predictions',
            torchObj=model,
            iters=5,
            verbose=1
        ).fit(data)

        res = stm.transform(data).take(1)
        self.assertTrue('predictions' in res[0])
        self.assertTrue(len(res[0]['predictions']) == 1)


if __name__ == '__main__':
    unittest.main()
