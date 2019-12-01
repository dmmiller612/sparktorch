"""
The MIT License

Copyright 2019 Derek Miller

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from sparktorch.pipeline_util import PysparkReaderWriter
from sparktorch.util import DataObj, load_torch_model
from sparktorch.server import Server
from sparktorch.hogwild import train, get_main
from sparktorch.async_server import train_async
import numpy as np
import torch
import codecs
import socket

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasPredictionCol, HasLabelCol
from pyspark.ml.base import Estimator
from pyspark.ml import Model
from pyspark.ml.util import Identifiable, MLReadable, MLWritable
from pyspark import keyword_only
import time
from pyspark.sql import functions as F
import dill
from pyspark.sql.types import DoubleType
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark import SparkContext


def handle_data(input_col, label_col):
    def inner(dataset):
        res = [
            DataObj(
                x_train=np.asarray(data[input_col].toArray()),
                y_train=data[label_col] if label_col else None,
                x_val=None,
                y_val=None
            ) for data in dataset
        ]
        return res

    return inner


class SparkTorchModel(Model, HasInputCol, HasPredictionCol, PysparkReaderWriter, MLReadable, MLWritable, Identifiable):

    modStr = Param(Params._dummy(), "modStr", "", typeConverter=TypeConverters.toString)
    useVectorOut = Param(Params._dummy(), "useVectorOut", "", typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(
        self,
        inputCol=None,
        predictionCol=None,
        modStr=None,
        useVectorOut=None
    ):
        super().__init__()
        self._setDefault(
            inputCol='encoded',
            predictionCol='predicted',
            modStr='',
            useVectorOut=False
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        inputCol=None,
        predictionCol=None,
        modStr=None,
        useVectorOut=None
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        inp = self.getOrDefault(self.inputCol)
        out = self.getOrDefault(self.predictionCol)
        mod_str = self.getOrDefault(self.modStr)
        use_vector_out = self.getOrDefault(self.useVectorOut)

        model = dill.loads(codecs.decode(mod_str.encode(), "base64"))
        model_broadcast = dataset._sc.broadcast(model)

        def predict_vec(data):
            features = data.toArray().reshape((1, len(data)))
            x_data = torch.from_numpy(features).float()
            model = model_broadcast.value
            model.eval()
            return Vectors.dense(model(x_data).detach().numpy().flatten())

        def predict_float(data):
            features = data.toArray().reshape((1, len(data)))
            x_data = torch.from_numpy(features).float()
            model = model_broadcast.value
            model.eval()
            raw_prediction = model(x_data).detach().numpy().flatten()
            if len(raw_prediction) > 1:
                return float(np.argmax(raw_prediction))
            return float(raw_prediction[0])

        if use_vector_out:
            udfGenerateCode = F.udf(predict_vec, VectorUDT())
        else:
            udfGenerateCode = F.udf(predict_float, DoubleType())

        return dataset.withColumn(out, udfGenerateCode(inp))


class SparkTorch(
    Estimator,
    HasInputCol,
    HasPredictionCol,
    HasLabelCol,
    PysparkReaderWriter,
    MLReadable,
    MLWritable,
    Identifiable
):

    torchObj = Param(Params._dummy(), "torchObj", "", typeConverter=TypeConverters.toString)
    mode = Param(Params._dummy(), "mode", "", typeConverter=TypeConverters.toString)
    iters = Param(Params._dummy(), "iters", "", typeConverter=TypeConverters.toInt)
    partitions = Param(Params._dummy(), "partitions", "", typeConverter=TypeConverters.toInt)
    verbose = Param(Params._dummy(), "verbose", "", typeConverter=TypeConverters.toInt)
    acquireLock = Param(Params._dummy(), "acquireLock", "", typeConverter=TypeConverters.toBoolean)
    partitionShuffles = Param(Params._dummy(), "partitionShuffles", "", typeConverter=TypeConverters.toInt)
    port = Param(Params._dummy(), "port", "", typeConverter=TypeConverters.toInt)
    useBarrier = Param(Params._dummy(), "useBarrier", "", typeConverter=TypeConverters.toBoolean)
    useVectorOut = Param(Params._dummy(), "useVectorOut", "", typeConverter=TypeConverters.toBoolean)
    earlyStopPatience = Param(Params._dummy(), "earlyStopPatience", "", typeConverter=TypeConverters.toInt)
    miniBatch = Param(Params._dummy(), "miniBatch", "", typeConverter=TypeConverters.toInt)
    validationPct = Param(Params._dummy(), "validationPct", "", typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(
        self,
        inputCol=None,
        labelCol=None,
        torchObj=None,
        iters=None,
        predictionCol=None,
        partitions=None,
        acquireLock=None,
        verbose=None,
        partitionShuffles=None,
        port=None,
        useBarrier=None,
        useVectorOut=None,
        earlyStopPatience=None,
        miniBatch=None,
        validationPct=None,
        mode=None
    ):
        super().__init__()
        self._setDefault(
            inputCol='features',
            labelCol=None,
            torchObj='',
            iters=10,
            predictionCol='predicted',
            partitions=-1,
            acquireLock=True,
            verbose=0,
            partitionShuffles=1,
            port=3000,
            useBarrier=False,
            useVectorOut=False,
            earlyStopPatience=-1,
            miniBatch=-1,
            validationPct=0.0,
            mode='async'
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        inputCol=None,
        labelCol=None,
        torchObj=None,
        iters=None,
        predictionCol=None,
        partitions=None,
        acquireLock=None,
        verbose=None,
        partitionShuffles=None,
        port=None,
        useBarrier=None,
        useVectorOut=None,
        earlyStopPatience=None,
        miniBatch=None,
        validationPct=None,
        mode=None
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getTorchObj(self):
        return self.getOrDefault(self.torchObj)

    def getEarlyStopPatience(self):
        return self.getOrDefault(self.earlyStopPatience)

    def getIters(self):
        return self.getOrDefault(self.iters)

    def getPartitions(self):
        return self.getOrDefault(self.partitions)

    def getVerbose(self):
        return self.getOrDefault(self.verbose)

    def getAqcuireLock(self):
        return self.getOrDefault(self.acquireLock)

    def getPartitionShuffles(self):
        return self.getOrDefault(self.partitionShuffles)

    def getPort(self):
        return self.getOrDefault(self.port)

    def getBarrier(self):
        return self.getOrDefault(self.useBarrier)

    def getVectorOut(self):
        return self.getOrDefault(self.useVectorOut)

    def getMiniBatch(self):
        return self.getOrDefault(self.miniBatch)

    def getValidationPct(self):
        return self.getOrDefault(self.validationPct)

    def getMode(self):
        return self.getOrDefault(self.mode)

    def _fit(self, dataset):
        inp_col = self.getInputCol()
        label = self.getLabelCol()
        prediction = self.getPredictionCol()

        torch_obj = self.getTorchObj()
        iters = self.getIters()
        partitions = self.getPartitions()
        acquire_lock = self.getAqcuireLock()
        verbose = self.getVerbose()
        partition_shuffles = self.getPartitionShuffles()
        port = self.getPort()
        barrier = self.getBarrier()
        use_vector_out = self.getVectorOut()
        early_stop_patience = self.getEarlyStopPatience()
        mini_batch = self.getMiniBatch()
        validation_pct = self.getValidationPct()
        mode = self.getMode()

        rdd = dataset.rdd.mapPartitions(handle_data(inp_col, label))

        if partitions > 0:
            rdd = rdd.repartition(partitions)

        partitions = partitions if partitions > 0 else rdd.getNumPartitions()

        master_url = SparkContext._active_spark_context.getConf().get("spark.driver.host").__str__()
        if mode == 'async':
            state_dict = train_async(
                rdd=rdd,
                torch_obj=torch_obj,
                master_url=master_url,
                iters=iters,
                partition_shuffles=1,
                verbose=verbose,
                mini_batch=mini_batch,
                validation_pct=validation_pct,
                world_size=partitions+1
            )
        elif mode == 'hogwild':
            if barrier:
                rdd = rdd.barrier()
            server = Server(
                torch_obj=torch_obj,
                master_url=master_url + ":" + str(port),
                port=port,
                acquire_lock=acquire_lock,
                early_stop_patience=early_stop_patience,
                window_len=partitions
            )

            server.start_server()
            time.sleep(5)
            print(f'Server is running {get_main(master_url + ":" + str(port))}')

            state_dict = train(
                rdd, torch_obj, server, iters,
                partition_shuffles, verbose=verbose,
                early_stop_patience=early_stop_patience,
                mini_batch=mini_batch,
                validation_pct=validation_pct
            )
        else:
            raise RuntimeError(f"mode: {mode} not recognized.")

        loaded = load_torch_model(torch_obj)
        model = loaded.model
        model.load_state_dict(state_dict)
        dumped_model = codecs.encode(dill.dumps(model), "base64").decode()

        return SparkTorchModel(
            inputCol=inp_col,
            predictionCol=prediction,
            modStr=dumped_model,
            useVectorOut=use_vector_out
        )
