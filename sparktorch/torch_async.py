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
import numpy as np
import torch
import codecs

from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasPredictionCol, HasLabelCol
from pyspark.ml.base import Estimator
from pyspark.ml import Model
from pyspark.ml.util import Identifiable, MLReadable, MLWritable
from pyspark import keyword_only
import time
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F
import dill

from pyspark import SparkContext


def handle_data(data, inp_col, label_col):
    return DataObj(
        np.asarray(data[inp_col]),
        data[label_col],
        None,
        None
    )


class SparkTorchModel(Model, HasInputCol, HasPredictionCol, PysparkReaderWriter, MLReadable, MLWritable, Identifiable):

    modStr = Param(Params._dummy(), "modStr", "", typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(
        self,
        inputCol=None,
        predictionCol=None,
        modStr=None
    ):
        super().__init__()
        self._setDefault(
            inputCol='encoded',
            predictionCol='predicted',
            modStr=''
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        inputCol=None,
        predictionCol=None,
        modStr=None
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        inp = self.getOrDefault(self.inputCol)
        out = self.getOrDefault(self.predictionCol)
        mod_str = self.getOrDefault(self.modStr)
        model = dill.loads(codecs.decode(mod_str.encode(), "base64"))
        model_broadcast = dataset._sc.broadcast(model)

        def predict_func(data):
            features = np.asarray(data).reshape((1, len(data)))
            x_data = torch.from_numpy(features).float()
            model = model_broadcast.value
            model.eval()
            return Vectors.dense(model(x_data).detach().numpy().flatten())

        udfGenerateCode = F.udf(predict_func, VectorUDT())

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
    iters = Param(Params._dummy(), "iters", "", typeConverter=TypeConverters.toInt)
    partitions = Param(Params._dummy(), "partitions", "", typeConverter=TypeConverters.toInt)
    verbose = Param(Params._dummy(), "verbose", "", typeConverter=TypeConverters.toInt)
    acquireLock = Param(Params._dummy(), "acquireLock", "", typeConverter=TypeConverters.toBoolean)
    shufflePerIter = Param(Params._dummy(), "shufflePerIter", "", typeConverter=TypeConverters.toBoolean)
    partitionShuffles = Param(Params._dummy(), "partitionShuffles", "", typeConverter=TypeConverters.toInt)
    port = Param(Params._dummy(), "port", "", typeConverter=TypeConverters.toInt)

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
        shufflePerIter=None,
        verbose=None,
        partitionShuffles=None,
        port=None
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
            shufflePerIter=True,
            verbose=0,
            partitionShuffles=1,
            port=3000
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
        shufflePerIter=None,
        verbose=None,
        partitionShuffles=None,
        port=None
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def getTorchObj(self):
        return self.getOrDefault(self.torchObj)

    def getIters(self):
        return self.getOrDefault(self.iters)

    def getPartitions(self):
        return self.getOrDefault(self.partitions)

    def getVerbose(self):
        return self.getOrDefault(self.verbose)

    def getAqcuireLock(self):
        return self.getOrDefault(self.acquireLock)

    def getShufflePerIter(self):
        return self.getOrDefault(self.shufflePerIter)

    def getPartitionShuffles(self):
        return self.getOrDefault(self.partitionShuffles)

    def getPort(self):
        return self.getOrDefault(self.port)

    def _fit(self, dataset):
        inp_col = self.getInputCol()
        label = self.getLabelCol()
        prediction = self.getPredictionCol()

        torch_obj = self.getTorchObj()
        iters = self.getIters()
        partitions = self.getPartitions()
        acquire_lock = self.getAqcuireLock()
        verbose = self.getVerbose()
        spi = self.getShufflePerIter()
        partition_shuffles = self.getPartitionShuffles()
        port = self.getPort()

        df = dataset.rdd.map(lambda x: handle_data(x, inp_col, label))
        if partitions > 0:
            df = df.coalesce(partitions) if partitions < df.getNumPartitions() else df

        master_url = SparkContext._active_spark_context.getConf().get("spark.driver.host").__str__() + ":" + str(port)
        server = Server(
            torch_obj=torch_obj,
            master_url=master_url,
            port=port,
            acquire_lock=acquire_lock
        )

        server.start_server()
        time.sleep(5)
        print(f'Server is running {get_main(master_url)}')

        state_dict = train(df, torch_obj, server, iters, partition_shuffles, verbose=verbose)
        loaded = load_torch_model(torch_obj)
        model = loaded.model
        model.load_state_dict(state_dict)
        dumped_model = codecs.encode(dill.dumps(model), "base64").decode()

        return SparkTorchModel(
            inputCol=inp_col,
            predictionCol=prediction,
            modStr=dumped_model
        )
