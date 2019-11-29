import dill
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.util import JavaMLReader, JavaMLWriter
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.wrapper import JavaParams
from pyspark.context import SparkContext
import zlib
import sys

"""
based off below stackoverflow thread. Changes were made for performance.
credit: https://stackoverflow.com/questions/41399399/serialize-a-custom-transformer-using-python-to-be-used-within-a-pyspark-ml-pipel
"""


class PysparkObjId(object):
    """
    A class to specify constants used to idenify and setup python
    Estimators, Transformers and Models so they can be serialized on there
    own and from within a Pipline or PipelineModel.
    """
    def __init__(self):
        super(PysparkObjId, self).__init__()

    @staticmethod
    def _getPyObjId():
        return '4c1740b00d3c4ff6806a1402321572cb'

    @staticmethod
    def _getCarrierClass(javaName=False):
        return 'org.apache.spark.ml.feature.StopWordsRemover' if javaName else StopWordsRemover


def load_byte_array(stop_words):
    swords = stop_words[0].split(',')[0:-1]
    if sys.version_info[0] < 3:
        lst = [chr(int(d)) for d in swords]
        dmp = ''.join(lst)
        dmp = zlib.decompress(dmp)
        py_obj = dill.loads(dmp)
        return py_obj
    dmp = bytes([int(i) for i in swords])
    dmp = zlib.decompress(dmp)
    py_obj = dill.loads(dmp)
    return py_obj


class PysparkPipelineWrapper(object):
    """
    A class to facilitate converting the stages of a Pipeline or PipelineModel
    that were saved from PysparkReaderWriter.
    """
    def __init__(self):
        super(PysparkPipelineWrapper, self).__init__()

    @staticmethod
    def unwrap(pipeline):
        if not (isinstance(pipeline, Pipeline) or isinstance(pipeline, PipelineModel)):
            raise TypeError("Cannot recognize a pipeline of type %s." % type(pipeline))

        stages = pipeline.getStages() if isinstance(pipeline, Pipeline) else pipeline.stages
        for i, stage in enumerate(stages):
            if (isinstance(stage, Pipeline) or isinstance(stage, PipelineModel)):
                stages[i] = PysparkPipelineWrapper.unwrap(stage)
            if isinstance(stage, PysparkObjId._getCarrierClass()) and stage.getStopWords()[-1] == PysparkObjId._getPyObjId():
                swords = stage.getStopWords()[:-1] # strip the id
                py_obj = load_byte_array(swords)
                stages[i] = py_obj

        if isinstance(pipeline, Pipeline):
            pipeline.setStages(stages)
        else:
            pipeline.stages = stages
        return pipeline


class PysparkReaderWriter(object):
    """
    A mixin class so custom pyspark Estimators, Transformers and Models may
    support saving and loading directly or be saved within a Pipline or PipelineModel.
    """
    def __init__(self):
        super(PysparkReaderWriter, self).__init__()

    def write(self):
        """Returns an MLWriter instance for this ML instance."""
        return JavaMLWriter(self)

    @classmethod
    def read(cls):
        """Returns an MLReader instance for our clarrier class."""
        return JavaMLReader(PysparkObjId._getCarrierClass())

    @classmethod
    def load(cls, path):
        """Reads an ML instance from the input path, a shortcut of `read().load(path)`."""
        swr_java_obj = cls.read().load(path)
        return cls._from_java(swr_java_obj)

    @classmethod
    def _from_java(cls, java_obj):
        """
        Get the dumby the stopwords that are the characters of the dills dump plus our guid
        and convert, via dill, back to our python instance.
        """
        swords = java_obj.getStopWords()[:-1] # strip the id
        return load_byte_array(swords)

    def _to_java(self):
        """
        Convert this instance to a dill dump, then to a list of strings with the unicode integer values of each character.
        Use this list as a set of dumby stopwords and store in a StopWordsRemover instance
        :return: Java object equivalent to this instance.
        """
        dmp = dill.dumps(self)
        dmp = zlib.compress(dmp)
        sc = SparkContext._active_spark_context
        pylist = [str(i) + ',' for i in bytearray(dmp)]
        # convert bytes to string integer list
        pylist = [''.join(pylist)]
        pylist.append(PysparkObjId._getPyObjId()) # add our id so PysparkPipelineWrapper can id us.
        java_class = sc._gateway.jvm.java.lang.String
        java_array = sc._gateway.new_array(java_class, len(pylist))
        java_array[0:2] = pylist[0:2]
        _java_obj = JavaParams._new_java_obj(PysparkObjId._getCarrierClass(javaName=True), self.uid)
        _java_obj.setStopWords(java_array)
        return _java_obj

