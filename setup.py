from setuptools import setup
from setuptools import find_packages

setup(name='sparktorch',
      version='0.1.2',
      description='Distributed training of PyTorch networks on Apache Spark with ML Pipeline support',
      keywords = ['pytorch', 'spark', 'sparktorch', 'machine learning', 'deep learning'],
      url='https://github.com/dmmiller612/sparktorch',
      download_url='https://github.com/dmmiller612/sparktorch/archive/0.1.2.tar.gz',
      author='Derek Miller',
      author_email='dmmiller612@gmail.com',
      long_description=open("README.md", "r", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      install_requires=['torch', 'flask', 'requests', 'dill'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
