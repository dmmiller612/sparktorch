from setuptools import setup
from setuptools import find_packages

setup(name='sparktorch',
      version='0.0.2',
      description='Deep learning on Apache Spark with Pytorch',
      keywords = ['pytorch', 'spark', 'sparktorch', 'machine learning', 'deep learning'],
      url='https://github.com/dmmiller612/sparktorch',
      download_url='https://github.com/dmmiller612/sparktorch/archive/0.0.2.tar.gz',
      author='Derek Miller',
      author_email='dmmiller612@gmail.com',
      long_description=open("README.md", "r", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      install_requires=['torch', 'flask', 'requests', 'dill'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
