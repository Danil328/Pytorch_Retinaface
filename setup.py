import codecs
import os
import re

from setuptools import find_packages, setup


def get_absolute_path(*args):
    """Transform relative pathnames into absolute pathnames."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *args)


def get_contents(*args):
    """Get the contents of a file relative to the source distribution directory."""
    with codecs.open(get_absolute_path(*args), 'r', 'UTF-8') as handle:
        return handle.read()


def get_version(*args):
    """Extract the version number from a Python module."""
    contents = get_contents(*args)
    metadata = dict(re.findall('__([a-z]+)__ = [\'"]([^\'"]+)', contents))
    return metadata['version']


setup(
    name='retinaface',
    version=get_version('retinaface', '__init__.py'),
    author='Akhmetov Danil',
    author_email='danil28644@gmail.com',
    description='RetinaFace PyTorch',
    long_description_content_type='text/markdown',
    url='https://github.com/Danil328/Pytorch_Retinaface.git',
    packages=find_packages(),
    install_requires=[
        'torchvision', 'torch>=1.1', "gdown"
    ],
    setup_requires=['pytest-runner'],
    python_requires='>=3.6.0'
)