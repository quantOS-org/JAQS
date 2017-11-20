from setuptools import setup, find_packages
from jaqs import __version__ as ver
import codecs
import os


def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()


def readme():
    with open('README.rst') as f:
        return f.read()


def read_install_requires():
    with open('requirements.txt', 'r') as f:
        res = f.readlines()
    res = list(map(lambda s: s.replace('\n', ''), res))
    return res


setup(
    name='jaqs',
    version=ver,
    description='Open source quantitative research&trading framework.',
    long_description = readme(),
    install_requires=read_install_requires(),
    license='Apache 2',
    classifiers=[
    'Programming Language :: Python :: 2.7',
    ],
    packages=find_packages(),
    package_data={'': ['*.json', '*.css', '*.html']},
    )