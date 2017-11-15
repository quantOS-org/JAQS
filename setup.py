from setuptools import setup, find_packages
import codecs
import os


def read(fname):
    return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='jaqs',
    version='0.4.0',
    description='Open source quantitative research&trading framework.',
    long_description = readme(),
    install_requires=[
						'pytest',
						'Jinja2',
						'matplotlib',
						'msgpack_python',
						'nose_parameterized',
						'seaborn',
						'six',
						'xarray',
						'pyzmq',
						'python-snappy',
    ],
    license='Apache 2',
    classifiers=[
    'Programming Language :: Python :: 2.7',
    ],
    packages=find_packages(),
    package_data={'': ['*.json', '*.css', '*.html']},
    )