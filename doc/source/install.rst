安装步骤
========

1、安装Python环境
-----------------

如果本地还没有安装Python环境，强烈建议安装
`Anaconda <http://www.continuum.io/downloads>`__\ 。

打开上面的网址，选择相应的操作系统，确定要按照的Python版本，一般建议用Python
2.7。

|image0|

下载完成以后，按照图形界面步骤完成安装。在默认情况下，Anaconda会自动设置PATH环境。

安装完成后，windows下我们可以在系统菜单中看如下程序目录：

|image1|

在cmd里执行ipyhont命令，可以调出IPython调试器。

|image2|

2、安装依赖包
-------------

如果Python环境不是类似Anaconda的集成开发环境，我们需要单独安装依赖包，在已经有pandas/numpy包前提下，还需要有以下几个包：

::

    pytest
    Jinja2
    matplotlib
    msgpack_python
    nose_parameterized
    seaborn
    six
    xarray
    pyzmq
    python-snappy

可以通过在jaqs程序目录下，执行 pip install -r requirements.txt
一次完成所有依赖的安装。

也可以通过单个安装完成，例如： pip install pytest

需要注意的是，python-snappy和msgpack-python这两个包在Windows上的安装需要比较多的编译依赖,建议从\ ` <http://www.lfd.uci.edu/~gohlke/pythonlibs>`__\ `http://www.lfd.uci.edu/~gohlke/pythonlibs <http://www.lfd.uci.edu/~gohlke/pythonlibs>`__
下载编译好的包，然后安装:

::

    pip install msgpack_python-0.4.8-cp27-cp27m-win_amd64.whl 

    pip install python_snappy-0.5.1-cp27-cp27m-win_amd64.whl

3、安装jaqs
-----------

安装方式主要有以下几种：

使用\ ``pip``\ 进行安装
-----------------------

::

    $ pip install jaqs

通过源代码安装
--------------

git clone
` <https://github.com/quantOS-org/jaqs.git>`__\ `https://github.com/quantOS-org/jaqs.git <https://github.com/quantOS-org/jaqs.git>`__
，进入到源文件目录，执行安装命令：

::

    $ python setup.py install

或者通过pypi地址\ ` <https://pypi.python.org/pypi/jaqs>`__\ `https://pypi.python.org/pypi/jaqs <https://pypi.python.org/pypi/jaqs>`__
下载,并执行上面安装命令。

代码升级
--------

::

    $ pip install jaqs --upgrade

完成安装以后，执行import确认安装是否成功。

|image3|

.. |image0| image:: https://github.com/quantOS-org/jaqs/blob/master/doc/img/anac.png
.. |image1| image:: https://github.com/quantOS-org/jaqs/blob/master/doc/img/anac_m.png
.. |image2| image:: https://github.com/quantOS-org/jaqs/blob/master/doc/img/anac_ipython.png
.. |image3| image:: https://github.com/quantOS-org/jaqs/blob/master/doc/img/jaqs_test.png
