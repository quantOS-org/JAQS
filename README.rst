Welcome
=======

在这里，你将可以获得：

-  使用数据API，轻松获取研究数据
-  根据策略模板，编写自己的量化策略
-  使用回测框架，对策略进行回测和验证

查看完整文档，请点击\ `连接 <http://jaqs.readthedocs.io>`__\ 

Dependencies
============

- pytest
- Jinja2
- matplotlib
- msgpack\_python
- nose\_parameterized
- seaborn
- six
- xarray
- pyzmq
- python-snappy

Installation
============

目前可以在如下操作系统上安装

-  Windows 64-bit
-  GNU/Linux 64-bit

如果还没有Python环境，建议先安装所对应操作系统的Python集成开发环境
`Anaconda <http://www.continuum.io/downloads>`__\ ，再安装jaqs。

安装方式主要有以下几种：

1、使用\ ``pip``\ 进行安装
--------------------------

    $ pip install jaqs

2、通过源代码安装
-----------------

git clone https://github.com/quantOS-org/jaqs.git
，进入到源文件目录，执行安装命令：

$ python setup.py install
或者通过pypi地址\ https://pypi.python.org/jaqs
下载,并执行上面安装命令。

3、代码升级
-----------

$ pip install jaqs --upgrade

Quickstart
==========

参见 `user guide <doc/source/user_guide.rst>`__

更多的示例保存在 ``jaqs/examples``


查看完整文档，请访问： `http://jaqs.readthedocs.io <http://jaqs.readthedocs.io>`__\ 

Questions?
==========

如果您发现任何问题，请到 \ `这里 <https://github.com/quantOSorg/jaqs/issues/new>`__\ 提交。

Change Logs
===========

0.2.0 2017/10/16
----------------

1、发布第一个版本

License
=======

Apache 2.0许可协议。版权所有(c)2017 quantOS-org(量化开源会)。
