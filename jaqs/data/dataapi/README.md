# DataApi

标准数据API定义。

# 安装步骤

## 1、安装Python环境

如果本地还没有安装Python环境，强烈建议安装Anaconda（Python的集成开发环境，包含众多常用包，且易于安装，避免不必要的麻烦）。打开[Anaconda官网](http://www.continuum.io/downloads)，选择相应的操作系统，确定要安装的Python版本，进行下载。

下载完成以后，按照图形界面步骤完成安装。在默认情况下，Anaconda会自动设置PATH环境。

***注***：如果安装过程遇到问题，或需要更详细的步骤，请参见[安装Anaconda Python环境教程](https://github.com/quantOS-org/JAQS/blob/master/doc/install.md#1安装python环境)

## 2、安装依赖包

如果Python环境不是类似Anaconda的集成开发环境，我们需要单独安装依赖包，在已经有pandas/numpy包前提下，还需要有以下几个包：
- `pyzmq`
- `msgpack_python`
- `python-snappy`

可以通过单个安装完成，例如： `pip install pyzmq`

需要注意的是，`python-snappy`的安装需要比较多的编译依赖，请按照[如何安装python-snappy包](https://github.com/quantOS-org/JAQS/blob/master/doc/install.md#如何安装python-snappy包)所述安装。


## 3、使用DataApi

```python
from DataApi import DataApi  # 这里假设项目目录名为DataApi, 且存放在工作目录下

api = DataApi(addr="tcp://data.tushare.org:8910")
result, msg = api.login("phone", "token") # 示例账户，用户需要改为自己在www.quantos.org上注册的账户
print(result)
print(msg)
```

