
# JAQS安装步骤



## 1、安装Python环境
运行JAQS需要Python环境，可通过在控制台（Windows系统为命令提示符，Linux系统为终端）运行`python`命令确定系统是否安装。

如果本地还没有安装Python环境，或已安装的Python不是[Anaconda](http://www.continuum.io/downloads "Anaconda")，强烈建议按下方步骤安装，Anaconda是集成开发环境，其中包含稳定版Python和众多常用包，且易于安装，避免不必要的麻烦；如果已经装好了Anaconda，可直接看下一步骤**安装依赖包**。

***如何安装Anaconda***：
1. 打开[Anaconda官网](http://www.continuum.io/downloads)，选择相应的操作系统，确定要按照的Python版本，一般建议用Python 2.7。
![anac](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/anac.png)
2. 下载完成以后，按照图形界面步骤完成安装。在默认情况下，Anaconda会自动设置PATH环境。
3. 安装完成后，  
    windows下我们可以在系统菜单中看如下程序目录：
![anacm](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/anac_m.png)
    在cmd里执行`ipython`命令，可以调出IPython调试器。
![anacipython](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/anac_ipython.png)


## 2、安装依赖包

除Anaconda中已包含的常用包外，JAQS还有些额外的依赖，这些依赖可在使用`pip`安装JAQS的过程中**自动安装**，唯一需要注意的是，`python-snappy`这个包在Windows（及部分Linux）系统上的安装需要比较多的编译依赖，建议从[这个网页](http://www.lfd.uci.edu/~gohlke/pythonlibs)下载编译好的包，然后安装:
```shell
pip install python_snappy-0.5.1-cp27-cp27m-win_amd64.whl # 具体文件名可能不同, 取决于系统版本
```

装好`python-snappy`后，即可使用`pip`直接安装JAQS和其他依赖包，见下一节**安装JAQS**。

如果希望手动安装依赖：
- 可以在jaqs程序目录下，执行 `pip install -r requirements.txt` 一次完成所有依赖的安装。
- 也可以通过单个安装完成，例如： `pip install pyzmq`


## 3、安装JAQS
可以使用`pip`安装或下载源代码安装，我们推荐使用`pip`.

### 使用`pip`进行安装
```sheel
pip install jaqs
```

### 通过源代码安装
首先克隆git仓库：
```shell
git clone https://github.com/quantOS-org/jaqs.git
```
然后进入到源文件目录，执行安装命令：
```shell
python setup.py install
```

也可以通过[PyPI地址](https://pypi.python.org/pypi/jaqs)下载,并执行上面安装命令。

### 4、确认安装成功
完成安装以后，在命令行中运行`python`并执行`import jaqs`确认安装是否成功：
![jaqstest](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/jaqs_test.png)

## 5、升级JAQS
如果有新的release，可通过如下命令升级：
```shell
pip uninstall jaqs
pip install jaqs
```

## 6、策略样例
策略参考样例，请访问[https://github.com/quantOS-org/JAQS/tree/release-0.5.0/example](https://github.com/quantOS-org/JAQS/tree/release-0.5.0/example)
