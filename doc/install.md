# JAQS安装步骤

## Windows下一键安装脚本
如果没有安装Python包的经验，或者极度不擅长相关操作，可以使用我们的一键安装脚本。该脚本仅适用于Windows系统，使用步骤为：
1. 先按照下方**如何安装Anaconda**操作，在系统中安装Anaconda；
2. 下载[一键安装脚本](http://www.quantos.org/downloads/install_scripts/onekey_install_jaqs.zip)，解压后运行`onekey_install_jaqs.bat`，看到"Successfully installed jaqs"字样，说明安装成功。

## 1、安装Python环境
运行JAQS需要Python环境，可通过在控制台（Windows系统为命令提示符，Linux系统为终端）运行`python`命令确定系统是否安装。

如果本地还没有安装Python环境，或已安装的Python不是[Anaconda](http://www.continuum.io/downloads "Anaconda")，强烈建议按下方步骤安装，Anaconda是集成开发环境，其中包含稳定版Python和众多常用包，且易于安装，避免不必要的麻烦；如果已经装好了Anaconda，可直接看下一步骤**安装依赖包**。

### 如何安装Anaconda

1. **下载安装包**：打开[Anaconda官网](http://www.continuum.io/downloads)，选择相应的操作系统，确定要安装的Python版本，目前JAQS同时支持Python2/3.
  ![anac](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/anac.png)

2. **安装**：下载完成后，运行下载好的文件，按提示完成安装（**注**：对于Windows系统，在默认情况下，Anaconda不会自动设置`PATH`环境，请确保选择“add Anaconda to system PATH”选项。）。具体安装教程参见[官方文档](https://conda.io/docs/user-guide/install/index.html#regular-installation)。

   

3. **检查安装是否成功**：安装完成后，

   - **windows**下，可以在系统菜单中看如下程序目录：![anacm](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/anac_m.png). 在cmd里执行`ipython`命令，可以调出IPython调试器：![anacipython](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/anac_ipython.png).
   - **Linux**下，可以在Terminal运行`Python`，可看到'Anaconda'字样。

## 2、安装依赖包

除Anaconda中已包含的常用包外，JAQS还有些额外的依赖，除`python-snappy`这个包外，其他依赖可在使用`pip install jaqs`安装JAQS的过程中**自动安装**。

请首先按照**如何安装`python-snappy`包**所述操作，然后直接安装JAQS。

如果希望手动安装依赖包（不推荐），请按照**如何手动安装依赖包**所述操作。

### 如何安装`python-snappy`包

这个包的安装需要比较多的编译依赖，直接使用`pip install python-snappy`安装可能会报错，请根据自己的操作系统，按下方步骤安装。

#### Windows

在Windows上准备编译环境较为复杂，建议从[这个网页](https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-snappys)下载编译好的包并直接安装。

下载时，需选择适合自己系统版本和Python版本的包，其中

- cp27代表Python2.7，cp36代表Python3.6，以此类推
- 64位系统选择带"amd64"字样的安装文件，32位系统选择不带的

安装时，使用`pip install 文件名`进行安装，如：

```shell
pip install python_snappy-0.5.1-cp27-cp27m-win_amd64.whl # 具体文件名可能不同, 取决于系统版本
```

#### Linux

首先，安装snappy的开发包`libsnappy-dev`：

```shell
sudo apt-get install libsnappy-dev
```

*注*：`libsnappy-dev`适用于Ubuntu系统，CentOS/RedHat/Fedora/SUSE等系统请使用`libsnappy-devel`，其他Linux系统请自行查找。

之后即可通过`pip`安装`python-snappy`：

```shell
pip install python-snappy
```
#### OS-X
参见[官方说明](https://github.com/andrix/python-snappy#frequently-asked-questions)


装好`python-snappy`后，即可使用`pip`直接安装JAQS和其他依赖包，见下一节**安装JAQS**。

### 如何手动安装依赖包

- 可以在jaqs程序目录下，执行 `pip install -r requirements.txt` 一次完成所有依赖的安装。
- 也可以通过单个安装完成，例如： `pip install pyzmq`


## 3、安装JAQS
可以使用`pip`安装或下载源代码安装。由于项目处于不断开发中，使用`pip`安装的一般为稳定版本；而使用源码安装，则可以自由选择版本（如最新开发版）。一般情况下，我们推荐使用`pip`. 

### 使用`pip`进行安装
```sheel
pip install jaqs
```

### 通过源代码安装
#### 1. 下载源代码
打开[项目的GitHub首页](https://github.com/quantOS-org/JAQS)，选择一个分支（稳定版为`release-x.x.x`，最新开发版为`master`），再点击“Clone or Download”-“Download ZIP”即可下载。或者，也可以通过Git命令克隆：
```shell
git clone https://github.com/quantOS-org/jaqs.git
```

#### 2. 从源代码安装
进入项目目录（如果是直接下载，项目目录名为`JAQS-分支名`，如果是git克隆，项目目录名为`JAQS`），打开命令提示符/终端，执行安装命令：
```shell
python setup.py install
```

***注***：如果已经通过`pip`安装了稳定版JAQS，希望通过源代码安装最新版本，则先通过`pip uninstall jaqs`卸载，在下载源代码安装即可。

### 4、确认安装成功
完成安装以后，在命令行中运行`python`并执行`import jaqs`确认安装是否成功：
![jaqstest](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/jaqs_test.png)

## 5、升级JAQS
如果有新的release，可通过如下命令升级：
```shell
pip uninstall jaqs
pip install jaqs
```

*注*：不使用`pip install --upgrade jaqs`是为了保护使用者现有环境，使用该命令会同时升级`pandas`等依赖包，导致改变用户当前包环境。

## 6、策略样例

策略参考样例，请访问[https://github.com/quantOS-org/JAQS/tree/master/example](https://github.com/quantOS-org/JAQS/tree/master/example)
