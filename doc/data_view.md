
## 数据视图(DataView)

数据视图(DataView)将各种行情数据和参考数据进行了封装，方便用户使用数据。

### DataView做什么

将频繁使用的`DataFrame`操作自动化，使用者操作数据时尽量只考虑业务需求而不是技术实现：
1. 根据字段名，自动从不同的数据api获取数据
2. 按时间、标的整理对齐（财务数据按发布日期对齐）
3. 在已有数据基础上，添加字段、加入自定义数据或根据公式计算新数据
4. 数据查询
5. 本地存储

### 初始化

DataView初始化工作主要包括创建DataView和DataService、初始化配置、数据准备三步。

#### 创建DataView和DataService


DataService提供原始的数据，目前jaqs已经提供远程数据服务类（RemoteDataService），可以通过互联网获取行情数据和参考数据。

```python
from jaqs.data.dataservice import RemoteDataService
from jaqs.data.dataview import DataView
dv = DataView()
ds = RemoteDataService()
```

#### 初始化配置
通过init_from_config函数进行初始化配置，配置参数如下表所示：

| 字段 | 类型 | 说明 | 缺省值 |
| ---| ---|---|---|
| symbol | string | universe标的代码，多标的以','隔开，如'000001.SH, 600300.SH'，指数代码不会被展开成对应成员股票代码|不可缺失，symbol与universe二选一 |
| universe | string |指数代码，单标的，将该指数的成员作为universe|不可缺省，symbol与universe二选一|
|start\_date|int|开始日期|不可缺省|
|end\_date|int|结束日期|不可缺省|
|fields|string|数据字段，多字段以','隔开，如'open,close,high,low'|不可缺省|
|freq|int|数据类型，目前只支持1，表示日线数据|1|

示例代码：

```python
dv = DataView()
ds = RemoteDataService()

secs = '600030.SH,000063.SZ,000001.SZ'
props = {'start_date': 20160601, 'end_date': 20170601, 'symbol': secs,
       'fields': 'open,close,high,low,volume,pb,net_assets,eps_basic',
       'freq': 1}
dv.init_from_config(props, data_api=ds)
```

#### 数据准备

从数据服务获取数据：
```python
dv.prepare_data()
```

### 获取数据

#### 数据结构说明

DataView用一个三维的数据结构保存的所需数据，其三维数据轴分别为：

1. 标的代码，如 600030.SH, 000002.SH
2. 交易日期，如 20150202, 20150203
3. 数据字段，如 open, high, low, close

如下图所示：
![dataview](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/dataview.png)

#### 根据日期获取数据:

使用get_snapshot()函数来获取某日的数据快照（在时间轴切片），输入参数见下表：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|snapshot_date|int|交易日|不可缺省|
|symbol|string|标的代码，多标的以','隔开，如'000001.SH, 600300.SH'|不可缺省|
|fields|string|数据字段，多字段以','隔开，如'open,close,high,low' |不可缺省|

返回在结果格式为pandas DataFrame，标的代码作为DataFrame的index，数据字段作为DataFrame的column。

示例代码：
```python
snap1 = dv.get_snapshot(20170504, symbol='600030.SH,000063.SZ', fields='close,pb')
```

返回结果示例：

#### 根据数据字段获取数据

使用get_ts()函数获取某个数据字段的时间序列（在字段轴切片），输入参数见下表：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|field|string|数据字段，多字段以','隔开|不可缺省|
|symbol|string|"标的代码，多标的以','隔开，如""000001.SH, 600300.SH"""|不可缺省|
|start\_date|int|开始日期|不可缺省|
|end\_date|int|结束日期|不可缺省|

返回结果格式为pandas DataFrame，交易日作为DataFrame的index，标的代码作为DataFrame的column

示例代码：
```python
ts1 = dv.get_ts('close', symbol='600030.SH,000063.SZ', 
            start_date=20170101, end_date=20170302)
```

### 数据视图及保存

- 可以读取修改后继续存储
- 默认覆盖

#### 保存DataView到文件

使用save_dataview()函数将当前数据视图保存到指定文件夹，保存格式为h5文件。函数输入参数如下：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|folder_path|string|文件保存主目录|不可缺省|
|sub_folder|string|文件保存子目录，缺省为'{start_date}_{end_date}_freq={freq}D',例如,若DataView初始参数为start_date=20120101,end_date=20120110,freq=1时，sub_folder为'20120101_20120110_freq=1D'|'{start_date}_{end_date}_freq={freq}D'|

示例代码：
```python
dv.save_dataview('prepared', 'demo')
```


    Store data...
    Dataview has been successfully saved to:
    /home/user/prepared/demo

    You can load it with load_dataview('/home/user/prepared/demo')


#### 读取已经保存的DataView
利用load_dataview()函数，DataView可以不经初始化，直接读取已经保存的DataView数据。函数输入参数如下所示：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|folder|string|DataView文件保存目录|不可缺省|


示例代码：
```python
dv = DataView()
dv.load_dataview('/home/user/prepared/demo')
```

    Dataview loaded successfully.


### 添加数据

- 从DataApi获取更多字段: `dv.add_field('roe')`
- 加入自定义DataFrame: `dv.append_df(name, df)`
- 根据公式计算衍生指标: `dv.add_formula(name, formula, is_quarterly=False)`

#### 添加字段
利用add_field()函数可以添加当前DataView没有包含的数据，输入参数如下：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|field_name|string|需要添加的字段名称|不可缺省|
|data_api | BaseDataServer |缺省时为None，即利用DataView初始化时传入的DataService添加数据；当DataView是从文件中读取得到时，该DataView没有DataService，需要外部传入一个DataService以添加数据。|None|

示例代码：

.

#### 添加自定义公式数据

利用add_formula()函数可以添加当前DataView添加自定义公式数据字段，输入参数如下所示：

|字段|类型|说明|缺省
| ---| ---|---|---|
|field\_name|string|字段名称|不可缺省|
|formula|string|公式表达式|不可缺省|
|is\_quarterly|bool|是否为季度数据，如财务季报数据|不可缺省|
|formula\_func\_name\_style|string|函数名大小写识别模式，'upper'：使用默认函数名，'lower'：formular里所有函数名都为应为小写。|'upper'|
|data\_api|BaseDataServer|数据服务|None|

示例代码：


```python
## 日频0/1指标：是否接近涨跌停
dv.add_formula('limit_reached', 'Abs((open - Delay(close, 1)) / Delay(close, 1)) > 0.095', is_quarterly=False)
dv.get_ts('limit_reached').iloc[:, 100:].head(2)
```



```python
## 日频指标：与52周高点的百分比
dv.add_formula('how_high_52w', 'close_adj / Ts_Max(close_adj, 252)', is_quarterly=False)
dv.get_ts('how_high_52w').tail().applymap(lambda x: round(100*x, 1))
```


```python
## 日频指标：量价背离
dv.add_formula('price_volume_divert', 'Correlation(vwap_adj, volume, 10)', is_quarterly=False)
dv.get_snapshot(20171009, fields='price_volume_divert')
```


```python
## 季频指标：eps增长率
dv.add_formula('eps_growth', 'Return(eps_basic, 4)', is_quarterly=True)
dv.get_ts('eps_growth', start_date=20160810).head()
```


```python
ds = RemoteDataService()
dv.add_field('total_share', ds)
```

目前支持的公式参见：[这里](https://github.com/quantOS-org/quantOSUserGuide/blob/master/jaqs/dataview_formula.md)。
