# 数据视图

数据视图(DataView)将各种行情数据和参考数据进行了封装，方便用户使用数据。

## 初始化

DataView初始化工作主要包括创建DataView和DataService、初始化配置、数据准备三步。

### 创建DataView和DataService


DataService提供原始的数据，目前jaqs已经提供远程数据服务类（RemoteDataService），可以通过互联网获取行情数据和参考数据。

```python
from jaqs.data.dataservice import RemoteDataService
from jaqs.data.dataview import DataView
dv = DataView()
ds = RemoteDataService()
```

### 初始化配置 
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
secs = '600030.SH,000063.SZ,000001.SZ'
props = {'start_date': 20160601, 'end_date': 20170601, 'symbol': secs,
       'fields': 'open,close,high,low,volume,pb,net_assets,ncf',
       'freq': 1}
dv.init_from_config(props, data_api=ds)
```

### 数据准备

从数据服务获取数据：
```python
dv.prepare_data()
```

## 获取数据

### 数据结构说明

DataView用一个三维的数据结构保存的所需数据，其三维数据轴分别为：

1. 标的代码，如 600030.SH, 000002.SH
2. 交易日期，如 20150202, 20150203
3. 数据字段，如 open, high, low, close

### 根据日期获取数据: 

使用get_snapshot()函数来获取某日的数据快照，输入参数见下表：

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

###根据数据字段获取数据

使用get_ts()函数获取某个数据字段的时间序列，输入参数见下表：

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

## 数据视图及保存
### 保存DataView到文件

使用save_dataview()函数将当前数据视图保存到指定文件夹，保存格式为h5文件。函数输入参数如下：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|folder_path|string|文件保存主目录|不可缺省|
|sub_folder|string|文件保存子目录，缺省为'{start_date}_{end_date}_freq={freq}D',例如,若DataView初始参数为start_date=20120101,end_date=20120110,freq=1时，sub_folder为'20120101_20120110_freq=1D'|'{start_date}_{end_date}_freq={freq}D'|

示例代码：
```python
folder_path = '../output/prepared'
dv.save_dataview(folder_path=folder_path)
```

### 读取已经保存的DataView
利用load_dataview()函数，DataView可以不经初始化，直接读取已经保存的DataView数据。函数输入参数如下所示：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|folder|string|DataView文件保存目录|不可缺省|


示例代码：
```python
dv = DataView()
folder_path = '../output/prepared/20160601_20170601_freq=1D'
dv.load_dataview(folder=folder_path)
```

## 添加数据
### 添加字段
利用add_field()函数可以添加当前DataView没有包含的数据，输入参数如下：

|字段|类型|说明|缺省值|
| ---| ---|---|---|
|field_name|string|需要添加的字段名称|不可缺省|
|data_api | BaseDataServer |缺省时为None，即利用DataView初始化时传入的DataService添加数据；当DataView是从文件中读取得到时，该DataView没有DataService，需要外部传入一个DataService以添加数据。|None|

示例代码：
```python
ds = RemoteDataService()
dv.add_field('total_share', ds)
```

### 添加自定义公式数据
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
dv.add_formula("myfactor", 'close / open', is_quarterly=False)
```
目前支持的公式如下表所示：

|公式|说明|示例|
| ---| ---|---|
|+|加法运算|close + open|
|-|减法运算|close - open|
|*|乘法运算|vwap * volume|
|/|除法运算|close / open|
|^|幂函数|close ^ 2|
|%|取余函数|oi % 10 |
|==|判断是否相等|close == open|
|!=|判断是否不等|close != open|
|>|大于|close > open|
|<|小于|close < open|
|>=|大于等于|close >= open|
|<=|小于等于|close <= open|
|&&|逻辑与|(close > open) && (close > vwap)|
|&#124;&#124;|逻辑或| (close > open) &#124;&#124;(close > vwap)|
|Sin(x)|正弦函数|Sin(close/open) |
|Cos(x)|余弦函数|Cos(close/open) |
|Tan(x)|正切函数|Tan(close/open) |
|Sqrt(x)|开平方函数|Sqrt(close^2 + open^2) |
|Abs(x)|绝对值函数|Abs(close-open)|
|Log(x)|自然对数|Log(close/open) |
|Ceil(x)|向上取整|Ceil(high) |
|Floor(x)|向下取整|Floor(low)|
|Round(x)|四舍五入|Round（close）|
|-x|对x取负|-close|
|!|逻辑非|!(close>open)|
|Sign(x)|取 x 正负号，返回以-1，0和1标志|Sign(close-open)|
|Max(x,y)|取 x 和 y 同位置上的较大值组成新的DataFrame返回|Max(close, open)|
|Min(x,y)|取 x 和 y 同位置上的较小值组成新的DataFrame返回|Min(close,open)|
|Delay(x,n)|时间序列函数， n 天前 x 的值|Delay(close,1) 表示前一天收盘价|
|Rank(x)|各标的根据给出的指标x的值，在横截面方向排名|Rank( close/Delay(close,1)-1 ) 表示按日收益率进行排名|
|GroupRank(x,g)|各标的根据指标 x 的值，在横截面方向进行按分组 g 进行分组排名。分组 DataFrame g 以int数据标志分组，例如三个标的在某一天的截面上的分组值都为2，则表示这三个标的在同一组|GroupRank(close/Delay(close,1)-1, g) 表示按分组g根据日收益率进行分组排名|
|ConditionRank(x,cond)|各标的根据条件 DataFrame cond,按照给出的指标 x 的值，在横截面方向排名，只有 cond 中值为True的标的参与排名。|GroupRank(close/Delay(close,1)-1, cond) 表示按条件cond根据日收益率进行分组排名|
|Quantile(x,n)|各标的按根据指标 x 的值，在横截面方向上进行分档，每档标的数量相同|Quantile( close/Delay(close,1)-1,5)表示按日收益率分为5档|
|GroupQuantile(x,g,n)|各标的根据指标 x 的值，在横截面方向上按分组 g 进行分组分档，分组 DataFrame g 以int数据标志分组，例如三个标的在某一天的截面上的分组值都为2，则表示这三个标的在同一组|GroupQuantile(close/Delay(close,1)-1,g,5) 表示按日收益率和分组g进行分档，每组分为5档|
|Standardize(x)|标准化，x值在横截面上减去平均值后再除以标准差|Standardize(close/Delay(close,1)-1) 表示日收益率的标准化|
|Cutoff(x,z_score)|x值在横截面上去极值，用MAD方法|Cutoff(close,3) 表示去掉z_score大于3的极值|
|Sum(x,n)|时间序列函数，x 指标在过去n天的和，类似于pandas的rolling_sum()函数|Sum(volume,5) 表示一周成交量|
|Product(x,n)|时间序列函数，计算 x 中的值在过去 n 天的积|Product(close/Delay(close,1),5) - 1 表示过去5天累计收益|
|CountNans(x,n)|时间序列函数，计算 x 中的值在过去 n 天中为 nan （非数字）的次数|CountNans((close-open)^0.5, 10) 表示过去10天内有几天close小于open|
|Ewma(x,halflife)|指数移动平均，以halflife的衰减对x进行指数移动平均|Ewma(x,3)|
|StdDev(x,n)|时间序列函数，计算 x 中的值在过去n天的标准差|StdDev(close/Delay(close,1)-1, 10)|
|Covariance(x,y,n)|时间序列函数，计算 x 中的值在过去n天的协方差|Covariance(close, open, 10)|
|Correlation(x,y,n)|时间序列函数，计算 x 中的值在过去n天的相关系数|Correlation(close,open, 10)|
|Delta(x,n)|时间序列函数，计算 x 当前值与n天前的值的差|Delta(close,5) |
|Return(x,n,log)|时间序列函数，计算x值n天的增长率，当log为False时，计算线性增长;当log为True时，计算对数增长|Return(close,5,True)计算一周对数收益|
|Ts_Mean(x，n)|时间序列函数，计算 x 中的值在过去n天的平均值|Ts_Mean(close,5)|
|Ts_Min(x，n)|时间序列函数，计算 x 中的值在过去n天的最小值|Ts_Mean(close，5)|
|Ts_Max(x，n)|时间序列函数，计算 x 中的值在过去n天的最大值|Ts_Min(close，5)|
|Ts_Skewness(x，n)|时间序列函数，计算 x 中的值在过去n天的偏度|Ts_Max(close，5)|
|Ts_Kurtosis(x，n)|时间序列函数，计算 x 中的值在过去n天的峰度|Ts_Skewness(close，20)|
|Tail(x，y， n)|如果 x 的值介于 lower 和 upper，则将其设定为 newval|Ts_Kurtosis(close，20)|
|Step(n)|Step(n) 为每个标的创建一个向量，向量中 n 代表最新日期，n-1 代表前一天，以此类推。|Step(30)|
|Decay_linear(x,n)|时间序列函数，过去n天的线性衰减函数。Decay_linear(x, n) = (x[date] * n + x[date - 1] * (n - 1) + … + x[date – n -| 1]) / (n + (n - 1) + … + 1)|Decay_linear(close,15)|
|Decay_exp(x,f,n)|时间序列函数, 过去 n 天的指数衰减函数，其中 f 是平滑因子。这里 f 是平滑因子，可以赋一个小于 1 的值。Decay_exp(x, |f, n) = (x[date] + x[date - 1] * f + … +x[date – n - 1] * (f ^ (n – 1))) / (1 + f + … + f ^ (n - 1))|Decay_exp(close,0.9,10)|
|Pow(x,y)|幂函数x^y|Pow(close,2)|
|SignedPower(x,e)|等价于Sign(x) * (Abs(x)^e)|SignedPower(close-open, 0.5)|
|If(cond,x,y)|cond为True取x的值，反之取y的值|If(close > open, close, open) 表示取open和close的较大值|



