# 参考数据

## 证券基础信息表

### 调用示例

```python
df, msg = api.query(
                view="lb.instrumentInfo", 
                fields="status,list_date, fullname_en, market", 
                filter="inst_type=&status=1&symbol=", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| symbol | string | 证券代码 |
| inst\_type | string | 证券类别 |
| start_delistdate | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| end_delistdate | int或者string | 结束日期，int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| inst\_type | string | 证券类别 |
| market | string | 交易所代码 |
| symbol | string | 证券代码 |
| name | string | 证券名称 |
| list\_date | string | 上市日期 |
| delist\_date | string | 退市日期 |
| cnspell | string | 拼音简写 |
| currency | string | 交易货币 |
| status | string | 上市状态 |
| buylot | INT | 最小买入单位 |
| selllot | INT | 最大买入单位 |
| pricetick | double | 最小变动单位 |
| product | string | 合约品种 |
| underlying | string | 对应标的 |
| multiplier | int | 合约乘数|

## 交易日历表

### 调用示例

```python
df, msg = api.query(
                view="jz.secTradeCal", 
                fields="date,istradeday,isweekday,isholiday", 
                filter="start_date=20170101&end_date=20170801", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| symbol | string | 证券代码 |
| start_date | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| end_date | int或者string | 结束日期，int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| trade\_date | string | 日历日期 |
| istradeday | string | 是否交易日 |
| isweekday | string | 是否工作日 |
| isweekend | string | 是否周末 |
| isholiday | string | 是否节假日 |



## 分配除权信息表

### 调用示例

```python
df, msg = api.query(
                view="lb.secDividend", 
                fields="", 
                filter="start_date=20170101&end_date=20170801", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| symbol | string | 证券代码 |
| start_date | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| end_date | int或者string | 结束日期，int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| ann\_date | string | 公告日期 |
| end\_date | string | 分红年度截至日 |
| process\_stauts | string | 事件进程 |
| publish\_date | string | 分红实施公告日 |
| record\_date | string | 股权登记日 |
| exdiv\_date | string | 除权除息日 |
| cash | double | 每股分红(税前) |
| cash\_tax | double | 每股分红(税后） |
| share\_ratio | double | 送股比例 |
| share\_trans\_ratio | double | 转赠比例 |
| cashpay\_date | string | 派现日 |
| bonus\_list\_date | string | 送股上市日 |



## 复权因子表

### 调用示例

```python
df, msg = api.query(
                view="lb.secAdjFactor", 
                fields="", 
                filter="symbol=002059.SZ&start_date=20170101&end_date=20170801", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| symbol | string | 证券代码 |
| start_date | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| end_date | int或者string | 结束日期，int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| trade\_date | string | 除权除息日 |
| adjust\_factor | double | 复权因子 |

## 停复牌信息表

### 调用示例

```python
df, msg = api.query(
                view="lb.secSusp", 
                fields="susp_time", 
                filter="symbol=002059", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| symbol | string | 证券代码 |
| start_date | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| end_date | int或者string | 结束日期，int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| ann\_date | string | 停牌公告日期 |
| susp\_date | string | 停牌开始日期 |
| susp\_time | string | 停牌开始时间 |
| resu\_date | string | 复牌日期 |
| resu\_time | string | 复牌时间 |
| susp\_reason | string | 停牌原因 |

## 行业分类表

### 调用示例

```python
df, msg = api.query(
                view="lb.secIndustry", 
                fields="", 
                filter="industry1_name=金融&industry2_name=金融&industry_src=中证", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| symbol | string | 证券代码 |
| industry1_name | string | 一级行业名称 例如：钢铁|
| industry2_name | string | 二级行业名称 |
| industry3_name | string | 三级行业名称 |
| industry4_name | string | 四级行业名称 |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |
! data_format | string | 格式 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| industry\_src | string | 行业分类来源 |
| in\_date | string | 纳入日期 |
| out\_date | string | 剔除日期 |
| is\_new | string | 是否最新 |
| industry1\_code | string | 一级行业代码 |
| industry1\_name | string | 一级行业名称 |
| industry2\_code | string | 二级行业代码 |
| industry2\_name | string | 二级行业名称 |
| industry3\_code | string | 三级行业代码 |
| industry3\_name | string | 三级行业名称 |
| industry4\_code | string | 四级行业代码 |
| industry4\_name | string | 四级行业名称 |


<!-- ## 每日涨跌停表

### 调用示例

```python
df, msg = api.query(
                view="lb.secLimit", 

                )
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| trade_date | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| trade\_date | string | 交易日期 |
| symbol | string | 证券代码 |
| name | string | 证券名称 |
| limit\_up | double | 涨停价格 |
| limit\_down | double | 跌停价格 | -->

<!-- ## 期货主力/连续合约表

### 调用示例

```python
df, msg = api.query(
                view="lb.futContract", 

                )
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| contract | string | 合约代码 |
| contract_type | int | 合约类型 |
| trade_date | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| contract | string | 合约信息 |
| market | string | 交易所代码 |
| trade\_date | string | 交易日期 |
| product | string | 标的合约代码 |
| contract\_type | string | 合约类型 | -->

## 常量参数表

### 调用示例

```python
df, msg = api.query(
                view="jz.sysConstants", 
                fields="", 
                filter="code_type=inst_type", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| code_type | string | 参数类型 |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| code\_type | string | 参数类型 |
| type\_name | string | 参数名称 |
| code | string | 参数代码 |
| value | string | 参数值 |

## 日行情估值表

### 调用示例

```python
df, msg = api.query(
                view="wd.secDailyIndicator",
                fields='pb,net_assets,ncf,price_level',
                filter='symbol=000063.SZ&start_date=20170605&end_date=20170701')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| view | string | 数据接口名 |
| symbol | string | 证券代码 |
| start_date | int或者string | 开始日期, int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| end_date | int或者string | 结束日期，int时为YYYYMMDD格式(如20170809)；string时为'YYYY-MM-DD'格式，如'2017-08-09' |
| fields | string | 需要返回字段，多字段以','隔开；为""时返回所有字段 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| trade\_date | string | 参数名称 |
| currency | string | 货币代码 |
| total\_market\_value | string | 当日总市值 |
| float\_market\_value | double | 当日流通市值 |
| high\_52w | double | 52周最高价 |
| low\_52w | double | 52周最低价 |
| pe | double | PE |
| pb | double | PB |
| pe\_ttm | double | PE TTM |
| pcf | double | PCF经营现金流 |
| pcf\_ttm | double | PCF经营现金流 TTM |
| ncf | double | 现金净流量 |
| ncf\_ttm | double | 现金净流量TTM |
| ps | double | 市销率 |
| ps\_ttm | double | 市销率 TTM |
| turnover\_ratio | double | 换手率 |
| turnover\_ratio\_float | double | 换手率(基准.自由流通股本) |
| share\_amount | double | 当日总股本 |
| share\_float | double | 当日流通股本 |
| close\_price | double | 当日收盘价 |
| price\_div\_dps | double | 股价/每股派息 (万股 ) |
| high\_52w\_adj | double | 52周最高价(复权) |
| low\_52w\_adj | double | 52周最低价（复权） |
| share\_float\_free | double | 当日自由流通股本 |
| nppc\_ttm | double | 归属母公司净利润 TTM |
| nppc\_lyr | double | 归属母公司净利润 LYR |
| net\_assets | double | 当日净资产 |
| ncfoa\_ttm | double | 经营活动产生的现金流量净额 TTM |
| ncfoa\_lyr | double | 经营活动产生的现金流量净额 LYR |
| rev\_ttm | double | 营业收入 TTM |
| rev\_lyr | double | 营业收入 LYR |
| nicce\_ttm | double | 现金及现金等价物净增加额(TTM) |
| nicce\_lyr | double | 现金及现金等价物净增加额(LYR) |
| limit\_status | string | 涨跌停状态 |
| price\_level | double | 最高最低价状态 |


## 资产负债表

### 调用示例

```python
df, msg = api.query(
                view="lb.balanceSheet", 
                fields="", 
                filter="symbol=002636.SZ",
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| start\_date | string | 公告开始日期 |
| end\_date | string | 公告结束日期 |
| comp\_type\_code | string | 公司类型代码 |
| start\_actdate | string | 实际公告开始日期 |
| end\_actdate | string | 实际公告结束日期 |
| start\_reportdate | string | 报告期开始日期 |
| start\_reportdate | string | 报告期结束日期 |
| report\_type | string | 报表类型 |
| update\_flag | int | 数据更新标记 |

### 输出参数


| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| ann\_date | string | 公告日期 |
| comp\_type\_code | string | 公司类型代码 |
| act\_ann\_date | string | 实际公告日期 |
| report\_date | string | 报告期 |
| report\_type | string | 报表类型 |
| currency | string | 货币代码 |
| monetary\_cap | double | 货币资金 |
| tradable\_assets | double | 交易性金融资产 |
| notes\_rcv | double | 应收票据 |
| acct\_rcv | double | 应收账款 |
| other\_rcv | double | 其他应收款 |
| pre\_pay | double | 预付款项 |
| dvd\_rcv | double | 应收股利 |
| int\_rcv | double | 应收利息 |
| inventories | double | 存货 |
| consumptive\_assets | double | 消耗性生物资产 |
| deferred\_exp | double | 待摊费用 |
| noncur\_assets\_due\_1y | double | 一年内到期的非流动资产 |
| settle\_rsrv | double | 结算备付金 |
| loans\_to\_banks | double | 拆出资金 |
| prem\_rcv | double | 应收保费 |
| rcv\_from\_reinsurer | double | 应收分保账款 |
| rcv\_from\_ceded\_insur\_cont\_rsrv | double | 应收分保合同准备金 |
| red\_monetary\_cap\_for\_sale | double | 买入返售金融资产 |
| other\_cur\_assets | double | 其他流动资产 |
| tot\_cur\_assets | double | 流动资产合计 |
| fin\_assets\_avail\_for\_sale | double | 可供出售金融资产 |
| held\_to\_mty\_invest | double | 持有至到期投资 |
| long\_term\_eqy\_invest | double | 长期股权投资 |
| invest\_real\_estate | double | 投资性房地产 |
| time\_deposits | double | 定期存款 |
| other\_assets | double | 其他资产 |
| long\_term\_rec | double | 长期应收款 |
| fix\_assets | double | 固定资产 |
| const\_in\_prog | double | 在建工程 |
| proj\_matl | double | 工程物资 |
| fix\_assets\_disp | double | 固定资产清理 |
| productive\_bio\_assets | double | 生产性生物资产 |
| oil\_and\_natural\_gas\_assets | double | 油气资产 |
| intang\_assets | double | 无形资产 |
| r\_and\_d\_costs | double | 开发支出 |
| goodwill | double | 商誉 |
| long\_term\_deferred\_exp | double | 长期待摊费用 |
| deferred\_tax\_assets | double | 递延所得税资产 |
| loans\_and\_adv\_granted | double | 发放贷款及垫款 |
| oth\_non\_cur\_assets | double | 其他非流动资产 |
| tot\_non\_cur\_assets | double | 非流动资产合计 |
| cash\_deposits\_central\_bank | double | 现金及存放中央银行款项 |
| asset\_dep\_oth\_banks\_fin\_inst | double | 存放同业和其它金融机构款项 |
| precious\_metals | double | 贵金属 |
| derivative\_fin\_assets | double | 衍生金融资产 |
| agency\_bus\_assets | double | 代理业务资产 |
| subr\_rec | double | 应收代位追偿款 |
| rcv\_ceded\_unearned\_prem\_rsrv | double | 应收分保未到期责任准备金 |
| rcv\_ceded\_claim\_rsrv | double | 应收分保未决赔款准备金 |
| rcv\_ceded\_life\_insur\_rsrv | double | 应收分保寿险责任准备金 |
| rcv\_ceded\_lt\_health\_insur\_rsrv | double | 应收分保长期健康险责任准备金 |
| mrgn\_paid | double | 存出保证金 |
| insured\_pledge\_loan | double | 保户质押贷款 |
| cap\_mrgn\_paid | double | 存出资本保证金 |
| independent\_acct\_assets | double | 独立账户资产 |
| clients\_cap\_deposit | double | 客户资金存款 |
| clients\_rsrv\_settle | double | 客户备付金 |
| incl\_seat\_fees\_exchange | double | 其中:交易席位费 |
| rcv\_invest | double | 应收款项类投资 |
| tot\_assets | double | 资产总计 |
| st\_borrow | double | 短期借款 |
| borrow\_central\_bank | double | 向中央银行借款 |
| deposit\_received\_ib\_deposits | double | 吸收存款及同业存放 |
| loans\_oth\_banks | double | 拆入资金 |
| tradable\_fin\_liab | double | 交易性金融负债 |
| notes\_payable | double | 应付票据 |
| acct\_payable | double | 应付账款 |
| adv\_from\_cust | double | 预收款项 |
| fund\_sales\_fin\_assets\_rp | double | 卖出回购金融资产款 |
| handling\_charges\_comm\_payable | double | 应付手续费及佣金 |
| empl\_ben\_payable | double | 应付职工薪酬 |
| taxes\_surcharges\_payable | double | 应交税费 |
| int\_payable | double | 应付利息 |
| dvd\_payable | double | 应付股利 |
| other\_payable | double | 其他应付款 |
| acc\_exp | double | 预提费用 |
| deferred\_inc | double | 递延收益 |
| st\_bonds\_payable | double | 应付短期债券 |
| payable\_to\_reinsurer | double | 应付分保账款 |
| rsrv\_insur\_cont | double | 保险合同准备金 |
| acting\_trading\_sec | double | 代理买卖证券款 |
| acting\_uw\_sec | double | 代理承销证券款 |
| non\_cur\_liab\_due\_within\_1y | double | 一年内到期的非流动负债 |
| other\_cur\_liab | double | 其他流动负债 |
| tot\_cur\_liab | double | 流动负债合计 |
| lt\_borrow | double | 长期借款 |
| bonds\_payable | double | 应付债券 |
| lt\_payable | double | 长期应付款 |
| specific\_item\_payable | double | 专项应付款 |
| provisions | double | 预计负债 |
| deferred\_tax\_liab | double | 递延所得税负债 |
| deferred\_inc\_non\_cur\_liab | double | 递延收益-非流动负债 |
| other\_non\_cur\_liab | double | 其他非流动负债 |
| tot\_non\_cur\_liab | double | 非流动负债合计 |
| liab\_dep\_other\_banks\_inst | double | 同业和其它金融机构存放款项 |
| derivative\_fin\_liab | double | 衍生金融负债 |
| cust\_bank\_dep | double | 吸收存款 |
| agency\_bus\_liab | double | 代理业务负债 |
| other\_liab | double | 其他负债 |
| prem\_received\_adv | double | 预收保费 |
| deposit\_received | double | 存入保证金 |
| insured\_deposit\_invest | double | 保户储金及投资款 |
| unearned\_prem\_rsrv | double | 未到期责任准备金 |
| out\_loss\_rsrv | double | 未决赔款准备金 |
| life\_insur\_rsrv | double | 寿险责任准备金 |
| lt\_health\_insur\_v | double | 长期健康险责任准备金 |
| independent\_acct\_liab | double | 独立账户负债 |
| incl\_pledge\_loan | double | 其中:质押借款 |
| claims\_payable | double | 应付赔付款 |
| dvd\_payable\_insured | double | 应付保单红利 |
| total\_liab | double | 负债合计 |
| capital\_stk | double | 股本 |
| capital\_reser | double | 资本公积金 |
| special\_rsrv | double | 专项储备 |
| surplus\_rsrv | double | 盈余公积金 |
| undistributed\_profit | double | 未分配利润 |
| less\_tsy\_stk | double | 减:库存股 |
| prov\_nom\_risks | double | 一般风险准备 |
| cnvd\_diff\_foreign\_curr\_stat | double | 外币报表折算差额 |
| unconfirmed\_invest\_loss | double | 未确认的投资损失 |
| minority\_int | double | 少数股东权益 |
| tot\_shrhldr\_eqy\_excl\_min\_int | double | 股东权益合计(不含少数股东权益) |
| tot\_shrhldr\_eqy\_incl\_min\_int | double | 股东权益合计(含少数股东权益) |
| tot\_liab\_shrhldr\_eqy | double | 负债及股东权益总计 |
| spe\_cur\_assets\_diff | double | 流动资产差额(特殊报表科目) |
| tot\_cur\_assets\_diff | double | 流动资产差额(合计平衡项目) |
| spe\_non\_cur\_assets\_diff | double | 非流动资产差额(特殊报表科目) |
| tot\_non\_cur\_assets\_diff | double | 非流动资产差额(合计平衡项目) |
| spe\_bal\_assets\_diff | double | 资产差额(特殊报表科目) |
| tot\_bal\_assets\_diff | double | 资产差额(合计平衡项目) |
| spe\_cur\_liab\_diff | double | 流动负债差额(特殊报表科目) |
| tot\_cur\_liab\_diff | double | 流动负债差额(合计平衡项目) |
| spe\_non\_cur\_liab\_diff | double | 非流动负债差额(特殊报表科目) |
| tot\_non\_cur\_liab\_diff | double | 非流动负债差额(合计平衡项目) |
| spe\_bal\_liab\_diff | double | 负债差额(特殊报表科目) |
| tot\_bal\_liab\_diff | double | 负债差额(合计平衡项目) |
| spe\_bal\_shrhldr\_eqy\_diff | double | 股东权益差额(特殊报表科目) |
| tot\_bal\_shrhldr\_eqy\_diff | double | 股东权益差额(合计平衡项目) |
| spe\_bal\_liab\_eqy\_diff | double | 负债及股东权益差额(特殊报表项目) |
| tot\_bal\_liab\_eqy\_diff | double | 负债及股东权益差额(合计平衡项目) |
| lt\_payroll\_payable | double | 长期应付职工薪酬 |
| other\_comp\_income | double | 其他综合收益 |
| other\_equity\_tools | double | 其他权益工具 |
| other\_equity\_tools\_p\_shr | double | 其他权益工具:优先股 |
| lending\_funds | double | 融出资金 |
| accounts\_receivable | double | 应收款项 |
| st\_financing\_payable | double | 应付短期融资款 |
| payables | double | 应付款项 |
| update\_flag | int | 数据更新标记 |

## 利润表

### 调用示例

```python
df, msg = api.query(
                view="lb.income", 
                fields="", 
                filter="symbol=600030.SH,000063.SZ,000001.SZ&report_type=408002000&start_date=20160601&end_date=20170601", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| start\_date | string | 公告开始日期 |
| end\_date | string | 公告结束日期 |
| comp\_type\_code | string | 公司类型代码 |
| start\_actdate | string | 实际公告开始日期 |
| end\_actdate | string | 实际公告结束日期 |
| start\_reportdate | string | 报告期开始日期 |
| start\_reportdate | string | 报告期结束日期 |
| report\_type | string | 报表类型 |
| update\_flag | int | 数据更新标记 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| ann\_date | string | 公告日期 |
| comp\_type\_code | string | 公司类型代码 |
| act\_ann\_date | string | 实际公告日期 |
| report\_date | string | 报告期 |
| report\_type | string | 报表类型 |
| currency | string | 货币代码 |
| total\_oper\_rev | double | 营业总收入 |
| oper\_rev | double | 营业收入 |
| int\_income | double | 利息收入 |
| net\_int\_income | double | 利息净收入 |
| insur\_prem\_unearned | double | 已赚保费 |
| handling\_chrg\_income | double | 手续费及佣金收入 |
| net\_handling\_chrg\_income | double | 手续费及佣金净收入 |
| net\_inc\_other\_ops | double | 其他经营净收益 |
| plus\_net\_inc\_other\_bus | double | 加:其他业务净收益 |
| prem\_income | double | 保费业务收入 |
| less\_ceded\_out\_prem | double | 减:分出保费 |
| chg\_unearned\_prem\_res | double | 提取未到期责任准备金 |
| incl\_reinsurance\_prem\_inc | double | 其中:分保费收入 |
| net\_inc\_sec\_trading\_brok\_bus | double | 代理买卖证券业务净收入 |
| net\_inc\_sec\_uw\_bus | double | 证券承销业务净收入 |
| net\_inc\_ec\_asset\_mgmt\_bus | double | 受托客户资产管理业务净收入 |
| other\_bus\_income | double | 其他业务收入 |
| plus\_net\_gain\_chg\_fv | double | 加:公允价值变动净收益 |
| plus\_net\_invest\_inc | double | 加:投资净收益 |
| incl\_inc\_invest\_assoc\_jv\_entp | double | 其中:对联营企业和合营企业的投资收益 |
| plus\_net\_gain\_fx\_trans | double | 加:汇兑净收益 |
| tot\_oper\_cost | double | 营业总成本 |
| less\_oper\_cost | double | 减:营业成本 |
| less\_int\_exp | double | 减:利息支出 |
| less\_handling\_chrg\_comm\_exp | double | 减:手续费及佣金支出 |
| less\_taxes\_surcharges\_ops | double | 减:营业税金及附加 |
| less\_selling\_dist\_exp | double | 减:销售费用 |
| less\_gerl\_admin\_exp | double | 减:管理费用 |
| less\_fin\_exp | double | 减:财务费用 |
| less\_impair\_loss\_assets | double | 减:资产减值损失 |
| prepay\_surr | double | 退保金 |
| tot\_claim\_exp | double | 赔付总支出 |
| chg\_insur\_cont\_rsrv | double | 提取保险责任准备金 |
| dvd\_exp\_insured | double | 保户红利支出 |
| reinsurance\_exp | double | 分保费用 |
| oper\_exp | double | 营业支出 |
| less\_claim\_recb\_reinsurer | double | 减:摊回赔付支出 |
| less\_ins\_rsrv\_recb\_reinsurer | double | 减:摊回保险责任准备金 |
| less\_exp\_recb\_reinsurer | double | 减:摊回分保费用 |
| other\_bus\_cost | double | 其他业务成本 |
| oper\_profit | double | 营业利润 |
| plus\_non\_oper\_rev | double | 加:营业外收入 |
| less\_non\_oper\_exp | double | 减:营业外支出 |
| il\_net\_loss\_disp\_noncur\_asset | double | 其中:减:非流动资产处置净损失 |
| tot\_profit | double | 利润总额 |
| inc\_tax | double | 所得税 |
| unconfirmed\_invest\_loss | double | 未确认投资损失 |
| net\_profit\_incl\_min\_int\_inc | double | 净利润(含少数股东损益) |
| net\_profit\_excl\_min\_int\_inc | double | 净利润(不含少数股东损益) |
| minority\_int\_inc | double | 少数股东损益 |
| other\_compreh\_inc | double | 其他综合收益 |
| tot\_compreh\_inc | double | 综合收益总额 |
| tot\_compreh\_inc\_parent\_comp | double | 综合收益总额(母公司) |
| tot\_compreh\_inc\_min\_shrhldr | double | 综合收益总额(少数股东) |
| ebit | double | 息税前利润 |
| ebitda | double | 息税折旧摊销前利润 |
| net\_profit\_after\_ded\_nr\_lp | double | 扣除非经常性损益后净利润 |
| net\_profit\_under\_intl\_acc\_sta | double | 国际会计准则净利润 |
| s\_fa\_eps\_basic | double | 基本每股收益 |
| s\_fa\_eps\_diluted | double | 稀释每股收益 |
| insurance\_expense | double | 保险业务支出 |
| spe\_bal\_oper\_profit | double | 营业利润差额(特殊报表科目) |
| tot\_bal\_oper\_profit | double | 营业利润差额(合计平衡项目) |
| spe\_bal\_tot\_profit | double | 利润总额差额(特殊报表科目) |
| tot\_bal\_tot\_profit | double | 利润总额差额(合计平衡项目) |
| spe\_bal\_net\_profit | double | 净利润差额(特殊报表科目) |
| tot\_bal\_net\_profit | double | 净利润差额(合计平衡项目) |
| undistributed\_profit | double | 年初未分配利润 |
| adjlossgain\_prevyear | double | 调整以前年度损益 |
| transfer\_from\_surplusreserve | double | 盈余公积转入 |
| transfer\_from\_housingimprest | double | 住房周转金转入 |
| transfer\_from\_others | double | 其他转入 |
| distributable\_profit | double | 可分配利润 |
| withdr\_legalsurplus | double | 提取法定盈余公积 |
| withdr\_legalpubwelfunds | double | 提取法定公益金 |
| workers\_welfare | double | 职工奖金福利 |
| withdr\_buzexpwelfare | double | 提取企业发展基金 |
| withdr\_reservefund | double | 提取储备基金 |
| distributable\_profit\_shrhder | double | 可供股东分配的利润 |
| prfshare\_dvd\_payable | double | 应付优先股股利 |
| withdr\_othersurpreserve | double | 提取任意盈余公积金 |
| comshare\_dvd\_payable | double | 应付普通股股利 |
| capitalized\_comstock\_div | double | 转作股本的普通股股利 |
| update\_flag | double | 数据更新标记 |


## 现金流量表

### 调用示例

```python
df, msg = api.query(
                view="lb.cashFlow", 
                fields="", 
                filter="symbol=002548.SZ", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| start\_date | string | 公告开始日期 |
| end\_date | string | 公告结束日期 |
| comp\_type\_code | string | 公司类型代码 |
| start\_actdate | string | 实际公告开始日期 |
| end\_actdate | string | 实际公告结束日期 |
| start\_reportdate | string | 报告期开始日期 |
| start\_reportdate | string | 报告期结束日期 |
| report\_type | string | 报表类型 |
| update\_flag | int | 数据更新标记 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| ann\_date | string | 公告日期 |
| comp\_type\_code | string | 公司类型代码 |
| act\_ann\_date | string | 实际公告日期 |
| report\_date | string | 报告期 |
| report\_type | string | 报表类型 |
| currency | string | 货币代码 |
| cash\_recp\_sg\_and\_rs | double | 销售商品、提供劳务收到的现金 |
| recp\_tax\_rends | double | 收到的税费返还 |
| net\_incr\_dep\_cob | double | 客户存款和同业存放款项净增加额 |
| net\_incr\_loans\_central\_bank | double | 向中央银行借款净增加额 |
| net\_incr\_fund\_borr\_ofi | double | 向其他金融机构拆入资金净增加额 |
| cash\_recp\_prem\_orig\_inco | double | 收到原保险合同保费取得的现金 |
| net\_incr\_insured\_dep | double | 保户储金净增加额 |
| net\_cash\_received\_reinsu\_bus | double | 收到再保业务现金净额 |
| net\_incr\_disp\_tfa | double | 处置交易性金融资产净增加额 |
| net\_incr\_int\_handling\_chrg | double | 收取利息和手续费净增加额 |
| net\_incr\_disp\_faas | double | 处置可供出售金融资产净增加额 |
| net\_incr\_loans\_other\_bank | double | 拆入资金净增加额 |
| net\_incr\_repurch\_bus\_fund | double | 回购业务资金净增加额 |
| other\_cash\_recp\_ral\_oper\_act | double | 收到其他与经营活动有关的现金 |
| stot\_cash\_inflows\_oper\_act | double | 经营活动现金流入小计 |
| cash\_pay\_goods\_purch\_serv\_rec | double | 购买商品、接受劳务支付的现金 |
| cash\_pay\_beh\_empl | double | 支付给职工以及为职工支付的现金 |
| pay\_all\_typ\_tax | double | 支付的各项税费 |
| net\_incr\_clients\_loan\_adv | double | 客户贷款及垫款净增加额 |
| net\_incr\_dep\_cbob | double | 存放央行和同业款项净增加额 |
| cash\_pay\_claims\_orig\_inco | double | 支付原保险合同赔付款项的现金 |
| handling\_chrg\_paid | double | 支付手续费的现金 |
| comm\_insur\_plcy\_paid | double | 支付保单红利的现金 |
| other\_cash\_pay\_ral\_oper\_act | double | 支付其他与经营活动有关的现金 |
| stot\_cash\_outflows\_oper\_act | double | 经营活动现金流出小计 |
| net\_cash\_flows\_oper\_act | double | 经营活动产生的现金流量净额 |
| cash\_recp\_disp\_withdrwl\_invest | double | 收回投资收到的现金 |
| cash\_recp\_return\_invest | double | 取得投资收益收到的现金 |
| net\_cash\_recp\_disp\_fiolta | double | 处置固定资产、无形资产和其他长期资产收回的现金净额 |
| net\_cash\_recp\_disp\_sobu | double | 处置子公司及其他营业单位收到的现金净额 |
| other\_cash\_recp\_ral\_inv\_act | double | 收到其他与投资活动有关的现金 |
| stot\_cash\_inflows\_inv\_act | double | 投资活动现金流入小计 |
| cash\_pay\_acq\_const\_fiolta | double | 购建固定资产、无形资产和其他长期资产支付的现金 |
| cash\_paid\_invest | double | 投资支付的现金 |
| net\_cash\_pay\_aquis\_sobu | double | 取得子公司及其他营业单位支付的现金净额 |
| other\_cash\_pay\_ral\_inv\_act | double | 支付其他与投资活动有关的现金 |
| net\_incr\_pledge\_loan | double | 质押贷款净增加额 |
| stot\_cash\_outflows\_inv\_act | double | 投资活动现金流出小计 |
| net\_cash\_flows\_inv\_act | double | 投资活动产生的现金流量净额 |
| cash\_recp\_cap\_contrib | double | 吸收投资收到的现金 |
| incl\_cash\_rec\_saims | double | 其中:子公司吸收少数股东投资收到的现金 |
| cash\_recp\_borrow | double | 取得借款收到的现金 |
| proc\_issue\_bonds | double | 发行债券收到的现金 |
| other\_cash\_recp\_ral\_fnc\_act | double | 收到其他与筹资活动有关的现金 |
| stot\_cash\_inflows\_fnc\_act | double | 筹资活动现金流入小计 |
| cash\_prepay\_amt\_borr | double | 偿还债务支付的现金 |
| cash\_pay\_dist\_dpcp\_int\_exp | double | 分配股利、利润或偿付利息支付的现金 |
| incl\_dvd\_profit\_paid\_sc\_ms | double | 其中:子公司支付给少数股东的股利、利润 |
| other\_cash\_pay\_ral\_fnc\_act | double | 支付其他与筹资活动有关的现金 |
| stot\_cash\_outflows\_fnc\_act | double | 筹资活动现金流出小计 |
| net\_cash\_flows\_fnc\_act | double | 筹资活动产生的现金流量净额 |
| eff\_fx\_flu\_cash | double | 汇率变动对现金的影响 |
| net\_incr\_cash\_cash\_equ | double | 现金及现金等价物净增加额 |
| cash\_cash\_equ\_beg\_period | double | 期初现金及现金等价物余额 |
| cash\_cash\_equ\_end\_period | double | 期末现金及现金等价物余额 |
| net\_profit | double | 净利润 |
| unconfirmed\_invest\_loss | double | 未确认投资损失 |
| plus\_prov\_depr\_assets | double | 加:资产减值准备 |
| depr\_fa\_coga\_dpba | double | 固定资产折旧、油气资产折耗、生产性生物资产折旧 |
| amort\_intang\_assets | double | 无形资产摊销 |
| amort\_lt\_deferred\_exp | double | 长期待摊费用摊销 |
| decr\_deferred\_exp | double | 待摊费用减少 |
| incr\_acc\_exp | double | 预提费用增加 |
| loss\_disp\_fiolta | double | 处置固定、无形资产和其他长期资产的损失 |
| loss\_scr\_fa | double | 固定资产报废损失 |
| loss\_fv\_chg | double | 公允价值变动损失 |
| fin\_exp | double | 财务费用 |
| invest\_loss | double | 投资损失 |
| decr\_deferred\_inc\_tax\_assets | double | 递延所得税资产减少 |
| incr\_deferred\_inc\_tax\_liab | double | 递延所得税负债增加 |
| decr\_inventories | double | 存货的减少 |
| decr\_oper\_payable | double | 经营性应收项目的减少 |
| incr\_oper\_payable | double | 经营性应付项目的增加 |
| others | double | 其他 |
| im\_net\_cash\_flows\_oper\_act | double | 间接法-经营活动产生的现金流量净额 |
| conv\_debt\_into\_cap | double | 债务转为资本 |
| conv\_corp\_bonds\_due\_within\_1y | double | 一年内到期的可转换公司债券 |
| fa\_fnc\_leases | double | 融资租入固定资产 |
| end\_bal\_cash | double | 现金的期末余额 |
| less\_beg\_bal\_cash | double | 减:现金的期初余额 |
| plus\_end\_bal\_cash\_equ | double | 加:现金等价物的期末余额 |
| less\_beg\_bal\_cash\_equ | double | 减:现金等价物的期初余额 |
| im\_net\_incr\_cash\_cash\_equ | double | 间接法-现金及现金等价物净增加额 |
| free\_cash\_flow | double | 企业自由现金流量 |
| spe\_bal\_cash\_inflows\_oper | double | 经营活动现金流入差额(特殊报表科目) |
| tot\_bal\_cash\_inflows\_oper | double | 经营活动现金流入差额(合计平衡项目) |
| spe\_bal\_cash\_outflows\_oper | double | 经营活动现金流出差额(特殊报表科目) |
| tot\_bal\_cash\_outflows\_oper | double | 经营活动现金流出差额(合计平衡项目) |
| tot\_bal\_netcash\_outflows\_oper | double | 经营活动产生的现金流量净额差额(合计平衡项目) |
| spe\_bal\_cash\_inflows\_inv | double | 投资活动现金流入差额(特殊报表科目) |
| tot\_bal\_cash\_inflows\_inv | double | 投资活动现金流入差额(合计平衡项目) |
| spe\_bal\_cash\_outflows\_inv | double | 投资活动现金流出差额(特殊报表科目) |
| tot\_bal\_cash\_outflows\_inv | double | 投资活动现金流出差额(合计平衡项目) |
| tot\_bal\_netcash\_outflows\_inv | double | 投资活动产生的现金流量净额差额(合计平衡项目) |
| spe\_bal\_cash\_inflows\_fnc | double | 筹资活动现金流入差额(特殊报表科目) |
| tot\_bal\_cash\_inflows\_fnc | double | 筹资活动现金流入差额(合计平衡项目) |
| spe\_bal\_cash\_outflows\_fnc | double | 筹资活动现金流出差额(特殊报表科目) |
| tot\_bal\_cash\_outflows\_fnc | double | 筹资活动现金流出差额(合计平衡项目) |
| tot\_bal\_netcash\_outflows\_fnc | double | 筹资活动产生的现金流量净额差额(合计平衡项目) |
| spe\_bal\_netcash\_inc | double | 现金净增加额差额(特殊报表科目) |
| tot\_bal\_netcash\_inc | double | 现金净增加额差额(合计平衡项目) |
| spe\_bal\_netcash\_equ\_undir | double | 间接法-经营活动现金流量净额差额(特殊报表科目) |
| tot\_bal\_netcash\_equ\_undir | double | 间接法-经营活动现金流量净额差额(合计平衡项目) |
| spe\_bal\_netcash\_inc\_undir | double | 间接法-现金净增加额差额(特殊报表科目) |
| spe\_bal\_netcash\_inc\_undir | double | 间接法-现金净增加额差额(合计平衡项目) |
| update\_flag | int | 数据更新标记 |

## 业绩快报

### 调用示例

```python
df, msg = api.query(
                view="lb.profitExpress", 
                fields="", 
                filter="start_anndate=20170101&end_anndate=20171010", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| start\_anndate | string | 公告开始日期 |
| end\_anndate | string | 公告结束日期 |
| start\_reportdate | string | 报告开始期 |
| end\_reportdate | string | 报告结束期 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| ann\_date | string | 公告日期 |
| report\_date | string | 报告期 |
| oper\_rev | double | 营业收入 |
| oper\_profit | double | 营业利润 |
| total\_profit | double | 利润总额 |
| net\_profit\_int\_inc | double | 净利润 |
| total\_assets | double | 总资产 |
| tot\_shrhldr\_int | double | 股东权益合计 |
| eps\_diluted | double | 每股收益(摊薄) |
| roe\_diluted | double | 净资产收益率(摊薄) |
| is\_audit | double | 是否审计 |
| yoy\_int\_inc | double | 去年同期修正后净利润 |


<!-- ## 均值因子（暂不提供）

### 调用示例

```python
df, msg = api.query(
                view="jz.axiomaFactor", 
                fields="value,size", 
                filter="riskmodel=CNAxiomaMH&start_date=20170901", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| tradedate | string | start\_date |
| tradedate | string | end\_date |
| riskmodel | string | riskmodel |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| tradedate | string | tradedate |
| riskmodel | string | riskmodel |
| value | double | value |
| leverage | double | leverage |
| growth | double | growth |
| size | double | size |
| liquidity | double | liquidity |
| shorttermmomentum | double | shorttermmomentum |
| mediumtermmomentum | double | mediumtermmomentum |
| volatility | double | volatility |
| exchangeratesensitivity | double | exchangeratesensitivity |
| bsharemarket | double | bsharemarket |
| coalconsumablefuels | double | coalconsumablefuels |
| energyexcoal | double | energyexcoal |
| steel | double | steel |
| chemicals | double | chemicals |
| constructionmaterials | double | constructionmaterials |
| metalsminingexsteel | double | metalsminingexsteel |
| paperforestproducts | double | paperforestproducts |
| commercialprofessionalservices | double | commercialprofessionalservices |
| electricalequipment | double | electricalequipment |
| constructionengineering | double | constructionengineering |
| transportationnoninfrastructure | double | transportationnoninfrastructure |
| machinery | double | machinery |
| tradingcompaniesdistributorsconglomerates | double | tradingcompaniesdistributorsconglomerates |
| transportationinfrastructure | double | transportationinfrastructure |
| media | double | media |
| retailing | double | retailing |
| textilesapparelluxurygoods | double | textilesapparelluxurygoods |
| automobiles | double | automobiles |
| householddurables | double | householddurables |
| autocomponents | double | autocomponents |
| consumerservices | double | consumerservices |
| foodproducts | double | foodproducts |
| beveragestobacco | double | beveragestobacco |
| healthcare | double | healthcare |
| realestate | double | realestate |
| financials | double | financials |
| softwareservices | double | softwareservices |
| computersperipherals | double | computersperipherals |
| communicationsequipment | double | communicationsequipment |
| semiconductorselectronics | double | semiconductorselectronics |
| telecommunicationservices | double | telecommunicationservices |
| utilities | double | utilities | -->


## 限售股解禁表

### 调用示例

```python
df, msg = api.query(
                view="lb.secRestricted", 
                fields="", 
                filter="start_date=20170101&end_date=20171011",
                data_format='pandas') 
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| start\_date | string | 开始日期 |
| end\_date | string | 结束日期 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| list\_date | string | 本期解禁流通日期 |
| lifted\_reason | string | 本期解禁原因（来源） |
| lifted\_shares | double | 本期解禁数量 |
| lifted\_ratio | double | 可流通占A股总数比例 |

<!-- ## 财务指标表

### 调用示例

```python
df, msg = api.query(
                view="lb.finindicator", 
)
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| start\_date | string | 公告开始日期 |
| end\_date | string | 公告结束日期 |
| start\_reportdate | string | 报告期开始日期 |
| start\_reportdate | string | 报告期结束日期 |
| update\_flag | int | 数据更新标记 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| ann\_date | string | 公告日期 |
| report\_date | string | 报告期 |
| crncy\_code | string | 货币代码 |
| extraordinary | double | 非经常性损益 |
| deductedprofit | double | 扣除非经常性损益后的净利润 |
| grossmargin | double | 毛利 |
| operateincome | double | 经营活动净收益 |
| investincome | double | 价值变动净收益 |
| stmnote\_finexp | double | 利息费用 |
| stm\_is | double | 折旧与摊销 |
| ebit | double | 息税前利润 |
| ebitda | double | 息税折旧摊销前利润 |
| fcff | double | 企业自由现金流量 |
| fcfe | double | 股权自由现金流量 |
| exinterestdebt\_current | double | 无息流动负债 |
| exinterestdebt\_noncurrent | double | 无息非流动负债 |
| interestdebt | double | 带息债务 |
| netdebt | double | 净债务 |
| tangibleasset | double | 有形资产 |
| workingcapital | double | 营运资金 |
| networkingcapital | double | 营运流动资本 |
| investcapital | double | 全部投入资本 |
| retainedearnings | double | 留存收益 |
| eps\_basic | double | 基本每股收益 |
| eps\_diluted | double | 稀释每股收益 |
| eps\_diluted2 | double | 期末摊薄每股收益 |
| bps | double | 每股净资产 |
| ocfps | double | 每股经营活动产生的现金流量净额 |
| grps | double | 每股营业总收入 |
| orps | double | 每股营业收入 |
| surpluscapitalps | double | 每股资本公积 |
| surplusreserveps | double | 每股盈余公积 |
| undistributedps | double | 每股未分配利润 |
| retainedps | double | 每股留存收益 |
| cfps | double | 每股现金流量净额 |
| ebitps | double | 每股息税前利润 |
| fcffps | double | 每股企业自由现金流量 |
| fcfeps | double | 每股股东自由现金流量 |
| netprofitmargin | double | 销售净利率 |
| grossprofitmargin | double | 销售毛利率 |
| cogstosales | double | 销售成本率 |
| expensetosales | double | 销售期间费用率 |
| profittogr | double | 净利润/营业总收入 |
| saleexpensetogr | double | 销售费用/营业总收入 |
| adminexpensetogr | double | 管理费用/营业总收入 |
| finaexpensetogr | double | 财务费用/营业总收入 |
| impairtogr\_ttm | double | 资产减值损失/营业总收入 |
| gctogr | double | 营业总成本/营业总收入 |
| optogr | double | 营业利润/营业总收入 |
| ebittogr | double | 息税前利润/营业总收入 |
| roe | double | 净资产收益率 |
| roe\_deducted | double | 净资产收益率(扣除非经常损益) |
| roa2 | double | 总资产报酬率 |
| roa | double | 总资产净利润 |
| roic | double | 投入资本回报率 |
| roe\_yearly | double | 年化净资产收益率 |
| roa2\_yearly | double | 年化总资产报酬率 |
| roe\_avg | double | 平均净资产收益率(增发条件) |
| operateincometoebt | double | 经营活动净收益/利润总额 |
| investincometoebt | double | 价值变动净收益/利润总额 |
| nonoperateprofittoebt | double | 营业外收支净额/利润总额 |
| taxtoebt | double | 所得税/利润总额 |
| deductedprofittoprofit | double | 扣除非经常损益后的净利润/净利润 |
| salescashintoor | double | 销售商品提供劳务收到的现金/营业收入 |
| ocftoor | double | 经营活动产生的现金流量净额/营业收入 |
| ocftooperateincome | double | 经营活动产生的现金流量净额/经营活动净收益 |
| capitalizedtoda | double | 资本支出/折旧和摊销 |
| debttoassets | double | 资产负债率 |
| assetstoequity | double | 权益乘数 |
| dupont\_assetstoequity | double | 权益乘数(用于杜邦分析) |
| catoassets | double | 流动资产/总资产 |
| ncatoassets | double | 非流动资产/总资产 |
| tangibleassetstoassets | double | 有形资产/总资产 |
| intdebttototalcap | double | 带息债务/全部投入资本 |
| equitytototalcapital | double | 归属于母公司的股东权益/全部投入资本 |
| currentdebttodebt | double | 流动负债/负债合计 |
| longdebtodebt | double | 非流动负债/负债合计 |
| current | double | 流动比率 |
| quick | double | 速动比率 |
| cashratio | double | 保守速动比率 |
| ocftoshortdebt | double | 经营活动产生的现金流量净额/流动负债 |
| debttoequity | double | 产权比率 |
| equitytodebt | double | 归属于母公司的股东权益/负债合计 |
| equitytointerestdebt | double | 归属于母公司的股东权益/带息债务 |
| tangibleassettodebt | double | 有形资产/负债合计 |
| tangassettointdebt | double | 有形资产/带息债务 |
| tangibleassettonetdebt | double | 有形资产/净债务 |
| ocftodebt | double | 经营活动产生的现金流量净额/负债合计 |
| ocftointerestdebt | double | 经营活动产生的现金流量净额/带息债务 |
| ocftonetdebt | double | 经营活动产生的现金流量净额/净债务 |
| ebittointerest | double | 已获利息倍数(EBIT/利息费用) |
| longdebttoworkingcapital | double | 长期债务与营运资金比率 |
| ebitdatodebt | double | 息税折旧摊销前利润/负债合计 |
| turndays | double | 营业周期 |
| invturndays | double | 存货周转天数 |
| arturndays | double | 应收账款周转天数 |
| invturn | double | 存货周转率 |
| arturn | double | 应收账款周转率 |
| caturn | double | 流动资产周转率 |
| faturn | double | 固定资产周转率 |
| assetsturn | double | 总资产周转率 |
| roa\_yearly | double | 年化总资产净利率 |
| dupont\_roa | double | 总资产净利率(杜邦分析) |
| s\_stm\_bs | double | 固定资产合计 |
| prefinexpense\_opprofit | double | 扣除财务费用前营业利润 |
| nonopprofit | double | 非营业利润 |
| optoebt | double | 营业利润／利润总额 |
| noptoebt | double | 非营业利润／利润总额 |
| ocftoprofit | double | 经营活动产生的现金流量净额／营业利润 |
| cashtoliqdebt | double | 货币资金／流动负债 |
| cashtoliqdebtwithinterest | double | 货币资金／带息流动负债 |
| optoliqdebt | double | 营业利润／流动负债 |
| optodebt | double | 营业利润／负债合计 |
| roic\_yearly | double | 年化投入资本回报率 |
| tot\_faturn | double | 固定资产合计周转率 |
| profittoop | double | 利润总额／营业收入 |
| qfa\_operateincome | double | 单季度.经营活动净收益 |
| qfa\_investincome | double | 单季度.价值变动净收益 |
| qfa\_deductedprofit | double | 单季度.扣除非经常损益后的净利润 |
| qfa\_eps | double | 单季度.每股收益 |
| qfa\_netprofitmargin | double | 单季度.销售净利率 |
| qfa\_grossprofitmargin | double | 单季度.销售毛利率 |
| qfa\_expensetosales | double | 单季度.销售期间费用率 |
| qfa\_profittogr | double | 单季度.净利润／营业总收入 |
| qfa\_saleexpensetogr | double | 单季度.销售费用／营业总收入 |
| qfa\_adminexpensetogr | double | 单季度.管理费用／营业总收入 |
| qfa\_finaexpensetogr | double | 单季度.财务费用／营业总收入 |
| qfa\_impairtogr\_ttm | double | 单季度.资产减值损失／营业总收入 |
| qfa\_gctogr | double | 单季度.营业总成本／营业总收入 |
| qfa\_optogr | double | 单季度.营业利润／营业总收入 |
| qfa\_roe | double | 单季度.净资产收益率 |
| qfa\_roe\_deducted | double | 单季度.净资产收益率(扣除非经常损益) |
| qfa\_roa | double | 单季度.总资产净利润 |
| qfa\_operateincometoebt | double | 单季度.经营活动净收益／利润总额 |
| qfa\_investincometoebt | double | 单季度.价值变动净收益／利润总额 |
| qfa\_deductedprofittoprofit | double | 单季度.扣除非经常损益后的净利润／净利润 |
| qfa\_salescashintoor | double | 单季度.销售商品提供劳务收到的现金／营业收入 |
| qfa\_ocftosales | double | 单季度.经营活动产生的现金流量净额／营业收入 |
| qfa\_ocftoor | double | 单季度.经营活动产生的现金流量净额／经营活动净收益 |
| yoyeps\_basic | double | 同比增长率-基本每股收益(%) |
| yoyeps\_diluted | double | 同比增长率-稀释每股收益(%) |
| yoyocfps | double | 同比增长率-每股经营活动产生的现金流量净额(%) |
| yoyop | double | 同比增长率-营业利润(%) |
| yoyebt | double | 同比增长率-利润总额(%) |
| yoynetprofit | double | 同比增长率-归属母公司股东的净利润(%) |
| yoynetprofit\_deducted | double | 同比增长率-归属母公司股东的净利润-扣除非经常损益(%) |
| yoyocf | double | 同比增长率-经营活动产生的现金流量净额(%) |
| yoyroe | double | 同比增长率-净资产收益率(摊薄)(%) |
| yoybps | double | 相对年初增长率-每股净资产(%) |
| yoyassets | double | 相对年初增长率-资产总计(%) |
| yoyequity | double | 相对年初增长率-归属母公司的股东权益(%) |
| yoy\_tr | double | 营业总收入同比增长率(%) |
| yoy\_or | double | 营业收入同比增长率(%) |
| qfa\_yoygr | double | 单季度.营业总收入同比增长率(%) |
| qfa\_cgrgr | double | 单季度.营业总收入环比增长率(%) |
| qfa\_yoysales | double | 单季度.营业收入同比增长率(%) |
| qfa\_cgrsales | double | 单季度.营业收入环比增长率(%) |
| qfa\_yoyop | double | 单季度.营业利润同比增长率(%) |
| qfa\_cgrop | double | 单季度.营业利润环比增长率(%) |
| qfa\_yoyprofit | double | 单季度.净利润同比增长率(%) |
| qfa\_cgrprofit | double | 单季度.净利润环比增长率(%) |
| qfa\_yoynetprofit | double | 单季度.归属母公司股东的净利润同比增长率(%) |
| qfa\_cgrnetprofit | double | 单季度.归属母公司股东的净利润环比增长率(%) |
| yoy\_equity | double | 净资产(同比增长率) |
| rd\_expense | double | 研发费用 |
| waa\_roe | double | 加权平均净资产收益率 |
| update\_flag | int | 数据更新标记 | -->

## 指数基本信息表

### 调用示例

```python
df, msg = api.query(
                view="lb.indexCons", 
                fields="", 
                filter="index_code=000001.SH&start_date=20170113&end_date=20171010", # this api must add start and end date both
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| symbol | string | 证券代码 |
| name | string | 证券简称 |
| compname | string | 指数名称 |
| exchmarket | string | 交易所 |
| index\_baseper | string | 基期 |
| index\_basept | double | 基点 |
| listdate | string | 发布日期 |
| index\_weightsrule | string | 加权方式 |
| publisher | string | 发布方 |
| indexcode | int | 指数类别代码 |
| indexstyle | string | 指数风格 |
| index\_intro | string | 指数简介 |
| weight\_type | int | 权重类型 |
| expire\_date | string | 终止发布日期 |

## 指数成份股表

### 调用示例

```python
df, msg = api.query(
                view="lb.indexCons", 
                fields="", 
                filter="index_code=399001&is_new=Y", 
                data_format='pandas')
```

### 输入参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| index\_code | string | 指数代码 |

### 输出参数

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| index\_code | string | 指数代码 |
| symbol | string | 证券代码 |
| in\_date | string | 纳入日期 |
| out\_date | string | 剔除日期 |
| is\_new | int | 最新标志 |
