
## Backtest

这里回测指**基于权重调仓**的Alpha策略回测，支持自定义**选股**和自定义**信号**。

### 回测&结果分析示例代码

处理：
- 分红除息再投资
- 退市清仓
- 指数成分

理念：
不在`on_bar`中进行发单，而是给出选股条件（boolean series）和信号（float series）权重




```python
dv.add_formula('my_signal', 'Quantile(price_volume_divert, 5)', is_quarterly=False)
```


```python
def my_singal(context, user_options=None):
    res = -context.snapshot_sub.loc[:, 'price_volume_divert']
    return res


def test_alpha_strategy_dataview():
##     dv = DataView()

##     fullpath = '/home/bliu/pytrade_dir/ipynb/prepared/compare'
##     dv.load_dataview(folder=fullpath)
    props = {
        "benchmark": "000300.SH",
        "universe": ','.join(dv.symbol),

        "start_date": dv.start_date,
        "end_date": dv.end_date,

        "period": "month",
        "days_delay": 0,
        "n_periods": 1,

        "init_balance": 1e9,
        "position_ratio": 0.5,
    }

    gateway = DailyStockSimGateway()
    gateway.init_from_config(props)

    context = model.Context(dataview=dv, gateway=gateway)

    signal_model = model.FactorRevenueModel(context)
    signal_model.add_signal('my_singal', my_singal)

    strategy = AlphaStrategy(revenue_model=signal_model, pc_method='equal_weight')

    bt = AlphaBacktestInstance()
    bt.init_from_config(props, strategy, context=context)

    bt.run_alpha()

    bt.save_results('output/divert')

test_alpha_strategy_dataview()
```

```python
def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    #data_service = RemoteDataService()

    out_folder = "output/jli"

    ta.initialize(dataview=dv, file_folder=out_folder)

    print "process trades..."
    ta.process_trades()
    print "get daily stats..."
    ta.get_daily()
    print "calc strategy return..."
    ta.get_returns(compound_return=False)
    # position change info is huge!
    # print "get position change..."
    # ta.get_pos_change_info()

    selected_sec = [] # list(ta.universe)[:5]
    if len(selected_sec) > 0:
        print "Plot single securities PnL"
        for symbol in selected_sec:
            df_daily = ta.daily.get(symbol, None)
            if df_daily is not None:
                ana.plot_trades(df_daily, symbol=symbol, save_folder=out_folder)

    print "Plot strategy PnL..."
    ta.plot_pnl(out_folder)

    print "generate report..."
    static_folder = fileio.join_relative_path("trade/analyze/static")
    ta.gen_report(source_dir=static_folder, template_fn='report_template.html',
                  out_folder=out_folder,
                  selected=selected_sec)

test_backtest_analyze()
```

    process trades...
    get daily stats...
    calc strategy return...
    Plot strategy PnL...
    generate report...
    HTML report: /home/bliu/pytrade_dir/ipynb/output/jli/report.html



![analyze](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/analyze.png)


### 格雷厄姆选股策略


主要介绍基于回测框架实现格雷厄姆模型。格雷厄姆模型分为两步，首先是条件选股，其次按照市值从小到大排序，选出排名前五的股票。
#### 一. 数据准备
我们选择如下指标，对全市场的股票进行筛选，实现过程如下：
a. 首先在数据准备模块save_dataview()中通过props设置数据起止日期，股票版块，以及所需变量
```python
props = {
'start_date': 20150101,
'end_date': 20170930,
'universe':'000905.SH',
'fields': ('tot_cur_assets,tot_cur_liab,inventories,pre_pay,deferred_exp, eps_basic,ebit,pe,pb,float_mv,sw1'),
'freq': 1
}
```
b. 接着创建0-1变量表示某只股票是否被选中，并通过add_formula将变量添加到dataview中
> * 市盈率（pe ratio）低于 20
> * 市净率（pb ratio）低于 2
> * 同比每股收益增长率（inc_earning_per_share）大于 0
> * 税前同比利润增长率（inc_profit_before_tax）大于 0
> * 流动比率（current_ratio）大于 2
> * 速动比率（quick_ratio）大于 1

```python
factor_formula = 'pe < 20'
dv.add_formula('pe_condition', factor_formula, is_quarterly=False)
factor_formula = 'pb < 2'
dv.add_formula('pb_condition', factor_formula, is_quarterly=False)
factor_formula = 'Return(eps_basic, 4) > 0'
dv.add_formula('eps_condition', factor_formula, is_quarterly=True)
factor_formula = 'Return(ebit, 4) > 0'
dv.add_formula('ebit_condition', factor_formula, is_quarterly=True)
factor_formula = 'tot_cur_assets/tot_cur_liab > 2'
dv.add_formula('current_condition', factor_formula, is_quarterly=True)
factor_formula = '(tot_cur_assets - inventories - pre_pay - deferred_exp)/tot_cur_liab > 1'
dv.add_formula('quick_condition', factor_formula, is_quarterly=True)
```
需要注意的是，涉及到的财务数据若不在secDailyIndicator表中，需将is_quarterly设置为True，表示该变量为季度数据。
c. 由于第二步中需要按流通市值排序，我们将这一变量也放入dataview中
```python
dv.add_formula('mv_rank', 'Rank(float_mv)', is_quarterly=False)
```
#### 二. 条件选股
条件选股在my_selector函数中完成：
> * 首先我们将上一步计算出的0/1变量提取出来，格式为Series
> * 接着我们对所有变量取交集，选中的股票设为1，未选中的设为0，并将结果通过DataFrame形式返回
```python
def my_selector(context, user_options=None):
    #
    pb_selector      = context.snapshot['pb_condition']
    pe_selector      = context.snapshot['pe_condition']
    eps_selector     = context.snapshot['eps_condition']
    ebit_selector    = context.snapshot['ebit_condition']
    current_selector = context.snapshot['current_condition']
    quick_selector   = context.snapshot['quick_condition']
    #
    merge = pd.concat([pb_selector, pe_selector, eps_selector,     ebit_selector, current_selector, quick_selector], axis=1)

    result = np.all(merge, axis=1)
    mask = np.all(merge.isnull().values, axis=1)
    result[mask] = False
    return pd.DataFrame(result, index=merge.index, columns=['selector'])
```
#### 三、按市值排序
按市值排序功能在signal_size函数中完成。我们根据流通市值排序变量'mv_rank'对所有股票进行排序，并选出市值最小的5只股票。
```python
def signal_size(context, user_options = None):
    mv_rank = context.snapshot_sub['mv_rank']
    s = np.sort(mv_rank.values)[::-1]
    if len(s) > 0:
        critical = s[-5] if len(s) > 5 else np.min(s)
        mask = mv_rank < critical
        mv_rank[mask] = 0.0
        mv_rank[~mask] = 1.0
    return mv_rank
```
#### 四、回测
我们在test_alpha_strategy_dataview()模块中实现回测功能
##### 1. 载入dataview，设置回测参数
该模块首先载入dataview并允许用户设置回测参数，比如基准指数，起止日期，换仓周期等。
```python
dv = DataView()

fullpath = fileio.join_relative_path('../output/prepared', dv_subfolder_name)
dv.load_dataview(folder=fullpath)

props = {
    "benchmark": "000905.SH",
    "universe": ','.join(dv.symbol),

    "start_date": dv.start_date,
    "end_date": dv.end_date,

    "period": "week",
    "days_delay": 0,

    "init_balance": 1e8,
    "position_ratio": 1.0,
}
```
##### 2. StockSelector选股模块
接着我们使用StockSelector选股模块，将之前定义的my_selector载入
```python
stock_selector = model.StockSelector(context)
stock_selector.add_filter(name='myselector', func=my_selector)
```
##### 3. FactorRevenueModel模块
在进行条件选股后，使用FactorRevenueModel模块对所选股票进行排序
```python
signal_model = model.FactorRevenueModel(context)
signal_model.add_signal(name='signalsize', func = signal_size)
```
##### 4. 策略回测模块
将上面定义的stockSelector和FactorRevenueModel载入AlphaStrategy函数进行回测
```python
    strategy = AlphaStrategy(
                stock_selector=stock_selector,
                revenue_model=signal_model，
                pc_method='factor_value_weight')
```
##### 5. 启动数据准备及回测模块
```python
t_start = time.time()

test_save_dataview()
test_alpha_strategy_dataview()
test_backtest_analyze()

t3 = time.time() - t_start
print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
```

#### 五、回测结果

回测的参数如下：

| 指标             | 值   |
| --------         | --:  |
| Beta             | 0.87 |
| Annual Return    | 0.08 |
| Annual Volatility| 0.29 |
| Sharpe Ratio     | 0.28 |

回测的净值曲线图如下：

![backtestgraham](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/backtest_Graham_result.png)

### 基于因子IC的多因子选股模型


主要介绍基于回测框架实现基于因子IC的因子权重优化模型。
#### 一. 因子IC定义及优化模型
##### 1. 因子IC的定义方法
首先介绍一下因子IC（Information Coefficient）的定义。传统意义上，因子在某一期的IC为该期因子与股票下期收益率的秩相关系数，即：
$$IC_t = RankCorrelation(\vec{f_t}, \vec{r_{t+1}})$$
其中$\vec{f_t}$为所有股票在t期的因子值向量，$\vec{r_{t+1}}$为所有股票在t到t+1期的收益率向量。秩相关系数直接反映了因子的预测能力：IC越高，说明该因子对接下里一期股票收益的预测能力越强。
##### 2. 因子的获取及计算方法
在本示例中我们简单选取了几个因子，更多的因子可以在股票因子数据中找到：
> * Turnover, 换手率
> * BP, Book-to-Market Ratio
> * MOM20, 过去20天收益率
> * LFMV, 对数流通市值

实现过程如下：
a. 首先在数据准备模块save_dataview()中通过props设置数据起止日期，股票版块，以及所需变量
```python
props = {'start_date': 20150101, 'end_date': 20170930, 'universe':
'000905.SH', 'fields': ('turnover,float_mv,close_adj,pe'), 'freq': 1}
```
b. 接着计算因子，进行标准化和去极值处理后通过add_formula()将因子添加到变量列表中
```python
factor_formula = 'Cutoff(Standardize(turnover / 10000 / float_mv), 2)'
dv.add_formula('TO', factor_formula, is_quarterly=False)

factor_formula = 'Cutoff(Standardize(1/pb), 2)'
dv.add_formula('BP', factor_formula, is_quarterly = False)

factor_formula = 'Cutoff(Standardize(Return(close_adj, 20)), 2)'
dv.add_formula('REVS20', factor_formula, is_quarterly=False)

factor_formula = 'Cutoff(Standardize(Log(float_mv)), 2)'
dv.add_formula('float_mv_factor', factor_formula, is_quarterly=False)
```
其中Standardize()和Cutoff()均为内置函数。Standardize作用是将序列做去均值并除以标准差的标准化处理，Cutoff作用是将序列中的极值拉回正常范围内。
之后将因子名称保存在外部文件中，以便后续计算使用
```python
factorList = ['TO', 'BP', 'REVS20', 'float_mv_factor']
factorList_adj = [x + '_adj' for x in factorList]
from jaqs.util import fileio
fileio.save_json(factorList_adj, '.../myCustomData.json')
```
c. 由于多个因子间可能存在多重共线性，我们对因子进行施密特正交化处理，并将处理后的因子添加到变量列表中。
```python
### add the orthogonalized factor to dataview
for trade_date in dv.dates:
    snapshot = dv.get_snapshot(trade_date)
    factorPanel = snapshot[factorList]
    factorPanel = factorPanel.dropna()

    if len(factorPanel) != 0:
        orthfactorPanel = Schmidt(factorPanel)
        orthfactorPanel.columns = [x + '_adj' for x in factorList]

        snapshot = pd.merge(left = snapshot, right = orthfactorPanel,
                            left_index = True, right_index = True, how = 'left')

        for factor in factorList:
            orthFactor_dic[factor][trade_date] = snapshot[factor]

for factor in factorList:
    dv.append_df(pd.DataFrame(orthFactor_dic[factor]).T, field_name = factor + '_adj', is_quarterly=False)
```
##### 3. 计算因子IC
从dataview中提取所有交易日，在每个交易日计算每个因子的IC
```python
def get_ic(dv):
    """
    Calculate factor IC on all dates and save it in a DataFrame
    :param dv:
    :return: DataFrame recording factor IC on all dates
    """
    factorList = fileio.read_json('.../myCustomData.json')
    ICPanel = {}
    for singleDate in dv.dates:
        singleSnapshot = dv.get_snapshot(singleDate)
        ICPanel[singleDate] = ic_calculation(singleSnapshot, factorList)

    ICPanel = pd.DataFrame(ICPanel).T
    return ICPanel
```
其中计算IC的函数为ic_calculation()
```python
def ic_calculation(snapshot, factorList):
    """
    Calculate factor IC on single date
    :param snapshot:
    :return: factor IC on single date
    """
    ICresult = []
    for factor in factorList:
        # drop na
        factorPanel = snapshot[[factor, 'NextRet']]
        factorPanel = factorPanel.dropna()
        ic, _ = stats.spearmanr(factorPanel[factor], factorPanel['NextRet'])
        ICresult.append(ic)
    return ICresult
```
##### 4. 因子权重优化
我们将因子IR设为因子权重优化的目标，因子IR（信息比）定义为因子IC的均值与因子IC的标准差的比值，IR值越高越好。假设我们有k个因子，其IC的均值向量为$\vec{IC}=(\overline{IC_1}, \overline{IC_2}, \cdots, \overline{IC_k},)'$，相应协方差矩阵为$\Sigma$，因子的权重向量为$\vec{v}=(\overline{V_1}, \overline{V_2},\cdots, \overline{V_k})'$。则所有因子的复合IR值为
$$IR = \frac{\vec{v}'\vec{IC}}{\sqrt{\vec{v}' \Sigma \vec{v}}}$$
我们的目标是通过调整$\vec{v}$使IR最大化。经简单计算我们可以直接求出$\vec{v}$的解析解，则最优权重向量为：
$$\vec{v}^* = \Sigma^{-1}\vec{IC}$$
具体实现过程如下：
```python
def store_ic_weight():
    """
    Calculate IC weight and save it to file
    """
    dv = DataView()
    fullpath = fileio.join_relative_path('../output/prepared', dv_subfolder_name)
    dv.load_dataview(folder=fullpath)

    w = get_ic_weight(dv)

    store = pd.HDFStore('/home/lli/ic_weight.hd5')
    store['ic_weight'] = w
    store.close()
```
其中使用到了get_ic_weight()函数，其作用是计算每个因子IC对应的weight
```python
def get_ic_weight(dv):
    """
    Calculate factor IC weight on all dates and save it in a DataFrame
    :param dv: dataview
    :return: DataFrame containing the factor IC weight, with trading date as index and factor name as columns
    """
    ICPanel = get_ic(dv)
    ICPanel = ICPanel.dropna()
    N = 10
    IC_weight_Panel = {}
    for i in range(N, len(ICPanel)):
        ICPanel_sub = ICPanel.iloc[i-N:i, :]
        ic_weight = ic_weight_calculation(ICPanel_sub)
        IC_weight_Panel[ICPanel.index[i]] = ic_weight
    IC_weight_Panel = pd.DataFrame(IC_weight_Panel).T
    return IC_weight_Panel
```
我们在计算weight时需要确定一个rolling window，这里选择N=10。
```python
def ic_weight_calculation(icpanel):
    """
    Calculate factor IC weight on single date
    :param icpanel:
    :return: a vector containing all factor IC weight
    """
    mat = np.mat(icpanel.cov())
    mat = nlg.inv(mat)
    weight = mat * np.mat(icpanel.mean()).reshape(len(mat), 1)
    weight = np.array(weight.reshape(len(weight), ))[0]
    return weight
```
#### 二. 基于因子IC及相应权重的选股模型
在介绍选股模型的具体实现之前，我们首先熟悉一下策略模块test_alpha_strategy_dataview()。该模块的功能是基于dataview对具体策略进行回测。
##### 1. 载入dataview，设置回测参数
该模块首先载入dataview并允许用户设置回测参数，比如基准指数，起止日期，换仓周期等。
```python
dv = DataView()

fullpath = fileio.join_relative_path('../output/prepared', dv_subfolder_name)
dv.load_dataview(folder=fullpath)

props = {
    "benchmark": "000905.SH",
    "universe": ','.join(dv.symbol),

    "start_date": dv.start_date,
    "end_date": dv.end_date,

    "period": "week",
    "days_delay": 0,

    "init_balance": 1e8,
    "position_ratio": 1.0,
}
```
##### 2. 载入context
context是一个类用来保存一些中间结果，可在程序中任意位置调用，并将之前算出的ic_weight放入context中。
```python
context = model.Context(dataview=dv, gateway=gateway)
store = pd.HDFStore('.../ic_weight.hd5')
context.ic_weight = store['ic_weight']
store.close()
```
##### 3. StockSelector选股模块
接着我们使用StockSelector选股模块。基于因子IC及相应权重的选股过程在my_selector中实现。
```python
stock_selector = model.StockSelector(context)
stock_selector.add_filter(name='myselector', func=my_selector)
```
a.首先载入因子ic的权重context.ic_weight，回测日期列表context.trade_date记忆因子名称列表factorList
```python
ic_weight = context.ic_weight
t_date = context.trade_date
current_ic_weight = np.mat(ic_weight.loc[t_date,]).reshape(-1,1)
factorList = fileio.read_json('.../myCustomData.json')

factorPanel = {}
for factor in factorList:
    factorPanel[factor] = context.snapshot[factor]

factorPanel = pd.DataFrame(factorPanel)
```
b.接着根据各因子IC的权重，对当天各股票的IC值进行加权求和，选出得分最高的前30只股票。最后返回一个列表，1代表选中，0代表未选中。
```python
factorResult = pd.DataFrame(np.mat(factorPanel) * np.mat(current_ic_weight), index = factorPanel.index)

factorResult = factorResult.fillna(-9999)
s = factorResult.sort_values(0)[::-1]

critical = s.values[30]
mask = factorResult > critical
factorResult[mask] = 1.0
factorResult[~mask] = 0.0
```
##### 4. 启动数据准备及回测模块
```python
t_start = time.time()

test_save_dataview()
store_ic_weight()
test_alpha_strategy_dataview()
test_backtest_analyze()

t3 = time.time() - t_start
print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
```
#### 三、回测结果

回测的参数如下：

| 指标             | 值   |
| --------         | ---  |
| Beta             | 0.92 |
| Annual Return    | 0.19 |
| Annual Volatility| 0.16 |
| Sharpe Ratio     | 1.21 |

回测的净值曲线图如下：
![backtesticmodel](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/backtest_ICModel_result.png)

#### 四、参考文献

1. [基于因子IC的多因子模型](https://uqer.io/community/share/57b540ef228e5b79a4759398)
2. 《安信证券－多因子系列报告之一：基于因子IC的多因子模型》
