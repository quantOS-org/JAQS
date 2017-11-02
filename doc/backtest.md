### 数据 - DataView

#### 数据准备
- 选择：起止时间、投资标的/基准
- 输入需要的字段：依照数据说明（日行情、财务报表、行业分类信息等）
- 自动获取：上市、退市信息，每日指数成分变化，复权因子


```python
ds = RemoteDataService()
dv = DataView()

props = {'start_date': 20160101, 'end_date': 20171029, 'universe': '000300.SH',
         'fields': 'sw2,volume,turnover,eps_basic,float_mv',
         'freq': 1}

dv.init_from_config(props, ds)
dv.prepare_data()
```

#### DataView做什么
将频繁使用的`DataFrame`操作自动化，使用者操作数据时尽量只考虑业务需求而不是技术实现：
1. 根据字段名，自动从不同的数据api获取数据
2. 按时间、标的整理对齐（财务数据按发布日期对齐）
3. 在已有数据基础上，添加字段、加入自定义数据或根据公式计算新数据
4. 数据查询
5. 本地存储

#### 数据获取

底层存储为3维：时间、标的、字段，可获取任意2维切片：

获取**快照**（在时间轴切片）：`get_snapshot(trade_date, symbol=, fields=)`


```python
dv.get_snapshot(20170712).head(3)
```




获取**时间序列**（在字段轴切片）：`get_ts(field_name, symbol=, start_date=, end_date=)`


```python
dv.get_ts('eps_basic', start_date=20170419).head(8)
```




#### 数据添加

- 从DataApi获取更多字段: `dv.add_field('roe')`
- 加入自定义DataFrame: `dv.append_df(name, df)`
- 根据公式计算衍生指标: `dv.add_formula(name, formula, is_quarterly=False)`


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



#### 数据存储与读取
- 可以读取修改后继续存储
- 默认覆盖


```python
dv.save_dataview('prepared', 'demo')
```

    
    Store data...
    Dataview has been successfully saved to:
    /home/bliu/pytrade_dir/ipynb/prepared/demo
    
    You can load it with load_dataview('/home/bliu/pytrade_dir/ipynb/prepared/demo')



```python
dv = DataView()
dv.load_dataview('/home/bliu/pytrade_dir/ipynb/prepared/demo')
## dv_12_17
```

    Dataview loaded successfully.


### 单信号测试 `SignalDigger`模块

#### 功能

- 收益分析：分组收益、加权组合收益等
- 相关性分析：每日IC、IC分布等
- 排除涨跌停、停牌、非指数成分等

#### 特性
- 计算与绘图分离
- 绘图输出格式可选、可关闭，数据计算结果可返回


#### 测试量价背离因子

- **输入**：两个`DataFrame`：因子值，标的价格/收益
- **设置**：period，quantile个数

```python
factor = -dv.get_ts('price_volume_divert').shift(1, axis=0)  # avoid look-ahead bias
price = dv.get_ts('close_adj')
price_bench = dv.data_benchmark

my_period = 5
obj = signaldigger.digger.SignalDigger(output_folder='.', output_format='plot')
obj.process_factor_before_analysis(factor, price=price,
                                   mask=mask_all,
                                   n_quantiles=5, period=my_period,
                                   benchmark_price=price_bench,
                                   )
res = obj.create_full_report()
```

![](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/returns_report.png)



![](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/ic_report.png)


#### 利用输出数据做进一步分析


```python
def performance(ret):
    cum = ret.add(1.0).cumprod(axis=0)
    std = np.std(ret)
    
    start = pd.to_datetime(ser.index[0], format="%Y%m%d")
    end = pd.to_datetime(ser.index[-1], format="%Y%m%d")
    years = (end - start).days / 365.0

    yearly_return = np.power(cum.values[-1], 1. / years) - 1
    yearly_vol = std * np.sqrt(225.)
    # beta = np.corrcoef(df_returns.loc[:, 'bench'], df_returns.loc[:, 'strat'])[0, 1]
    sharpe = yearly_return / yearly_vol
    print "ann. ret = {:.1f}%; ann. vol = {:.1f}%, sharpe = {:.2f}".format(yearly_return*100, yearly_vol*100, sharpe)
    
```


```python
ser = res['quantile_active_ret_correct'][1]['mean']#.iloc[90:]
print ser.index[0], ser.index[-1]
plt.figure(figsize=(14, 5))
plt.plot(ser.add(1.0).cumprod().values)
performance(ser)
```

    20160105 20171013
    ann. ret = -17.2%; ann. vol = 3.7%, sharpe = -4.63



![](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/further_analysis.png)


### 回测

这里指**基于权重调仓**的Alpha策略回测，支持自定义**选股**和自定义**信号**。

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



![](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/analyze.png)

