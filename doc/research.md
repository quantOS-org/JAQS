## 研究

信号研究与回测: `SignalDigger`模块

### 功能

- 收益分析：分组收益、加权组合收益等
- 相关性分析：每日IC、IC分布等
- 排除涨跌停、停牌、非指数成分等

### 特性
- 计算与绘图分离
- 绘图输出格式可选、可关闭，数据计算结果可返回

完整代码及样例见[这里](https://github.com/quantOS-org/JAQS/blob/release-0.6.0/example/research/signal_return_ic_analysis.py)，安装JAQS后即可直接运行。**请勿直接复制下方代码运行**。

### 测试量价背离因子

- **输入**：两个`DataFrame`：因子值，标的价格/收益
- **设置**：period，quantile个数

```python
factor = -dv.get_ts('price_volume_divert').shift(1, axis=0)  # avoid look-ahead bias
price = dv.get_ts('close_adj')
price_bench = dv.data_benchmark

my_period = 5
obj = SignalDigger(output_folder='.', output_format='plot')
obj.process_signal_before_analysis(factor, price=price,
                                   mask=mask_all,
                                   n_quantiles=5, period=my_period,
                                   benchmark_price=price_bench,
                                   )
res = obj.create_full_report()
```

![returnsreport](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/returns_report.png)



![icreport](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/ic_report.png)


### 利用输出数据做进一步分析


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


![furtheranalysis](https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/further_analysis.png)
