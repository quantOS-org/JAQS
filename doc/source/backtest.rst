回测
----

JAQS支持\ **Alpha选股策略**\ 和\ **事件驱动择时策略**\ ，两种策略使用不同方法回测。

对于入门用户，推荐首先查看\ `快速入门 <https://github.com/quantOS-org/quantOSUserGuide>`__\ ，关于JAQS策略系统的介绍，见\ `这里 <https://github.com/quantOS-org/quantOSUserGuide/blob/master/jaqs.md>`__\ 。本文在以上教程的基础上，举出更多策略样例。

***注***\ ：本文所用例子的完整代码见\ `这里 <https://github.com/quantOS-org/JAQS/tree/master/example>`__\ ，安装JAQS后即可直接运行。\ **请勿直接复制下方代码运行**\ 。

格雷厄姆选股策略
~~~~~~~~~~~~~~~~

本策略完整实现代码见
`这里 <https://github.com/quantOS-org/JAQS/blob/master/example/alpha/Graham.py>`__\ 。

主要介绍基于回测框架实现格雷厄姆模型。格雷厄姆模型分为两步，首先是条件选股，其次按照市值从小到大排序，选出排名前五的股票。

一. 数据准备
^^^^^^^^^^^^

我们选择如下指标，对全市场的股票进行筛选，实现过程如下：

a.
首先在数据准备模块save\_dataview()中通过props设置数据起止日期，股票版块，以及所需变量

.. code:: python

    props = {
    'start_date': 20150101,
    'end_date': 20170930,
    'universe':'000905.SH',
    'fields': ('tot_cur_assets,tot_cur_liab,inventories,pre_pay,deferred_exp, eps_basic,ebit,pe,pb,float_mv,sw1'),
    'freq': 1
    }

b.
接着创建0-1变量表示某只股票是否被选中，并通过add\_formula将变量添加到dataview中

    -  市盈率（pe ratio）低于 20
    -  市净率（pb ratio）低于 2
    -  同比每股收益增长率（inc\_earning\_per\_share）大于 0
    -  税前同比利润增长率（inc\_profit\_before\_tax）大于 0
    -  流动比率（current\_ratio）大于 2
    -  速动比率（quick\_ratio）大于 1

.. code:: python

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

需要注意的是，涉及到的财务数据若不在secDailyIndicator表中，需将is\_quarterly设置为True，表示该变量为季度数据。

c. 由于第二步中需要按流通市值排序，我们将这一变量也放入dataview中

.. code:: python

    dv.add_formula('mv_rank', 'Rank(float_mv)', is_quarterly=False)

二. 条件选股
^^^^^^^^^^^^

条件选股在my\_selector函数中完成：

    -  首先我们将上一步计算出的0/1变量提取出来，格式为Series
    -  接着我们对所有变量取交集，选中的股票设为1，未选中的设为0，并将结果通过DataFrame形式返回

.. code:: python

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

三、按市值排序
^^^^^^^^^^^^^^

按市值排序功能在signal\_size函数中完成。我们根据流通市值排序变量'mv\_rank'对所有股票进行排序，并选出市值最小的5只股票。

.. code:: python

    def signal_size(context, user_options = None):
        mv_rank = context.snapshot_sub['mv_rank']
        s = np.sort(mv_rank.values)[::-1]
        if len(s) > 0:
            critical = s[-5] if len(s) > 5 else np.min(s)
            mask = mv_rank < critical
            mv_rank[mask] = 0.0
            mv_rank[~mask] = 1.0
        return mv_rank

四、回测
^^^^^^^^

我们在test\_alpha\_strategy\_dataview()模块中实现回测功能

1. 载入dataview，设置回测参数
'''''''''''''''''''''''''''''

该模块首先载入dataview并允许用户设置回测参数，比如基准指数，起止日期，换仓周期等。

.. code:: python

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

2. StockSelector选股模块
''''''''''''''''''''''''

接着我们使用StockSelector选股模块，将之前定义的my\_selector载入

.. code:: python

    stock_selector = model.StockSelector
    stock_selector.add_filter(name='myselector', func=my_selector)

3. FactorSignalModel模块
''''''''''''''''''''''''

在进行条件选股后，使用FactorSignalModel模块对所选股票进行排序

.. code:: python

    signal_model = model.FactorSignalModel(context)
    signal_model.add_signal(name='signalsize', func = signal_size)

4. 策略回测模块
'''''''''''''''

将上面定义的stockSelector和FactorSignalModel载入AlphaStrategy函数进行回测

.. code:: python

        strategy = AlphaStrategy(
                    stock_selector=stock_selector,
                    signal_model=signal_model，
                    pc_method='factor_value_weight')

5. 启动数据准备及回测模块
'''''''''''''''''''''''''

.. code:: python

    t_start = time.time()

    test_save_dataview()
    test_alpha_strategy_dataview()
    test_backtest_analyze()

    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)

五、回测结果
^^^^^^^^^^^^

回测的参数如下：

+---------------------+--------+
| 指标                | 值     |
+=====================+========+
| Beta                | 0.87   |
+---------------------+--------+
| Annual Return       | 0.08   |
+---------------------+--------+
| Annual Volatility   | 0.29   |
+---------------------+--------+
| Sharpe Ratio        | 0.28   |
+---------------------+--------+

回测的净值曲线图如下：

|backtestgraham|

基于因子IC的多因子选股模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

本策略完整实现代码见
`这里 <https://github.com/quantOS-org/JAQS/blob/master/example/alpha/ICCombine.py>`__\ 。

主要介绍基于回测框架实现基于因子IC的因子权重优化模型。

一. 因子IC定义及优化模型
^^^^^^^^^^^^^^^^^^^^^^^^

1. 因子IC的定义方法
'''''''''''''''''''

| 首先介绍一下因子IC (Information
Coefficient)的定义。传统意义上，因子在某一期的IC为该期因子与股票下期收益率的秩相关系数，即：
| $$IC\_t = RankCorrelation(\\vec{f\_t}, \\vec{r\_{t+1}})$$
| 其中$\\vec{f\_t}$为所有股票在t期的因子值向量，$\\vec{r\_{t+1}}$为所有股票在t到t+1期的收益率向量。秩相关系数直接反映了因子的预测能力：IC越高，说明该因子对接下里一期股票收益的预测能力越强。

2. 因子的获取及计算方法
'''''''''''''''''''''''

在本示例中我们简单选取了几个因子，更多的因子可以在股票因子数据中找到：

    -  Turnover, 换手率
    -  BP, Book-to-Market Ratio
    -  MOM20, 过去20天收益率
    -  LFMV, 对数流通市值

实现过程如下：

a.
首先在数据准备模块save\_dataview()中通过props设置数据起止日期，股票版块，以及所需变量

.. code:: python

    props = {'start_date': 20150101, 'end_date': 20170930, 'universe':
    '000905.SH', 'fields': ('turnover,float_mv,close_adj,pe'), 'freq': 1}

b.
接着计算因子，进行标准化和去极值处理后通过add\_formula()将因子添加到变量列表中

.. code:: python

    factor_formula = 'Cutoff(Standardize(turnover / 10000 / float_mv), 2)'
    dv.add_formula('TO', factor_formula, is_quarterly=False)

    factor_formula = 'Cutoff(Standardize(1/pb), 2)'
    dv.add_formula('BP', factor_formula, is_quarterly = False)

    factor_formula = 'Cutoff(Standardize(Return(close_adj, 20)), 2)'
    dv.add_formula('REVS20', factor_formula, is_quarterly=False)

    factor_formula = 'Cutoff(Standardize(Log(float_mv)), 2)'
    dv.add_formula('float_mv_factor', factor_formula, is_quarterly=False)

| 其中Standardize()和Cutoff()均为内置函数。Standardize作用是将序列做去均值并除以标准差的标准化处理，Cutoff作用是将序列中的极值拉回正常范围内。
| 之后将因子名称保存在外部文件中，以便后续计算使用

.. code:: python

    factorList = ['TO', 'BP', 'REVS20', 'float_mv_factor']
    factorList_adj = [x + '_adj' for x in factorList]
    from jaqs.util import fileio
    fileio.save_json(factorList_adj, '.../myCustomData.json')

c.
由于多个因子间可能存在多重共线性，我们对因子进行施密特正交化处理，并将处理后的因子添加到变量列表中。

.. code:: python

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

3. 计算因子IC
'''''''''''''

从dataview中提取所有交易日，在每个交易日计算每个因子的IC

.. code:: python

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

其中计算IC的函数为ic\_calculation()

.. code:: python

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

4. 因子权重优化
'''''''''''''''

| 我们将因子IR设为因子权重优化的目标，因子IR（信息比）定义为因子IC的均值与因子IC的标准差的比值，IR值越高越好。假设我们有k个因子，其IC的均值向量为$\\vec{IC}=(\\overline{IC\_1},
\\overline{IC\_2}, \\cdots,
\\overline{IC\_k},)'$，相应协方差矩阵为$\\Sigma$，因子的权重向量为$\\vec{v}=(\\overline{V\_1},
\\overline{V\_2},\\cdots, \\overline{V\_k})'$。则所有因子的复合IR值为
| $$IR = \\frac{\\vec{v}'\\vec{IC}}{\\sqrt{\\vec{v}' \\Sigma
\\vec{v}}}$$
| 我们的目标是通过调整$\\vec{v}$使IR最大化。经简单计算我们可以直接求出$\\vec{v}$的解析解，则最优权重向量为：
| $$\\vec{v}^\* = \\Sigma^{-1}\\vec{IC}$$
| 具体实现过程如下：

.. code:: python

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

其中使用到了get\_ic\_weight()函数，其作用是计算每个因子IC对应的weight

.. code:: python

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

我们在计算weight时需要确定一个rolling window，这里选择N=10。

.. code:: python

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

二. 基于因子IC及相应权重的选股模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

在介绍选股模型的具体实现之前，我们首先熟悉一下策略模块test\_alpha\_strategy\_dataview()。该模块的功能是基于dataview对具体策略进行回测。

1. 载入dataview，设置回测参数
'''''''''''''''''''''''''''''

该模块首先载入dataview并允许用户设置回测参数，比如基准指数，起止日期，换仓周期等。

.. code:: python

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

2. 载入context
''''''''''''''

context是一个类用来保存一些中间结果，可在程序中任意位置调用，并将之前算出的ic\_weight放入context中。

.. code:: python

    context = model.Context(dataview=dv, gateway=gateway)
    store = pd.HDFStore('.../ic_weight.hd5')
    context.ic_weight = store['ic_weight']
    store.close()

3. StockSelector选股模块
''''''''''''''''''''''''

接着我们使用StockSelector选股模块。基于因子IC及相应权重的选股过程在my\_selector中实现。

.. code:: python

    stock_selector = model.StockSelector(context)
    stock_selector.add_filter(name='myselector', func=my_selector)

a.首先载入因子ic的权重context.ic\_weight，回测日期列表context.trade\_date记忆因子名称列表factorList

.. code:: python

    ic_weight = context.ic_weight
    t_date = context.trade_date
    current_ic_weight = np.mat(ic_weight.loc[t_date,]).reshape(-1,1)
    factorList = fileio.read_json('.../myCustomData.json')

    factorPanel = {}
    for factor in factorList:
        factorPanel[factor] = context.snapshot[factor]

    factorPanel = pd.DataFrame(factorPanel)

b.接着根据各因子IC的权重，对当天各股票的IC值进行加权求和，选出得分最高的前30只股票。最后返回一个列表，1代表选中，0代表未选中。

.. code:: python

    factorResult = pd.DataFrame(np.mat(factorPanel) * np.mat(current_ic_weight), index = factorPanel.index)

    factorResult = factorResult.fillna(-9999)
    s = factorResult.sort_values(0)[::-1]

    critical = s.values[30]
    mask = factorResult > critical
    factorResult[mask] = 1.0
    factorResult[~mask] = 0.0

4. 启动数据准备及回测模块
'''''''''''''''''''''''''

.. code:: python

    t_start = time.time()

    test_save_dataview()
    store_ic_weight()
    test_alpha_strategy_dataview()
    test_backtest_analyze()

    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)

三、回测结果
^^^^^^^^^^^^

回测的参数如下：

+---------------------+--------+
| 指标                | 值     |
+=====================+========+
| Beta                | 0.92   |
+---------------------+--------+
| Annual Return       | 0.19   |
+---------------------+--------+
| Annual Volatility   | 0.16   |
+---------------------+--------+
| Sharpe Ratio        | 1.21   |
+---------------------+--------+

| 回测的净值曲线图如下：
| |backtesticmodel|

四、参考文献
^^^^^^^^^^^^

#. `基于因子IC的多因子模型 <https://uqer.io/community/share/57b540ef228e5b79a4759398>`__
#. 《安信证券－多因子系列报告之一：基于因子IC的多因子模型》

Calendar Spread交易策略
~~~~~~~~~~~~~~~~~~~~~~~

本策略完整实现代码见
`这里 <https://github.com/quantOS-org/JAQS/blob/master/example/alpha/CalendarSpread.py>`__\ 。

本帖主要介绍了基于事件驱动回测框架实现calendar spread交易策略。

一. 策略介绍
^^^^^^^^^^^^

| 在商品期货市场中，同一期货品种不同到期月份合约间的价格在短期内的相关性较稳定。该策略就利用这一特性，在跨期基差稳定上升时进场做多基差，反之做空基差。
| 在本文中我们选择了天然橡胶作为交易品种，时间范围从2017年7月到2017年11月，选择的合约为RU1801.SHF和RU1805.SHF，将基差定义为近期合约价格减去远期合约价格。

二. 参数准备
^^^^^^^^^^^^

我们在test\_spread\_commodity.py文件中的test\_spread\_trading()函数中设置策略所需参数，例如交易标的，策略开始日期，终止日期，换仓频率等。

.. code:: python

    props = {
             "symbol"                : "ru1801.SHF,ru1805.SHF",
             "start_date"            : 20170701,
             "end_date"              : 20171109,
             "bar_type"              : "DAILY",
             "init_balance"          : 2e4,
             "bufferSize"            : 20,
             "future_commission_rate": 0.00002,
             "stock_commission_rate" : 0.0001,
             "stock_tax_rate"        : 0.0000
             }

三. 策略实现
^^^^^^^^^^^^

策略实现全部在spread\_commodity.py中完成，创建名为SpreadCommodity()的class继承EventDrivenStrategy，具体分为以下几个步骤：

1. 策略初始化
'''''''''''''

这里将后续步骤所需要的变量都创建好并初始化。

.. code:: python

    def __init__(self):
        EventDrivenStrategy.__init__(self)

        self.symbol      = ''
        self.s1          = ''
        self.s2          = ''
        self.quote1      = None
        self.quote2      = None

        self.bufferSize  = 0
        self.bufferCount = 0
        self.spreadList  = ''

2. 从props中得到变量值
''''''''''''''''''''''

这里将props中设置的参数传入。其中，self.spreadList记录了最近$n$天的spread值，$n$是由self.bufferSize确定的。

.. code:: python

    def init_from_config(self, props):
        super(SpreadCommodity, self).init_from_config(props)
        self.symbol       = props.get('symbol')
        self.init_balance = props.get('init_balance')
        self.bufferSize   = props.get('bufferSize')
        self.s1, self.s2  = self.symbol.split(',')
        self.spreadList = np.zeros(self.bufferSize)

3. 策略实现
'''''''''''

| 策略的主体部分在on\_bar()函数中实现。因为我们选择每日调仓，所以会在每天调用on\_bar()函数。
| 首先将两个合约的quote放入self.quote1和self.quote2中，并计算当天的spread

.. code:: python

    q1 = quote_dic.get(self.s1)
    q2 = quote_dic.get(self.s2)
    self.quote1 = q1
    self.quote2 = q2
    spread = q1.close - q2.close

接着更新self.spreadList。因为self.spreadList为固定长度，更新方法为将第2个到最后1个元素向左平移1位，并将当前的spread放在队列末尾。

.. code:: python

    self.spreadList[0:self.bufferSize - 1] = self.spreadList[1:self.bufferSize]
    self.spreadList[-1] = spread
    self.bufferCount += 1

接着将self.spreadList中的数据对其对应的编号（例如从1到20）做regression，观察回归系数的pvalue是否显著，比如小于0.05。如果结果不显著，则不对仓位进行操作；如果结果显著，再判断系数符号，如果系数大于0则做多spread，反之做空spread。

.. code:: python

    X, y = np.array(range(self.bufferSize)), np.array(self.spreadList)
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X = sm.add_constant(X)

    est = sm.OLS(y, X)
    est = est.fit()

    if est.pvalues[1] < 0.05:
        if est.params[1] < 0:
            self.short_spread(q1, q2)
        else:
            self.long_spread(q1, q2)

四. 回测结果
^^^^^^^^^^^^

|calendarspreadresult|

商品期货的Dual Thrust日内交易策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本策略完整实现代码见
`这里 <https://github.com/quantOS-org/JAQS/blob/master/example/alpha/DualThrust.py>`__\ 。

本帖主要介绍了基于事件驱动回测框架实现Dual Thrust日内交易策略。

一. 策略介绍
^^^^^^^^^^^^

| Dual
Thrust是一个趋势跟踪策略，具有简单易用、适用度广的特点，其思路简单、参数较少，配合不同的参数、止盈止损和仓位管理，可以为投资者带来长期稳定的收益，被投资者广泛应用于股票、货币、贵金属、债券、能源及股指期货市场等。
| 在本文中，我们将Dual Thrust应用于商品期货市场中。
| 简而言之，该策略的逻辑原型是较为常见的开盘区间突破策略，以今日开盘价加减一定比例确定上下轨。日内突破上轨时平空做多，突破下轨时平多做空。
| 在Dual
Thrust交易系统中，对于震荡区间的定义非常关键，这也是该交易系统的核心和精髓。Dual
Thrust系统使用
| $$Range = Max(HH-LC,HC-LL)$$
| 来描述震荡区间的大小。其中HH是过去N日High的最大值，LC是N日Close的最小值，HC是N日Close的最大值，LL是N日Low的最小值。

二. 参数准备
^^^^^^^^^^^^

我们在test\_spread\_commodity.py文件中的test\_spread\_trading()函数中设置策略所需参数，例如交易标的，策略开始日期，终止日期，换仓频率等，其中$k1，k2$为确定突破区间上下限的参数。

.. code:: python

    props = {
             "symbol"                : "rb1710.SHF",
             "start_date"            : 20170510,
             "end_date"              : 20170930,
             "buffersize"            : 2,
             "k1"                    : 0.7,
             "k2"                    : 0.7,
             "bar_type"              : "MIN",
             "init_balance"          : 1e5,
             "future_commission_rate": 0.00002,
             "stock_commission_rate" : 0.0001,
             "stock_tax_rate"        : 0.0000
             }

三. 策略实现
^^^^^^^^^^^^

策略实现全部在DualThrust.py中完成，创建名为DualThrustStrategy()的class继承EventDrivenStrategy，具体分为以下几个步骤：

1. 策略初始化
'''''''''''''

这里将后续步骤所需要的变量都创建好并初始化。其中self.bufferSize为窗口期长度，self.pos记录了实时仓位，self.Upper和self.Lower记录了突破区间上下限。

.. code:: python

    def __init__(self):
        EventDrivenStrategy.__init__(self)
        self.symbol      = ''
        self.quote       = None
        self.bufferCount = 0
        self.bufferSize  = ''
        self.high_list   = ''
        self.close_list  = ''
        self.low_list    = ''
        self.open_list   = ''
        self.k1          = ''
        self.k2          = ''
        self.pos         = 0
        self.Upper       = 0.0
        self.Lower       = 0.0

2. 从props中得到变量值
''''''''''''''''''''''

这里将props中设置的参数传入。其中，self.high\_list为固定长度的list，保存了最近$N$天的日最高价，其他变量类似。

.. code:: python

    def init_from_config(self, props):
        super(DualThrustStrategy, self).init_from_config(props)

        self.symbol       = props.get('symbol')
        self.init_balance = props.get('init_balance')
        self.bufferSize   = props.get('buffersize')
        self.k1           = props.get('k1')
        self.k2           = props.get('k2')
        self.high_list    = np.zeros(self.bufferSize)
        self.close_list   = np.zeros(self.bufferSize)
        self.low_list     = np.zeros(self.bufferSize)
        self.open_list    = np.zeros(self.bufferSize)

3. 策略实现
'''''''''''

在每天开始时，首先调用initialize()函数，得到当天的open，close，high和low的值，并对应放入list中。

.. code:: python

    def initialize(self):
        self.bufferCount += 1

        # get the trading date
        td = self.ctx.trade_date
        ds = self.ctx.data_api

        # get the daily data
        df, msg = ds.daily(symbol=self.symbol, start_date=td, end_date=td)

        # put the daily value into the corresponding list
        self.open_list[0:self.bufferSize - 1] =
                       self.open_list[1:self.bufferSize]
        self.open_list[-1] = df.high
        self.high_list[0:self.bufferSize - 1] =
                       self.high_list[1:self.bufferSize]
        self.high_list[-1] = df.high
        self.close_list[0:self.bufferSize - 1] =
                       self.close_list[1:self.bufferSize]
        self.close_list[-1] = df.close
        self.low_list[0:self.bufferSize - 1] =
                       self.low_list[1:self.bufferSize]
        self.low_list[-1] = df.low

策略的主体部分在on\_bar()函数中实现。因为我们选择分钟级回测，所以会在每分钟调用on\_bar()函数。

首先取到当日的quote，并计算过去$N$天的HH，HC，LC和LL，并据此计算Range和上下限Upper，Lower

.. code:: python

    HH = max(self.high_list[:-1])
    HC = max(self.close_list[:-1])
    LC = min(self.close_list[:-1])
    LL = min(self.low_list[:-1])

    Range = max(HH - LC, HC - LL)
    Upper = self.open_list[-1] + self.k1 * Range
    Lower = self.open_list[-1] - self.k2 * Range

| 几个关键变量的意义如下图所示：
| |illustrationdual|

我们的交易时间段为早上9:01:00到下午14:28:00,交易的逻辑为：

#. 当分钟Bar的open向上突破上轨时，如果当时持有空单，则先平仓，再开多单；如果没有仓位，则直接开多单；
#. 当分钟Bar的open向下突破下轨时，如果当时持有多单，则先平仓，再开空单；如果没有仓位，则直接开空单；

   .. code:: python

       if self.pos == 0:
           if self.quote.open > Upper:
               self.short(self.quote, self.quote.close, 1)
           elif self.quote.open < Lower:
               self.buy(self.quote, self.quote.close, 1)
       elif self.pos < 0:
           if self.quote.open < Lower:
               self.cover(self.quote, self.quote.close, 1)
               self.long(self.quote, self.quote.close, 1)
       else:
           if self.quote.open > Upper:
               self.sell(self.quote, self.quote.close, 1)
               self.short(self.quote, self.quote.close, 1)

   由于我们限制该策略为日内策略，故当交易时间超过14:28:00时，进行强行平仓。

   .. code:: python

       elif self.quote.time > 142800:
           if self.pos > 0:
               self.sell(self.quote, self.quote.close, 1)
           elif self.pos < 0:
               self.cover(self.quote, self.quote.close, 1)

   我们在下单后，可能由于市场剧烈变动导致未成交，因此在on\_trade\_ind()函数中记录具体成交情况，当空单成交时，self.pos减一，当多单成交时，self.pos加一。

   .. code:: python

       def on_trade_ind(self, ind):
           if ind.entrust_action == 'sell' or ind.entrust_action == 'short':
               self.pos -= 1
           elif ind.entrust_action == 'buy' or ind.entrust_action == 'cover':
               self.pos += 1
           print(ind)

四. 回测结果
^^^^^^^^^^^^

| 回测结果如下图所示：
| |dualthrustresult|

五、参考文献
^^^^^^^^^^^^

版块内股票轮动策略
~~~~~~~~~~~~~~~~~~

本策略完整实现代码见
`这里 <https://github.com/quantOS-org/JAQS/blob/master/example/alpha/SectorRolling.py>`__\ 。

本帖主要介绍了基于事件驱动回测框架实现版块内股票轮动策略。

一. 策略介绍
^^^^^^^^^^^^

| 该轮动策略如下：在策略开始执行时等价值买入版块内所有股票，每天 $t$
计算各股在过去$m$天相对板块指数的收益率
| $$R^A\_{i,t} =
(lnP\_{i,t}-lnP\_{i,t-m}）-（lnP\_{B,t}-lnP\_{B,t-m}）$$
| 其中$P\_{i,t}$为股票$i$在$t$天的收盘价，$P\_{B,t}$为板块指数在$t$天的收盘价。每天检查持仓，若持仓股$R^A\_{i,t}$超过过去$n$天均值加$k$倍标准差，则卖出；反之，若有未持仓股$R^A\_{i,t}$小于过去$n$天均值减$k$倍标准差，则买入。

二. 参数准备
^^^^^^^^^^^^

我们在test\_roll\_trading.py文件中的test\_strategy()函数中设置策略所需参数。首先确定策略开始日期，终止日期以及板块指数。在本文中，我们选择券商指数399975.SZ，并听过data\_service得到该指数中所有成份股。

.. code:: python

    start_date = 20150901
    end_date = 20171030
    index = '399975.SZ'
    data_service = RemoteDataService()
    symbol_list = data_service.get_index_comp(index, start_date, start_date)

接着在props中设置参数

.. code:: python

    symbol_list.append(index)
    props = {"symbol": ','.join(symbol_list),
             "start_date": start_date,
             "end_date": end_date,
             "bar_type": "DAILY",
             "init_balance": 1e7,
             "std multiplier": 1.5,
             "m": 10,
             "n": 60,
             "future_commission_rate": 0.00002,
             "stock_commission_rate": 0.0001,
             "stock_tax_rate": 0.0000}

我们可以在bar\_type中设置换仓周期，现在支持分钟和日换仓，本例中选择每日调仓。

三. 策略实现
^^^^^^^^^^^^

策略实现全部在roll.py中完成，创建名为RollStrategy()的class继承EventDrivenStrategy，具体分为以下几个步骤：

1. 策略初始化
'''''''''''''

这里将后续步骤所需要的变量都创建好并初始化。

.. code:: python

    def __init__(self):
        EventDrivenStrategy.__init__(self)
        self.symbol = ''
        self.benchmark_symbol = ''
        self.quotelist = ''
        self.startdate = ''
        self.bufferSize = 0
        self.rollingWindow = 0
        self.bufferCount = 0
        self.bufferCount2 = 0
        self.closeArray = {}
        self.activeReturnArray = {}
        self.std = ''
        self.balance = ''
        self.multiplier = 1.0
        self.std_multiplier = 0.0

2. 从props中得到变量值
''''''''''''''''''''''

这里将props中设置的参数传入。其中，self.closeArray和self.activeReturnArray数据类型为dict，key为股票代码，value分别为最近$m$天的收盘价和最近$n$天的active
return。

.. code:: python

    def init_from_config(self, props):
        super(RollStrategy, self).init_from_config(props)
        self.symbol = props.get('symbol').split(',')
        self.init_balance = props.get('init_balance')
        self.startdate = props.get('start_date')
        self.std_multiplier = props.get('std multiplier')
        self.bufferSize = props.get('n')
        self.rollingWindow = props.get('m')
        self.benchmark_symbol = self.symbol[-1]
        self.balance = self.init_balance

        for s in self.symbol:
            self.closeArray[s] = np.zeros(self.rollingWindow)
            self.activeReturnArray[s] = np.zeros(self.bufferSize)

3. 策略实现
'''''''''''

| 策略的主体部分在on\_bar()函数中实现。因为我们选择每日调仓，所以会在每天调用on\_bar()函数。
| 首先将版块内所有股票的quote放入self.quotelist中，

.. code:: python

    self.quotelist = []
    for s in self.symbol:
        self.quotelist.append(quote_dic.get(s))

接着对每只股票更新self.closeArray。因为self.closeArray为固定长度，更新方法为将第2个到最后1个元素向左平移1位，并将当前quote中最新的close放在末尾。

.. code:: python

    for stock in self.quotelist:
        self.closeArray[stock.symbol][0:self.rollingWindow - 1] =  self.closeArray[stock.symbol][1:self.rollingWindow]
        self.closeArray[stock.symbol][-1] = stock.close

计算每只股票在过去$m$天的active return，存入self.activeReturnArray。

.. code:: python

    ### calculate active return for each stock
    benchmarkReturn = np.log(self.closeArray[self.benchmark_symbol][-1])
                     -np.log(self.closeArray[self.benchmark_symbol][0])
    for stock in self.quotelist:
        stockReturn = np.log(self.closeArray[stock.symbol][-1])
                     -np.log(self.closeArray[stock.symbol][0])
        activeReturn = stockReturn - benchmarkReturn
        self.activeReturnArray[stock.symbol][0:self.bufferSize - 1]
                     = self.activeReturnArray[stock.symbol][1:self.bufferSize]
        self.activeReturnArray[stock.symbol][-1] = activeReturn

在策略首次执行时，默认等价值持有版块中所有的股票。

.. code:: python

    ### On the first day of strategy, buy in equal value stock in the universe
    stockvalue = self.balance/len(self.symbol)
    for stock in self.quotelist:
        if stock.symbol != self.benchmark_symbol:
            self.buy(stock, stock.close,
                     np.floor(stockvalue/stock.close/self.multiplier))

在其他日期，当策略开始执行时，首先通过self.pm.holding\_securities检查持有的股票代码，并与版块成分比较确定未持有的股票代码。

.. code:: python

    stockholdings = self.pm.holding_securities
    noholdings = set(self.symbol) - stockholdings
    stockvalue = self.balance/len(noholdings)

对于已持有的股票，计算最近$m$天的active
return，若超过self.activeReturnArray均值的一定范围，就将该股票卖出。

.. code:: python

    for stock in list(stockholdings):
        curRet = self.activeReturnArray[stock][-1]
        avgRet = np.mean(self.activeReturnArray[stock][:-1])
        stdRet = np.std(self.activeReturnArray[stock][:-1])
        if curRet >= avgRet + self.std_multiplier * stdRet:
            curPosition = self.pm.positions[stock].curr_size
            stock_quote = quote_dic.get(stock)
            self.sell(stock_quote, stock_quote.close, curPosition)

反之，对于未持有的股票，若其active
return低于均值的一定范围，就将其买入。

.. code:: python

    for stock in list(noholdings):
        curRet = self.activeReturnArray[stock][-1]
        avgRet = np.mean(self.activeReturnArray[stock][:-1])
        stdRet = np.std(self.activeReturnArray[stock][:-1])
        if curRet < avgRet - self.std_multiplier * stdRet:
            stock_quote = quote_dic.get(stock)
            self.buy(stock_quote, stock_quote.close,
                     np.floor(stockvalue/stock_quote.close/self.multiplier))

此外，我们在框架中on\_trade\_ind()中实现了仓位管理。在策略初始化时，我们将组合中的现金设为初始资金。

.. code:: python

    self.init_balance = props.get('init_balance')
    self.balance = self.init_balance

此后，每买入一只股票，我们将self.balance减去相应市值；每卖出一只股票，将self.balance加上相应市值。

.. code:: python

    def on_trade_ind(self, ind):
        if ind.entrust_action == 'buy':
            self.balance -= ind.fill_price * ind.fill_size * self.multiplier
        elif ind.entrust_action == 'sell':
            self.balance += ind.fill_price * ind.fill_size * self.multiplier
        print(ind)

四. 回测结果
^^^^^^^^^^^^

| 该策略的回测结果如下图所示：
| |rollwithinsectorresult|

| 回测的参数如下：
| \| 指标 \| 值 \|
| \| -------- \| --: \|
| \| Beta \| 0.70 \|
| \| Annual Return \| 0.05 \|
| \| Annual Volatility\| 0.17 \|
| \| Sharpe Ratio \| 0.29 \|

.. |backtestgraham| image:: https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/backtest_Graham_result.png
.. |backtesticmodel| image:: https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/backtest_ICModel_result.png
.. |calendarspreadresult| image:: https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/event_driven_calendar_spread_result.png
.. |illustrationdual| image:: https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/event_drivent_illustration_dual.png
.. |dualthrustresult| image:: https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/event_drivent_dual_thrust_result.png
.. |rollwithinsectorresult| image:: https://raw.githubusercontent.com/quantOS-org/jaqs/master/doc/img/event_driven_roll_within_sector_result.png
