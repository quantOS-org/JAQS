quantOS量化策略样例(1)：选股
============================

本文主要内容：

-  用\ **JAQS的股票Alpha框架**\ 做股票策略回测；
-  用\ **TradeSim**\ 在线仿真交易。

代码量：<100行；预计阅读时间：5分钟。

完整代码及样例见\ `这里 <https://github.com/quantOS-org/JAQS/blob/master/example/alpha/first_example.py>`__\ ，安装JAQS后即可直接运行（安装教程，参见
`JAQS安装指南 <https://github.com/quantOS-org/JAQS/blob/master/doc/install.md>`__\ ）。

策略描述
--------

-  每月调仓一次；
-  每次调仓时，将食品饮料行业（中证申万食品饮料指数）的所有成份股按照市值权重构建目标投资组合，依照此目标进行买卖；
-  初始资金1亿，回测期间收益均再投资；
-  策略目标是获得比沪深300更好的收益。

策略回测和结果分析
------------------

我们使用\ `JAQS <https://www.quantos.org/jaqs/index.html>`__\ 来完成数据获取和回测。关于JAQS的介绍，见本文最后部分。

策略实现
~~~~~~~~

实现并回测该策略的代码可分为两部分：\ **准备数据**\ 和\ **利用准备好的数据进行回测**\ ，这是因为同样的策略可以在不同数据上测试，同样的数据也可以用于不同数据的回测。JAQS也依照这一理念进行了模块化设计，具体请看下方例子。

准备数据
^^^^^^^^

根据策略描述，我们需要以下数据：

-  投资范围(universe)：中证申万食品饮料指数
   ``000807.SH``\ 的所有成份股；
-  业绩比较基准(benchmark)：沪深300指数 ``000300.SH``\ ；
-  市值：投资范围内所有股票的市值；
-  起止时间：回测运行的时间段；
-  日行情：包括高开低收、成交量等，这是回测必然会用到的数据。

使用JAQS提供的\ ``DataView``\ 工具，可方便地获取以上数据：

.. code:: python

    dataview_props = {# Start and end date of back-test
                      'start_date': 20170101, 'end_date': 20171030,
                      # Investment universe and performance benchmark
                      'universe': UNIVERSE, 'benchmark': '000300.SH',
                      # Data fields that we need
                      'fields': 'total_mv,turnover',
                      # freq = 1 means we use daily data. Please do not change this.
                      'freq': 1}

    # RemoteDataService communicates with a remote server to fetch data
    ds = RemoteDataService()
    # Use username and password in data_config to login
    ds.init_from_config(data_config)

    # DataView utilizes RemoteDataService to get various data and store them
    dv = DataView()
    dv.init_from_config(dataview_props, ds)
    dv.prepare_data()
    dv.save_dataview(folder_path=dataview_store_folder)

运行后，在输出中看到：

.. code:: shell

    Dataview has been successfully saved to:
    'dataview_store_folder'

    You can load it with load_dataview('dataview_store_folder')

则说明\ ``DataView``\ 已经自动获取数据并存储在我们设置的存储路径\ ``dataview_store_folder``\ 里。

数据存储好后，可多次读取使用，无需再连接数据服务器。

策略回测
^^^^^^^^

首先需要读取保存好的数据文件：

.. code:: python

    # Load local data file that we just stored.
    dv = DataView()
    dv.load_dataview(folder_path=dataview_store_folder)

读取后会看到如下输出：

.. code:: shell

    Dataview loaded successfully.

接下来是策略部分。我们使用JAQS的股票Alpha策略框架\ ``AlphaStrategy``\ 、回测框架\ ``AlphaBacktestInstance``\ 实现。

框架允许用户任意指定投资范围内每只股票的权重，同时对于等权重、市值权重等常用情况，JAQS已经内置了相应函数，用户无需自己实现。

我们需要设置起止时间、初始资金等回测配置\ ``backtest_props``\ ，建立策略对象\ ``AlphaStrategy``\ ，回测实例对象\ ``AlphaBacktestInstance``\ 等。此外还需要建立运行上下文\ ``context``\ ，用于放置一些全局变量。

.. code:: python

    backtest_props = {# start and end date of back-test
                      "start_date": dv.start_date,
                      "end_date": dv.end_date,
                      # re-balance period length
                      "period": "month",
                      # benchmark and universe
                      "benchmark": dv.benchmark,
                      "universe": dv.universe,
                      # Amount of money at the start of back-test
                      "init_balance": 1e8}

    # This is our strategy
    strategy = AlphaStrategy(pc_method='market_value_weight')

    # BacktestInstance is in charge of running the back-test
    bt = AlphaBacktestInstance()

    # Public variables are stored in context. We can also store anything in it
    context = model.Context(dataview=dv, instance=bt, strategy=strategy, trade_api=trade_api, pm=pm)

准备好这些对象后，即可运行回测并储存结果：

.. code:: python

    bt.init_from_config(backtest_props)
    bt.run_alpha()

    # After finishing back-test, we save trade results into a folder
    bt.save_results(folder_path=backtest_result_folder)

回测过程中会实时输出回测进度及资金情况，如：

.. code:: shell

    AlphaStrategy Initialized.

    =======new day 20170103
    Before 20170103 re-balance: available cash all = 1.0000e+08

    =======new day 20170203
    Before 20170203 re-balance: available cash all = 1.0054e+08

回测完成后，会有如下提示：

.. code:: shell

    Backtest done. 240 days, 5.27e+02 trades in total.
    Backtest results has been successfully saved to:
    'backtest_result_folder'

即回测的结果（交易记录）和回测相关配置已成功存储在\ ``backtest_result_folder``\ 内。用户可自行查看，也可使用我们提供的分析工具进行分析，见下一节。

结果分析
~~~~~~~~

我们使用JAQS的\ ``Analyzer``\ 工具。分析时同样基于保存好的\ ``DataView``\ 和回测结果进行：

.. code:: python

    # Analyzer help us calculate various trade statistics according to trade results.
    # All the calculation results will be stored as its members.
    ta = ana.AlphaAnalyzer()
    ta.initialize(dataview=dv, file_folder=backtest_result_folder)

    ta.do_analyze(result_dir=backtest_result_folder,
                  selected_sec=list(ta.universe)[:3])

其中\ ``selected_sec``\ 参数是一个\ ``list``\ ，存放标的代码，其中存放的标的的买卖详情会绘制在回测报告中。

分析完成后，会生成HTML格式的回测报告，并输出报告所在路径。报告样例截图如下：

仿真交易
--------

若想将以上策略接入仿真交易，仍使用JAQS运行策略，最后用\ `TradeSim <https://www.quantos.org/tradesim/index.html>`__\ 进行仿真交易与撮合。无需修改策略代码，只需修改主程序

，只需更换\ ``AlphaTradeApi``\ 和\ ``AlphaBacktestInstance``\ ：

.. code:: python

    livetrade_props = {"period": "day",
                      "strategy_no": 1044,
                       "init_balance" 1e6}

    strategy = AlphaStrategy(pc_method='market_value_weight')

    bt = AlphaLiveTradeInstance()
    trade_api = RealTimeTradeApi(props)
    ds = RemoteDataService()

    context = model.Context(dataview=dv, instance=bt, strategy=strategy, trade_api=trade_api, pm=pm, data_api=ds)

    bt.init_from_config(props)
    bt.run_alpha()

以上代码中，我们将\ ``AlphaBacktestInstance``\ 更换为\ ``AlphaLiveTradeInstance``\ ，将\ ``AlphaTradeApi``\ 更换为\ ``RealTimeTradeApi``\ ，并使用\ ``RemoteDataService``\ 以获取最新数据。注意\ ``livetrade_props``\ 中需要填写\ ``strategy_no``\ 项，这是我们的策略号，不同用户不同。

运行\ ``run_alpha``\ 后，策略会根据最新数据产生目标投资组合，可从\ ``strategy.goal_positions``\ 中取出并发单：

.. code:: python

    goal_positions = strategy.goal_positions
    task_id, msg = trade_api.basket_order(goal_positions)

发单成功后，\ ``trade_api``\ 为该任务的编号，msg为返回信息。

具体订单、持仓、盈亏等可在\ `仿真交易网站 <https://www.quantos.org/tradesim/trade.html>`__\ 查看，也可使用\ `vnTrade <https://github.com/quantOS-org/TradeSim/tree/master/vnTrader>`__\ 客户端查看，如下图：

***注**\ ：以上只展示了部分核心代码段，\ **无法直接运行**\ 。完整、可运行代码可在\ `这里 <https://github.com/quantOS-org/JAQS/blob/master/example/alpha/first_example.py>`__\ 下载。*

附：JAQS简介
------------

策略开发的完整流程一般是以下四步的循环往复，JAQS在这四步都提供了支持：

#. **数据**\ 收集处理：我们提供了标准接口\ ``DataApi``,
   便利接口\ ``DataService``, 高效工具\ ``DataView``
#. 对数据进行\ **研究**\ ：我们提供进行信号/事件研究的\ ``SignalDigger``
#. 根据研究结果开发策略并\ **回测**\ ：我们提供两种策略回测框架，Alpha选股和事件驱动择时（如CTA、套利）
#. 对回测结果进行\ **分析**\ ：我们提供直观简洁的报告\ ``Report``\ ，以及分析内容丰富、可进一步开发的分析器\ ``Analyzer``

本文作为入门系列，主要围绕具体样例，介绍了回测部分，更多资料，参见\ `官方网站。 <https://www.quantos.org/jaqs/index.html>`__
