实盘交易
--------

使用\ **JAQS**\ 进行回测与实盘运行的代码具有高一致性，回测满意后，只需以下几点改动，即可接入实盘/模拟撮合：

#. 使用实盘交易的交易接口：将\ ``BacktestTradeApi``\ 替换为\ ``RealTimeTradeApi``
#. 使用实盘交易主程序：将\ ``EventBacktestInstance``\ 替换为\ ``EventLiveTradeInstance``
#. 在数据接口\ ``RemoteDataService``\ 中订阅所要交易品种的行情
#. 在主程序最后添加\ ``time.sleep(9999999)``.
   保证在事件循环运行中，主程序不会提前终止
#. 实时行情均为逐笔或Tick数据，即使订阅多个品种，行情数据仍会逐个到达\ ``strategy.on_tick()``\ 函数

这里我们以双均线策略为例，展示实盘运行的代码：

.. code:: python

    props = {'symbol': 'rb1801.SHF'}
    tapi = RealTimeTradeApi()
    ins = EventLiveTradeInstance()

    tapi.use_strategy(3)

    ds = RemoteDataService()
    strat = DoubleMaStrategy()
    pm = PortfolioManager()

    context = model.Context(data_api=ds, trade_api=tapi,
                            instance=ins, strategy=strat, pm=pm)

    ins.init_from_config(props)
    ds.subscribe(props['symbol'])

    ins.run()

    time.sleep(999999)

程序运行后，行情、策略、交易会在不同的线程内运行，我们只需要在\ ``on_tick``\ 中进行下单，在\ ``on_trade``,
``on_order_status``\ 中处理交易回报即可。

正常收到行情如下：

::

    rb1801.SHF  20171116-211319500       (BID)    229@3889.00 | 3891.00@12     (ASK)
    Fast MA = 3889.89     Slow MA = 3889.81
    rb1801.SHF  20171116-211320000       (BID)    224@3889.00 | 3891.00@13     (ASK)
    Fast MA = 3889.93     Slow MA = 3889.83
    rb1801.SHF  20171116-211320500       (BID)    223@3889.00 | 3891.00@5      (ASK)
    Fast MA = 3889.89     Slow MA = 3889.85
    rb1801.SHF  20171116-211321000       (BID)    223@3889.00 | 3891.00@5      (ASK)
    Fast MA = 3889.93     Slow MA = 3889.88
    rb1801.SHF  20171116-211321500       (BID)     24@3890.00 | 3891.00@5      (ASK)
    Fast MA = 3890.00     Slow MA = 3889.92

如果发生下单、成交，则会收到如下回报：

::

    rb1801.SHF  20171116-211319000       (BID)    230@3889.00 | 3891.00@6      (ASK)
    Fast MA = 3889.93     Slow MA = 3889.79

    Strategy on order status: 
    New       |  20171115(  211319) Buy    rb1801.SHF@3.000  size = 1

    Strategy on order status: 
    Accepted  |  20171115(  211319) Buy    rb1801.SHF@3.000  size = 1

    Strategy on trade: 
    20171115(  211319) Buy    rb1801.SHF@3.000  size = 1

    Strategy on order status: 
    Filled    |  20171115(  211319) Buy    rb1801.SHF@3.000  size = 1

    Strategy on task ind: 
    task_id = 81116000001  |  task_status = Stopped  |  task_algo = 
    task_msg = 
    DONE Execution Report (00'00"039):
    +---------+------+----------+------+----+---------+-----+----------+----+---------+------------+-------+-------+--------+---------+
    |underlyer| ba_id|    symbol|jzcode|side|is_filled|price|fill_price|size|fill_size|pending_size|cancels|rejects|entrusts| duration|
    +---------+------+----------+------+----+---------+-----+----------+----+---------+------------+-------+-------+--------+---------+
    |        0|800000|rb1801.SHF| 27509|Long|        Y|  3.0|    3.0000|   1|        1|           0|      0|      0|       1|00'00"033|
    +---------+------+----------+------+----+---------+-----+----------+----+---------+------------+-------+-------+--------+---------+
