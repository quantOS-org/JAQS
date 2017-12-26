交易API
-------

本产品提供了交易API，可用于委托、撤单、查询账户、查询持仓、查询委托、查询成交、批量下单等，以下是用法

导入接口
~~~~~~~~

在python程序里面导入module，然后用注册的用户帐号登录就可以使用交易接口进行交易。

引入模块
^^^^^^^^

.. code:: python

    from jaqs.trade.tradeapi import TradeApi

登录
^^^^

.. code:: python

    tapi = TradeApi(addr="tcp://gw.quantos.org:8901") # tcp://gw.quantos.org:8901是仿真系统地址
    user_info, msg = tapi.login("demo", "666666")     # 示例账户，用户需要改为自己注册的账户

使用用户名、密码登陆， 如果成功，返回用户可用的策略帐号列表

定义并设置回调接口
~~~~~~~~~~~~~~~~~~

TradeApi通过回调函数方式通知用户事件。事件包括三种：订单状态、成交回报、委托任务执行状态。

-  订单状态推送

   .. code:: python

       def on_orderstatus(order):
           print "on_orderstatus:" #, order
           for key in order:    print "%20s : %s" % (key, str(order[key]))
           print ""

-  成交回报推送

   .. code:: python

       def on_trade(trade):
           print "on_trade:"
           for key in trade:    print "%20s : %s" % (key, str(trade[key]))
           print ""

-  委托任务执行状态推送，通常可以忽略该回调函数

   .. code:: python

       def on_taskstatus(task):
           print "on_taskstatus:"
           for key in task:    print "%20s : %s" % (key, str(task[key]))
           print ""

设置回调函数

.. code:: python

    tapi.set_ordstatus_callback(on_orderstatus)
    tapi.set_trade_callback(on_trade)
    tapi.set_task_callback(on_taskstatus)

选择使用的策略帐号
~~~~~~~~~~~~~~~~~~

| 该函数成功后，下单、查持仓等和策略帐号有关的操作都和该策略帐号绑定。
| 没有必要每次下单、查询都调用该函数。重复调用该函数可以选择新的策略帐号。

| 如果成功，返回(strategy\_id, msg)
| 否则返回 (0, err\_msg)

.. code:: python

    sid, msg = tapi.use_strategy(1)
    print "msg: ", msg
    print "sid: ", sid    

查询账户信息
~~~~~~~~~~~~

返回当前的策略帐号的账户资金信息。

.. code:: python

    df, msg = tapi.query_account()
    print "msg: ", msg
    print df    

查询Portfolio
~~~~~~~~~~~~~

返回当前的策略帐号的Universe中所有标的的净持仓，包括持仓为0的标的。

.. code:: python

    df, msg = tapi.query_portfolio()
    print "msg: ", msg
    print df    

查询当前策略帐号的所有持仓
~~~~~~~~~~~~~~~~~~~~~~~~~~

| 和 query\_portfolio接口不一样。如果莫个期货合约 Long,
Short两个方向都有持仓，这里是返回两条记录
| 返回的 size 不带方向，全部为 正

.. code:: python

    df, msg = tapi.query_position()
    print "msg: ", msg
    print df

单标的下单
~~~~~~~~~~

| task\_id, msg = place\_order(code, action, price, size )
| action: Buy, Short, Cover, Sell, CoverToday, CoverYesterday,
SellToday, SellYesterday
| 返回 task\_id

.. code:: python

    task_id, msg = tapi.place_order("000025.SZ", "Buy", 57, 100)
    print "msg:", msg
    print "task_id:", task_id

撤单
~~~~

cancel\_order(task\_id)

.. code:: python

    tapi.cancel_order(task_id)

查询委托
~~~~~~~~

返回委托信息

.. code:: python

    df, msg = tapi.query_order(task_id = task_id, format = 'pandas')

查询成交
~~~~~~~~

返回成交信息

.. code:: python

    df, msg = tapi.query_trade(task_id = task_id, format = 'pandas')

目标持仓下单
~~~~~~~~~~~~

.. code:: python

    #  goal_protfolio
    #  参数：目标持仓
    #  返回：(result, msg)
    #     result:  成功或失败
    #     msg:     错误原因
    #  注意：目标持仓中必须包括所有的代码的持仓，即使不修改

    # 先查询当前的持仓, 
    portfolio, msg = tapi.goal_portfolio(goal, algo, algo_param)
    print "msg", msg
    print "portfolio", portfolio

portfolio撤单
~~~~~~~~~~~~~

.. code:: python

    # stop_portfolio
    # 撤单, 撤销所有portfolio订单
    tapi.stop_portfolio()

批量下单(1)
~~~~~~~~~~~

place\_batch\_order，指定绝对size和交易类型

.. code:: python

    # place_batch_order
    # 返回task_id, msg。
    orders = [ 
        {"security":"600030.SH", "action" : "Buy", "price": 16, "size":1000},
        {"security":"600519.SH", "action" : "Buy", "price": 320, "size":1000},
        ]

    task_id, msg = tapi.place_batch_order(orders)
    print task_id
    print msg    

批量下单(2)
~~~~~~~~~~~

basket\_order，指定变化量，不指定交易方向，由系统根据正负号来确定

.. code:: python

    # 批量下单2：basket_order
    #
    # 返回task_id, msg。
    orders = [ 
        {"security":"601857.SH", "ref_price": 8.40, "inc_size":1000},
        {"security":"601997.SH",  "ref_price": 14.540, "inc_size":20000},
        ]

    task_id, msg = tapi.basket_order(orders)
    print task_id
    print msg
