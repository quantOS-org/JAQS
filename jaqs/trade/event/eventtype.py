# encoding: UTF-8

from jaqs.trade import common

'''
本文件仅用于存放对于事件类型常量的定义。

由于python中不存在真正的常量概念，因此选择使用全大写的变量名来代替常量。
这里设计的命名规则以EVENT_前缀开头。

常量的内容通常选择一个能够代表真实意义的字符串（便于理解）。

建议将所有的常量定义放在该文件中，便于检查是否存在重复的现象。
'''


class EVENT_TYPE(common.ReprStrEnum):
    TIMER = 'timer'  # 计时器事件，每隔1秒发送一次
    MARKET_DATA = 'market_data'  # 行情事件

    ORDER_RSP = 'order_rsp'
    TASK_RSP = 'task_rsp'
    TASK_STATUS_IND = 'task_callback'
    TRADE_IND = 'trade_ind'  # 成交回报
    ORDER_STATUS_IND = 'order_status_ind'  # 状态回报
    
    PLACE_ORDER = 'place_order'
    CANCEL_ORDER = 'cancel_order'
    
    QUERY_ACCOUNT = 'query_account'
    QUERY_UNIVERSE = 'query_universe'
    QUERY_PORTFOLIO = 'query_portfolio'
    QUERY_POSITION = 'query_position'
    QUERY_ORDER = 'query_order'
    QUERY_TASK = 'query_task'
    QUERY_TRADE = 'query_trade'
    
    GOAL_PORTFOLIO = 'goal_portfolio'
    STOP_PORTFOLIO = 'stop_portfolio'
    
    TRADE_API_DISCONNECTED = 'trade_api_disconnected'
    TRADE_API_CONNECTED = 'trade_api_connected'
