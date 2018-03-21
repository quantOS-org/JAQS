# encoding:utf-8
"""
Classes defined in trade module are relevant to trades, including:
- Trade
- TradeInd
- TradeStat

"""

from .order import Order


class Trade(object):
    """
    Trade represents fill/partial fill of an order.

    Attributes
    ----------
    task_id : str
        Id of the task.
    entrust_no : str
        ID of the order.
    entrust_action : str
    symbol : str
    fill_price : float
    fill_size : int
    fill_date : int
    fill_time : int
    fill_no : str
        ID of this trade.

    """
    
    def __init__(self, order=None):
        self.task_id = 0
        self.entrust_no = ""
        
        self.entrust_action = ""
        
        self.symbol = ""

        self.fill_no = ""
        self.fill_price = 0.0
        self.fill_size = 0
        self.fill_date = 0
        self.fill_time = 0
        
        self.commission = 0.0

        if order is not None:
            if isinstance(order, Order):
                self.init_from_order(order)
            else:
                raise ValueError("init_from_order only accept argument of type Order.")
        
    def init_from_order(self, order):
        """Update information from a given order."""
        self.task_id = order.task_id
        self.entrust_no = order.entrust_no
        self.symbol = order.symbol
        self.entrust_action = order.entrust_action
    
    def set_fill_info(self, price, size, date, time, no, trade_date=0):
        """Update filling information."""
        self.fill_price = price
        self.fill_size = size
        self.fill_date = date
        self.fill_time = time
        self.fill_no = no
        self.trade_date = trade_date if trade_date else date
        
    @classmethod
    def create_from_dict(cls, dic):
        trade_ind = cls()
        trade_ind.__dict__.update(dic)
        return trade_ind
    
    def __repr__(self):
        return "{0.fill_date:8d}({0.fill_time:8d}) " \
               "{0.entrust_action:6s} {0.symbol:10s}@{0.fill_price:.3f}" \
               "  size = {0.fill_size}".format(self)
    
    def __str__(self):
        return self.__repr__()


class TaskInd(object):
    """
    TaskInd is a indication of status change of a task.
    
    Attributes
    ----------
    task_id : int
        Id of the task.
    task_status : str
        Current status of the task.
    task_algo : str
        Algorithm of the task.
    task_msg : str
        Message relevant to the task.
    
    """
    def __init__(self, task_id=0, task_status="", task_algo="", task_msg=""):
        self.task_id = task_id
        self.task_status = task_status
        self.task_algo = task_algo
        self.task_msg = task_msg

    @classmethod
    def create_from_dict(cls, dic):
        rsp = cls()
        rsp.__dict__.update(dic)
        return rsp
    
    def __repr__(self):
        return "task_id = {0.task_id:d}  |  task_status = {0.task_status:s}  |  task_algo = {0.task_algo:s}" \
               "\ntask_msg = \n{0.task_msg:s}".format(self)

    def __str__(self):
        return self.__repr__()


class TradeStat(object):
    """
    TradeStat stores statistics of trading of a certain symbol.
    
    Attributes
    ----------
    symbol : str
    buy_filled_size : int
        Amount of long position that is already filled.
    buy_want_size : int
        Amount of long position that is not yet filled.
    sell_filled_size : int
        Amount of short position that is already filled.
    sell_want_size : int
        Amount of short position that is not yet filled.
    
    """
    def __init__(self, symbol=""):
        self.symbol = symbol
        self.buy_filled_size = 0
        self.buy_want_size = 0
        self.sell_filled_size = 0
        self.sell_want_size = 0
    
    def __repr__(self):
        return ("        Want Size      Filled Size  \n"
                "====================================\n"
                "Buy     {0:8.0f}         {1:8.0f}   \n"
                "------------------------------------\n"
                "Sell    {2:8.0f}         {3:8.0f}   \n"
                "".format(self.buy_want_size, self.buy_filled_size,
                          self.sell_want_size, self.sell_filled_size))
    
    def __str__(self):
        return self.__repr__()
