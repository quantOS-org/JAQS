# encoding:utf-8


class Trade(object):
    """
    Basic order class.

    Attributes
    ----------
    task_id : str
        id of the task.
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

    Methods
    -------


    """
    
    def __init__(self):
        self.task_id = ""
        self.entrust_no = ""
        
        self.entrust_action = ""
        
        self.symbol = ""
        
        self.fill_price = 0.0
        self.fill_size = 0
        self.fill_date = 0
        self.fill_time = 0
        self.fill_no = ""
        
        # TODO attributes below only for backward compatibility
        self.order_price = 0.0
        self.order_size = 0
        
        self.refquote = None
        self.refcode = ''
        
        self.entrust_no = ''
    
    def init_from_order(self, order):
        self.task_id = order.task_id
        self.entrust_no = order.entrust_no
        self.symbol = order.symbol
        self.entrust_action = order.entrust_action
    
    def send_fill_info(self, price, size, date, time, no):
        self.fill_price = price
        self.fill_size = size
        self.fill_date = date
        self.fill_time = time
        self.fill_no = no
    
    def __repr__(self):
        return "{0.fill_date:8d}({0.fill_time:8d}) " \
               "{0.entrust_action:6s} {0.fill_size:5d} of " \
               "{0.symbol:10s}@{0.fill_price:.3f}".format(self)
    
    def __str__(self):
        return self.__repr__()
