# encoding:utf-8

from jaqs.trade import common


class Order(object):
    """
    Basic order class.

    Attributes
    ----------
    task_id : str
        id of the task.
    entrust_no : str
        ID of the order.
    symbol : str
    entrust_action : str
        Action of the trade.
    entrust_price : double
        Price of the order.
    entrust_size : int
        Quantity of the order.
    entrust_date : int
        Date of the order.
    entrust_time : int
        Time of the order.
    sub_seq : int
        Number of sub-orders, start with 0.
    sub_total : int
        Total number of sub-orders.
    batch_no : int
        Number of batch.
    order_status : str
    fill_price : float
    fill_size : int
    algo : str
    order_type : str (common.ORDER_TYPE)
        market, limit, stop, etc.
    time_in_force : str (common.ORDER_TIME_IN_FORCE)
        FAK, FOK, GTM, etc.

    Methods
    -------

    """
    
    def __init__(self):
        self.task_id = 0
        self.entrust_no = ""
        
        self.symbol = ""
        
        self.entrust_action = ""
        self.entrust_price = 0.0
        self.entrust_size = 0
        self.entrust_date = 0
        self.entrust_time = 0
        
        self.sub_seq = 0
        self.sub_total = 0
        self.batch_no = 0
        
        self.order_status = ""
        self.fill_price = 0.0
        self.fill_size = 0
        
        self.algo = ""
        
        self.order_type = ""
        self.time_in_force = ""
        
        # TODO attributes below only for backward compatibility
        self.errmsg = ""
        self.cancel_size = 0
        self.entrust_no = ''
    
    def __eq__(self, other):
        return self.entrust_no == other.entrust_no
    
    def __cmp__(self, other):
        return cmp(self.entrust_no, other.entrust_no)

    def __repr__(self):
        return "{0.entrust_date:8d}({0.entrust_time:8d}) " \
               "{0.entrust_action:6s} {0.symbol:10s}@{0.entrust_price:.3f}" \
               "  size = {0.fill_size} / {0.entrust_size}".format(self)

    def __str__(self):
        return self.__repr__()

    def copy(self, order):
        self.task_id = order.task_id
        self.entrust_no = order.entrust_no
        
        self.symbol = order.symbol
        
        self.entrust_action = order.entrust_action
        self.entrust_price = order.entrust_price
        self.entrust_size = order.entrust_size
        self.entrust_date = order.entrust_date
        self.entrust_time = order.entrust_time
        
        self.sub_seq = order.sub_seq
        self.sub_total = order.sub_total
        self.batch_no = order.batch_no
        
        self.order_status = order.order_status
        self.fill_size = order.fill_size
        self.fill_price = order.fill_price
        
        self.algo = order.algo
        
        self.order_type = order.order_type
        self.time_in_force = order.time_in_force
        
        self.cancel_size = order.cancel_size
        self.entrust_no = order.entrust_no
        self.errmsg = order.errmsg
    
    @property
    def is_finished(self):
        return (self.order_status == common.ORDER_STATUS.FILLED
                or self.order_status == common.ORDER_STATUS.CANCELLED
                or self.order_status == common.ORDER_STATUS.REJECTED)
    
    @classmethod
    def new_order(cls, symbol, action, price, size, date, time):
        o = cls()
        o.symbol = symbol
        o.entrust_action = action
        o.entrust_price = price
        o.entrust_size = size
        o.entrust_date = date
        o.entrust_time = time
        o.order_status = common.ORDER_STATUS.NEW
        return o


class FixedPriceTypeOrder(Order):
    """
    This type of order aims to be matched at a given price type, eg. CLOSE, OPEN, VWAP, etc.
    Only used in daily resolution trade.

    Attributes
    ----------
    price_target : str
        The type of price we want.

    Methods
    -------

    """
    
    def __init__(self, target=""):
        Order.__init__(self)
        
        self.price_target = target


class VwapOrder(Order):
    """
    This type of order will only be matched once a day.
    Only used in daily resolution trade.

    Attributes
    ----------
    start : int
        The start of matching time range.
        If start = -1, end will be ignored and the order will be matched in the whole trading session.
    end : int
        The end of matching time range.

    """
    
    def __init__(self, start=-1, end=-1):
        Order.__init__(self)
        
        self.start = start
        self.end = end
    
    @property
    def time_range(self):
        return self.start, self.end


class OrderStatusInd(object):
    def __init__(self):
        self.ba_id = 0
        self.sa_id = 0
        
        self.task_id = 0
        self.entrust_no = ''
        
        self.algo = ''
        self.batch_no = 0
        self.order_seq = 0
        
        self.symbol = ''
        
        self.entrust_action = ''
        self.entrust_price = 0.0
        self.entrust_size = 0
        self.entrust_date = 0
        self.entrust_time = 0
        
        self.order_status = ''
        
        self.fill_size = 0
        self.fill_price = 0.0
        self.commission = 0.0
        
        self.is_finished = False
    
    def init_from_order(self, order):
        self.entrust_no = order.entrust_no
        
        self.symbol = order.symbol
        
        self.entrust_action = order.entrust_action
        self.entrust_price = order.entrust_price
        self.entrust_size = order.entrust_size
        self.entrust_date = order.entrust_date
        self.entrust_time = order.entrust_time
        
        self.order_status = order.order_status
        
        self.fill_size = order.fill_size
        self.fill_price = order.fill_price

    @classmethod
    def create_from_dict(cls, dic):
        ind = cls()
        ind.__dict__.update(dic)
        return ind
    
    def __repr__(self):
        return "{0.order_status:8s}  |  " \
               "{0.entrust_date:8d}({0.entrust_time:8d}) " \
               "{0.entrust_action:6s} {0.symbol:10s}@{0.entrust_price:.3f}" \
               "  size = {0.entrust_size}".format(self)

    def __str__(self):
        return self.__repr__()


class OrderRsp(object):
    def __init__(self, entrust_no="", msg="", task_id=0):
        self.msg = msg
        self.entrust_no = entrust_no
        self.task_id = task_id

    @classmethod
    def create_from_dict(cls, dic):
        rsp = cls()
        rsp.__dict__.update(dic)
        return rsp
    
    def __repr__(self):
        return "entrust_no = {0.entrust_no:s}  |  task_id = {0.task_id:d}" \
               "\nmsg = {0.msg:s}".format(self)

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    order = FixedPriceTypeOrder.new_order('cu', 'buy', 1.0, 100, 20170505, 130524)
    print order.__dict__
