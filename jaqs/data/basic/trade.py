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
