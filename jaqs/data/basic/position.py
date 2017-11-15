# encoding:utf-8


class Position(object):
    """
    Basic position class.

    Attributes
    ----------
    symbol : str
        List of securities.
    side : str
        ("Long", "Short"). Positions of different sides will not be merged.
    cost_price : float
        Average cost price of current net position.
    close_pnl : float
    float_pnl : float
    trading_pnl : float
    holding_pnl : float
    init_size : int
        Position at the start of the day.
    enable_size : int
        Position that can be closed.
    frozen_size : int
        Positions to be closed.
    want_size : int
        Positions to be opened.
    pre_size : int
        Last day's position.
    today_size : int
        Today's position.
    current_size : int
        Current position.

    Methods
    -------

    """
    '''
    __slots__ = ['symbol', 'side', 'cost_price',
                 'close_pnl', 'float_pnl', 'trading_pnl', 'holding_pnl',
                 'enable_size', 'frozen_size', 'want_size',
                 'today_size', 'pre_size', 'current_size', 'init_size',
                 'commission']
    '''
    
    def __init__(self, symbol=""):
        self.symbol = symbol
        
        self.side = ""
        self.cost_price = 0.0
        
        self.close_pnl = 0.0
        self.float_pnl = 0.0
        self.trading_pnl = 0.0
        self.holding_pnl = 0.0
        
        self.enable_size = 0
        self.frozen_size = 0
        self.want_size = 0
        
        self.today_size = 0
        self.pre_size = 0
        self.current_size = 0
        self.init_size = 0
        
        self.commission = 0.0

    def __repr__(self):
        return "{0.side:7s} {0.current_size:5d} of {0.symbol:10s}".format(self)

    def __str__(self):
        return self.__repr__()

    @classmethod
    def create_from_df(cls, df):
        bar_list = []
        for _, row in df.iterrows():
            dic = row.to_dict()
            bar = cls.create_from_dict(dic)
            bar_list.append(bar)
        return bar_list

    @classmethod
    def create_from_dict(cls, dic):
        bar = cls()
        bar.__dict__.update(dic)
        return bar


class GoalPosition(object):
    """
    Used in goal_portfolio function to generate orders.

    Attributes
    ----------
    symbol : str
    ref_price : float
        Reference price, used by risk management, not by order.
    size : int
        Target position size.
    urgency : int
        The urgency to adjust position, used to determine trading algorithm.

    """
    
    def __init__(self):
        self.symbol = ""
        self.ref_price = 0.0
        self.size = 0
        self.urgency = 0
    
    def __repr__(self):
        return "{0.size:5d} of {0.symbol:10s}".format(self)

    def __str__(self):
        return self.__repr__()
