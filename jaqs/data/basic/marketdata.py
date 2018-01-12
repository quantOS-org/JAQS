# encoding: utf-8
"""
Classes defined in marketdata module represent different market data, including:
- Bar
- Quote

"""


class Bar(object):
    """
    A Bar is a summary of information of price and volume
    during a certain length of time span.
    
    Attributes
    ----------
    symbol : str
    open : float
        The first price in the time span.
    high : float
        The highest price in the time span.
    low  : float
        The lowest price in the time span.
    close : float
        The last price in the time span.
    vwap : float
        The volume-weighted average price of the time span.
    volume : int
        Trading volume within or at then end of the time span.
    oi : int
        Open interest at the end of the time span.
    trade_date : int
        %YY%mm%dd
    time : int
        %HH%mm%ss
    
    """
    def __init__(self):
        self.symbol = ""
        self.open = 0.
        self.close = 0.
        self.high = 0.
        self.low = 0.
        self.volume = 0
        self.oi = 0
        self.vwap = 0.
        
        self.trade_date = 0
        self.time = 0
        self._bar_keys = ['open', 'close', 'high', 'low', 'vwap',
                          'volume', 'oi',
                          'trade_date', 'time']
    
    @classmethod
    def create_from_df(cls, df):
        """
        Create a list of Bars from a DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Index does not matter. Each row contains information of a Bar.

        Returns
        -------
        bar_list : list of Bar

        """
        bar_list = []
        for _, row in df.iterrows():
            dic = row.to_dict()
            bar = Bar.create_from_dict(dic)
            bar_list.append(bar)
        return bar_list

    @classmethod
    def create_from_dict(cls, dic):
        """
        Create a list of Bars from a dic.
        
        Parameters
        ----------
        dic : dict
            Keys are attribute names of Bar, values are corresponding values.

        Returns
        -------
        bar : Bar

        """
        bar = cls()
        bar.__dict__.update(dic)
        return bar

    def __repr__(self):
        return "{0.trade_date:8d}-{0.time:6d} " \
               "{0.volume:13.2f} of " \
               "{0.symbol:10s}@{0.close:.3f}".format(self)

    def __str__(self):
        return self.__repr__()


class Quote(object):
    """
    Quote represents a snapshot of price and volume information.
    
    Attributes
    ----------
    open : float
        The daily open price.
    high : float
        The daily highest price.
    low  : float
        The daily lowest price.
    close : float
    volume : int
        Trading volume till now.
    turnover : int
        Trading turnover till now.
    oi : int
        Open interest till now.
    trade_date : int
        %YY%mm%dd
    date : int
        %YY%mm%dd, natural date.
    time : int
        %HH%mm%ss
    last : float
        Latest trading price.
    askprice1 : float
        Best ask price.
    bidprice1 : float
        Best bid price.
    askvolume1 : float
        Sum of tradable volume of all orders at the best ask price.
    bidvolume1 : float
        Sum of tradable volume of all orders at the best bid price.
    
    
    """
    def __init__(self):
        self.trade_date = 0
        self.date = 0
        self.time = 0
        
        self.open = 0.
        self.close = 0.
        self.high = 0.
        self.low = 0.
        self.vwap = 0.
        
        self.settle = 0.0
        
        self.volume = 0
        self.turnover = 0
        self.oi = 0
        
        self.preclose = 0.0
        self.presettle = 0.0
        self.preoi = 0
        
        self.last = 0.0
        self.bidprice1 = 0.0
        self.bidprice2 = 0.0
        self.bidprice3 = 0.0
        self.bidprice4 = 0.0
        self.bidprice5 = 0.0
        self.askprice1 = 0.0
        self.askprice2 = 0.0
        self.askprice3 = 0.0
        self.askprice4 = 0.0
        self.askprice5 = 0.0

        self.bidvolume1 = 0.0
        self.bidvolume2 = 0.0
        self.bidvolume3 = 0.0
        self.bidvolume4 = 0.0
        self.bidvolume5 = 0.0
        self.askvolume1 = 0.0
        self.askvolume2 = 0.0
        self.askvolume3 = 0.0
        self.askvolume4 = 0.0
        self.askvolume5 = 0.0
        
        self.limit_up = 0.0
        self.limit_down = 0.0

    @classmethod
    def create_from_dict(cls, dic):
        quote = cls()
        quote.__dict__.update(dic)
        return quote

    def __repr__(self):
        return ("{0.symbol:s}  {0.trade_date:8d}-{0.time:6d}      "
                " (BID) {0.bidvolume1:6.0f}@{0.bidprice1:6.2f}"
                " | "
                "{0.askprice1:6.2f}@{0.askvolume1:<6.0f} (ASK)".format(self))

    def __str__(self):
        return self.__repr__()

