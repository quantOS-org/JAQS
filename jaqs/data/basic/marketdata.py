# encoding: utf-8


class Bar(object):
    def __init__(self):
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
        bar_list = []
        for _, row in df.iterrows():
            dic = row.to_dict()
            bar = Bar.create_from_dict(dic)
            bar_list.append(bar)
        return bar_list

    @classmethod
    def create_from_dict(cls, dic):
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

