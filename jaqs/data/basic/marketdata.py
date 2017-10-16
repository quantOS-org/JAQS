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
    
    @classmethod
    def create_from_df(cls, df):
        bar_list = []
        for _, row in df.iterrows():
            bar = cls()
            
            dic = row.to_dict()
            bar.__dict__.update(dic)
            
            bar_list.append(bar)
        return bar_list
