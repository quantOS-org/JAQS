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
        dic_of_dic = df.to_dict(orient='index')
        
        bar_list = []
        for v in dic_of_dic.values():
            bar = cls()
            bar.__dict__.update(v)
            bar_list.append(bar)
        return bar_list
