# encoding: UTF-8

from jaqs.data.dataservice import RemoteDataService


class Instrument(object):
    def __init__(self):
        self.symbol = ""
        self.multiplier = 0.0
        self.inst_type = 0
        self.market = ""
    
    def is_stock(self):
        return self.inst_type == 1
    
    def is_future(self):
        res = (self.inst_type == 101
               or self.inst_type == 102
               or self.inst_type == 103)
        return res


class InstManager(object):
    def __init__(self):
        self.data_api = RemoteDataService()
        self.inst_map = {}
        self.load_instruments()
    
    def load_instruments(self):
        fields = ['symbol', 'inst_type', 'market', 'status', 'multiplier']
        res, msg = self.data_api.query_inst_info(symbol='', fields=','.join(fields), inst_type="")
        
        for _, row in res.iterrows():
            inst = Instrument()
            dic = row.to_dict()
            dic = {k: v for k, v in dic.iteritems() if k in fields}
            inst.__dict__.update(dic)
            self.inst_map[inst.symbol] = inst
    
    def get_intruments(self, code):
        return self.inst_map.get(code, None)
