# encoding: UTF-8
from __future__ import print_function


class Instrument(object):
    def __init__(self):
        self.symbol = ""
        self.multiplier = 0.0
        self.inst_type = 0
        self.market = ""
        self.list_date = 0
        self.delist_date = 99999999
    
    @property
    def is_stock(self):
        return self.inst_type == 1
    
    @property
    def is_future(self):
        return (self.inst_type == 101
                or self.inst_type == 102
                or self.inst_type == 103)


class InstManager(object):
    def __init__(self, data_api, inst_type="", symbol=""):
        self.data_api = data_api
        
        self.inst_map = {}
        self.load_instruments(inst_type=inst_type, symbol=symbol)
    
    def load_instruments(self, inst_type="", symbol=""):
        fields = ['symbol', 'inst_type', 'market', 'status', 'multiplier', 'list_date', 'delist_date']
        res = self.data_api.query_inst_info(symbol=symbol, fields=','.join(fields), inst_type=inst_type)
        res = res.reset_index()

        dic_of_dic = res.to_dict(orient='index')
        res = {v['symbol']: {v_key: v_value for v_key, v_value in v.items() if v_key in fields}
               for _, v in dic_of_dic.items()}
        for k, v in res.items():
            inst = Instrument()
            inst.__dict__.update(v)
            self.inst_map[k] = inst
    
    def get_instrument(self, code):
        return self.inst_map.get(code, None)
