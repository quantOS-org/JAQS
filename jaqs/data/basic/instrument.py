# encoding: UTF-8

from jaqs.data.dataservice import RemoteDataService


class Instrument(object):
    def __init__(self):
        self.symbol = ""
        self.multiplier = 0.0
        self.inst_type = 0
        self.market = ""
        self.list_date = 0
        self.delist_date = 99999999
    
    def is_stock(self):
        return self.inst_type == 1
    
    def is_future(self):
        res = (self.inst_type == 101
               or self.inst_type == 102
               or self.inst_type == 103)
        return res


class InstManager(object):
    def __init__(self, inst_type="", symbol="", data_api=None):
        if data_api is None:
            self.data_api = RemoteDataService()
            self.data_api.init_from_config()
        else:
            self.data_api = data_api
        
        self.inst_map = {}
        self.load_instruments(inst_type=inst_type, symbol=symbol)
    
    def load_instruments(self, inst_type="", symbol=""):
        fields = ['symbol', 'inst_type', 'market', 'status', 'multiplier', 'list_date', 'delist_date']
        res = self.data_api.query_inst_info(symbol=symbol, fields=','.join(fields), inst_type=inst_type)
        res = res.reset_index()

        dic_of_dic = res.to_dict(orient='index')
        res = {v['symbol']: {v_key: v_value for v_key, v_value in v.viewitems() if v_key in fields}
               for _, v in dic_of_dic.viewitems()}
        for k, v in res.viewitems():
            inst = Instrument()
            inst.__dict__.update(v)
            self.inst_map[k] = inst
        print
    
    def get_instrument(self, code):
        return self.inst_map.get(code, None)
