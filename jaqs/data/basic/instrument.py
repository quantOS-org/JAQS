# encoding: UTF-8
from __future__ import print_function


class Instrument(object):
    """
    An Instrument represents a specific financial contract.
    
    Attributes
    ----------
    symbol : str
        Symbol (ticker) of the instrument.
    multiplier : float
        Contract multiplier (size).
    inst_type : int
        Integer used to represent category of the instrument.
    market : str
        The market (exchange) where the instrument is traded.
    list_date : int
        Date when the instrument is listed.
    delist_date : int
        Date when the instrument is de-listed.
        
    """
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
    """
    InstManager query information of instruments from data server and store them.
    
    Attributes
    ----------
    data_api : RemoteDataService
        Used to query data.
    _inst_map : dict
        Used to store information of instruments.
    
    Methods
    -------
    load_instruments
    
    """
    def __init__(self, data_api, inst_type="", symbol=""):
        self.data_api = data_api
        
        self._inst_map = {}
        self.load_instruments(inst_type=inst_type, symbol=symbol)
    
    @property
    def inst_map(self):
        return self._inst_map
    
    def load_instruments(self, inst_type="", symbol=""):
        fields = ['symbol', 'name', 'inst_type', 'market',
                  'status', 'multiplier', 'list_date', 'delist_date', 'pricetick', 'buylot', 'selllot']
        res = self.data_api.query_inst_info(symbol=symbol, fields=','.join(fields), inst_type=inst_type)
        res = res.reset_index()

        dic_of_dic = res.to_dict(orient='index')
        res = {v['symbol']: {v_key: v_value for v_key, v_value in v.items() if v_key in fields}
               for _, v in dic_of_dic.items()}
        for k, v in res.items():
            inst = Instrument()
            inst.__dict__.update(v)
            self._inst_map[k] = inst
    
    def get_instrument(self, code):
        return self._inst_map.get(code, None)
