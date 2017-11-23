# encoding: utf-8
from jaqs.data.basic import Bar, InstManager
from jaqs.data import RemoteDataService
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)


def test_inst_manager():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    mgr = InstManager(ds)
    mgr.load_instruments()
    sym = '000001.SZ'
    inst_obj = mgr.get_instrument(sym)
    assert inst_obj.market == 'SZ'
    assert inst_obj.symbol == sym
    assert inst_obj.multiplier == 1
    assert inst_obj.inst_type == 1


def test_bar():
    from jaqs.data import RemoteDataService
    from jaqs.trade.common import QUOTE_TYPE
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    df_quotes, msg = ds.bar(symbol='rb1710.SHF,hc1710.SHF', start_time=200000, end_time=160000,
                                           trade_date=20170704, freq=QUOTE_TYPE.MIN)
    quotes_list = Bar.create_from_df(df_quotes)


if __name__ == "__main__":
    test_inst_manager()
    test_bar()
