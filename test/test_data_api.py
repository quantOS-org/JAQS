# encoding: UTF-8

from __future__ import print_function
from jaqs.data import DataApi
from jaqs.trade import common
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)


def test_data_api():
    address = data_config.get("remote.data.address", None)
    username = data_config.get("remote.data.username", None)
    password = data_config.get("remote.data.password", None)
    if address is None or username is None or password is None:
        raise ValueError("no data service config available!")
    
    api = DataApi(address, use_jrpc=False)
    login_msg = api.login(username=username, password=password)
    print(login_msg)
    
    daily, msg = api.daily(symbol="600030.SH,000002.SZ", start_date=20170103, end_date=20170708,
                           fields="open,high,low,close,volume,last,trade_date,settle")
    daily2, msg2 = api.daily(symbol="600030.SH", start_date=20170103, end_date=20170708,
                             fields="open,high,low,close,volume,last,trade_date,settle")
    # err_code, err_msg = msg.split(',')
    assert msg == '0,'
    assert msg2 == '0,'
    assert daily.shape == (248, 9)
    assert daily2.shape == (124, 9)
    
    df, msg = api.bar(symbol="600030.SH", trade_date=20170904, freq=common.QUOTE_TYPE.MIN, start_time=90000, end_time=150000)
    print(df.columns)
    assert df.shape == (240, 15)
    
    print("test passed")
    

if __name__ == "__main__":
    test_data_api()
