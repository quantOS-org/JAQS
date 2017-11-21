# encoding: UTF-8

import jaqs.util as jutil
from jaqs.data import DataApi
from jaqs.trade import common


def test_data_api():
    dic = jutil.read_json(jutil.join_relative_path('etc/data_config.json'))
    address = dic.get("remote.address", None)
    username = dic.get("remote.username", None)
    password = dic.get("remote.password", None)
    if address is None or username is None or password is None:
        raise ValueError("no data service config available!")
    
    api = DataApi(address, use_jrpc=False)
    login_msg = api.login(username=username, password=password)
    print login_msg
    
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
    print df.columns
    assert df.shape == (240, 15)
    
    print "test passed"
    

if __name__ == "__main__":
    test_data_api()
