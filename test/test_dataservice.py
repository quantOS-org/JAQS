# encoding: UTF-8

from __future__ import print_function
from jaqs.data import RemoteDataService
try:
    import pytest
except ImportError as e:
    if __name__ == "__main__":
        pass
    else:
        raise e
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)


def test_remote_data_service_daily():
    # test daily
    res, msg = ds.daily('rb1710.SHF,600662.SH', fields="",
                        start_date=20170828, end_date=20170831,
                        adjust_mode=None)
    assert msg == '0,'
    
    rb = res.loc[res.loc[:, 'symbol'] == 'rb1710.SHF', :]
    stk = res.loc[res.loc[:, 'symbol'] == '600662.SH', :]
    assert set(rb.columns) == {'close', 'code', 'high', 'low', 'oi', 'open', 'settle', 'symbol', 'freq',
                               'trade_date', 'trade_status', 'turnover', 'volume', 'vwap', 'presettle'}
    assert rb.shape == (4, 15)
    assert rb.loc[:, 'volume'].values[0] == 189616
    assert stk.loc[:, 'volume'].values[0] == 7174813


def test_remote_data_service_daily_quited():
    # test daily
    res, msg = ds.daily('600832.SH', fields="",
                        start_date=20140828, end_date=20170831,
                        adjust_mode=None)
    assert msg == '0,'
    assert res.shape == (175, 15)


def test_remote_data_service_bar():
    # test bar
    res2, msg2 = ds.bar('rb1710.SHF,600662.SH', start_time=200000, end_time=160000, trade_date=20170831, fields="")
    assert msg2 == '0,'
    
    rb2 = res2.loc[res2.loc[:, 'symbol'] == 'rb1710.SHF', :]
    stk2 = res2.loc[res2.loc[:, 'symbol'] == '600662.SH', :]
    assert set(rb2.columns) == {u'close', u'code', u'date', u'freq', u'high', u'low', u'oi', u'open',
                                u'settle', u'symbol', u'time', u'trade_date', u'turnover', u'volume',
                                u'vwap'}
    assert abs(rb2.loc[:, 'settle'].values[0] - 0.0) < 1e-3
    assert rb2.shape == (345, 15)
    assert stk2.shape == (240, 15)
    assert rb2.loc[:, 'volume'].values[344] == 3366


def test_remote_data_serviece_quote():
    res, msg = ds.quote('000001.SH')
    assert msg == '0,'
    

def test_remote_data_service_lb():
    # test lb.secDailyIndicator
    fields = "pb,pe,free_share,net_assets,limit_status"
    for res3, msg3 in [ds.query("lb.secDailyIndicator", fields=fields,
                                filter="symbol=600030.SH&start_date=20170907&end_date=20170907",
                                orderby="trade_date"),
                       ds.query_lb_dailyindicator('600030.SH', 20170907, 20170907, fields)]:
        assert msg3 == '0,'
        assert abs(res3.loc[0, 'pb'] - 1.5135) < 1e-4
        assert abs(res3.loc[0, 'free_share'] - 781496.5954) < 1e-4
        assert abs(res3.loc[0, 'net_assets'] - 1.437e11) < 1e8
        assert res3.loc[0, 'limit_status'] == 0
    
    # test lb.income
    for res4, msg4 in [ds.query("lb.income", fields="",
                                filter="symbol=600000.SH&start_date=20150101&end_date=20170101&report_type=408001000",
                                order_by="report_date"),
                       ds.query_lb_fin_stat('income', '600000.SH', 20150101, 20170101, fields="")]:
        assert msg4 == '0,'
        assert res4.shape == (8, 12)
        assert abs(res4.loc[4, 'oper_rev'] - 120928000000) < 1


def test_remote_data_service_daily_ind_performance():
    hs300 = ds.query_index_member('000300.SH', 20151001, 20170101)
    hs300_str = ','.join(hs300)
    
    fields = "pb,pe,share_float_free,net_assets,limit_status"
    res, msg = ds.query("lb.secDailyIndicator", fields=fields,
                          filter=("symbol=" + hs300_str
                                  + "&start_date=20160907&end_date=20170907"),
                          orderby="trade_date")
    assert msg == '0,'


def test_remote_data_service_components():
    res = ds.query_index_member_daily(index='000300.SH', start_date=20140101, end_date=20170505)
    assert res.shape == (814, 430)
    
    arr = ds.query_index_member(index='000300.SH', start_date=20140101, end_date=20170505)
    assert len(arr) == 430


def test_remote_data_service_industry():
    from jaqs.data.align import align
    import pandas as pd
    
    arr = ds.query_index_member(index='000300.SH', start_date=20130101, end_date=20170505)
    df = ds.query_industry_raw(symbol=','.join(arr), type_='SW')
    df = ds.query_industry_raw(symbol=','.join(arr), type_='ZZ')
    
    # errors
    try:
        ds.query_industry_raw(symbol=','.join(arr), type_='ZZ', level=5)
    except ValueError:
        pass
    try:
        ds.query_industry_raw(symbol=','.join(arr), type_='blabla')
    except ValueError:
        pass
    
    # df_ann = df.loc[:, ['in_date', 'symbol']]
    # df_ann = df_ann.set_index(['symbol', 'in_date'])
    # df_ann = df_ann.unstack(level='symbol')
    
    from jaqs.data import DataView
    dic_sec = jutil.group_df_to_dict(df, by='symbol')
    dic_sec = {sec: df.reset_index() for sec, df in dic_sec.items()}
    
    df_ann = pd.concat([df.loc[:, 'in_date'].rename(sec) for sec, df in dic_sec.items()], axis=1)
    df_value = pd.concat([df.loc[:, 'industry1_code'].rename(sec) for sec, df in dic_sec.items()], axis=1)
    
    dates_arr = ds.query_trade_dates(20140101, 20170505)
    res = align(df_value, df_ann, dates_arr)
    # df_ann = df.pivot(index='in_date', columns='symbol', values='in_date')
    # df_value = df.pivot(index=None, columns='symbol', values='industry1_code')
    
    def align_single_df(df_one_sec):
        df_value = df_one_sec.loc[:, ['industry1_code']]
        df_ann = df_one_sec.loc[:, ['in_date']]
        res = align(df_value, df_ann, dates_arr)
        return res
    # res_list = [align_single_df(df) for sec, df in dic_sec.items()]
    res_list = [align_single_df(df) for df in list(dic_sec.values())[:10]]
    res = pd.concat(res_list, axis=1)
    
    
def test_remote_data_service_industry_df():
    # from jaqs.data import Calendar
    
    arr = ds.query_index_member(index='000300.SH', start_date=20130101, end_date=20170505)
    symbol_arr = ','.join(arr)
    
    sec = '000008.SZ'
    type_ = 'ZZ'
    df_raw = ds.query_industry_raw(symbol=sec, type_=type_)
    df = ds.query_industry_daily(symbol=symbol_arr,
                                 start_date=df_raw['in_date'].min(), end_date=20170505,
                                 type_=type_, level=1)
    
    for idx, row in df_raw.iterrows():
        in_date = row['in_date']
        value = row['industry1_code']
        if in_date in df.index:
            assert df.loc[in_date, sec] == value
        else:
            idx = ds.query_next_trade_date(in_date)
            assert df.loc[idx, sec] == value
        

def test_remote_data_service_fin_indicator():
    symbol = '000008.SZ'
    filter_argument = ds._dic2url({'symbol': symbol})
    
    df_raw, msg = ds.query("lb.finIndicator", fields="",
                           filter=filter_argument, orderby="symbol")


def test_remote_data_service_adj_factor():
    arr = ds.query_index_member(index='000300.SH', start_date=20160101, end_date=20170505)
    symbol_arr = ','.join(arr)
    
    res = ds.query_adj_factor_daily(symbol_arr, start_date=20160101, end_date=20170101, div=False)
    assert abs(res.loc[20160408, '300024.SZ'] - 10.735) < 1e-3
    assert abs(res.loc[20160412, '300024.SZ'] - 23.658) < 1e-3
    
    res = ds.query_adj_factor_daily(symbol_arr, start_date=20160101, end_date=20170101, div=True)


def test_remote_data_service_dividend():
    arr = ds.query_index_member(index='000300.SH', start_date=20160101, end_date=20170505)
    symbol_arr = ','.join(arr)
    
    df, msg = ds.query_dividend(symbol_arr, start_date=20160101, end_date=20170101)
    df2 = df.pivot(index='exdiv_date', columns='symbol', values='share_ratio')
    assert abs(df.loc[(df['exdiv_date'] == 20160504) & (df['symbol'] == '002085.SZ'), 'share_ratio'] - 0.20).iat[0] < 1e-2


def test_remote_data_service_inst_info():
    sec = '000001.SZ'
    res = ds.query_inst_info(sec, fields='status,selllot,buylot,pricetick,multiplier,product')
    assert res.at[sec, 'multiplier'] == 1
    assert abs(res.at[sec, 'pricetick'] - 0.01) < 1e-2
    assert res.at[sec, 'buylot'] == 100

    res = ds.query_inst_info('000001.SH')
    assert not res.empty


def test_remote_data_service_index_weight():
    df = ds.query_index_weights_raw(index='000300.SH', trade_date=20140101)
    assert df.shape[0] == 300
    assert abs(df['weight'].sum() - 1.0) < 1.0

    df = ds.query_index_weights_range(index='000300.SH', start_date=20140101, end_date=20140305)

    df = ds.query_index_weights_raw(index='000016.SH', trade_date=20140101)
    assert df.shape[0] == 50
    assert abs(df['weight'].sum() - 1.0) < 1.0
    
    df = ds.query_index_weights_daily(index='000300.SH', start_date=20150101, end_date=20151221)
    assert abs(df.at[20150120, '000001.SZ'] - 1.07e-2) < 1e-2
    assert df.shape == (236, 321)


def test_remote_data_service_initialize():
    import jaqs.data.dataservice as jads
    data_config2 = {k: v for k, v in data_config.items()}
    
    data_config2['remote.data.password'] = ''
    try:
        ds.init_from_config(data_config2)
    except jads.InitializeError:
        pass
    
    data_config2['remote.data.password'] = '123'
    msg = ds.init_from_config(data_config2)
    assert msg.split(',')[0] == '-1000'
    try:
        ds.daily('000001.SH', start_date=20170101, end_date=20170109)
    except jads.NotLoginError:
        pass
    
    msg = ds.init_from_config(data_config)
    assert msg.split(',')[0] == '0'
    msg = ds.init_from_config(data_config)
    assert msg.split(',')[0] == '0'
    

def test_remote_data_service_subscribe():
    ds.subscribe('000001.SH')


def test_remote_data_bar_quote():
    df, msg = ds.bar_quote('000001.SZ', trade_date=20171009, freq='1M')
    assert msg == '0,'
    assert df['askvolume1'].all()
    assert abs(df['bidprice1'].iat[1] - 11.52) < 1e-2
    

def test_remote_data_service_mkt_data_callback():
    from jaqs.data.basic import Quote
    q = Quote()
    ds.mkt_data_callback(key='quote', quote=q)


def test_calendar():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    res1 = ds.query_trade_dates(20121224, 20130201)
    assert len(res1) == 27
    
    day_zero = 20170102
    res2 = ds.query_next_trade_date(day_zero)
    assert res2 == 20170103
    res2_last = ds.query_last_trade_date(res2)
    assert res2_last == 20161230
    
    res3 = ds.query_next_trade_date(20170104)
    assert res3 == 20170105
    res4 = ds.query_last_trade_date(res3)
    assert res4 == 20170104
    
    res11 = ds.query_trade_dates(20161224, 20170201)
    assert len(res11) == 23
    
    assert not ds.is_trade_date(20150101)
    assert not ds.is_trade_date(20130501)


'''
def test_remote_data_service_exception():
    from jaqs.data.dataservice import NotLoginError, InitializeError
    
    del ds
    ds2 = RemoteDataService()
    try:
        ds2.daily('000001.SH', 20170101, 20170109)
    except NotLoginError:
        pass
    except Exception as exc:
        raise exc
    
    try:
        ds2.init_from_config({'remote.data.address': 'blabla'})
    except InitializeError:
        pass
    except Exception as exc:
        raise exc

'''


@pytest.fixture(autouse=True)
def my_globals(request):
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    request.function.__globals__.update({'ds': ds})


if __name__ == "__main__":
    import time
    t_start = time.time()

    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    g = globals()
    g = {k: v for k, v in g.items() if k.startswith('test_') and callable(v)}

    for test_name, test_func in g.items():
        print("\n==========\nTesting {:s}...".format(test_name))
        test_func()
    print("Test Complete.")
    
    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))
