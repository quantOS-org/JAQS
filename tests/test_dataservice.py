# encoding: UTF-8

from jaqs.data.dataservice import RemoteDataService


def test_remote_data_service_daily():
    ds = RemoteDataService()
    
    # test daily
    res, msg = ds.daily('rb1710.SHF,600662.SH', fields="",
                        start_date=20170828, end_date=20170831,
                        adjust_mode=None)
    assert msg == '0,'
    
    rb = res.loc[res.loc[:, 'symbol'] == 'rb1710.SHF', :]
    stk = res.loc[res.loc[:, 'symbol'] == '600662.SH', :]
    assert set(rb.columns) == {'close', 'code', 'high', 'low', 'oi', 'open', 'settle', 'symbol',
                               'trade_date', 'trade_status', 'turnover', 'volume', 'vwap'}
    assert rb.shape == (4, 13)
    assert rb.loc[:, 'volume'].values[0] == 189616
    assert stk.loc[:, 'volume'].values[0] == 7174813


def test_remote_data_service_daily_quited():
    ds = RemoteDataService()
    
    # test daily
    res, msg = ds.daily('600832.SH', fields="",
                        start_date=20140828, end_date=20170831,
                        adjust_mode=None)
    assert msg == '0,'
    assert res.shape == (175, 13)


def test_remote_data_service_bar():
    ds = RemoteDataService()
    
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
    
    
def test_remote_data_service_lb():
    ds = RemoteDataService()
    
    # test lb.secDailyIndicator
    fields = "pb,pe,share_float_free,net_assets,limit_status"
    for res3, msg3 in [ds.query("lb.secDailyIndicator", fields=fields,
                                filter="symbol=600030.SH&start_date=20170907&end_date=20170907",
                                orderby="trade_date"),
                       ds.query_lb_dailyindicator('600030.SH', 20170907, 20170907, fields)]:
        assert msg3 == '0,'
        assert abs(res3.loc[0, 'pb'] - 1.5135) < 1e-4
        assert abs(res3.loc[0, 'share_float_free'] - 781496.5954) < 1e-4
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
    ds = RemoteDataService()
    
    hs300 = ds.get_index_comp('000300.SH', 20140101, 20170101)
    hs300_str = ','.join(hs300)
    
    fields = "pb,pe,share_float_free,net_assets,limit_status"
    res, msg = ds.query("lb.secDailyIndicator", fields=fields,
                          filter=("symbol=" + hs300_str
                                  + "&start_date=20160907&end_date=20170907"),
                          orderby="trade_date")
    assert msg == '0,'


def test_remote_data_service_components():
    ds = RemoteDataService()
    res = ds.get_index_comp_df(index='000300.SH', start_date=20140101, end_date=20170505)
    assert res.shape == (814, 430)
    
    arr = ds.get_index_comp(index='000300.SH', start_date=20140101, end_date=20170505)
    assert len(arr) == 430


def test_remote_data_service_industry():
    from jaqs.data.align import align
    import pandas as pd
    
    ds = RemoteDataService()
    arr = ds.get_index_comp(index='000300.SH', start_date=20130101, end_date=20170505)
    df = ds.get_industry_raw(symbol=','.join(arr), type_='ZZ')
    df = df.astype(dtype={'in_date': int})
    
    # df_ann = df.loc[:, ['in_date', 'symbol']]
    # df_ann = df_ann.set_index(['symbol', 'in_date'])
    # df_ann = df_ann.unstack(level='symbol')
    
    from jaqs.data.dataview import DataView
    dic_sec = DataView._group_df_to_dict(df, by='symbol')
    dic_sec = {sec: df.reset_index() for sec, df in dic_sec.viewitems()}
    
    df_ann = pd.concat([df.loc[:, 'in_date'].rename(sec) for sec, df in dic_sec.viewitems()], axis=1)
    df_value = pd.concat([df.loc[:, 'industry1_code'].rename(sec) for sec, df in dic_sec.viewitems()], axis=1)
    
    dates_arr = ds.get_trade_date(20140101, 20170505)
    res = align(df_value, df_ann, dates_arr)
    # df_ann = df.pivot(index='in_date', columns='symbol', values='in_date')
    # df_value = df.pivot(index=None, columns='symbol', values='industry1_code')
    
    def align_single_df(df_one_sec):
        df_value = df_one_sec.loc[:, ['industry1_code']]
        df_ann = df_one_sec.loc[:, ['in_date']]
        res = align(df_value, df_ann, dates_arr)
        return res
    # res_list = [align_single_df(df) for sec, df in dic_sec.viewitems()]
    res_list = [align_single_df(df) for df in dic_sec.values()[:10]]
    res = pd.concat(res_list, axis=1)
    
    
def test_remote_data_service_industry_df():
    from jaqs.data.calendar import Calendar
    cal = Calendar()
    
    ds = RemoteDataService()
    arr = ds.get_index_comp(index='000300.SH', start_date=20130101, end_date=20170505)
    symbol_arr = ','.join(arr)
    
    sec = '000008.SZ'
    type_ = 'ZZ'
    df_raw = ds.get_industry_raw(symbol=sec, type_=type_)
    df = ds.get_industry_daily(symbol=symbol_arr,
                               start_date=df_raw['in_date'].min(), end_date=20170505,
                               type_=type_, level=1)
    
    for idx, row in df_raw.iterrows():
        in_date = row['in_date']
        value = row['industry1_code']
        if in_date in df.index:
            assert df.loc[in_date, sec] == value
        else:
            idx = cal.get_next_trade_date(in_date)
            assert df.loc[idx, sec] == value
        

def test_remote_data_service_fin_indicator():
    ds = RemoteDataService()
    
    symbol = '000008.SZ'
    filter_argument = ds._dic2url({'symbol': symbol})
    
    df_raw, msg = ds.query("lb.finIndicator", fields="",
                           filter=filter_argument, orderby="symbol")


def test_remote_data_service_adj_factor():
    ds = RemoteDataService()

    arr = ds.get_index_comp(index='000300.SH', start_date=20130101, end_date=20170505)
    symbol_arr = ','.join(arr)
    
    res = ds.get_adj_factor_daily(symbol_arr, start_date=20130101, end_date=20170101, div=False)
    assert abs(res.loc[20160408, '300024.SZ'] - 10.735) < 1e-3
    assert abs(res.loc[20160412, '300024.SZ'] - 23.658) < 1e-3


def test_remote_data_service_inst_info():
    ds = RemoteDataService()
    
    sec = '000001.SZ'
    res = ds.query_inst_info(sec, fields='status,selllot,buylot,pricetick,multiplier,product')
    assert res.at[sec, 'multiplier'] == 1
    assert abs(res.at[sec, 'pricetick'] - 0.01) < 1e-2
    assert res.at[sec, 'buylot'] == 100
    
if __name__ == "__main__":
    import time
    t_start = time.time()
    
    g = globals()
    g = {k: v for k, v in g.viewitems() if k.startswith('test_') and callable(v)}

    for test_name, test_func in g.viewitems():
        print "\nTesting {:s}...".format(test_name)
        test_func()
    print "Test Complete."
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
