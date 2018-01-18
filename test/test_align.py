# encoding: utf-8
from __future__ import print_function
import pandas as pd
from jaqs.data import RemoteDataService
from jaqs.data import Parser
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)


def test_align():
    # -------------------------------------------------------------------------------------
    # input and pre-process demo data
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    raw, msg = ds.query_lb_fin_stat('income', '600000.SH', 20151225, 20170501, 'oper_rev')
    assert msg == '0,'
    
    idx_list = ['report_date', 'symbol']
    raw_idx = raw.set_index(idx_list)
    raw_idx.sort_index(axis=0, level=idx_list, inplace=True)
    
    df_ann = raw_idx.loc[pd.IndexSlice[:, :], 'ann_date']
    df_ann = df_ann.unstack(level=1)

    df_value = raw_idx.loc[pd.IndexSlice[:, :], 'oper_rev']
    df_value = df_value.unstack(level=1)
    
    date_arr = ds.query_trade_dates(20160101, 20170501)
    df_close = pd.DataFrame(index=date_arr, columns=df_value.columns, data=1e3)
    
    # -------------------------------------------------------------------------------------
    # demo usage of parser
    parser = Parser()
    parser.register_function('Myfunc', lambda x: x * 0 + 1)  # simultaneously test register function and align
    expr_formula = 'signal / Myfunc(close)'
    expression = parser.parse(expr_formula)
    for i in range(100):
        df_res = parser.evaluate({'signal': df_value, 'close': df_close}, df_ann, date_arr)
    
    # -------------------------------------------------------------------------------------
    sec = '600000.SH'
    """
    # print to validate results
    print "\n======Expression Formula:\n{:s}".format(expr_formula)
    
    print "\n======Report date, ann_date and evaluation value:"
    tmp = pd.concat([df_ann.loc[:, sec], df_value.loc[:, sec]], axis=1)
    tmp.columns = ['df_ann', 'df_value']
    print tmp
    
    print "\n======Selection of result of expansion:"
    print "20161028  {:.4f}".format(df_res.loc[20161028, sec])
    print "20161031  {:.4f}".format(df_res.loc[20161031, sec])
    print "20170427  {:.4f}".format(df_res.loc[20170427, sec])
    
    """
    assert abs(df_res.loc[20161028, sec] - 82172000000) < 1
    assert abs(df_res.loc[20161031, sec] - 120928000000) < 1
    assert abs(df_res.loc[20170427, sec] - 42360000000) < 1


if __name__ == "__main__":
    import time
    t_start = time.time()
    
    test_align()
    
    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))
