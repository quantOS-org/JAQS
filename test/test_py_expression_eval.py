# encoding: UTF-8

from __future__ import print_function
import pandas as pd
import numpy as np
try:
    import pytest
except ImportError as e:
    if __name__ == "__main__":
        pass
    else:
        raise e
from jaqs.data import RemoteDataService
from jaqs.data import Parser
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)


def test_group_rank():
    shape = (500, 3000)
    df_val = pd.DataFrame(np.random.rand(*shape))
    df_group = pd.DataFrame(np.random.randint(1, 5, size=shape[0] * shape[1]).reshape(*shape))
    expr = parser.parse('GroupRank(val, mygroup)')
    res = parser.evaluate({'val': df_val, 'mygroup': df_group})


def test_group_quantile():
    shape = (500, 3000)
    df_val = pd.DataFrame(np.random.rand(*shape))
    df_group = pd.DataFrame(np.random.randint(1, 5, size=shape[0] * shape[1]).reshape(*shape))
    expr = parser.parse('GroupQuantile(val, mygroup, 23)')
    res = parser.evaluate({'val': df_val, 'mygroup': df_group})
    n = 100
    df_val = pd.DataFrame(np.arange(n).reshape(2, -1))
    df_group = pd.DataFrame(np.array([1] * 25 + [2] * 25 + [2] * 20 + [3] * 20 + [9] * 10).reshape(2, -1))
    expr = parser.parse('GroupQuantile(val, mygroup, 5)')
    res = parser.evaluate({'val': df_val, 'mygroup': df_group})
    n1 = 5
    n2 = 4
    n3 = 2
    res_correct = np.array([0.] * n1 + [1.] * n1 + [2.] * n1 + [3.] * n1 + [4.] * n1
                           + [0.] * n1 + [1.] * n1 + [2.] * n1 + [3.] * n1 + [4.] * n1
                           + [0.] * n2 + [1.] * n2 + [2.] * n2 + [3.] * n2 + [4.] * n2
                           + [0.] * n2 + [1.] * n2 + [2.] * n2 + [3.] * n2 + [4.] * n2
                           + [0.] * n3 + [1.] * n3 + [2.] * n3 + [3.] * n3 + [4.] * n3).reshape(2, -1) + 1.0
    assert np.abs(res.values - res_correct).flatten().sum() < 1e-6


def test_quantile():
    val = pd.DataFrame(np.random.rand(500, 3000))
    expr = parser.parse('Quantile(val, 12)')
    res = parser.evaluate({'val': val})
    assert np.nanmean(val[res == 1].values.flatten()) < 0.11

    val = pd.DataFrame(np.random.rand(1000, 100))
    expr = parser.parse('Ts_Quantile(val, 500, 12)')
    res = parser.evaluate({'val': val})
    assert np.nanmean(val[res == 1].values.flatten()) < 0.11


def test_ttm():
    from jaqs.data import DataView
    
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    props = {'start_date': 20120101, 'end_date': 20170601, 'universe': '000016.SH',
             'fields': ('net_profit_incl_min_int_inc'), 'freq': 1}
    dv.init_from_config(props, ds)
    dv.prepare_data()

    dv.add_formula('single', 'TTM(net_profit_incl_min_int_inc)', is_quarterly=True)
    
    
def test_logical_and_or():
    parser.parse('open + 3 && 1')
    res = parser.evaluate({'open': dfx})
    assert np.all(res.values.flatten())

    parser.parse('open + 3 && 0.0')
    res = parser.evaluate({'open': dfx})
    assert not np.all(res.values.flatten())


def test_plus_minus_mul_div():
    expression = parser.parse('close * open + close / open - close^3 % open')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_eq_neq():
    expression = parser.parse('(close == open) && (close != open) && (!close)')
    res = parser.evaluate({'close': dfy, 'open': dfx})
    
    expression = parser.parse('(close > open)')
    res = parser.evaluate({'close': dfy, 'open': dfx})
    
    expression = parser.parse('(close >= open)')
    res = parser.evaluate({'close': dfy, 'open': dfx})

    expression = parser.parse('(close < open)')
    res = parser.evaluate({'close': dfy, 'open': dfx})

    expression = parser.parse('(close <= open)')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_cutoff_standardize():
    expression = parser.parse('Standardize(Cutoff(close, 2.8))')
    res = parser.evaluate({'close': dfy, 'open': dfx})
    

def test_moving_avg():
    expression = parser.parse('Ewma(close, 5)')
    res = parser.evaluate({'close': dfy})
    expression = parser.parse('Ts_Mean(close, 5)')
    res = parser.evaluate({'close': dfy})
    expression = parser.parse('Ts_Min(close, 5)')
    res = parser.evaluate({'close': dfy})
    expression = parser.parse('Ts_Max(close, 5)')
    res = parser.evaluate({'close': dfy})


def test_cov_corr():
    expression = parser.parse('Correlation(close, open, 5)')
    res = parser.evaluate({'close': dfy, 'open': dfx})
    expression = parser.parse('Covariance(close, open, 5)')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_return_delay_delta():
    expression = parser.parse('Delta(close, 5)')
    res = parser.evaluate({'close': dfy})
    expression = parser.parse('Delay(close, 5)')
    res = parser.evaluate({'close': dfy})
    expression = parser.parse('Return(close, 5)')
    res = parser.evaluate({'close': dfy})
    
    
def test_skew():
    expression = parser.parse('Ts_Skewness(close,4)')
    res = parser.evaluate({'close': dfy})
    expression = parser.parse('Ts_Kurtosis(close,4)')
    res = parser.evaluate({'close': dfy})


def test_variables():
    expression = parser.parse('Ts_Skewness(open,4)+close / what')
    res = set(expression.variables()) == {'open', 'close', 'what'}
    
    
def test_product():
    # parser.set_capital('lower')
    expression = parser.parse('Product(open,2)')
    res = parser.evaluate({'close': dfy, 'open': dfx})
    # parser.set_capital('upper')


def test_rank():
    expression = parser.parse('Rank(close)')
    res = parser.evaluate({'close': dfy, 'open': dfx})
    
    expression = parser.parse('Ts_Rank(close, 8)')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_tail():
    expression = parser.parse('Tail(close/open,0.99,1.01,1.0)')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_step():
    expression = parser.parse('Step(close,10)')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_decay_linear():
    expression = parser.parse('Decay_linear(open,2)')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_decay_exp():
    expression = parser.parse('Decay_exp(open, 0.5, 2)')
    res = parser.evaluate({'close': dfy, 'open': dfx})


def test_signed_power():
    expression = parser.parse('SignedPower(close-open, 2)')
    res = parser.evaluate({'close': dfx, 'open': dfy})


def test_ewma():
    expr = parser.parse('Ewma(close, 3)')
    res = parser.evaluate({'close': dfx})
    assert abs(res.loc[20170801, '000001.SH'] - 3292.6) < 1e-1


def test_if():
    expr = parser.parse('If(close > 20, 3, -3)')
    res = parser.evaluate({'close': dfx})
    assert res.iloc[0, 0] == 3.
    assert res.iloc[0, 2] == -3.


'''
def test_group_apply():
    import numpy as np
    np.random.seed(369)
    
    n = 20
    
    dic = {c: np.random.rand(n) for c in 'abcdefghijklmnopqrstuvwxyz'[:n]}
    df_value = pd.DataFrame(index=range(n), data=dic)
    
    r = np.random.randint(0, 5, n * df_value.shape[0]).reshape(df_value.shape[0], n)
    cols = df_value.columns.values.copy()
    np.random.shuffle(cols)
    
    df_group = pd.DataFrame(index=df_value.index, columns=cols, data=r)

    parser = Parser()
    expr = parser.parse('GroupApply(Standardize, GroupApply(Cutoff, close, 2.8))')
    res = parser.evaluate({'close': df_value}, df_group=df_group)
    
    assert abs(res.iloc[3, 6] - (-1.53432)) < 1e-5
    assert abs(res.iloc[19, 18] - (-1.17779)) < 1e-5


'''


def test_calc_return():
    expr = parser.parse('Return(close, 2, 0)')
    res = parser.evaluate({'close': dfx})
    assert abs(res.loc[20170808, '000001.SH'] - 0.006067) < 1e-6

    expr = parser.parse('Return(close, 2, 1)')
    res = parser.evaluate({'close': dfx})


@pytest.fixture(autouse=True)
def my_globals(request):
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    df, msg = ds.daily("000001.SH, 600030.SH, 000300.SH", start_date=20170801, end_date=20170820,
                       fields="open,high,low,close,vwap,preclose")
    
    multi_index_names = ['trade_date', 'symbol']
    df_multi = df.set_index(multi_index_names, drop=False)
    df_multi.sort_index(axis=0, level=multi_index_names, inplace=True)
    
    dfx = df_multi.loc[pd.IndexSlice[:, :], pd.IndexSlice['close']].unstack()
    dfy = df_multi.loc[pd.IndexSlice[:, :], pd.IndexSlice['open']].unstack()
    
    parser = Parser()
    request.function.__globals__.update({'parser': parser, 'dfx': dfx, 'dfy': dfy})


if __name__ == "__main__":
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    df, msg = ds.daily("000001.SH, 600030.SH, 000300.SH", start_date=20170801, end_date=20170820,
                       fields="open,high,low,close,vwap,preclose")
    ds.data_api.close()
    
    multi_index_names = ['trade_date', 'symbol']
    df_multi = df.set_index(multi_index_names, drop=False)
    df_multi.sort_index(axis=0, level=multi_index_names, inplace=True)
    
    dfx = df_multi.loc[pd.IndexSlice[:, :], pd.IndexSlice['close']].unstack()
    dfy = df_multi.loc[pd.IndexSlice[:, :], pd.IndexSlice['open']].unstack()
    
    parser = Parser()
    
    g = globals()
    g = {k: v for k, v in g.items() if k.startswith('test_') and callable(v)}
    
    for test_name, test_func in g.items():
        print("\n==========\nTesting {:s}...".format(test_name))
        # try:
        test_func()
        # print "Successfully tested {:s}.".format(test_name)
        # except Exception, e:
    print("Test Complete.")
