# encoding: utf-8

from __future__ import print_function
from jaqs.data import RemoteDataService
from jaqs.data import DataView
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)

daily_path = '../output/tests/test_dataview_d'
quarterly_path = '../output/tests/test_dataview_q'


def test_write():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    secs = '600030.SH,000063.SZ,000001.SZ'
    props = {'start_date': 20160601, 'end_date': 20170601, 'symbol': secs,
             'fields': 'open,close,high,low,volume,pb,net_assets,pcf_ncf',
             'freq': 1}

    dv.init_from_config(props, data_api=ds)
    dv.prepare_data()
    assert dv.data_d.shape == (281, 54)
    assert dv.dates.shape == (281, )
    # TODO
    """
    PerformanceWarning:
    your performance may suffer as PyTables will pickle object types that it cannot
    map directly to c-types [inferred_type->mixed,key->block1_values] [items->[('000001.SZ', 'int_income'), ('000001.SZ', 'less_handling_chrg_comm_exp'), ('000001.SZ', 'net_int_income'), ('000001.SZ', 'oper_exp'), ('000001.SZ', 'symbol'), ('000063.SZ', 'int_income'), ('000063.SZ', 'less_handling_chrg_comm_exp'), ('000063.SZ', 'net_int_income'), ('000063.SZ', 'oper_exp'), ('000063.SZ', 'symbol'), ('600030.SH', 'int_income'), ('600030.SH', 'less_handling_chrg_comm_exp'), ('600030.SH', 'net_int_income'), ('600030.SH', 'oper_exp'), ('600030.SH', 'symbol')]]
    """
    
    dv.save_dataview(folder_path=daily_path)


def test_write_future():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    secs = 'rb1710.SHF,j1710.DCE'
    props = {'start_date': 20170401, 'end_date': 20170901, 'symbol': secs,
             'fields': 'open,close,high,low,volume,oi',
             'freq': 1, 'all_price': False}
    
    dv.init_from_config(props, data_api=ds)
    dv.prepare_data()
    assert dv.data_d.shape == (145, 14)
    
    
def test_load():
    dv = DataView()
    dv.load_dataview(folder_path=daily_path)
    
    assert dv.start_date == 20160601 and set(dv.symbol) == set('000001.SZ,600030.SH,000063.SZ'.split(','))

    # test get_snapshot
    snap1 = dv.get_snapshot(20170504, symbol='600030.SH,000063.SZ', fields='close,pb')
    assert snap1.shape == (2, 2)
    assert set(snap1.columns.values) == {'close', 'pb'}
    assert set(snap1.index.values) == {'600030.SH', '000063.SZ'}
    
    # test get_ts
    ts1 = dv.get_ts('close', symbol='600030.SH,000063.SZ', start_date=20170101, end_date=20170302)
    assert ts1.shape == (38, 2)
    assert set(ts1.columns.values) == {'600030.SH', '000063.SZ'}
    assert ts1.index.values[-1] == 20170302


def test_add_field():
    dv = DataView()
    dv.load_dataview(folder_path=daily_path)
    nrows, ncols = dv.data_d.shape
    n_securities = len(dv.data_d.columns.levels[0])
    
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv.add_field('total_share', ds)
    assert dv.data_d.shape == (nrows, ncols + 1 * n_securities)


def test_add_formula_directly():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    secs = '600030.SH,000063.SZ,000001.SZ'
    props = {'start_date': 20160601, 'end_date': 20170601, 'symbol': secs,
             'fields': 'open,close',
             'freq': 1}
    dv.init_from_config(props, data_api=ds)
    dv.prepare_data()
    
    dv.add_formula("myfactor", 'close / open', is_quarterly=False)
    assert dv.data_d.shape == (281, 39)


def test_add_formula():
    dv = DataView()
    dv.load_dataview(folder_path=daily_path)
    nrows, ncols = dv.data_d.shape
    n_securities = len(dv.data_d.columns.levels[0])
    
    formula = 'Delta(high - close, 1)'
    dv.add_formula('myvar1', formula, is_quarterly=False)
    assert dv.data_d.shape == (nrows, ncols + 1 * n_securities)
    
    formula2 = 'myvar1 - close'
    dv.add_formula('myvar2', formula2, is_quarterly=False)
    assert dv.data_d.shape == (nrows, ncols + 2 * n_securities)


def test_dataview_universe():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    props = {'start_date': 20170227, 'end_date': 20170327, 'universe': '000016.SH',
             # 'symbol': 'rb1710.SHF,rb1801.SHF',
             'fields': ('open,high,low,close,vwap,volume,turnover,'
                        + 'sw1,zz2,'
                        + 'roe,net_assets,'
                        + 'total_oper_rev,oper_exp,tot_profit,int_income'
                        ),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    data_bench = dv.data_benchmark.copy()
    dv.data_benchmark = data_bench
    
    try:
        dv.data_benchmark = data_bench.iloc[3:]
    except ValueError:
        pass
    
    dv.remove_field('roe,net_assets')
    dv.remove_field('close')


# quarterly
def test_q():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    secs = '600030.SH,000063.SZ,000001.SZ'
    props = {'start_date': 20160609, 'end_date': 20170601, 'symbol': secs,
             'fields': ('open,close,'
                        + 'pb,net_assets,'
                        + 'total_oper_rev,oper_exp,'
                        + 'cash_paid_invest,'
                        + 'capital_stk,'
                        + 'roe'), 'freq': 1}
    
    dv.init_from_config(props, data_api=ds)
    dv.prepare_data()
    dv.save_dataview(folder_path=quarterly_path)


def test_q_get():
    dv = DataView()
    dv.load_dataview(folder_path=quarterly_path)
    res = dv.get("", 0, 0, 'total_oper_rev')
    assert set(res.index.values) == set(dv.dates[dv.dates >= dv.start_date])


def test_q_add_field():
    dv = DataView()
    dv.load_dataview(folder_path=quarterly_path)
    nrows, ncols = dv.data_q.shape
    n_securities = len(dv.data_d.columns.levels[0])
    
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv.add_field('net_inc_other_ops', ds)
    """
    dv.add_field('oper_rev', ds)
    dv.add_field('turnover', ds)
    """
    assert dv.data_q.shape == (nrows, ncols + 1 * n_securities)


def test_q_add_formula():
    dv = DataView()
    folder_path = '../output/prepared/20160609_20170601_freq=1D'
    dv.load_dataview(folder_path=quarterly_path)
    nrows, ncols = dv.data_d.shape
    n_securities = len(dv.data_d.columns.levels[0])
    
    formula = 'total_oper_rev / close'
    dv.add_formula('myvar1', formula, is_quarterly=False)
    df1 = dv.get_ts('myvar1')
    assert not df1.empty
    
    formula2 = 'Delta(oper_exp * myvar1 - open, 3)'
    dv.add_formula('myvar2', formula2, is_quarterly=False)
    df2 = dv.get_ts('myvar2')
    assert not df2.empty


if __name__ == "__main__":
    g = globals()
    g = {k: v for k, v in g.items() if k.startswith('test_') and callable(v)}

    # for test_name, test_func in g.items():
    for test_name in ['test_write', 'test_load', 'test_add_field', 'test_add_formula_directly',
                      'test_add_formula', 'test_dataview_universe',
                      'test_q', 'test_q_get', 'test_q_add_field', 'test_q_add_formula',
                      ]:
        test_func = g[test_name]
        print("\n==========\nTesting {:s}...".format(test_name))
        test_func()
    print("Test Complete.")
