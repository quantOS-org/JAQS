# encoding: utf-8

from __future__ import unicode_literals
import numpy as np
import pandas as pd

from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs.research import SignalDigger
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)

dataview_folder = '../output/prepared/test_signal'


def test_save_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    props = {'start_date': 20170301, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': 'volume,turnover,float_mv,pb,total_mv',
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    dv.save_dataview(dataview_folder)


def test_analyze_signal():
    # --------------------------------------------------------------------------------
    # Step.1 load dataview
    dv = DataView()
    dv.load_dataview(dataview_folder)

    # --------------------------------------------------------------------------------
    # Step.2 calculate mask (to mask those ill data points)
    trade_status = dv.get_ts('trade_status')
    mask_sus = trade_status == u'停牌'

    df_index_member = dv.get_ts('index_member')
    mask_index_member = ~(df_index_member > 0)

    dv.add_formula('limit_reached', 'Abs((open - Delay(close, 1)) / Delay(close, 1)) > 0.095', is_quarterly=False)
    df_limit_reached = dv.get_ts('limit_reached')
    mask_limit_reached = df_limit_reached > 0

    mask_all = np.logical_or(mask_sus, np.logical_or(mask_index_member, mask_limit_reached))

    # --------------------------------------------------------------------------------
    # Step.3 get signal, benchmark and price data
    dv.add_formula('divert', '- Correlation(vwap_adj, volume, 10)', is_quarterly=False)
    
    signal = dv.get_ts('divert').shift(1, axis=0)  # avoid look-ahead bias
    price = dv.get_ts('close_adj')
    price_bench = dv.data_benchmark

    # Step.4 analyze!
    my_period = 5
    obj = SignalDigger(output_folder='../output/test_signal', output_format='pdf')
    obj.process_signal_before_analysis(signal, price=price,
                                       mask=mask_all,
                                       n_quantiles=5, period=my_period,
                                       benchmark_price=price_bench,
                                       )
    res = obj.create_full_report()
    

if __name__ == "__main__":
    test_save_dataview()
    test_analyze_signal()
