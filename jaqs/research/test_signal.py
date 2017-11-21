# encoding: utf-8

import numpy as np
import pandas as pd

from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs.research import SignalDigger
import jaqs.util as jutil

dataview_folder = jutil.join_relative_path('../output/prepared', 'test_signal')

def save_dataview():
    ds = RemoteDataService()
    ds.init_from_config()
    dv = DataView()
    
    props = {'start_date': 20140101, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': 'volume,turnover,float_mv,pb,total_mv',
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    # for convenience to check limit reachers
    dv.add_formula('limit_reached', 'Abs((open - Delay(close, 1)) / Delay(close, 1)) > 0.095', is_quarterly=False)
    
    dv.add_formula('random', 'StdDev(volume, 20)', is_quarterly=False)
    dv.add_formula('momentum', 'Return(close_adj, 20)', is_quarterly=False)
    # dv.add_formula('size', '', is_quarterly=False)
    
    dv.save_dataview(dataview_folder)


def analyze_signal():
    # --------------------------------------------------------------------------------
    # Step.1 load dataview
    dv = DataView()
    dv.load_dataview(dataview_folder)

    # --------------------------------------------------------------------------------
    # Step.2 calculate mask (to mask those ill data points)
    trade_status = dv.get_ts('trade_status')
    mask_sus = trade_status == u'停牌'.encode('utf-8')

    df_index_member = dv.get_ts('index_member')
    mask_index_member = ~(df_index_member > 0)

    dv.add_formula('limit_reached', 'Abs((open - Delay(close, 1)) / Delay(close, 1)) > 0.095', is_quarterly=False)
    df_limit_reached = dv.get_ts('limit_reached')
    mask_limit_reached = df_limit_reached > 0

    mask_all = np.logical_or(mask_sus, np.logical_or(mask_index_member, mask_limit_reached))

    # --------------------------------------------------------------------------------
    # Step.3 get signal, benchmark and price data
    # dv.add_formula('illi_daily', '(high - low) * 1000000000 / turnover', is_quarterly=False)
    # dv.add_formula('illi', 'Ewma(illi_daily, 11)', is_quarterly=False)
    
    # dv.add_formula('size', 'Log(float_mv)', is_quarterly=False)
    # dv.add_formula('value', '-1.0/pb', is_quarterly=False)
    # dv.add_formula('liquidity', 'Ts_Mean(volume, 22) / float_mv', is_quarterly=False)
    dv.add_formula('divert', '- Correlation(vwap_adj, volume, 10)', is_quarterly=False)
    
    signal = dv.get_ts('divert').shift(1, axis=0)  # avoid look-ahead bias
    price = dv.get_ts('close_adj')
    price_bench = dv.data_benchmark

    # Step.4 analyze!
    my_period = 5
    obj = SignalDigger(output_folder=jutil.join_relative_path('../output'),
                       output_format='pdf')
    obj.process_signal_before_analysis(signal, price=price,
                                       mask=mask_all,
                                       n_quantiles=5, period=my_period,
                                       benchmark_price=price_bench,
                                       )
    res = obj.create_full_report()
    
    # import cPickle as pickle
    # pickle.dump(res, open('_res.pic', 'w'))


if __name__ == "__main__":
    # save_dataview()
    analyze_signal()
