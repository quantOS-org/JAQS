# encoding: utf-8

import numpy as np
import pandas as pd

from jaqs.data.dataview import DataView
from jaqs.data.dataservice import RemoteDataService
from jaqs.research import signaldigger
from jaqs.util import fileio


def save_dataview(data_folder_name):
    ds = RemoteDataService()
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
    
    dv.save_dataview(folder_path=fileio.join_relative_path('../output/prepared'), sub_folder=data_folder_name)


def analyze_factor(data_folder_name):
    # --------------------------------------------------------------------------------
    # Step.1 load dataview
    dv = DataView()
    fullpath = fileio.join_relative_path('../output/prepared', data_folder_name)
    dv.load_dataview(folder=fullpath)

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
    # Step.3 get factor, benchmark and price data
    # dv.add_formula('illi_daily', '(high - low) * 1000000000 / turnover', is_quarterly=False)
    # dv.add_formula('illi', 'Ewma(illi_daily, 11)', is_quarterly=False)
    
    # dv.add_formula('size', 'Log(float_mv)', is_quarterly=False)
    # dv.add_formula('value', '-1.0/pb', is_quarterly=False)
    # dv.add_formula('liquidity', 'Ts_Mean(volume, 22) / float_mv', is_quarterly=False)
    
    factor = dv.get_ts('close').shift(1, axis=0)  # avoid look-ahead bias
    price = dv.get_ts('close_adj')
    price_bench = dv.data_benchmark

    # Step.4 analyze!
    my_period = 22
    obj = signaldigger.digger.SignalDigger(output_folder=fileio.join_relative_path('../output'),
                                           output_format='pdf')
    obj.process_factor_before_analysis(factor, price=price,
                                       mask=mask_all,
                                       n_quantiles=5, period=my_period,
                                       benchmark_price=price_bench,
                                       )
    res = obj.create_full_report()
    
    # import cPickle as pickle
    # pickle.dump(res, open('_res.pic', 'w'))


if __name__ == "__main__":
    sub_folder_name = 'test_signal'
    # save_dataview(sub_folder_name)
    analyze_factor(sub_folder_name)
