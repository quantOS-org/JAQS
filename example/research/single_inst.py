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

dataview_folder = '../../output/prepared/test_signal'


def save_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    props = {'start_date': 20160101, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': 'volume,turnover',
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    # for convenience to check limit reachers
    dv.add_formula('limit_reached', 'Abs((open - Delay(close, 1)) / Delay(close, 1)) > 0.095', is_quarterly=False)
    dv.add_formula('mask_limit_reached', 'limit_reached > 0', is_quarterly=False)
    
    dv.add_formula('mask_index_member', '!(index_member > 0)', is_quarterly=False)
    
    trade_status = dv.get_ts('trade_status')
    mask_sus = trade_status == u'停牌'
    dv.append_df(mask_sus, 'mask_sus', is_quarterly=False)
    
    # dv.add_formula('size', '', is_quarterly=False)
    
    dv.save_dataview(dataview_folder)


def analyze_event():
    # --------------------------------------------------------------------------------
    # Step.1 load dataview
    dv = DataView()
    dv.load_dataview(dataview_folder)

    # --------------------------------------------------------------------------------
    # Step.2 calculate mask (to mask those ill data points)
    mask_limit_reached = dv.get_ts('mask_limit_reached')
    mask_index_member = dv.get_ts('mask_index_member')
    mask_sus = dv.get_ts('mask_sus')
    
    mask_all = np.logical_or(mask_sus, np.logical_or(mask_index_member, mask_limit_reached))
    
    # --------------------------------------------------------------------------------
    # Step.3 get signal, benchmark and price data
    target_symbol = '600519.SH'
    price = dv.get_ts('close_adj', symbol=target_symbol)
    dv.add_formula('in_', 'open_adj / Delay(close_adj, 1)', is_quarterly=False)
    signal = dv.get_ts('in_', symbol=target_symbol).shift(1, axis=0)  # avoid look-ahead bias
    # Step.4 analyze!
    #from jaqs.research.signaldigger.digger import single_inst
    obj = SignalDigger(output_folder='../../output', output_format='pdf')
    obj.single_inst(signal, price, [1, 5, 9, 21], 6, mask=None,
                    buy_condition={'cond1': {'column': 'quantile',
                                    'filter': lambda x: x > 3,
                                    'hold': 5},
                          'cond2': {'column': 'quantile',
                                    'filter': lambda x: x > 5,
                                    'hold': 5},
                          'cond3': {'column': 'quantile',
                                    'filter': lambda x: x > 5,
                                    'hold': 9},
                                   })
    print('done')


if __name__ == "__main__":
    #save_dataview()
    analyze_event()
