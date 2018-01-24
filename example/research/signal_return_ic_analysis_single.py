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
    
    dv.save_dataview(dataview_folder)


def analyze_event():
    # --------------------------------------------------------------------------------
    # Step.1 load dataview
    dv = DataView()
    dv.load_dataview(dataview_folder)
    
    # --------------------------------------------------------------------------------
    # Step.3 get signal, benchmark and price data
    target_symbol = '600519.SH'
    price = dv.get_ts('close_adj', symbol=target_symbol)
    dv.add_formula('in_', 'open_adj / Delay(close_adj, 1)', is_quarterly=False)
    signal = dv.get_ts('in_', symbol=target_symbol).shift(1, axis=0)  # avoid look-ahead bias
    
    # Step.4 analyze!
    obj = SignalDigger(output_folder='../../output', output_format='pdf')

    obj.create_single_signal_report(signal, price, [1, 5, 9, 21], 6, mask=None,
                                    trade_condition={'cond1': {'column': 'quantile',
                                                             'filter': lambda x: x > 3,
                                                             'hold': 5},
                                                   'cond2': {'column': 'quantile',
                                                             'filter': lambda x: x > 5,
                                                             'hold': 5},
                                                   'cond3': {'column': 'quantile',
                                                             'filter': lambda x: x > 5,
                                                             'hold': 9},
                                                     })


if __name__ == "__main__":
    save_dataview()
    analyze_event()
