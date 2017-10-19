# encoding: utf-8

import numpy as np
import pandas as pd

from jaqs.data.dataview import DataView
from jaqs.data.dataservice import RemoteDataService
from jaqs.util import fileio


def build_stock_selection_factor():
    ds = RemoteDataService()
    dv = DataView()

    props = {'start_date': 20120101, 'end_date': 20170901, 'universe': '000300.SH',
             # 'symbol': 'rb1710.SHF,rb1801.SHF',
             'fields': ('open,high,low,close,vwap,volume,turnover,'
                        # + 'pb,net_assets,'
                        + 's_fa_eps_basic,oper_exp,tot_profit,int_income'
                        ),
             'freq': 1}

    dv.init_from_config(props, ds)
    dv.prepare_data()

    dv.add_formula('eps_ret', 'Return(s_fa_eps_basic, 4)', is_quarterly=True)
    dv.add_formula('rule1', '(eps_ret > 0.2) && (Delay(eps_ret, 1) > 0.2)', is_quarterly=True)
    dv.add_formula('rule2', 'close > Ts_Max(close, 120)', is_quarterly=False)
    # dv.add_formula('ytan', 'rule1 && rule2', is_quarterly=False)

    dv.add_formula('ret20', 'Delay(Return(close_adj, 20), -20)', is_quarterly=False)

    dv.save_dataview(folder_path=fileio.join_relative_path('../output/prepared'))
