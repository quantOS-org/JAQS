# -*- encoding: utf-8 -*-

import time

from jaqs.data.dataservice import RemoteDataService
from jaqs.trade.strategy import AlphaStrategy

from jaqs.util import fileio
import jaqs.trade.analyze.analyze as ana
from jaqs.trade.backtest import AlphaBacktestInstance
from jaqs.trade.gateway import DailyStockSimGateway
from jaqs.trade import model
from jaqs.data.dataview import DataView


def save_dataview(sub_folder='test_dataview'):
    ds = RemoteDataService()
    dv = DataView()
    
    props = {'start_date': 20160104, 'end_date': 20171012, 'universe': '000300.SH',
             'fields': ('open,high,low,close,vwap,volume,turnover,'
                        # + 'pb,net_assets,'
                        + 's_fa_eps_basic,oper_exp,tot_profit,int_income'
                        ),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'close >= Delay(Ts_Max(close, 20), 1)'  # 20 days new high
    factor_name = 'new_high'
    dv.add_formula(factor_name, factor_formula, is_quarterly=False)
    
    dv.add_formula('total_profit_growth', formula='Return(tot_profit, 4)', is_quarterly=True)

    dv.add_formula('eps_ret', 'Return(s_fa_eps_basic, 4)', is_quarterly=True)
    dv.add_formula('rule1', '(eps_ret > 0.2)', is_quarterly=True)
    dv.add_formula('rule2', '(Delay(eps_ret, 1) > 0.2)', is_quarterly=True)
    '''
    df_rule1 = dv.get_ts('rule1')
    df_rule2 = dv.get_ts('rule2')
    import numpy as np
    df_mask = np.logical_and(df_rule1, df_rule2)
    daily_dic = dict()
    for idx, row in df_mask.iterrows():
        daily_dic[idx] = row.index.values[row.values]
    '''
    
    dv.save_dataview(folder_path=fileio.join_relative_path('../output/prepared'), sub_folder=sub_folder)


def test_alpha_strategy_dataview():
    dv_subfolder_name = 'test_dataview'
    save_dataview(sub_folder=dv_subfolder_name)
    
    dv = DataView()
    fullpath = fileio.join_relative_path('../output/prepared', dv_subfolder_name)
    dv.load_dataview(folder=fullpath)
    
    # ------
    df_rule1 = dv.get_ts('rule1')
    df_rule2 = dv.get_ts('rule2')
    import numpy as np
    df_mask = np.logical_and(df_rule1, df_rule2)
    from collections import OrderedDict
    daily_dic = OrderedDict()
    for idx, row in df_mask.iterrows():
        daily_dic[idx] = row.index.values[row.values]
    # ------
    print


if __name__ == "__main__":
    test_alpha_strategy_dataview()
