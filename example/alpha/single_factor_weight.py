# -*- encoding: utf-8 -*-

"""
Signal_model : 流通市值
按月调仓，factor_value_weight

universe : hs300
init_balance = 1e8
start_date 20170101
end_date   20171001

"""
from __future__ import print_function
from __future__ import absolute_import
import time

from jaqs.data import RemoteDataService
from jaqs.trade import AlphaBacktestInstance
from jaqs.trade import PortfolioManager

import jaqs.util as jutil
import jaqs.trade.analyze as ana
from jaqs.trade import AlphaStrategy
from jaqs.trade import AlphaTradeApi
from jaqs.trade import model
from jaqs.data import DataView

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

dataview_dir_path = '../../output/single_factor_weight/dataview'
backtest_result_dir_path = '../../output/single_factor_weight'


def test_save_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()

    props = {'start_date': 20170201, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': ('float_mv,sw2,sw1'),
             'freq': 1}

    dv.init_from_config(props, ds)
    dv.prepare_data()

    factor_formula = 'GroupQuantile(float_mv, sw1, 10)'
    dv.add_formula('gq30', factor_formula, is_quarterly=False)

    dv.save_dataview(folder_path=dataview_dir_path)


def test_alpha_strategy_dataview():
    dv = DataView()

    dv.load_dataview(folder_path=dataview_dir_path)

    props = {
        "benchmark": "000300.SH",
        "universe": ','.join(dv.symbol),

        "start_date": 20170131,
        "end_date": dv.end_date,

        "period": "month",
        "days_delay": 0,

        "init_balance": 1e9,
        "position_ratio": 1.0,
    }
    props.update(data_config)
    props.update(trade_config)

    trade_api = AlphaTradeApi()

    def singal_gq30(context, user_options=None):
        import numpy as np
        res = np.power(context.snapshot['gq30'], 8)
        return res
    
    signal_model = model.FactorSignalModel()
    signal_model.add_signal('signal_gq30', singal_gq30)

    strategy = AlphaStrategy(signal_model=signal_model, pc_method='factor_value_weight')
    pm = PortfolioManager()

    bt = AlphaBacktestInstance()
    
    context = model.Context(dataview=dv, instance=bt, strategy=strategy, trade_api=trade_api, pm=pm)
    
    signal_model.register_context(context)

    bt.init_from_config(props)

    bt.run_alpha()

    bt.save_results(folder_path=backtest_result_dir_path)
    

def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    dv = DataView()
    dv.load_dataview(folder_path=dataview_dir_path)

    ta.initialize(dataview=dv, file_folder=backtest_result_dir_path)

    ta.do_analyze(result_dir=backtest_result_dir_path, selected_sec=list(ta.universe)[:3])


if __name__ == "__main__":
    t_start = time.time()

    test_save_dataview()
    test_alpha_strategy_dataview()
    test_backtest_analyze()

    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))
