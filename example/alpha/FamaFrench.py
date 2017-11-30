# -*- encoding: utf-8 -*-

"""
Weekly rebalance

1. size: float market value
2. value: book equity: stockholders equity + redemption/liquidation + deferred tax - postretirement benefits

universe : hs300
init_balance = 1e8
start_date 20140101
end_date   20170301
"""
from __future__ import print_function
from __future__ import absolute_import
import time

import pandas as pd

import jaqs.trade.analyze as ana
from jaqs.data import RemoteDataService
from jaqs.data import DataView
from jaqs.trade import model
from jaqs.trade import PortfolioManager
from jaqs.trade import AlphaBacktestInstance
from jaqs.trade import AlphaTradeApi
from jaqs.trade import AlphaStrategy
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

dataview_dir_path = '../../output/fama_french/dataview'
backtest_result_dir_path = '../../output/fama_french'


def test_save_dataview(sub_folder='test_dataview'):
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    props = {'start_date': 20150101, 'end_date': 20170930, 'universe': '000905.SH',
             'fields': ('float_mv,tot_shrhldr_eqy_excl_min_int,deferred_tax_assets,sw2'),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'Quantile(-float_mv,5)'
    dv.add_formula('rank_mv', factor_formula, is_quarterly=False)
    
    factor_formula = 'Quantile(float_mv/(tot_shrhldr_eqy_excl_min_int+deferred_tax_assets), 5)'
    dv.add_formula('rank_pb', factor_formula, is_quarterly=False)
    
    dv.save_dataview(folder_path=dataview_dir_path)


def my_selector(context, user_options=None):
    dv = context.dataview
    
    rank_mv = dv.get_snapshot(context.trade_date, fields='rank_mv')
    rank_pb = dv.get_snapshot(context.trade_date, fields='rank_pb')
    # rank_pe = dv.get_snapshot(context.trade_date, fields='rank_pe')
    
    rank = pd.DataFrame()
    rank['rank_total'] = rank_mv['rank_mv'] + rank_pb['rank_pb']
    
    rank = rank.sort_values('rank_total', ascending=True)
    length = int(rank.shape[0] * 0.2)
    return rank.isin(rank.head(length))


def test_alpha_strategy_dataview():
    dv = DataView()
    
    dv.load_dataview(folder_path=dataview_dir_path)
    
    props = {
        "start_date": dv.start_date,
        "end_date": dv.end_date,
        
        "period": "week",
        "days_delay": 0,
        
        "init_balance": 1e8,
        "position_ratio": 1.0,
    }
    props.update(data_config)
    props.update(trade_config)
    
    trade_api = AlphaTradeApi()
    
    stock_selector = model.StockSelector()
    stock_selector.add_filter(name='myselector', func=my_selector)
    
    strategy = AlphaStrategy(stock_selector=stock_selector, pc_method='equal_weight')
    pm = PortfolioManager()
    
    bt = AlphaBacktestInstance()
    
    context = model.Context(dataview=dv, instance=bt, strategy=strategy, trade_api=trade_api, pm=pm)
    stock_selector.register_context(context)

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
    dv_subfolder_name = 'graham'
    t_start = time.time()
    
    test_save_dataview()
    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))
