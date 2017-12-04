# -*- encoding: utf-8 -*-

"""
pe_ttm between 10 and 20
net profit growth between 0.2 and 4
equal weight
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

import jaqs.util as jutil
import jaqs.trade.analyze as ana
from jaqs.trade import PortfolioManager
from jaqs.trade import AlphaStrategy
from jaqs.trade import AlphaTradeApi
from jaqs.trade import model
from jaqs.data import DataView

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

dataview_dir_path = '../../output/select_stocks_pe_profit/dataview'
backtest_result_dir_path = '../../output/select_stocks_pe_profit'


def test_save_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    dv = DataView()
    
    props = {'start_date': 20170101, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': 'pe_ttm,net_profit_incl_min_int_inc',
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'Return(net_profit_incl_min_int_inc, 4)'
    factor_name = 'net_profit_growth'
    dv.add_formula(factor_name, factor_formula, is_quarterly=True)
    
    dv.save_dataview(folder_path=dataview_dir_path)


def test_alpha_strategy_dataview():
    dv = DataView()

    dv.load_dataview(folder_path=dataview_dir_path)
    
    props = {
        "benchmark": "000300.SH",
        "universe": ','.join(dv.symbol),
    
        "start_date": dv.start_date,
        "end_date": dv.end_date,
    
        "period": "month",
        "days_delay": 0,
    
        "init_balance": 1e8,
        "position_ratio": 1.0,
        }
    props.update(data_config)
    props.update(trade_config)

    trade_api = AlphaTradeApi()
    trade_api.init_from_config(props)

    def selector_growth(context, user_options=None):
        growth_rate = context.snapshot['net_profit_growth']
        return (growth_rate >= 0.2) & (growth_rate <= 4)
    
    def selector_pe(context, user_options=None):
        pe_ttm = context.snapshot['pe_ttm']
        return (pe_ttm >= 10) & (pe_ttm <= 20)
    
    stock_selector = model.StockSelector()
    stock_selector.add_filter(name='net_profit_growth', func=selector_growth)
    stock_selector.add_filter(name='pe', func=selector_pe)

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
    t_start = time.time()
    
    test_save_dataview()
    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))

