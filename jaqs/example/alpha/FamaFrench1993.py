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
import time

import pandas as pd

import jaqs.trade.analyze.analyze as ana
from jaqs.data.dataservice import RemoteDataService
from jaqs.data.dataview import DataView
from jaqs.trade import model
from jaqs.trade.backtest import AlphaBacktestInstance
from jaqs.trade.gateway import DailyStockSimGateway
from jaqs.trade.strategy import AlphaStrategy
from jaqs.util import fileio

dataview_dir_path = fileio.join_relative_path('../output/prepared/fama_french/dataview')
backtest_result_dir_path = fileio.join_relative_path('../output/fama_french')


def test_save_dataview(sub_folder='test_dataview'):
    ds = RemoteDataService()
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
    
    gateway = DailyStockSimGateway()
    gateway.init_from_config(props)
    
    context = model.Context(dataview=dv, gateway=gateway)
    
    stock_selector = model.StockSelector(context)
    stock_selector.add_filter(name='myselector', func=my_selector)
    
    strategy = AlphaStrategy(stock_selector=stock_selector, pc_method='equal_weight',
                             # revenue_model=signal_model
                             )
    
    bt = AlphaBacktestInstance()
    bt.init_from_config(props, strategy, context=context)
    
    bt.run_alpha()
    
    bt.save_results(folder_path=backtest_result_dir_path)


def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    dv = DataView()
    dv.load_dataview(folder_path=dataview_dir_path)
    
    ta.initialize(dataview=dv, file_folder=backtest_result_dir_path)
    
    print "process trades..."
    ta.process_trades()
    print "get daily stats..."
    ta.get_daily()
    #########################
    # ta.daily[symbol] # = df
    
    print "calc strategy return..."
    ta.get_returns()
    # position change info is huge!
    # print "get position change..."
    # ta.get_pos_change_info()
    
    selected_sec = []  # list(ta.universe)[:5]
    if len(selected_sec) > 0:
        print "Plot single securities PnL"
        for symbol in selected_sec:
            df_daily = ta.daily.get(symbol, None)
            if df_daily is not None:
                ana.plot_trades(df_daily, symbol=symbol, save_folder=backtest_result_dir_path)
    
    print "Plot strategy PnL..."
    ta.plot_pnl(backtest_result_dir_path)
    
    print "generate report..."
    static_folder = fileio.join_relative_path("trade/analyze/static")
    ta.gen_report(source_dir=static_folder, template_fn='report_template.html',
                  out_folder=backtest_result_dir_path,
                  selected=selected_sec)


if __name__ == "__main__":
    dv_subfolder_name = 'graham'
    t_start = time.time()
    
    test_save_dataview()
    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
