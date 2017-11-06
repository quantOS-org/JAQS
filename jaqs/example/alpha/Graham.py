# -*- encoding: utf-8 -*-

"""
Weekly rebalance
1. pe ratio < 15
2. pb ratio < 1.5
3. inc_earning_per_share > 0
4. inc_profit_before_tax > 0
5. current_ratio > 2
6. quick_ratio > 1

universe : hs300
init_balance = 1e8
start_date 20140101
end_date   20170301
"""
import time

import numpy as np
import pandas as pd

import jaqs.trade.analyze.analyze as ana
from jaqs.data.dataservice import RemoteDataService
from jaqs.data.dataview import DataView
from jaqs.trade import model
from jaqs.trade.backtest import AlphaBacktestInstance
from jaqs.trade.gateway import DailyStockSimGateway
from jaqs.trade.strategy import AlphaStrategy
from jaqs.util import fileio

dataview_dir_path = fileio.join_relative_path('../output/prepared/Graham_dataview')
backtest_result_dir_path = fileio.join_relative_path('../output/Graham')


def test_save_dataview():
    ds = RemoteDataService()
    dv = DataView()
    
    props = {'start_date': 20150101, 'end_date': 20170930, 'universe': '000905.SH',
             'fields': ('tot_cur_assets,tot_cur_liab,inventories,pre_pay,deferred_exp,'
                        'eps_basic,ebit,pe,pb,float_mv,sw1'),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'pe < 30'
    dv.add_formula('pe_condition', factor_formula, is_quarterly=False)
    factor_formula = 'pb < 3'
    dv.add_formula('pb_condition', factor_formula, is_quarterly=False)
    factor_formula = 'Return(eps_basic, 4) > 0'
    dv.add_formula('eps_condition', factor_formula, is_quarterly=True)
    factor_formula = 'Return(ebit, 4) > 0'
    dv.add_formula('ebit_condition', factor_formula, is_quarterly=True)
    factor_formula = 'tot_cur_assets/tot_cur_liab > 2'
    dv.add_formula('current_condition', factor_formula, is_quarterly=True)
    factor_formula = '(tot_cur_assets - inventories - pre_pay - deferred_exp)/tot_cur_liab > 1'
    dv.add_formula('quick_condition', factor_formula, is_quarterly=True)
    
    dv.add_formula('mv_rank', 'Rank(float_mv)', is_quarterly=False)
    
    dv.save_dataview(folder_path=dataview_dir_path)


def signal_size(context, user_options=None):
    mv_rank = context.snapshot_sub['mv_rank']
    s = np.sort(mv_rank.values)[::-1]
    if len(s) > 0:
        critical = s[-5] if len(s) > 5 else np.min(s)
        mask = mv_rank < critical
        mv_rank[mask] = 0.0
        mv_rank[~mask] = 1.0
    return mv_rank


def my_selector(context, user_options=None):
    #
    pb_selector = context.snapshot['pb_condition']
    pe_selector = context.snapshot['pe_condition']
    eps_selector = context.snapshot['eps_condition']
    ebit_selector = context.snapshot['ebit_condition']
    current_selector = context.snapshot['current_condition']
    quick_selector = context.snapshot['quick_condition']
    #
    # result = pb_selector & pe_selector & eps_selector & ebit_selector & current_selector & quick_selector
    merge = pd.concat([pb_selector,
                       pe_selector, eps_selector, ebit_selector, current_selector, quick_selector], axis=1)
    
    result = np.all(merge, axis=1)
    mask = np.all(merge.isnull().values, axis=1)
    result[mask] = False
    return pd.DataFrame(result, index=merge.index, columns=['lksjdf'])


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
    
    signal_model = model.FactorRevenueModel(context)
    signal_model.add_signal(name='signalsize', func=signal_size)
    
    strategy = AlphaStrategy(stock_selector=stock_selector, pc_method='factor_value_weight',
                             revenue_model=signal_model
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
    t_start = time.time()
    
    test_save_dataview()
    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
