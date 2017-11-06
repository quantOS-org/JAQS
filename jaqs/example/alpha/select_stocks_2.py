# -*- encoding: utf-8 -*-

"""
选择行业龙头股（价值白马）
1. 流通市值从大到小排列 （float_mv）
2. PB从小到大 （pb）
3. PE从小到大 （pe_ttm)
分别在行业内排序，然后计算每支股票的分数，取总分靠前的20%，
等权重，按月调仓
universe : hs300
init_balance = 1e8
start_date 20170101
end_date   20171001
"""
import time

from jaqs.data.dataservice import RemoteDataService
from jaqs.trade.backtest import AlphaBacktestInstance

from jaqs.util import fileio
import jaqs.trade.analyze.analyze as ana
from jaqs.trade.strategy import AlphaStrategy
from jaqs.trade.gateway import DailyStockSimGateway
from jaqs.trade import model
from jaqs.data.dataview import DataView

import pandas as pd

dataview_dir_path = fileio.join_relative_path('../output/prepared/select_stocks2')
backtest_result_dir_path = fileio.join_relative_path('../output/select_stocks2')


def test_save_dataview():
    ds = RemoteDataService()
    dv = DataView()

    props = {'start_date': 20170101, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': ('float_mv,pb,pe_ttm,sw2'),
             'freq': 1}

    dv.init_from_config(props, ds)
    dv.prepare_data()

    factor_formula = 'GroupQuantile(-float_mv, sw2, 10)'
    dv.add_formula('rank_mv', factor_formula, is_quarterly=False)

    factor_formula = 'GroupQuantile(If(pb >= 0.2, pb, 100), sw2, 10)'
    dv.add_formula('rank_pb', factor_formula, is_quarterly=False)

    factor_formula = 'GroupQuantile(If(pe_ttm >= 3, pe_ttm, 9999.0), sw2, 10)'
    dv.add_formula('rank_pe', factor_formula, is_quarterly=False)

    dv.save_dataview(folder_path=dataview_dir_path)


def my_selector(context, user_options=None):
    rank_mv = context.snapshot['rank_mv']
    rank_pb = context.snapshot['rank_pb']
    rank_pe = context.snapshot['rank_pe']

    rank = pd.DataFrame()
    rank['rank_total'] = rank_mv + rank_pb + rank_pe

    rank = rank.sort_values('rank_total', ascending=True)
    length = int(rank.shape[0] * 0.2)
    rank.iloc[: length] = 1.0
    rank.iloc[length: ] = 0.0
    return rank


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

    gateway = DailyStockSimGateway()
    gateway.init_from_config(props)

    context = model.Context(dataview=dv, gateway=gateway)

    stock_selector = model.StockSelector(context)
    stock_selector.add_filter(name='myrank', func=my_selector)

    strategy = AlphaStrategy(stock_selector=stock_selector, pc_method='equal_weight')

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
    print "get position change..."
    ta.get_pos_change_info()

    selected_sec = list(ta.universe)[:3]
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
