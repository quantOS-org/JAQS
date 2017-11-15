# encoding: utf-8

import time
from jaqs.data.dataservice import RemoteDataService
from jaqs.trade import model, common
from jaqs.trade.realinstance import RealInstance
from jaqs.trade.backtest import EventBacktestInstance
from jaqs.example.eventdriven.market_making import RealStrategy
from jaqs.trade.gateway import RealTimeTradeApi, BacktestTradeApi
from jaqs.trade.portfoliomanager import PortfolioManager
import jaqs.util as jutil
import jaqs.trade.analyze.analyze as ana

result_dir_path = jutil.join_relative_path('../output/test_consistency')
is_backtest = True


def consis():
    
    if is_backtest:
        props = {"symbol": "rb1710.SHF,hc1710.SHF",
                 "start_date": 20170510,
                 "end_date": 20170530,
                 "bar_type": "MIN",
                 "init_balance": 2e4,
                 "future_commission_rate": 0.00002,
                 "stock_commission_rate": 0.0001,
                 "stock_tax_rate": 0.0000}

        # props['bar_type'] = 'DAILY'

        enum_props = {'bar_type': common.QUOTE_TYPE}
        for k, v in enum_props.iteritems():
            props[k] = v.to_enum(props[k])

        tapi = BacktestTradeApi()
        ins = EventBacktestInstance()
        
    else:
        props = {'symbol': 'IC1712.CFE,rb1801.SHF'}
        tapi = RealTimeTradeApi()
        ins = RealInstance()

    tapi.use_strategy(3)
    
    ds = RemoteDataService()
    strat = RealStrategy()
    pm = PortfolioManager(strategy=strat)
    
    context = model.Context(data_api=ds, trade_api=tapi, gateway=None, instance=ins,
                            strategy=strat, pm=pm)
    
    if not is_backtest:
        ds.subscribe(props['symbol'])

    ins.init_from_config(props, strat)
    ins.run()
    if not is_backtest:
        time.sleep(1000)
    ins.save_results(folder_path=result_dir_path)
    
    # order dependent is not good! we should make it not dependent or hard-code the order
    # props = {'symbol': '000001.SZ,600001.SH'}
    # props = {'symbol': 'CFCICA.JZ,600001.SH'}
    
    # ds.subscribe('CFCICA.JZ,000001.SZ')
    # ds.subscribe('rb1710.SHF')


def test_backtest_analyze():
    ta = ana.EventAnalyzer()
    
    ds = RemoteDataService()
    
    ta.initialize(data_server_=ds, file_folder=result_dir_path)
    
    print "process trades..."
    ta.process_trades()
    print "get daily stats..."
    ta.get_daily()
    print "calc strategy return..."
    ta.get_returns(consider_commission=False)
    # position change info is huge!
    # print "get position change..."
    # ta.get_pos_change_info()
    
    selected_sec = list(ta.universe)[:2]
    if len(selected_sec) > 0:
        print "Plot single securities PnL"
        for symbol in selected_sec:
            df_daily = ta.daily.get(symbol, None)
            if df_daily is not None:
                ana.plot_trades(df_daily, symbol=symbol, save_folder=result_dir_path)
    
    print "Plot strategy PnL..."
    ta.plot_pnl(result_dir_path)
    
    print "generate report..."
    static_folder = jutil.join_relative_path("trade/analyze/static")
    ta.gen_report(source_dir=static_folder, template_fn='report_template.html',
                  out_folder=result_dir_path,
                  selected=selected_sec)


if __name__ == "__main__":
    consis()
    test_backtest_analyze()
