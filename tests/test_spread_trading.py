# -*- encoding: utf-8 -*-

import json


from jaqs.util import fileio
from jaqs.trade import model
from jaqs.trade import common
import jaqs.trade.analyze.analyze as ana
from jaqs.data.dataservice import RemoteDataService
from jaqs.example.eventdriven.spread import SpreadStrategy
from jaqs.trade.backtest import EventBacktestInstance
from jaqs.trade.gateway import BarSimulatorGateway

backtest_result_dir_path = fileio.join_relative_path('../output/event_driven')


def test_double_ma():
    # prop_file_path = fileio.join_relative_path("etc/backtest.json")
    # print prop_file_path
    # prop_file = open(prop_file_path, 'r')
    
    # props = json.load(prop_file)
    props = {"symbol": "rb1710.SHF,hc1710.SHF",
             "start_date": 20170510,
             "end_date": 20170930,
             "bar_type": "MIN",
             "init_balance": 1e7,
             "future_commission_rate": 0.00002,
             "stock_commission_rate": 0.0001,
             "stock_tax_rate": 0.0000}
    
    # props['bar_type'] = 'DAILY'
    
    enum_props = {'bar_type': common.QUOTE_TYPE}
    for k, v in enum_props.iteritems():
        props[k] = v.to_enum(props[k])
    
    strategy = SpreadStrategy()
    gateway = BarSimulatorGateway()
    data_service = RemoteDataService()

    context = model.Context(data_api=data_service, gateway=gateway)
    
    bt = EventBacktestInstance()
    bt.init_from_config(props, strategy, context=context)
    
    bt.run()
    
    bt.save_results(folder_path=backtest_result_dir_path)
    
    report = bt.generate_report(output_format="plot")
    # print report.trades[:100]
    # for pnl in report.daily_pnls:
    #     print pnl.date, pnl.trade_pnl, pnl.hold_pnl,pnl.total_pnl, pnl.positions.get('600030.SH')


def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    
    ds = RemoteDataService()
    
    ta.initialize(data_server_=ds, file_folder=backtest_result_dir_path)
    
    print "process trades..."
    ta.process_trades()
    print "get daily stats..."
    ta.get_daily()
    print "calc strategy return..."
    ta.get_returns(consider_commission=True)
    # position change info is huge!
    # print "get position change..."
    # ta.get_pos_change_info()
    
    selected_sec = list(ta.universe)[:2]
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
    test_double_ma()
    # test_backtest_analyze()
    print "test success."
