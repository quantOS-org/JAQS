# -*- encoding: utf-8 -*-

import json


from jaqs.util import fileio
from jaqs.trade import model
from jaqs.trade import common
from jaqs.data.dataservice import RemoteDataService
from jaqs.example.eventdriven.doubleMaStrategy import DoubleMaStrategy
from jaqs.trade.backtest import EventBacktestInstance
from jaqs.trade.gateway import BacktestTradeApi


def test_double_ma():
    prop_file_path = fileio.join_relative_path("etc/backtest.json")
    print prop_file_path
    prop_file = open(prop_file_path, 'r')
    
    props = json.load(prop_file)
    
    enum_props = {'bar_type': common.QUOTE_TYPE}
    for k, v in enum_props.iteritems():
        props[k] = v.to_enum(props[k])
    
    # strategy   = CtaStrategy()
    strategy = DoubleMaStrategy()
    gateway = BacktestTradeApi()
    data_service = RemoteDataService()

    context = model.Context(data_api=data_service, gateway=gateway)
    
    backtest = EventBacktestInstance()
    backtest.init_from_config(props, strategy, context=context)
    
    # trade.run()
    backtest.run()
    report = backtest.generate_report(output_format="")
    # print report.trades[:100]
    # for pnl in report.daily_pnls:
    #     print pnl.date, pnl.trade_pnl, pnl.hold_pnl,pnl.total_pnl, pnl.positions.get('600030.SH')


if __name__ == "__main__":
    test_double_ma()
    print "test success."
