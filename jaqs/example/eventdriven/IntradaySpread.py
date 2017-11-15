# encoding: utf-8

import numpy as np

from jaqs.trade.strategy import EventDrivenStrategy
from jaqs.trade import common, model

from jaqs.data.dataservice import RemoteDataService
from jaqs.trade.backtest import EventBacktestInstance
from jaqs.trade.tradegateway import BacktestTradeApi
from jaqs.trade.portfoliomanager import PortfolioManager
import jaqs.util as jutil
import jaqs.trade.analyze.analyze as ana

result_dir_path = jutil.join_relative_path('../output/calendar_spread')


class SpreadStrategy(EventDrivenStrategy):
    """"""
    def __init__(self):
        super(SpreadStrategy, self).__init__()
        self.symbol = ''
        
        self.s1 = ''
        self.s2 = ''
        self.quote1 = None
        self.quote2 = None

        self.window = 8
        self.idx = -1
        self.spread_arr = np.empty(self.window, dtype=float)
        
        self.threshold = 6.5
        
    def init_from_config(self, props):
        super(SpreadStrategy, self).init_from_config(props)
        self.symbol = props.get('symbol')
        self.init_balance = props.get('init_balance')
        
        self.s1, self.s2 = self.symbol.split(',')
    
    def on_cycle(self):
        pass
    
    def long_spread(self, quote1, quote2):
        self.ctx.trade_api.place_order(quote1.symbol, common.ORDER_ACTION.BUY, quote1.close + 1, 1)
        self.ctx.trade_api.place_order(quote2.symbol, common.ORDER_ACTION.SELL, quote2.close - 1, 1)

    def short_spread(self, quote1, quote2):
        self.ctx.trade_api.place_order(quote2.symbol, common.ORDER_ACTION.BUY, quote2.close + 1, 1)
        self.ctx.trade_api.place_order(quote1.symbol, common.ORDER_ACTION.SELL, quote1.close - 1, 1)

    def on_quote(self, quote_dic):
        q1 = quote_dic.get(self.s1)
        q2 = quote_dic.get(self.s2)
        self.quote1 = q1
        self.quote2 = q2

        if self.quote1.time > 142800 and self.quote1.time < 200000:
            # self.cancel_all_orders()
            # self.liquidate(self.quote1, 3, tick_size=1.0, pos=self.ctx.pm.get_pos(self.quote1.symbol))
            # self.liquidate(self.quote2, 3, tick_size=1.0, pos=self.ctx.pm.get_pos(self.quote2.symbol))
            # return
            pass
        
        spread = q1.close - q2.close
        self.idx += 1
        self.spread_arr[self.idx % self.window] = spread
        
        if self.idx <= self.window:
            return
        
        mean = self.spread_arr.mean()
        if (spread - mean) > self.threshold:
            self.short_spread(q1, q2)
        elif (spread - mean) < -self.threshold:
            self.long_spread(q1, q2)
    
    def on_trade(self, ind):
        print(ind)
        print(self.ctx.pm.get_position(ind.symbol))


props = {
    "symbol"                : "rb1801.SHF,hc1801.SHF",
    "start_date"            : 20170801,
    "end_date"              : 20170910,
    "bar_type"              : "1M",
    "init_balance"          : 3e4,
    "commission_rate": 2E-4
}


def run_strategy():
    tapi = BacktestTradeApi()
    ins = EventBacktestInstance()
    
    ds = RemoteDataService()
    strat = SpreadStrategy()
    pm = PortfolioManager()
    
    context = model.Context(data_api=ds, trade_api=tapi, instance=ins,
                            strategy=strat, pm=pm)
    
    ins.init_from_config(props)
    ins.run()
    ins.save_results(folder_path=result_dir_path)


def analyze():
    ta = ana.EventAnalyzer()
    
    ds = RemoteDataService()
    ds.init_from_config()
    
    ta.initialize(data_server_=ds, file_folder=result_dir_path)
    
    ta.do_analyze(result_dir=result_dir_path, selected_sec=props['symbol'].split(','))


if __name__ == "__main__":
    run_strategy()
    analyze()
