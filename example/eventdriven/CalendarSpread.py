# encoding: utf-8

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import statsmodels.api as sm

from jaqs.trade import EventDrivenStrategy
from jaqs.trade import common, model

from jaqs.data import RemoteDataService
from jaqs.trade import EventBacktestInstance
from jaqs.trade import BacktestTradeApi
from jaqs.trade import PortfolioManager
import jaqs.trade.analyze as ana
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

result_dir_path = '../../output/calendar_spread'


class CalendarSpread(EventDrivenStrategy):
    def __init__(self):
        super(CalendarSpread, self).__init__()
        
        self.symbol      = ''
        self.s1          = ''
        self.s2          = ''
        self.quote1      = None
        self.quote2      = None
        
        self.bufferSize  = 0
        self.bufferCount = 0
        self.spreadList  = ''
    
    def init_from_config(self, props):
        super(CalendarSpread, self).init_from_config(props)
        
        self.symbol       = props.get('symbol')
        self.init_balance = props.get('init_balance')
        self.bufferSize   = props.get('bufferSize')
        self.s1, self.s2  = self.symbol.split(',')
        self.spreadList = np.zeros(self.bufferSize)
        
        self.output = True

    def on_cycle(self):
        pass
    
    def on_tick(self, quote):
        pass

    def buy(self, quote, price, size):
        self.ctx.trade_api.place_order(quote.symbol, 'Buy', price, size)

    def sell(self, quote, price, size):
        self.ctx.trade_api.place_order(quote.symbol, 'Sell', price, size)

    def long_spread(self, quote1, quote2):
        self.buy(quote1, quote1.close, 1)
        self.sell(quote2, quote2.close, 1)

    def short_spread(self, quote1, quote2):
        self.buy(quote2, quote2.close, 1)
        self.sell(quote1, quote1.close, 1)
    
    def on_bar(self, quote):
    
        q1 = quote.get(self.s1)
        q2 = quote.get(self.s2)
        self.quote1 = q1
        self.quote2 = q2
    
        spread = q1.close - q2.close
    
        self.spreadList[0:self.bufferSize - 1] = self.spreadList[1:self.bufferSize]
        self.spreadList[-1] = spread
        self.bufferCount += 1
    
        if self.bufferCount <= self.bufferSize:
            return
    
        X, y = np.array(range(self.bufferSize)), np.array(self.spreadList)
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        X = sm.add_constant(X)
    
        est = sm.OLS(y, X)
        est = est.fit()
    
        if est.pvalues[1] < 0.05:
            if est.params[1] < 0:
                self.short_spread(q1, q2)
            else:
                self.long_spread(q1, q2)
                
    def on_trade(self, ind):
        if self.output:
            print("\nStrategy on trade: ")
            print(ind)
            print(self.ctx.pm.get_trade_stat(ind.symbol))
        self.pos = self.ctx.pm.get_pos(ind.symbol)

    def on_order_status(self, ind):
        if self.output:
            print("\nStrategy on order status: ")
            print(ind)
        
    def on_task_rsp(self, rsp):
        if self.output:
            print("\nStrategy on task rsp: ")
            print(rsp)

    def on_order_rsp(self, rsp):
        if self.output:
            print("\nStrategy on order rsp: ")
            print(rsp)
    
    def on_task_status(self, ind):
        if self.output:
            print("\nStrategy on task ind: ")
            print(ind)


props = {
    "symbol"                : "ru1801.SHF,ru1805.SHF",
    "start_date"            : 20170701,
    "end_date"              : 20171030,
    "bar_type"              : "1d",
    "init_balance"          : 2e4,
    "bufferSize"            : 20,
    "commission_rate": 2E-4
}
props.update(data_config)
props.update(trade_config)


def run_strategy():
    tapi = BacktestTradeApi()
    ins = EventBacktestInstance()

    ds = RemoteDataService()
    strat = CalendarSpread()
    pm = PortfolioManager()

    context = model.Context(data_api=ds, trade_api=tapi, instance=ins,
                            strategy=strat, pm=pm)

    ins.init_from_config(props)
    ins.run()
    ins.save_results(folder_path=result_dir_path)


def analyze():
    ta = ana.EventAnalyzer()
    
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    ta.initialize(data_server_=ds, file_folder=result_dir_path)
    
    ta.do_analyze(result_dir=result_dir_path, selected_sec=props['symbol'].split(','))


if __name__ == "__main__":
    run_strategy()
    analyze()
