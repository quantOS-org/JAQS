# encoding: utf-8

from __future__ import print_function
from __future__ import absolute_import
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
import numpy as np

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


class QuoteBuffer(object):
    __slots__ = ['_window', '_cnt', '_queue', '_prices', 'symbol', 'tick_size', 'q', 'p']
    
    def __init__(self, window, symbol="", tick_size=1.0):
        self._window = window
        self._cnt = 0
        self._queue = Queue()
        self._prices = np.empty(self.window, dtype=float)
        
        self.symbol = symbol
        self.tick_size = tick_size
        
        self.q = None  # latest quote
        self.p = 0.0  # latest price
        
    @property
    def window(self):
        return self._window
    
    @property
    def full(self):
        return self._cnt > self._window
    
    def enqueue(self, quote):
        self.q = quote
        p = self.q.vwap
        if np.isnan(p) or np.isinf(p) or np.abs(p) < 1e-8:
            pass
        else:
            self.p = p
            
        self._queue.put(self.q)
        self._cnt += 1
        if self._cnt > self._window:
            self._queue.get()

        self._prices[: -1] = self._prices[1:]
        self._prices[-1] = self.p

    
class SpreadBuffer(object):
    __slots__ = ['_window', 'prices', 's1', 's2',
                 'tick_size1', 'tick_size2',
                 'quote_buffer1', 'quote_buffer2']
    
    def __init__(self, window, symbols, tick_sizes):
        self._window = window
        self.prices = np.empty(self._window, dtype=float)
        
        if isinstance(symbols, str):
            self.s1, self.s2 = symbols.split(',')
        elif isinstance(symbols, (tuple, list)):
            self.s1, self.s2 = symbols
        else:
            raise ValueError()
        
        self.tick_size1, self.tick_size2 = tick_sizes
        self.quote_buffer1 = QuoteBuffer(self._window, self.s1, self.tick_size1)
        self.quote_buffer2 = QuoteBuffer(self._window, self.s2, self.tick_size2)
    
    def put(self, quote_dic):
        quote1 = quote_dic.get(self.s1)
        quote2 = quote_dic.get(self.s2)
        self.quote_buffer1.enqueue(quote1)
        self.quote_buffer2.enqueue(quote2)
        
        self.prices[: -1] = self.prices[1:]
        self.prices[-1] = self.quote_buffer1.p - self.quote_buffer2.p

    @property
    def price(self):
        return self.prices[-1]
    
    @property
    def full(self):
        return self.quote_buffer1.full
    
    @property
    def mean(self):
        return self.prices.mean()
    
    @property
    def std(self):
        return self.prices.std()
    
    
class SpreadStrategy(EventDrivenStrategy):
    """"""
    def __init__(self):
        super(SpreadStrategy, self).__init__()
        self.symbol = ''
        
        self.window = 55
        
        self.threshold = 6.5
        
        self.tick_sizes = []
        self.spread_buffer = None
        
        self.signal = 0.0
        
        self.pos = 0
        self.pos1 = 0
        self.pos2 = 0
        
        self.size = 10
        
    def init_from_config(self, props):
        super(SpreadStrategy, self).init_from_config(props)
        self.symbol = props.get('symbol')
        self.init_balance = props.get('init_balance')
        self.tick_sizes = props.get('tick_sizes')
        
        self.s1, self.s2 = self.symbol.split(',')
        
        self.spread_buffer = SpreadBuffer(self.window, self.symbol, self.tick_sizes)
        self.pos = self.ctx.pm.get_pos(self.s1)
        
    def on_cycle(self):
        pass
    
    def long_spread(self, pos, n=1):
        quote1 = self.spread_buffer.quote_buffer1.q
        quote2 = self.spread_buffer.quote_buffer2.q
        self.ctx.trade_api.place_order(quote1.symbol, common.ORDER_ACTION.BUY,
                                       quote1.close + n * self.tick_sizes[0], pos)
        self.ctx.trade_api.place_order(quote2.symbol, common.ORDER_ACTION.SHORT,
                                       quote2.close - n * self.tick_sizes[1], pos)

    def short_spread(self, pos, n=1):
        quote1 = self.spread_buffer.quote_buffer1.q
        quote2 = self.spread_buffer.quote_buffer2.q
        self.ctx.trade_api.place_order(quote2.symbol, common.ORDER_ACTION.BUY,
                                       quote2.close + n * self.tick_sizes[1], pos)
        self.ctx.trade_api.place_order(quote1.symbol, common.ORDER_ACTION.SHORT,
                                       quote1.close - n * self.tick_sizes[0], pos)

    def target_position(self, target):
        diff = target - self.pos
        n = 1
        if abs(diff) < 5:
            return
        else:
            print("{:8d} - {:6d}        "
                  "Signal = {: 4.2f},   "
                  "Spread = {: 5.0f},   "
                  "Target pos {:4d} = {:4d} + {:4d}".format(self.ctx.trade_date,
                                                            self.ctx.time,
                                                            self.signal, self.spread_buffer.price,
                                                            target, self.pos, diff))
            if diff > 0:
                self.long_spread(diff, n)
            elif diff < 0:
                self.short_spread(-diff, n)
            
    def on_bar(self, quote_dic):
        self.spread_buffer.put(quote_dic)
        
        if self.ctx.time > 145500 and self.ctx.time < 200000:
            pass
            # self.liquidate(self.spread_buffer.quote_buffer1.q, 5, self.tick_sizes[0], self.pos)
            # self.liquidate(self.spread_buffer.quote_buffer2.q, 5, self.tick_sizes[1], self.pos)
        
        if not self.spread_buffer.full:
            return
        
        spread = self.spread_buffer.price
        mean, std = self.spread_buffer.mean, self.spread_buffer.std
        self.signal = (spread - mean) / self.threshold

        target_pos = int(round(self.signal * self.size)) * -1
        if abs(self.signal) > 1.5:
            self.target_position(target_pos)

        if self.pos1 + self.pos2 != 0:
            print("BROKEN LEG {}  {}".format(self.pos1, self.pos2))
            
    def on_trade(self, ind):
        # print(ind)
        # print(self.ctx.pm.get_position(ind.symbol))
        self.pos1 = self.ctx.pm.get_pos(self.s1)
        self.pos2 = self.ctx.pm.get_pos(self.s2)
        self.pos = self.pos1


props = {
    "symbol"                : "rb1710.SHF,hc1710.SHF",
    "start_date"            : 20170501,
    "end_date"              : 20170901,
    "bar_type"              : "1M",
    "init_balance"          : 3e4,
    "commission_rate": 2E-4,
    "tick_sizes": [1.0, 1.0]
}
props.update(data_config)
props.update(trade_config)


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
    ds.init_from_config(data_config)
    
    ta.initialize(data_server_=ds, file_folder=result_dir_path)
    
    ta.do_analyze(result_dir=result_dir_path, selected_sec=props['symbol'].split(','))


if __name__ == "__main__":
    run_strategy()
    analyze()
