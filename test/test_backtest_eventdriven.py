# encoding: utf-8

from __future__ import print_function
from __future__ import absolute_import
import time

import numpy as np

from jaqs.trade import common
from jaqs.trade import EventDrivenStrategy
from jaqs.data import RemoteDataService
from jaqs.data.basic import Bar, Quote
from jaqs.trade import model
from jaqs.trade import EventLiveTradeInstance
from jaqs.trade import EventBacktestInstance
from jaqs.trade import RealTimeTradeApi, BacktestTradeApi
from jaqs.trade import PortfolioManager
import jaqs.trade.analyze as ana
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

result_dir_path = '../output/double_ma'


class DoubleMaStrategy(EventDrivenStrategy):
    """"""
    def __init__(self):
        super(DoubleMaStrategy, self).__init__()
        self.symbol = ''
        
        self.fast_ma_len = 13
        self.slow_ma_len = 23
        
        self.window_count = 0
        self.window = self.slow_ma_len + 1
        
        self.price_arr = np.zeros(self.window)
        self.fast_ma = 0
        self.slow_ma = 0
        self.pos = 0
        
        self.buy_size_unit = 1
        self.output = True
    
    def init_from_config(self, props):
        super(DoubleMaStrategy, self).init_from_config(props)
        self.symbol = props.get('symbol')
        self.init_balance = props.get('init_balance')
    
    def buy(self, quote, size=1):
        if isinstance(quote, Quote):
            ref_price = (quote.bidprice1 + quote.askprice1) / 2.0
        else:
            ref_price = quote.close
            
        task_id, msg = self.ctx.trade_api.place_order(quote.symbol, common.ORDER_ACTION.BUY, ref_price + 3, self.buy_size_unit * size)
        if (task_id is None) or (task_id == 0):
            print("place_order FAILED! msg = {}".format(msg))
    
    def sell(self, quote, size=1):
        if isinstance(quote, Quote):
            ref_price = (quote.bidprice1 + quote.askprice1) / 2.0
        else:
            ref_price = quote.close
    
        task_id, msg = self.ctx.trade_api.place_order(quote.symbol, common.ORDER_ACTION.SHORT, ref_price - 3, self.buy_size_unit * size)
        if (task_id is None) or (task_id == 0):
            print("place_order FAILED! msg = {}".format(msg))
    
    """
    `on_tick` accepts single Quote, while `on_bar` accepts
    'on_tick' is used for real-time trading, while 'on_bar' is used for backtest
    """
    def on_tick(self, quote):
        # 'quote' can be:
        #     - Quote (at real-time trading, data comes as ticks)
        #     - Bar (backtest on minutely or daily frequency)
        # we use 'isinstance' to check whether 'quote' is Quote or Bar
        if isinstance(quote, Quote):
            bid, ask = quote.bidprice1, quote.askprice1
            if bid > 0 and ask > 0:
                mid = (quote.bidprice1 + quote.askprice1) / 2.0
            else:
                # price reached high_limit or low_limit, we do not trade
                return
        else:
            mid = quote.close
        
        self.price_arr[0: self.window - 1] = self.price_arr[1: self.window]
        self.price_arr[-1] = mid
        self.window_count += 1
    
        if self.window_count <= self.window:
            return
    
        self.fast_ma = np.mean(self.price_arr[-self.fast_ma_len - 1:])
        self.slow_ma = np.mean(self.price_arr[-self.slow_ma_len - 1:])
    
        print(quote)
        print("Fast MA = {:.2f}     Slow MA = {:.2f}".format(self.fast_ma, self.slow_ma))
        if self.fast_ma > self.slow_ma:
            if self.pos == 0:
                self.buy(quote, 1)
        
            elif self.pos < 0:
                self.buy(quote, 2)
    
        elif self.fast_ma < self.slow_ma:
            if self.pos == 0:
                self.sell(quote, 1)
            elif self.pos > 0:
                self.sell(quote, 2)

    def on_bar(self, quote_dic):
        quote = quote_dic.get(self.symbol)
        self.on_tick(quote)

    def on_trade(self, ind):
        print("\nStrategy on trade: ")
        print(ind)
        self.pos = self.ctx.pm.get_pos(self.symbol)

    def on_order_status(self, ind):
        if self.output:
            print("\nStrategy on order status: ")
            print(ind)

    def on_task_status(self, ind):
        if self.output:
            print("\nStrategy on task ind: ")
            print(ind)


def test_backtest():
    props = {"symbol": "rb1710.SHF",
             "start_date": 20170710,
             "end_date": 20170730,
             "bar_type": "1M",  # '1d'
             "init_balance": 2e4}

    tapi = BacktestTradeApi()
    ins = EventBacktestInstance()
        
    props.update(data_config)
    props.update(trade_config)
    
    ds = RemoteDataService()
    strat = DoubleMaStrategy()
    pm = PortfolioManager()
    
    context = model.Context(data_api=ds, trade_api=tapi, instance=ins,
                            strategy=strat, pm=pm)
    
    ins.init_from_config(props)

    ins.run()
    ins.save_results(result_dir_path)
    do_analyze()


def test_livetrade():
    props = {'symbol': 'rb1801.SHF',
             'strategy_no': 1044}
    tapi = RealTimeTradeApi(trade_config)
    ins = EventLiveTradeInstance()
    
    props.update(data_config)
    props.update(trade_config)
    
    ds = RemoteDataService()
    strat = DoubleMaStrategy()
    pm = PortfolioManager()
    
    context = model.Context(data_api=ds, trade_api=tapi, instance=ins,
                            strategy=strat, pm=pm)
    
    ins.init_from_config(props)
    ds.subscribe(props['symbol'])
    
    ins.run()
    time.sleep(3)
    ins.stop()
    ins.save_results(result_dir_path)
    do_analyze()


def do_analyze():
    from jaqs.trade.analyze.analyze import TradeRecordEmptyError
    ta = ana.EventAnalyzer()
    
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    try:
        ta.initialize(data_server_=ds, file_folder=result_dir_path)
        
        ta.do_analyze(result_dir=result_dir_path, selected_sec=[])
        
    except TradeRecordEmptyError:
        pass


if __name__ == "__main__":
    test_backtest()
    test_livetrade()
