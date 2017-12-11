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

result_dir_path = '../../output/sector_rolling'


class SectorRolling(EventDrivenStrategy):
    def __init__(self):
        super(SectorRolling, self).__init__()
        self.symbol = ''
        self.benchmark_symbol = ''
        self.quotelist = ''
        self.startdate = ''
        self.bufferSize = 0
        self.rollingWindow = 0
        self.bufferCount = 0
        self.bufferCount2 = 0
        self.closeArray = {}
        self.activeReturnArray = {}
        self.std = ''
        self.balance = ''
        self.multiplier = 1.0
        self.std_multiplier = 0.0
    
    def init_from_config(self, props):
        super(SectorRolling, self).init_from_config(props)
        self.symbol = props.get('symbol').split(',')
        self.init_balance = props.get('init_balance')
        self.startdate = props.get('start_date')
        self.std_multiplier = props.get('std multiplier')
        self.bufferSize = props.get('n')
        self.rollingWindow = props.get('m')
        self.benchmark_symbol = self.symbol[-1]
        self.balance = self.init_balance
        
        for s in self.symbol:
            self.closeArray[s] = np.zeros(self.rollingWindow)
            self.activeReturnArray[s] = np.zeros(self.bufferSize)
        
        self.output = True
    
    def on_cycle(self):
        pass
    
    def on_tick(self, quote):
        pass

    def buy(self, quote, price, size):
        self.ctx.trade_api.place_order(quote.symbol, 'Buy', price, size)

    def sell(self, quote, price, size):
        self.ctx.trade_api.place_order(quote.symbol, 'Sell', price, size)

    def on_bar(self, quote):
        # 1 is for stock, 2 is for convertible bond
        self.bufferCount += 1
        self.quotelist = []
    
        for s in self.symbol:
            self.quotelist.append(quote.get(s))
    
        for stock in self.quotelist:
            self.closeArray[stock.symbol][0:self.rollingWindow - 1] = self.closeArray[stock.symbol][1:self.rollingWindow]
            self.closeArray[stock.symbol][-1] = stock.close
    
        if self.bufferCount < self.rollingWindow:
            return
    
        elif self.bufferCount >= self.rollingWindow:
            self.bufferCount2 += 1
            # calculate active return for each stock
            benchmarkReturn = np.log(self.closeArray[self.benchmark_symbol][-1]) - np.log(self.closeArray[self.benchmark_symbol][0])
            for stock in self.quotelist:
                stockReturn = np.log(self.closeArray[stock.symbol][-1]) - np.log(self.closeArray[stock.symbol][0])
                activeReturn = stockReturn - benchmarkReturn
                self.activeReturnArray[stock.symbol][0:self.bufferSize - 1] = self.activeReturnArray[stock.symbol][1:self.bufferSize]
                self.activeReturnArray[stock.symbol][-1] = activeReturn
        
            if self.bufferCount2 < self.bufferSize:
                return
        
            elif self.bufferCount2 == self.bufferSize:
                # if it's the first date of strategy, we will buy equal value stock in the universe
                stockvalue = self.balance/len(self.symbol)
                for stock in self.quotelist:
                    if stock.symbol != self.benchmark_symbol:
                        self.buy(stock, stock.close, np.floor(stockvalue/stock.close/self.multiplier))
                return
            else:
                stockholdings = self.ctx.pm.holding_securities
                noholdings = set(self.symbol) - stockholdings
                stockvalue = self.balance/len(noholdings)
            
                for stock in list(stockholdings):
                    curRet = self.activeReturnArray[stock][-1]
                    avgRet = np.mean(self.activeReturnArray[stock][:-1])
                    stdRet = np.std(self.activeReturnArray[stock][:-1])
                    if curRet >= avgRet + self.std_multiplier * stdRet:
                        curPosition = self.ctx.pm.positions[stock].current_size
                        stock_quote = quote.get(stock)
                        self.sell(stock_quote, stock_quote.close, curPosition)
            
                for stock in list(noholdings):
                    curRet = self.activeReturnArray[stock][-1]
                    avgRet = np.mean(self.activeReturnArray[stock][:-1])
                    stdRet = np.std(self.activeReturnArray[stock][:-1])
                    if curRet < avgRet - self.std_multiplier * stdRet:
                        stock_quote = quote.get(stock)
                        self.buy(stock_quote, stock_quote.close, np.floor(stockvalue/stock_quote.close/self.multiplier))

    def on_trade(self, ind):
        print("\nStrategy on trade: ")
        print(ind)
        self.pos = self.ctx.pm.get_pos(ind.symbol)
        print(self.ctx.pm.get_trade_stat(ind.symbol))

        if common.ORDER_ACTION.is_positive(ind.entrust_action):
            self.balance -= ind.fill_price * ind.fill_size * self.multiplier
        else:
            self.balance += ind.fill_price * ind.fill_size * self.multiplier

    def on_order_status(self, ind):
        if self.output:
            print("\nStrategy on order status: ")
            print(ind)
    
    def on_task_status(self, ind):
        if self.output:
            print("\nStrategy on task ind: ")
            print(ind)


def run_strategy():

    start_date = 20150501
    end_date = 20171030
    index = '399975.SZ'
    
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    symbol_list = ds.get_index_comp(index, start_date, start_date)

    # add the benchmark index to the last position of symbol_list
    symbol_list.append(index)
    props = {"symbol": ','.join(symbol_list),
             "start_date": start_date,
             "end_date": end_date,
             "bar_type": "1d",
             "init_balance": 1e7,
             "std multiplier": 1.5,
             "m": 10,
             "n": 60,
             "commission_rate": 2E-4}
    props.update(data_config)
    props.update(trade_config)

    tapi = BacktestTradeApi()
    ins = EventBacktestInstance()

    strat = SectorRolling()
    pm = PortfolioManager()

    context = model.Context(data_api=ds, trade_api=tapi, instance=ins,
                            strategy=strat, pm=pm)

    ins.init_from_config(props)
    ins.run()
    ins.save_results(folder_path=result_dir_path)

    ta = ana.EventAnalyzer()

    ta.initialize(data_server_=ds, file_folder=result_dir_path)
    df_bench, _ = ds.daily(index, start_date=start_date, end_date=end_date)
    ta.data_benchmark = df_bench.set_index('trade_date').loc[:, ['close']]

    ta.do_analyze(result_dir=result_dir_path, selected_sec=props['symbol'].split(',')[:2])


if __name__ == "__main__":
    run_strategy()
