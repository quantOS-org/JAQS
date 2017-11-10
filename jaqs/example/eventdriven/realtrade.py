import numpy as np

from jaqs.trade import common
from jaqs.trade.strategy import EventDrivenStrategy
from jaqs.trade.event import Event, EVENT_TYPE


class RealStrategy(EventDrivenStrategy):
    """
    
    """
    def __init__(self):
        super(RealStrategy, self).__init__()
        self.symbol = ''
        
        self.s1 = ''
        self.s2 = ''
        self.quote1 = None
        self.quote2 = None

        self.window = 13
        self.pos = -1
        self.spread_arr = np.empty(self.window, dtype=float)
        
        self.threshold = 8
        
        self.counter = 0
        
    def init_from_config(self, props):
        super(RealStrategy, self).init_from_config(props)
        self.symbol = props.get('symbol')
        self.init_balance = props.get('init_balance')
        
        # self.s1, self.s2 = self.symbol.split(',')
    
    def on_cycle(self):
        pass
    
    def on_quote(self, quote):
        self.counter += 1
        print(quote.time, quote.symbol, self.counter)
        if self.counter == 2:
            print("counter = 10, let's trade.")
            print(quote)
            # self.place_order(quote.symbol, 'Buy', quote.askprice1 + 1.0, 1)
            self.place_order('IF1712.CFE', 'Buy', 9999.9, 1)
            
        return
        
        q1 = quote_dic.get(self.s1)
        q2 = quote_dic.get(self.s2)
        self.quote1 = q1
        self.quote2 = q2
        
        spread = q1.close - q2.close
        self.pos += 1
        self.spread_arr[self.pos % self.window] = spread
        
        if self.pos <= self.window:
            return
        
        mean = self.spread_arr.mean()
        if (spread - mean) > self.threshold:
            self.short_spread(q1, q2)
        elif (spread - mean) < -self.threshold:
            self.long_spread(q1, q2)
    
    def on_trade(self, ind):
        print "\nStrategy on trade: "
        print(ind)
    
    def on_order_status(self, ind):
        print "\nStrategy on order status: "
        print(ind)
    
    def on_order_rsp(self, rsp):
        print "\nStrategy on order rsp: "
        print(rsp)
    
    def on_task_status(self, ind):
        print "\nStrategy on task ind: "
        print(ind)
