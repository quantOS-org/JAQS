from __future__ import print_function
import numpy as np

from jaqs.trade import EventDrivenStrategy
from jaqs.trade import common


class RealStrategy(EventDrivenStrategy):
    """
    
    """
    def __init__(self):
        super(RealStrategy, self).__init__()
        self.symbol = ''
        self.symbol_list = []
        self.s1 = ""
        
        self.pos = 0
        
        self.bid_orders = []
        self.ask_orders = []
        
        self.counter = 0
        
    def init_from_config(self, props):
        super(RealStrategy, self).init_from_config(props)
        self.symbol = props.get('symbol')
        self.init_balance = props.get('init_balance')
        
        # self.s1, self.s2 = self.symbol.split(',')
        self.symbol_list = self.symbol.split(',')
        self.s1 = self.symbol_list[0]
        
        self.tick_size1 = 1.0
        self.open_size = 1
        
        self.output = True
        
        # self.liquidate()
        self._update_pos()
    
    def on_cycle(self):
        pass
    
    def on_tick(self, quote):
        if quote.symbol != self.s1:
            return
        
        self.counter += 1
        
        # if self.counter % 5 == 0:
        if self.counter == 1:
            print(quote)
            # task_id, msg = self.ctx.trade_api.place_order(quote.symbol, 'Buy', quote.askprice1, 1)
            # if task_id is None or task_id == 0:
            #     print(msg)
            # else:
            #     print("   place_order: task_id = {}".format(task_id))
            task_id, msg = self.ask(quote)
            if task_id is None or task_id == 0:
                print(msg)
            else:
                print("   place_order: task_id = {}".format(task_id))
            
        return

    def bid(self, quote):
        if hasattr(quote, 'bidprice1'):
            price = quote.bidprice1
        else:
            price = quote.vwap - self.tick_size1
        action = common.ORDER_ACTION.BUY
        self.ctx.trade_api.place_order(quote.symbol, action, price, self.open_size)

    def ask(self, quote):
        if hasattr(quote, 'askprice1'):
            price = quote.askprice1
        else:
            price = quote.vwap + self.tick_size1
        action = common.ORDER_ACTION.SHORT
        return self.ctx.trade_api.place_order(quote.symbol, action, price, self.open_size)

    def cancel_all_orders(self):
        for task_id, task in self.ctx.pm.tasks.items():
            if task.trade_date == self.ctx.trade_date and not task.is_finished:
                self.ctx.trade_api.cancel_order(task_id)
    
    def liquidate(self, quote, n):
        self._update_pos()
        # self.cancel_all_orders()
        
        if self.pos == 0:
            return
        
        ref_price = quote.close
        if self.pos < 0:
            action = common.ORDER_ACTION.BUY
            price = ref_price + n * self.tick_size1
        else:
            action = common.ORDER_ACTION.SELL
            price = ref_price - n * self.tick_size1
        self.ctx.trade_api.place_order(quote.symbol, action, price, abs(self.pos))
    
    def on_bar(self, quote):
        quote = quote[self.s1]
        if quote.time > 145600 and quote.time < 200000:
            self.liquidate(quote, 5)
            return

        ts = self.ctx.pm.get_trade_stat(self.s1)
        if (abs(self.pos) > 5
            or ((ts is not None) and (ts.buy_want_size > 5 or ts.sell_want_size > 5))):
            # self.cancel_all_orders()
            self.liquidate(quote, 1)
        else:
            self.bid(quote)
            self.ask(quote)
        
        if self.output:
            pass
            # self.show()
    
    def show(self):
        ts = self.ctx.pm.get_trade_stat(self.s1)
        p = self.ctx.pm.get_position(self.s1)
        print(ts)
        print(p)
    
    def _update_pos(self):
        p = self.ctx.pm.get_position(self.s1)
        if p is None:
            self.pos = 0
        else:
            self.pos = p.current_size
    
    def on_trade(self, ind):
        print("\nStrategy on trade: ")
        self._update_pos()
        # print(ind)
        
        # print(p)
        ts = self.ctx.pm.get_trade_stat(self.s1)
        print(ts)
    
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
