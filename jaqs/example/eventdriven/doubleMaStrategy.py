import numpy as np

from jaqs.trade import common
from jaqs.trade.strategy import EventDrivenStrategy
from jaqs.data.basic.order import Order


class DoubleMaStrategy(EventDrivenStrategy):
    """"""
    def __init__(self):
        EventDrivenStrategy.__init__(self)
        self.symbol = ''
        self.fastN = 3
        self.slowN = 8
        self.bar = None
        self.bufferCount = 0
        self.bufferSize = 20
        self.closeArray = np.zeros(self.bufferSize)
        self.fastMa = 0
        self.slowMa = 0
        self.pos = 0
    
    def init_from_config(self, props):
        super(DoubleMaStrategy, self).init_from_config(props)
        self.symbol = props.get('symbol')
        self.init_balance = props.get('init_balance')
    
    def on_cycle(self):
        pass
    
    def create_order(self, quote, price, size):
        order = Order.new_order(quote.symbol, "", price, size, quote.trade_date, quote.time,
                                order_type=common.ORDER_TYPE.LIMIT)
        return order
    
    def buy(self, quote, price, size):
        order = self.create_order(quote, price, size)
        order.entrust_action = common.ORDER_ACTION.BUY
        self.ctx.gateway.send_order(order, '', '')
    
    def sell(self, quote, price, size):
        order = self.create_order(quote, price, size)
        order.entrust_action = common.ORDER_ACTION.SELL
        self.ctx.gateway.send_order(order, '', '')
    
    def cover(self, quote, price, size):
        order = self.create_order(quote, price, size)
        order.entrust_action = common.ORDER_ACTION.BUY
        self.ctx.gateway.send_order(order, '', '')
    
    def short(self, quote, price, size):
        order = self.create_order(quote, price, size)
        order.entrust_action = common.ORDER_ACTION.SELL
        self.ctx.gateway.send_order(order, '', '')
    
    def on_quote(self, quote_dic):
        quote = quote_dic.values()[0]
        quote_date = quote.trade_date
        p = self.ctx.pm.get_position(quote.symbol)
        if p is None:
            self.pos = 0
        else:
            self.pos = p.current_size
        
        self.closeArray[0:self.bufferSize - 1] = self.closeArray[1:self.bufferSize]
        self.closeArray[-1] = quote.close
        self.bufferCount += 1
        
        if self.bufferCount <= self.bufferSize:
            return
        self.fastMa = self.closeArray[-self.fastN - 1:-1].mean()
        self.slowMa = self.closeArray[-self.slowN - 1:-1].mean()
        
        if self.fastMa > self.slowMa:
            if self.pos == 0:
                self.buy(quote, quote.close + 3, 1)
            
            elif self.pos < 0:
                self.cover(quote, quote.close + 3, 1)
                self.buy(quote, quote.close + 3, 1)
        
        elif self.fastMa < self.slowMa:
            if self.pos == 0:
                self.short(quote, quote.close - 3, 1)
            elif self.pos > 0:
                self.sell(quote, quote.close - 3, 1)
                self.short(quote, quote.close - 3, 1)
