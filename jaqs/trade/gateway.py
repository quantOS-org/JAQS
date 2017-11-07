# encoding: UTF-8

from abc import abstractmethod
import abc
from six import with_metaclass

import numpy as np

from jaqs.data.basic.order import *
from jaqs.data.basic.position import Position
from jaqs.data.basic.trade import Trade
from jaqs.util.sequence import SequenceGenerator


class OrderStatusInd(object):
    def __init__(self):
        self.entrust_no = ''
        
        self.symbol = ''
        
        self.entrust_action = ''
        self.entrust_price = 0.0
        self.entrust_size = 0
        self.entrust_date = 0
        self.entrust_time = 0
        
        self.order_status = ''
        
        self.fill_size = 0
        self.fill_price = 0.0
    
    def init_from_order(self, order):
        self.entrust_no = order.entrust_no
        
        self.symbol = order.symbol
        
        self.entrust_action = order.entrust_action
        self.entrust_price = order.entrust_price
        self.entrust_size = order.entrust_size
        self.entrust_date = order.entrust_date
        self.entrust_time = order.entrust_time
        
        self.order_status = order.order_status
        
        self.fill_size = order.fill_size
        self.fill_price = order.fill_price


class TradeCallback(with_metaclass(abc.ABCMeta)):
    @abstractmethod
    def on_trade_ind(self, trade):
        pass
    
    @abstractmethod
    def on_order_status(self, orderstatus):
        pass
    
    @abstractmethod
    def on_order_rsp(self, order, result, msg):
        pass


class TradeStat(object):
    def __init__(self):
        self.symbol = ""
        self.buy_filled_size = 0
        self.buy_want_size = 0
        self.sell_filled_size = 0
        self.sell_want_size = 0


class PortfolioManager_RAW(TradeCallback):
    """
    Used to store relevant context of the strategy.

    Attributes
    ----------
    orders : list of jaqs.data.basic.Order objects
    trades : list of jaqs.data.basic.Trade objects
    positions : dict of {symbol + trade_date : jaqs.data.basic.Position}
    strategy : Strategy
    holding_securities : set of securities

    Methods
    -------

    """
    
    # TODO want / frozen update
    def __init__(self, strategy=None):
        self.orders = {}
        self.trades = []
        self.positions = {}
        self.holding_securities = set()
        self.tradestat = {}
        self.strategy = strategy
    
    @staticmethod
    def _make_position_key(symbol, trade_date):
        return '@'.join((symbol, str(trade_date)))

    @staticmethod
    def _make_order_key(entrust_id, trade_date):
        return '@'.join((str(entrust_id), str(trade_date)))
    
    def on_order_rsp(self, order, result, msg):
        if result:
            self.add_order(order)
    
    def get_position(self, symbol, date):
        key = self._make_position_key(symbol, date)
        position = self.positions.get(key, None)
        return position
    
    def on_new_day(self, date, pre_date):
        for key, pos in self.positions.viewitems():
            sec, td = key.split('@')
            if str(pre_date) == td:
                new_key = self._make_position_key(sec, date)
                pre_position = pos
                
                new_position = Position()
                new_position.curr_size = pre_position.curr_size
                new_position.init_size = new_position.curr_size
                new_position.symbol = pre_position.symbol
                new_position.trade_date = date
                self.positions[new_key] = new_position
        
        """
        for sec in self.holding_securities:
            pre_key = self._make_position_key(sec, pre_date)
            new_key = self._make_position_key(sec, date)
            if pre_key in self.positions:
                pre_position = self.positions.get(pre_key)
                new_position = Position()
                new_position.curr_size = pre_position.curr_size
                new_position.init_size = new_position.curr_size
                new_position.symbol = pre_position.symbol
                new_position.trade_date = date
                self.positions[new_key] = new_position
        """
    
    def add_order(self, order):
        """
        Add order to orders, create position and tradestat if necessary.

        Parameters
        ----------
        order : Order

        """
        if order.entrust_no in self.orders:
            print 'duplicate entrust_no {}'.format(order.entrust_no)
            return False
        
        new_order = Order()
        new_order.copy(order)  # TODO why copy?
        self.orders[self._make_order_key(order.entrust_no, self.strategy.ctx.trade_date)] = new_order
        
        position_key = self._make_position_key(order.symbol, self.strategy.ctx.trade_date)
        if position_key not in self.positions:
            position = Position()
            position.symbol = order.symbol
            self.positions[position_key] = position
        
        if order.symbol not in self.tradestat:
            tradestat = TradeStat()
            tradestat.symbol = order.symbol
            self.tradestat[order.symbol] = tradestat
        
        tradestat = self.tradestat.get(order.symbol)
        
        if order.entrust_action == common.ORDER_ACTION.BUY:
            tradestat.buy_want_size += order.entrust_size
        else:
            tradestat.sell_want_size += order.entrust_size
    
    def on_order_status(self, ind):
        if ind.order_status is None:
            return
        
        if ind.order_status == common.ORDER_STATUS.CANCELLED or ind.order_status == common.ORDER_STATUS.REJECTED:
            entrust_no = ind.entrust_no
            order = self.orders.get(self._make_order_key(entrust_no, self.strategy.ctx.trade_date), None)
            if order is not None:
                order.order_status = ind.order_status
                
                tradestat = self.tradestat.get(ind.symbol)
                release_size = ind.entrust_size - ind.fill_size
                
                if ind.entrust_action == common.ORDER_ACTION.BUY:
                    tradestat.buy_want_size -= release_size
                else:
                    tradestat.sell_want_size -= release_size
            else:
                raise ValueError("order {} does not exist".format(entrust_no))
    
    def set_position(self, symbol, date, ratio=1):
        """Modify latest (thus date might not be necessary) position by a ratio."""
        pos_key = self._make_position_key(symbol, date)
        pos = self.positions.get(pos_key)

        pos.curr_size *= ratio
        pos.init_size *= ratio
        self.positions[pos_key] = pos
        
    def on_trade_ind(self, ind):
        entrust_no = ind.entrust_no
        
        order = self.orders.get(self._make_order_key(entrust_no, self.strategy.ctx.trade_date), None)
        if order is None:
            print 'cannot find order for entrust_no' + entrust_no
            return
        
        self.trades.append(ind)
        
        order.fill_size += ind.fill_size
        
        if order.fill_size == order.entrust_size:
            order.order_status = common.ORDER_STATUS.FILLED
        else:
            order.order_status = common.ORDER_STATUS.ACCEPTED
        
        position_key = self._make_position_key(ind.symbol, self.strategy.ctx.trade_date)
        position = self.positions.get(position_key)
        tradestat = self.tradestat.get(ind.symbol)
        
        if (ind.entrust_action == common.ORDER_ACTION.BUY
            or ind.entrust_action == common.ORDER_ACTION.COVER
            or ind.entrust_action == common.ORDER_ACTION.COVERYESTERDAY
            or ind.entrust_action == common.ORDER_ACTION.COVERTODAY):
            
            tradestat.buy_filled_size += ind.fill_size
            tradestat.buy_want_size -= ind.fill_size
            
            position.curr_size += ind.fill_size
        
        elif (ind.entrust_action == common.ORDER_ACTION.SELL
              or ind.entrust_action == common.ORDER_ACTION.SELLTODAY
              or ind.entrust_action == common.ORDER_ACTION.SELLYESTERDAY
              or ind.entrust_action == common.ORDER_ACTION.SHORT):
            
            tradestat.sell_filled_size += ind.fill_size
            tradestat.sell_want_size -= ind.fill_size
            
            position.curr_size -= ind.fill_size
        
        if position.curr_size != 0:
            self.holding_securities.add(ind.symbol)
        else:
            self.holding_securities.remove(ind.symbol)
    
    def market_value(self, ref_date, ref_prices, suspensions=None):
        """
        Calculate total market value according to all current positions.
        NOTE for now this func only support stocks.

        Parameters
        ----------
        ref_date : int
            The date we refer to to get symbol position.
        ref_prices : dict of {symbol: price}
            The prices we refer to to get symbol price.
        suspensions : list of securities
            Securities that are suspended.

        Returns
        -------
        market_value : float

        """
        # TODO some securities could not be able to be traded
        if suspensions is None:
            suspensions = []
        
        market_value = 0.0
        for sec in self.holding_securities:
            if sec in suspensions:
                continue
            
            size = self.get_position(sec, ref_date).curr_size
            # TODO PortfolioManager object should not access price
            price = ref_prices[sec]
            market_value += price * size * 100
        
        return market_value


class PortfolioManager(TradeCallback):
    """
    Used to store relevant context of the strategy.

    Attributes
    ----------
    orders : list of jaqs.data.basic.Order objects
    trades : list of jaqs.data.basic.Trade objects
    positions : dict of {symbol + trade_date : jaqs.data.basic.Position}
    strategy : Strategy
    holding_securities : set of securities

    Methods
    -------

    """
    
    # TODO want / frozen update
    def __init__(self, strategy=None):
        self.orders = {}
        self.trades = []
        self.positions = {}
        self.holding_securities = set()
        self.tradestat = {}
        self.strategy = strategy
    
    @staticmethod
    def _make_position_key(symbol, trade_date=0):
        # return '@'.join(symbol)
        return symbol
    
    @staticmethod
    def _make_order_key(entrust_id, trade_date):
        return '@'.join((str(entrust_id), str(trade_date)))
    
    def on_order_rsp(self, order, result, msg):
        if result:
            self.add_order(order)
    
    def get_position(self, symbol, date=0):
        key = self._make_position_key(symbol)
        position = self.positions.get(key, None)
        return position
    
    def on_new_day(self, date, pre_date):
        pass
    
    def add_order(self, order):
        """
        Add order to orders, create position and tradestat if necessary.

        Parameters
        ----------
        order : Order

        """
        if order.entrust_no in self.orders:
            print 'duplicate entrust_no {}'.format(order.entrust_no)
            return False
        
        new_order = Order()
        new_order.copy(order)  # TODO why copy?
        self.orders[self._make_order_key(order.entrust_no, self.strategy.ctx.trade_date)] = new_order
        
        position_key = self._make_position_key(order.symbol, self.strategy.ctx.trade_date)
        if position_key not in self.positions:
            position = Position()
            position.symbol = order.symbol
            self.positions[position_key] = position
        
        if order.symbol not in self.tradestat:
            tradestat = TradeStat()
            tradestat.symbol = order.symbol
            self.tradestat[order.symbol] = tradestat
        
        tradestat = self.tradestat.get(order.symbol)
        
        if order.entrust_action == common.ORDER_ACTION.BUY:
            tradestat.buy_want_size += order.entrust_size
        else:
            tradestat.sell_want_size += order.entrust_size
    
    def on_order_status(self, ind):
        if ind.order_status is None:
            return
        
        if ind.order_status == common.ORDER_STATUS.CANCELLED or ind.order_status == common.ORDER_STATUS.REJECTED:
            entrust_no = ind.entrust_no
            order = self.orders.get(self._make_order_key(entrust_no, self.strategy.ctx.trade_date), None)
            if order is not None:
                order.order_status = ind.order_status
                
                tradestat = self.tradestat.get(ind.symbol)
                release_size = ind.entrust_size - ind.fill_size
                
                if ind.entrust_action == common.ORDER_ACTION.BUY:
                    tradestat.buy_want_size -= release_size
                else:
                    tradestat.sell_want_size -= release_size
            else:
                raise ValueError("order {} does not exist".format(entrust_no))
    
    def set_position(self, symbol, date, ratio=1):
        """Modify latest (thus date might not be necessary) position by a ratio."""
        pos_key = self._make_position_key(symbol, date)
        pos = self.positions.get(pos_key)
        
        pos.curr_size *= ratio
        pos.init_size *= ratio
        self.positions[pos_key] = pos
    
    def on_trade_ind(self, ind):
        # record trades
        self.trades.append(ind)

        # change order status
        entrust_no = ind.entrust_no
        if entrust_no == 101010 or 202020:  # trades generate by system
            pass
        else:
            order = self.orders.get(self._make_order_key(entrust_no, self.strategy.ctx.trade_date), None)
            if order is None:
                print 'cannot find order for entrust_no' + entrust_no
                return
            
            order.fill_size += ind.fill_size
            
            if order.fill_size == order.entrust_size:
                order.order_status = common.ORDER_STATUS.FILLED
            else:
                order.order_status = common.ORDER_STATUS.ACCEPTED
        
        # change position and trade stats
        position_key = self._make_position_key(ind.symbol, self.strategy.ctx.trade_date)
        position = self.positions.get(position_key)
        tradestat = self.tradestat.get(ind.symbol)
        
        if (ind.entrust_action == common.ORDER_ACTION.BUY
            or ind.entrust_action == common.ORDER_ACTION.COVER
            or ind.entrust_action == common.ORDER_ACTION.COVERYESTERDAY
            or ind.entrust_action == common.ORDER_ACTION.COVERTODAY):
            
            tradestat.buy_filled_size += ind.fill_size
            tradestat.buy_want_size -= ind.fill_size
            
            position.curr_size += ind.fill_size
        
        elif (ind.entrust_action == common.ORDER_ACTION.SELL
              or ind.entrust_action == common.ORDER_ACTION.SELLTODAY
              or ind.entrust_action == common.ORDER_ACTION.SELLYESTERDAY
              or ind.entrust_action == common.ORDER_ACTION.SHORT):
            
            tradestat.sell_filled_size += ind.fill_size
            tradestat.sell_want_size -= ind.fill_size
            
            position.curr_size -= ind.fill_size
        
        if position.curr_size != 0:
            self.holding_securities.add(ind.symbol)
        else:
            self.holding_securities.remove(ind.symbol)
    
    def market_value(self, ref_date, ref_prices, suspensions=None):
        """
        Calculate total market value according to all current positions.
        NOTE for now this func only support stocks.

        Parameters
        ----------
        ref_date : int
            The date we refer to to get symbol position.
        ref_prices : dict of {symbol: price}
            The prices we refer to to get symbol price.
        suspensions : list of securities
            Securities that are suspended.

        Returns
        -------
        market_value : float

        """
        # TODO some securities could not be able to be traded
        if suspensions is None:
            suspensions = []
        
        market_value_float = 0.0
        market_value_frozen = 0.0  # suspended or high/low limit
        for sec in self.holding_securities:
            size = self.get_position(sec, ref_date).curr_size
            # TODO PortfolioManager object should not access price
            price = ref_prices[sec]
            mv_sec = price * size
            if sec in suspensions:
                market_value_frozen += mv_sec
            else:
                market_value_float += mv_sec
        
        return market_value_float, market_value_frozen


class BaseGateway(object):
    """
    Strategy communicates with Gateway using APIs defined by ourselves;
    Gateway communicates with brokers using brokers' APIs;
    Gateway can also communicate with simulator.
    
    Attributes
    ----------
    ctx : Context
        Trading context, including data_api, dataview, calendar, etc.

    Note: Gateway knows nothing about task_id but entrust_no,
          so does Simulator.

    """
    
    def __init__(self):
        self.callback = None
        self.cb_on_trade_ind = None
        self.cb_on_order_status = None
        self.cb_pm = None
        self.ctx = None
    
    @abstractmethod
    def init_from_config(self, props):
        pass
    
    def register_callback(self, type_, callback):
        '''
        
        Parameters
        ----------
        type_
        callback

        Returns
        -------

        '''
        '''
        if type_ == 'on_trade_ind':
            self.cb_on_trade_ind = callback
        elif type_ == 'on_order_status':
            self.cb_on_order_status = callback
        '''
        if type_ == 'portfolio manager':
            self.cb_pm = callback
        else:
            raise NotImplementedError("callback of type {}".format(type_))
    
    def register_context(self, context=None):
        self.ctx = context
        
    def on_new_day(self, trade_date):
        pass
    
    def place_order(self, order):
        """
        Send an order with determined task_id and entrust_no.

        Parameters
        ----------
        order : Order

        Returns
        -------
        task_id : str
            Task ID generated by entrust_order.
        err_msg : str.

        """
        # do sth.
        # return task_id, err_msg
        pass
    
    def cancel_order(self, task_id):
        """Cancel all want orders of a task according to its task ID.

        Parameters
        ----------
        task_id : str
            ID of the task.
            NOTE we CANNOT cancel order by entrust_no because this may break the execution of algorithm.

        Returns
        -------
        result : str
            Indicate whether the cancel succeed.
        err_msg : str

        """
        # do sth.
        # return result, err_msg
        pass
    
    def goal_portfolio(self, goals):
        """
        Let the system automatically generate orders according to portfolio positions goal.
        If there are want orders of any symbol in the strategy universe, this order will be rejected.

        Parameters
        -----------
        goals : list of GoalPosition
            This must include positions of all securities in the strategy universe.
            Use former value if there is no change.

        Returns
        --------
        result : bool
            Whether this command is accepted. True means the system's acceptance, instead of positions have changed.
        err_msg : str

        """
        pass
    

class DailyStockSimGateway(BaseGateway):
    def __init__(self):
        BaseGateway.__init__(self)
        
        self.simulator = DailyStockSimulator()
    
    def init_from_config(self, props):
        pass
    
    def on_new_day(self, trade_date):
        self.simulator.on_new_day(trade_date)

    def on_after_market_close(self):
        self.simulator.on_after_market_close()
    
    def place_order(self, order):
        err_msg = self.simulator.add_order(order)
        return err_msg
    
    def cancel_order(self, entrust_no):
        order_status_ind, err_msg = self.simulator.cancel_order(entrust_no)
        self.cb_on_order_status(order_status_ind)
        return err_msg
    
    @property
    def match_finished(self):
        return self.simulator.match_finished
    
    @abstractmethod
    def match(self, price_dict, time=0):
        """
        Match un-fill orders in simulator. Return trade indications.

        Parameters
        ----------
        price_dict : dict
        time : int
        # TODO: do we need time parameter?

        Returns
        -------
        list

        """
        return self.simulator.match(price_dict, date=self.ctx.trade_date, time=time)


class DailyStockSimulator(object):
    """This is not event driven!

    Attributes
    ----------
    __orders : list of Order
        Store orders that have not been filled.

    """
    
    def __init__(self):
        # TODO heap is better for insertion and deletion. We only need implement search of heapq module.
        self.__orders = dict()
        self.seq_gen = SequenceGenerator()
        
        self.date = 0
        self.time = 0
    
    def on_new_day(self, trade_date):
        self.date = trade_date
    
    def on_after_market_close(self):
        # self._refresh_orders() #TODO sometimes we do not want to refresh (multi-days match)
        pass
    
    def _refresh_orders(self):
        self.__orders.clear()
    
    def _next_fill_no(self):
        return str(np.int64(self.date) * 10000 + self.seq_gen.get_next('fill_no'))
    
    @property
    def match_finished(self):
        return len(self.__orders) == 0
    
    @staticmethod
    def _validate_order(order):
        # TODO to be enhanced
        assert order is not None
    
    @staticmethod
    def _validate_price(price_dic):
        # TODO to be enhanced
        assert price_dic is not None
    
    def add_order(self, order):
        """
        Add one order to the simulator.

        Parameters
        ----------
        order : Order

        Returns
        -------
        err_msg : str
            default ""

        """
        self._validate_order(order)
        
        if order.entrust_no in self.__orders:
            err_msg = "order with entrust_no {} already exists in simulator".format(order.entrust_no)
        self.__orders[order.entrust_no] = order
        err_msg = ""
        return err_msg
    
    def cancel_order(self, entrust_no):
        """
        Cancel an order.

        Parameters
        ----------
        entrust_no : str

        Returns
        -------
        err_msg : str
            default ""

        """
        popped = self.__orders.pop(entrust_no, None)
        if popped is None:
            err_msg = "No order with entrust_no {} in simulator.".format(entrust_no)
            order_status_ind = None
        else:
            err_msg = ""
            order_status_ind = OrderStatusInd()
            order_status_ind.init_from_order(popped)
            order_status_ind.order_status = common.ORDER_STATUS.CANCELLED
        return order_status_ind, err_msg
    
    def match(self, price_dic, date=19700101, time=150000):
        self._validate_price(price_dic)
        
        results = []
        for order in self.__orders.values():
            symbol = order.symbol
            symbol_dic = price_dic[symbol]
            
            # get fill price
            if isinstance(order, FixedPriceTypeOrder):
                price_target = order.price_target
                fill_price = symbol_dic[price_target]
            elif isinstance(order, VwapOrder):
                if order.start != -1:
                    raise NotImplementedError("Vwap of a certain time range")
                fill_price = symbol_dic['vwap']
            elif isinstance(order, Order):
                # TODO
                fill_price = symbol_dic['close']
            else:
                raise NotImplementedError("order class {} not support!".format(order.__class__))
            
            # get fill size
            fill_size = order.entrust_size - order.fill_size
            
            # create trade indication
            trade_ind = Trade()
            trade_ind.init_from_order(order)
            trade_ind.send_fill_info(fill_price, fill_size,
                                     date, time,
                                     self._next_fill_no())
            results.append(trade_ind)
            
            # update order status
            order.fill_price = (order.fill_price * order.fill_size
                                + fill_price * fill_size) / (order.fill_size + fill_size)
            order.fill_size += fill_size
            if order.fill_size == order.entrust_size:
                order.order_status = common.ORDER_STATUS.FILLED
        
        self.__orders = {k: v for k, v in self.__orders.viewitems() if not v.is_finished}
        # self.cancel_order(order.entrust_no)  # TODO DEBUG
        
        return results


class OrderBook(object):
    def __init__(self):
        self.orders = []
        self.trade_id = 0
        self.order_id = 0
        
        self.seq_gen = SequenceGenerator()
    
    def next_trade_id(self):
        return self.seq_gen.get_next('trade_id')
    
    def next_order_id(self):
        return self.seq_gen.get_next('order_id')
    
    def add_order(self, order):
        neworder = Order()
        # to do
        order.entrust_no = self.next_order_id()
        neworder.copy(order)
        self.orders.append(neworder)
    
    def make_trade(self, quote, freq):
        
        if freq == common.QUOTE_TYPE.TICK:
            # TODO
            return self.makeTickTrade(quote)
        
        elif (freq == common.QUOTE_TYPE.MIN
              or freq == common.QUOTE_TYPE.FIVEMIN
              or freq == common.QUOTE_TYPE.QUARTERMIN
              or freq == common.QUOTE_TYPE.SPECIALBAR):
            return self._make_trade_bar(quote)
        
        elif freq == common.QUOTE_TYPE.DAILY:
            return self._make_trade_bar(quote)
    
    def _make_trade_bar(self, quote):
        low = quote.low
        high = quote.high
        quote_date = quote.trade_date
        quote_time = quote.time
        quote_symbol = quote.symbol
        
        result = []
        # to be optimized
        for order in self.orders:
            if quote_symbol != order.symbol:
                continue
            if order.is_finished:
                continue
            
            if order.order_type == common.ORDER_TYPE.LIMIT:
                if order.entrust_action == common.ORDER_ACTION.BUY and order.entrust_price >= low:
                    trade = Trade()
                    trade.init_from_order(order)
                    trade.send_fill_info(order.entrust_price, order.entrust_size,
                                         quote_date, quote_time,
                                         self.next_trade_id())
                    
                    order.order_status = common.ORDER_STATUS.FILLED
                    order.fill_size = trade.fill_size
                    order.fill_price = trade.fill_price
                    
                    orderstatus_ind = OrderStatusInd()
                    orderstatus_ind.init_from_order(order)
                    result.append((trade, orderstatus_ind))
                    
                elif order.entrust_action == common.ORDER_ACTION.SELL and order.entrust_price <= high:
                    trade = Trade()
                    trade.init_from_order(order)
                    trade.send_fill_info(order.entrust_price, order.entrust_size,
                                         quote_date, quote_time,
                                         self.next_trade_id())
                    
                    order.order_status = common.ORDER_STATUS.FILLED
                    order.fill_size = trade.fill_size
                    order.fill_price = trade.fill_price
                    
                    orderstatus_ind = OrderStatusInd()
                    orderstatus_ind.init_from_order(order)
                    result.append((trade, orderstatus_ind))
            
            elif order.order_type == common.ORDER_TYPE.STOP:
                if order.entrust_action == common.ORDER_ACTION.BUY and order.entrust_price <= high:
                    trade = Trade()
                    trade.init_from_order(order)
                    trade.send_fill_info(order.entrust_price, order.entrust_size,
                                         quote_date, quote_time,
                                         self.next_trade_id())
                    
                    order.order_status = common.ORDER_STATUS.FILLED
                    order.fill_size = trade.fill_size
                    order.fill_price = trade.fill_price
                    orderstatus_ind = OrderStatusInd()
                    orderstatus_ind.init_from_order(order)
                    result.append((trade, orderstatus_ind))
                
                if order.entrust_action == common.ORDER_ACTION.SELL and order.entrust_price >= low:
                    trade = Trade()
                    trade.init_from_order(order)
                    trade.send_fill_info(order.entrust_price, order.entrust_size,
                                         quote_date, quote_time,
                                         self.next_trade_id())
                    
                    order.order_status = common.ORDER_STATUS.FILLED
                    order.fill_size = trade.fill_size
                    order.fill_price = trade.fill_price
                    orderstatus_ind = OrderStatusInd()
                    orderstatus_ind.init_from_order(order)
                    result.append((trade, orderstatus_ind))
        
        return result
    
    def cancel_order(self, entrust_no):
        for i in xrange(len(self.orders)):
            order = self.orders[i]
            
            if (order.is_finished):
                continue
            
            if (order.entrust_no == entrust_no):
                order.cancel_size = order.entrust_size - order.fill_size
                order.order_status = common.ORDER_STATUS.CANCELLED
            
            # todo
            orderstatus = OrderStatusInd()
            orderstatus.init_from_order(order)
            
            return orderstatus
    
    def cancel_all(self):
        result = []
        for order in self.orders:
            if order.is_finished:
                continue
            order.cancel_size = order.entrust_size - order.fill_size
            order.order_status = common.ORDER_STATUS.CANCELLED
            
            # todo
            orderstatus = OrderStatusInd()
            orderstatus.init_from_order(order)
            result.append(orderstatus)
        
        return result


class BarSimulatorGateway(BaseGateway):
    def __init__(self):
        super(BarSimulatorGateway, self).__init__()
        self.orderbook = OrderBook()
    
    def init_from_config(self, props):
        pass
    
    def on_new_day(self, trade_date):
        self.orderbook = OrderBook()
    
    def send_order(self, order, algo, param):
        self.orderbook.add_order(order)
        self.cb_pm.on_order_rsp(order, True, '')
    
    def process_quote(self, df_quote, freq):
        results = self.orderbook.make_trade(df_quote, freq)
        return results
