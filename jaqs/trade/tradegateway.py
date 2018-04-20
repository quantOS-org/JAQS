# encoding: UTF-8

from __future__ import print_function, unicode_literals

import copy
import time

import numpy as np

from jaqs.data.basic import *
from jaqs.data.basic import OrderStatusInd, Trade, TaskInd, Task
from jaqs.trade.tradeapi import TradeApi
from jaqs.util.sequence import SequenceGenerator
import jaqs.util as jutil


'''
class TradeCallback(with_metaclass(abc.ABCMeta)):
    @abstractmethod
    def on_trade(self, ind):
        pass
    
    @abstractmethod
    def on_order_status(self, ind):
        pass
    
    @abstractmethod
    def on_order_rsp(self, rsp):
        pass

'''


def calc_commission(trade_ind, commission_rate):
    turnover = abs(trade_ind.fill_price * trade_ind.fill_size)
    res = turnover * commission_rate
    return res


class BaseTradeApi(object):
    def __init__(self):
        super(BaseTradeApi, self).__init__()
        
        self._order_status_callback = None
        self._task_status_callback = None
        self._trade_callback = None
        self._on_connection_callback = None
        
        self.ctx = None
        
    def set_connection_callback(self, callback):
        self._on_connection_callback = callback
    
    def set_order_status_callback(self, callback):
        self._order_status_callback = callback
    
    def set_trade_callback(self, callback):
        self._trade_callback = callback
    
    def set_task_status_callback(self, callback):
        self._task_status_callback = callback
    
    def place_order(self, symbol, action, price, size, algo="", algo_param={}, userdata=""):
        """
        return (result, message)
        if result is None, message contains error information
        """
        pass
    
    def place_batch_order(self, orders, algo="", algo_param={}, userdata=""):
        """
        orders format:
            [ {"symbol": "000001.SZ", "action": "Buy", "price": 10.0, "size" : 100}, ... ]
        return (result, message)
        if result is None, message contains error information
        """
        pass
    
    def cancel_order(self, task_id):
        """
        return (result, message)
        if result is None, message contains error information
        """
        pass
    
    def query_account(self, format=""):
        """
            return pd.dataframe
        """
        pass
    
    def query_position(self, mode = "all", symbols = "", format=""):
        """
            symbols: seperate by ","
            return pd.dataframe
        """
        pass
    
    def query_net_position(self, mode = "all", symbols = "", format=""):
        """
            symbols: seperate by ","
            return pd.dataframe
        """
        pass
    
    def query_task(self, task_id = -1, format=""):
        """
            task_id: -1 -- all
            return pd.dataframe
        """
        pass
    
    def query_order(self, task_id = -1, format=""):
        """
            task_id: -1 -- all
            return pd.dataframe
        """
        pass
    
    def query_trade(self, task_id = -1, format=""):
        """
            task_id: -1 -- all
            return pd.dataframe
        """
        pass
    
    def query_portfolio(self, format=""):
        """
            return pd.dataframe
        """
        pass
    
    def goal_portfolio(self, positions, algo="", algo_param={}, userdata=""):
        """
        positions format:
            [ {"symbol": "000001.SZ", "ref_price": 10.0, "size" : 100}, ...]
        return (result, message)
        if result is None, message contains error information
        """
        pass
    
    def basket_order(self, orders, algo="", algo_param={}, userdata=""):
        """
        orders format:
            [ {"symbol": "000001.SZ", "ref_price": 10.0, "inc_size" : 100}, ...]
        return (result, message)
        if result is None, message contains error information
        """
        pass
    
    def stop_portfolio(self):
        """
        return (result, message)
        if result is None, message contains error information
        """
        pass
    
    def query_universe(self, format=""):
        pass


# DONT DELETE THIS: RealTimeTradeApi_async
'''
class RealTimeTradeApi_async(BaseTradeApi, EventEngine):
    """
    Attributes
    ----------
    _trade_api : TradeApi
    
    """
    def __init__(self):
        super(RealTimeTradeApi_async, self).__init__()
        
        self.ctx = None
        
        self._trade_api = None

        self._task_no_id_map = dict()

        self.seq_gen = SequenceGenerator()
        
    def init_from_config(self, props):
        """
        Instantiate TradeAPI and login.
        
        Parameters
        ----------
        props : dict

        """
        if self._trade_api is not None:
            self._trade_api.close()
    
        def get_from_list_of_dict(l, key, default=None):
            res = None
            for dic in l:
                res = dic.get(key, None)
                if res is not None:
                    break
            if res is None:
                res = default
            return res
    
        props_default = dict()  # jutil.read_json(jutil.join_relative_path('etc/trade_config.json'))
        dic_list = [props, props_default]
    
        address = get_from_list_of_dict(dic_list, "remote.trade.address", "")
        username = get_from_list_of_dict(dic_list, "remote.trade.username", "")
        password = get_from_list_of_dict(dic_list, "remote.trade.password", "")
        if address is None or username is None or password is None:
            raise ValueError("no address, username or password available!")
    
        tapi = TradeApi(address)
        self.set_trade_api_callbacks(tapi)

        # 使用用户名、密码登陆， 如果成功，返回用户可用的策略帐号列表
        print("\n{}@{} login...".format(username, address))
        user_info, msg = tapi.login(username, password)
        print("    Login msg: {:s}".format(msg))
        print("    Login user info: ", repr(user_info))
        print("")
        self._trade_api = tapi

        # event types and trade_api functions are one-to-one corresponded
        self._omni_api_map = {EVENT_TYPE.QUERY_ACCOUNT: self._trade_api.query_account,
                              EVENT_TYPE.QUERY_UNIVERSE: self._trade_api.query_universe,
                              EVENT_TYPE.QUERY_POSITION: self._trade_api.query_position,
                              EVENT_TYPE.QUERY_PORTFOLIO: self._trade_api.query_portfolio,
                              EVENT_TYPE.QUERY_TASK: self._trade_api.query_task,
                              EVENT_TYPE.QUERY_TRADE: self._trade_api.query_trade,
                              EVENT_TYPE.QUERY_ORDER: self._trade_api.query_order,
                              }
    
    # -------------------------------------------------------------------------------------------
    # On TradeAPI Callback: put a corresponding event to EventLiveTradeInstance

    def set_trade_api_callbacks(self, trade_api):
        trade_api.set_task_status_callback(self.on_task_status)
        trade_api.set_order_status_callback(self.on_order_status)
        trade_api.set_trade_callback(self.on_trade)
        trade_api.set_connection_callback(self.on_connection_callback)

    def on_connection_callback(self, connected):
        """
        
        Parameters
        ----------
        connected : bool

        """
        if connected:
            print("TradeAPI connected.")
            event_type = EVENT_TYPE.TRADE_API_CONNECTED
        else:
            print("TradeAPI disconnected.")
            event_type = EVENT_TYPE.TRADE_API_DISCONNECTED
        e = Event(event_type)
        self.ctx.instance.put(e)

    def on_trade(self, ind_dic):
        """
        
        Parameters
        ----------
        ind_dic : dict

        """
        # print("\nGateway on trade: ")
        # print(ind_dic)
        if 'security' in ind_dic:
            ind_dic['symbol'] = ind_dic.pop('security')
        
        ind = Trade.create_from_dict(ind_dic)
        ind.task_no = self._task_no_id_map[ind.task_id]
        
        e = Event(EVENT_TYPE.TRADE_IND)
        e.dic['ind'] = ind
        self.ctx.instance.put(e)

    def on_order_status(self, ind_dic):
        """
        
        Parameters
        ----------
        ind_dic : dict

        """
        # print("\nGateway on order status: ")
        # print(ind_dic)
        if 'security' in ind_dic:
            ind_dic['symbol'] = ind_dic.pop('security')
        
        ind = OrderStatusInd.create_from_dict(ind_dic)
        ind.task_no = self._task_no_id_map[ind.task_id]
        
        e = Event(EVENT_TYPE.ORDER_STATUS_IND)
        e.dic['ind'] = ind
        self.ctx.instance.put(e)
        
    def on_task_status(self, ind_dic):
        # print("\nGateway on task ind: ")
        # print(ind_dic)
        ind = TaskInd.create_from_dict(ind_dic)
        ind.task_no = self._task_no_id_map[ind.task_id]

        e = Event(EVENT_TYPE.TASK_STATUS_IND)
        e.dic['ind'] = ind
        self.ctx.instance.put(e)

    def on_omni_call(self, event):
        func = self._omni_api_map.get(event.type_, None)
        if func is None:
            print("{} not recgonized. Ignore.".format(event))
            return
    
        args = event.dic.get('args', None)
        func(args)

    def on_goal_portfolio(self, event):
        task = event.dic['task']
        positions = task.data
    
        # TODO: compatibility
        for dic in positions:
            dic['symbol'] = dic.pop('security')
    
        task_id, msg = self._trade_api.goal_portfolio(positions, algo=task.algo, algo_param=task.algo_param)
    
        self._generate_on_task_rsp(task.task_no, task_id, msg)

    def _generate_on_task_rsp(self, task_no, task_id, msg):
        # this rsp is generated by gateway itself
        rsp = TaskRsp(task_no=task_no, task_id=task_id, msg=msg)
        if rsp.success:
            self._task_no_id_map[task_id] = task_no
    
        # DEBUG
        print("\nGateway generate task_rsp {}".format(rsp))
        e = Event(EVENT_TYPE.TASK_RSP)
        e.dic['rsp'] = rsp
        self.ctx.instance.put(e)

    def on_place_order(self, event):
        task = event.dic['task']
        order = task.data
        task_id, msg = self._trade_api.place_order(order.symbol, order.entrust_action,
                                                   order.entrust_price, order.entrust_size,
                                                   task.algo, task.algo_param)
    
        self._generate_on_task_rsp(task.task_no, task_id, msg)

    # -------------------------------------------------------------------------------------------
    # Run
    
    def run(self):
        """
        Listen to certain events and run the EventEngine.
        Events include:
            1. placement & cancellation of orders
            2. query of universe, account, position and portfolio
            3. etc.

        """
        for e_type in self._omni_api_map.keys():
            self.register(e_type, self.on_omni_call)
        
        self.register(EVENT_TYPE.PLACE_ORDER, self.on_place_order)
        self.register(EVENT_TYPE.CANCEL_ORDER, self.on_omni_call)
        self.register(EVENT_TYPE.GOAL_PORTFOLIO, self.on_goal_portfolio)
        
        self.start(timer=False)
    
    # -------------------------------------------------------------------------------------------
    # API
    
    def _get_next_num(self, key):
        """used to generate id for orders and trades."""
        return str(np.int64(self.ctx.trade_date) * 10000 + self.seq_gen.get_next(key))

    def _get_next_task_no(self):
        return self._get_next_num('task_no')

    def publish_event(self, event):
        self.put(event)

    # ----------------------------------------------------------------------------------------
    # place & cancel

    def place_order(self, symbol, action, price, size, algo="", algo_param=None, userdata=""):
        if algo_param is None:
            algo_param = dict()
    
        # this order object is not for TradeApi, but for strategy itself to remember the order
        order = Order.new_order(symbol, action, price, size, self.ctx.trade_date, 0)
        order.entrust_no = self._get_next_num('entrust_no')
    
        task = Task(self._get_next_task_no(),
                    algo=algo, algo_param=algo_param,
                    data=order,
                    function_name="place_order")
        self.ctx.pm.add_task(task)
        # self.task_id_map[order.task_id].append(order.entrust_no)
    
        # self.pm.add_order(order)
    
        e = Event(EVENT_TYPE.PLACE_ORDER)
        e.dic['task'] = task
        self.publish_event(e)

    def cancel_order(self, entrust_no):
        e = Event(EVENT_TYPE.CANCEL_ORDER)
        e.dic['entrust_no'] = entrust_no
        self.publish_event(e)

    # ----------------------------------------------------------------------------------------
    # PMS

    def goal_portfolio(self, positions, algo="", algo_param=None, userdata=""):
        if algo_param is None:
            algo_param = dict()
    
        task = Task(self._get_next_task_no(), data=positions,
                    algo=algo, algo_param=algo_param,
                    function_name="goal_portfolio")
        self.ctx.pm.add_task(task)
    
        e = Event(EVENT_TYPE.GOAL_PORTFOLIO)
        e.dic['task'] = task
        self.publish_event(e)

    # ----------------------------------------------------------------------------------------
    # query account, universe, position, portfolio

    def query_account_async(self, userdata=""):
        args = locals()
        e = Event(EVENT_TYPE.QUERY_ACCOUNT)
        e.dic['args'] = args
        self.publish_event(e)

    def query_universe_async(self, userdata=""):
        args = locals()
        e = Event(EVENT_TYPE.QUERY_UNIVERSE)
        e.dic['args'] = args
        self.publish_event(e)

    def query_position_async(self, mode="all", symbols="", userdata=""):
        args = locals()
        e = Event(EVENT_TYPE.QUERY_POSITION)
        e.dic['args'] = args
        e.dic['securities'] = e.dic.pop['symbols']
        self.publish_event(e)

    def query_portfolio_async(self, userdata=""):
        args = locals()
        e = Event(EVENT_TYPE.QUERY_PORTFOLIO)
        e.dic['args'] = args
        self.publish_event(e)
    
    def query_account(self, format=""):
        return self._trade_api.query_account(format=format)

    def query_universe(self, format=""):
        return self._trade_api.query_universe(format=format)

    def query_position(self, mode = "all", symbols = "", format=""):
        return self._trade_api.query_position(mode=mode, securities=symbols, format=format)

    def query_portfolio(self, format=""):
        return self._trade_api.query_portfolio(format=format)

    # ----------------------------------------------------------------------------------------
    # Use Strategy

    def use_strategy(self, strategy_id) :
        return self._trade_api.use_strategy(strategy_id)
    
    # ----------------------------------------------------------------------------------------
    # query task, order, trade

    def query_task(self, task_id=-1, userdata=""):
        args = locals()
        e = Event(EVENT_TYPE.QUERY_TASK)
        e.dic['args'] = args
        self.publish_event(e)

    def query_order(self, task_id=-1, userdata=""):
        args = locals()
        e = Event(EVENT_TYPE.QUERY_ORDER)
        e.dic['args'] = args
        self.publish_event(e)

    def query_trade(self, task_id=-1, userdata=""):
        args = locals()
        e = Event(EVENT_TYPE.QUERY_TRADE)
        e.dic['args'] = args
        self.publish_event(e)


'''


class RealTimeTradeApi(TradeApi):
    def __init__(self, props, **trade_api_kwargs):
        address = props['remote.trade.address']
        
        super(RealTimeTradeApi, self).__init__(address, **trade_api_kwargs)
        
        self.ctx = None
        self.user_info = dict()
        
    def init_from_config(self, props):
        self.set_trade_api_callbacks()

        def get_from_list_of_dict(l, key, default=None):
            res = None
            for dic in l:
                res = dic.get(key, None)
                if res is not None:
                    break
            if res is None:
                res = default
            return res

        props_default = dict()  # jutil.read_json(jutil.join_relative_path('etc/trade_config.json'))
        dic_list = [props, props_default]

        address = get_from_list_of_dict(dic_list, "remote.trade.address", "")
        username = get_from_list_of_dict(dic_list, "remote.trade.username", "")
        password = get_from_list_of_dict(dic_list, "remote.trade.password", "")
        if address is None or username is None or password is None:
            raise ValueError("no address, username or password available!")
        
        # 使用用户名、密码登陆， 如果成功，返回用户可用的策略帐号列表
        print("\nTradeApi login {}@{}".format(username, address))
        user_info, msg = self.login(username, password)
        if not (msg == '0,'):
            print("    login failed: msg = '{}'\n".format(msg))
        else:
            print("    login success. user info: \n", repr(user_info), "\n")
            
        self.user_info = user_info
        
        strategy_no = get_from_list_of_dict(dic_list, "strategy_no", 0)
        sid, msg = self.use_strategy(strategy_no)
        if not msg.split(',')[0] == '0':
            raise RuntimeError("use strategy failed. Error msg: {}\n"
                               "Please re-try.".format(msg))
        time.sleep(0.1)
    
    def set_trade_api_callbacks(self):
        self.set_task_callback(self.on_task_status)
        self.set_ordstatus_callback(self.on_order_status)
        self.set_trade_callback(self.on_trade)
        self.set_connection_callback(self.on_connection_callback)

    def on_connection_callback(self, connected):
        """
        
        Parameters
        ----------
        connected : bool

        """
        if connected:
            print("TradeAPI connected.")
        else:
            print("TradeAPI disconnected.")

    def on_trade(self, ind_dic):
        """
        
        Parameters
        ----------
        ind_dic : dict

        """
        # print("\nGateway on trade: ")
        # print(ind_dic)
        if 'security' in ind_dic:
            ind_dic['symbol'] = ind_dic.pop('security')
        
        ind = Trade.create_from_dict(ind_dic)
        
        self.ctx.strategy.on_trade(ind)
    
    def on_order_status(self, ind_dic):
        """
        
        Parameters
        ----------
        ind_dic : dict

        """
        # print("\nGateway on order status: ")
        # print(ind_dic)
        if 'security' in ind_dic:
            ind_dic['symbol'] = ind_dic.pop('security')
        
        ind = OrderStatusInd.create_from_dict(ind_dic)
        
        self.ctx.strategy.on_order_status(ind)
    
    def on_task_status(self, ind_dic):
        # print("\nGateway on task ind: ")
        # print(ind_dic)
        ind = TaskInd.create_from_dict(ind_dic)
        
        self.ctx.strategy.on_task_status(ind)

    @staticmethod
    def _is_failed_task(task_id):
        return (task_id is None) or (task_id == 0)
    
    def place_order(self, symbol, action, price, size, algo="", algo_param={}, userdata=""):
        # Generate Task
        order = Order.new_order(symbol, action, price, size, self.ctx.trade_date, self.ctx.time,
                                order_type=common.ORDER_TYPE.LIMIT)
        
        task_id, msg = super(RealTimeTradeApi, self).place_order(symbol, action, price, size, algo, algo_param, userdata)
        if self._is_failed_task(task_id):
            return task_id, msg
    
        task = Task(task_id,
                    algo=algo, algo_param=algo_param, data=order,
                    function_name='place_order')
        
        self.ctx.pm.add_task(task)
    
        return task_id, msg
    

# ---------------------------------------------
# For Alpha Strategy

class AlphaTradeApi(BaseTradeApi):
    def __init__(self):
        super(AlphaTradeApi, self).__init__()
        self.ctx = None
        
        self._simulator = DailyStockSimulator()
        self.entrust_no_task_id_map = dict()
        self.seq_gen = SequenceGenerator()

        self.commission_rate = 0.0
        
        self.MATCH_TIME = 143000

    def _get_next_task_id(self):
        return np.int64(self.ctx.trade_date) * 10000 + self.seq_gen.get_next('task_id')

    def _add_task_id(self, ind):
        no = ind.entrust_no
        task_id = self.entrust_no_task_id_map[no]
        ind.task_id = task_id
        # ind.task_no = task_id

    def init_from_config(self, props):
        self.commission_rate = props.get('commission_rate', 0.0)
        
        self.set_order_status_callback(lambda ind: self.ctx.strategy.on_order_status(ind))
        self.set_trade_callback(lambda ind: self.ctx.strategy.on_trade(ind))
        self.set_task_status_callback(lambda ind: self.ctx.strategy.on_task_status(ind))
        
    def query_account(self, format=""):
        pass
    
    def on_new_day(self, trade_date):
        self._simulator.on_new_day(trade_date)

    def on_after_market_close(self):
        self._simulator.on_after_market_close()

    def place_order(self, security, action, price, size, algo="", algo_param={}, userdata=""):
        if size <= 0:
            print("Invalid size {}".format(size))
            return
    
        # Generate Task
        order = Order.new_order(security, action, price, size, self.ctx.trade_date, self.ctx.time,
                                order_type=common.ORDER_TYPE.LIMIT)
    
        task_id = self._get_next_task_id()
        order.task_id = task_id
    
        task = Task(task_id,
                    algo=algo, algo_param=algo_param, data=order,
                    function_name='place_order', trade_date=self.ctx.trade_date)
        # task.task_no = task_id
    
        # Send Order to Exchange
        entrust_no = self._simulator.add_order(order)
        task.data.entrust_no = entrust_no
    
        self.ctx.pm.add_task(task)
        self.entrust_no_task_id_map[entrust_no] = task.task_id
    
        order_status_ind = OrderStatusInd(order)
        order_status_ind.order_status = common.ORDER_STATUS.ACCEPTED
        self._order_status_callback(order_status_ind)
    
        return task_id, ""

    def cancel_order(self, task_id):
        task = self.ctx.pm.get_task(task_id)
        if task.function_name == 'place_order':
            order = task.data
            entrust_no = order.entrust_no
            order_status_ind, err_msg = self._simulator.cancel_order(entrust_no)
            task_id = self.entrust_no_task_id_map[entrust_no]
            order_status_ind.task_id = task_id
            # order_status_ind.task_no = task_id
            self._order_status_callback(order_status_ind)
        else:
            raise NotImplementedError("cancel task with function_name = {}".format(task.function_name))

    def goal_portfolio(self, positions, algo="", algo_param={}, userdata=""):
        # Generate Orders
        task_id = self._get_next_task_id()

        orders = {}
        for goal in positions:
            sec, goal_size = goal['symbol'], goal['size']
            if sec in self.ctx.pm.holding_securities:
                current_size = self.ctx.pm.get_position(sec).current_size
            else:
                current_size = 0
            diff_size = goal_size - current_size
            if diff_size != 0:
                action = common.ORDER_ACTION.BUY if diff_size > 0 else common.ORDER_ACTION.SELL
        
                order = FixedPriceTypeOrder.new_order(sec, action, 0.0, abs(diff_size), self.ctx.trade_date, 0)
                if algo == 'vwap':
                    order.price_target = 'vwap'  # TODO
                elif algo.startswith('limit:'):
                    order.price_target = algo.split(':')[1].strip()
                elif algo == '':
                    order.price_target = 'vwap'
                else:
                    raise NotImplementedError("goal_portfolio algo = {}".format(algo))

                order.task_id = task_id
                order.entrust_no = self._simulator.add_order(order)
                orders[order.entrust_no] = order

        # Generate Task
        task = Task(task_id,
                    algo=algo, algo_param=algo_param, data=orders,
                    function_name='goal_portfolio', trade_date=self.ctx.trade_date)


        self.ctx.pm.add_task(task)

        # Send Orders to Exchange
        for entrust_no, order in orders.items():
            self.entrust_no_task_id_map[entrust_no] = task.task_id

            order_status_ind = OrderStatusInd(order)
            order_status_ind.order_status = common.ORDER_STATUS.ACCEPTED

            self._order_status_callback(order_status_ind)
    
    def goal_portfolio_by_batch_order(self, goals):
        assert len(goals) == len(self.ctx.universe)
    
        orders = []
        for goal in goals:
            sec, goal_size = goal.symbol, goal.size
            if sec in self.ctx.pm.holding_securities:
                current_size = self.ctx.pm.get_position(sec).current_size
            else:
                current_size = 0
            diff_size = goal_size - current_size
            if diff_size != 0:
                action = common.ORDER_ACTION.BUY if diff_size > 0 else common.ORDER_ACTION.SELL
            
                order = FixedPriceTypeOrder.new_order(sec, action, 0.0, abs(diff_size), self.ctx.trade_date, 0)
                order.price_target = 'vwap'  # TODO
            
                orders.append(order)
        
        for order in orders:
            self._simulator.add_order(order)

    @property
    def match_finished(self):
        return self._simulator.match_finished
    
    '''
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
        return self._simulator.match(price_dict, date=self.ctx.trade_date, time=time)

    '''
    def _add_commission(self, ind):
        comm = calc_commission(ind, self.commission_rate)
        ind.commission = comm
        
    def match_and_callback(self, price_dict):
        results = self._simulator.match(price_dict, date=self.ctx.trade_date, time=self.MATCH_TIME)

        for trade_ind, order_status_ind in results:
            self._add_commission(trade_ind)
        
            task_id = self.entrust_no_task_id_map[trade_ind.entrust_no]
            self._add_task_id(trade_ind)
            self._add_task_id(order_status_ind)

            self._order_status_callback(order_status_ind)
            self._trade_callback(trade_ind)

            task = self.ctx.pm.get_task(task_id)
            if task.is_finished:
                task_ind = TaskInd(task_id, task_status=task.task_status,
                                   task_algo='', task_msg="")
                self._task_status_callback(task_ind)

        return results


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

    def _get_next_entrust_no(self):
        """used to generate id for orders and trades."""
        return str(self.seq_gen.get_next('entrust_no'))
    
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
        neworder = copy.copy(order)
        self._validate_order(order)

        entrust_no = self._get_next_entrust_no()
        neworder.entrust_no = entrust_no

        self.__orders[entrust_no] = neworder
        return entrust_no
    
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
        order = self.__orders.pop(entrust_no, None)
        if order is None:
            err_msg = "No order with entrust_no {} in simulator.".format(entrust_no)
            order_status_ind = None
        else:
            order.cancel_size = order.entrust_size - order.fill_size
            order.order_status = common.ORDER_STATUS.CANCELLED
            
            err_msg = ""
            order_status_ind = OrderStatusInd(order)
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
            trade_ind = Trade(order)
            trade_ind.set_fill_info(fill_price, fill_size,
                                    date, time,
                                    self._next_fill_no(),
                                    trade_date=date)
            
            # update order status
            order.fill_price = (order.fill_price * order.fill_size
                                + fill_price * fill_size) / (order.fill_size + fill_size)
            order.fill_size += fill_size
            if order.fill_size == order.entrust_size:
                order.order_status = common.ORDER_STATUS.FILLED
                
            order_status_ind = OrderStatusInd(order)
            
            results.append((trade_ind, order_status_ind))
        
        self.__orders = {k: v for k, v in self.__orders.items() if not v.is_finished}
        # self.cancel_order(order.entrust_no)  # TODO DEBUG
        
        return results


# ---------------------------------------------
# For Event-driven Strategy

class OrderBook(object):
    def __init__(self):
        self.orders = dict()
        
        self.seq_gen = SequenceGenerator()
        self.participation_rate = 1.0
    
    def _next_fill_no(self):
        return str(self.seq_gen.get_next('trade_id'))
    
    def _next_order_entrust_no(self):
        return str(self.seq_gen.get_next('order_id'))
    
    def add_order(self, order):
        neworder = copy.copy(order)
        
        entrust_no = self._next_order_entrust_no()
        neworder.entrust_no = entrust_no
        
        self.orders[entrust_no] = neworder
        
        return entrust_no
    
    def _make_tick_trade(self, quote):
        raise NotImplementedError()
    
    def make_trade(self, quote, freq):
        
        if freq == common.QUOTE_TYPE.TICK:
            # TODO
            return self._make_tick_trade(quote)
        
        elif (freq == common.QUOTE_TYPE.MIN
              or freq == common.QUOTE_TYPE.FIVEMIN
              or freq == common.QUOTE_TYPE.QUARTERMIN
              or freq == common.QUOTE_TYPE.SPECIALBAR):
            return self._make_trade_bar(quote)
        
        elif freq == common.QUOTE_TYPE.DAILY:
            return self._make_trade_bar(quote)
    
    def _make_trade_bar(self, quote_dic):
        
        result = []
        
        for entrust_no, order in self.orders.items():
            quote = quote_dic[order.symbol]
            low = quote.low
            high = quote.high
            #quote_date = quote.trade_date
            quote_date = quote.date
            quote_time = quote.time
            volume = quote.volume
            
            '''
            if order.order_type == common.ORDER_TYPE.LIMIT:
                if order.entrust_action == common.ORDER_ACTION.BUY and order.entrust_price >= low:
                    trade = Trade()
                    trade.init_from_order(order)
                    trade.send_fill_info(order.entrust_price, order.entrust_size,
                                         quote_date, quote_time,
                                         self._next_fill_no())
                    
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
                                         self._next_fill_no())
                    
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
                                         self._next_fill_no())
                    
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
                                         self._next_fill_no())
                    
                    order.order_status = common.ORDER_STATUS.FILLED
                    order.fill_size = trade.fill_size
                    order.fill_price = trade.fill_price
                    orderstatus_ind = OrderStatusInd()
                    orderstatus_ind.init_from_order(order)
                    result.append((trade, orderstatus_ind))
            '''
            
            entrust_price = order.entrust_price
            entrust_size = order.entrust_size
            
            fill_size = 0
            if order.order_type == common.ORDER_TYPE.LIMIT:
                if common.ORDER_ACTION.is_positive(order.entrust_action) and entrust_price >= low:
                    fill_price = min(entrust_price, high)
                    # fill_size = min(entrust_size, self.participation_rate * volume)
                    fill_size = entrust_size
                    
                elif common.ORDER_ACTION.is_negative(order.entrust_action) and order.entrust_price <= high:
                    fill_price = max(entrust_price, low)
                    # fill_size = min(entrust_size, self.participation_rate * volume)
                    fill_size = entrust_size

            elif order.order_type == common.ORDER_TYPE.STOP:
                if common.ORDER_ACTION.is_positive(order.entrust_action) and order.entrust_price <= high:
                    fill_price = max(entrust_price, low)
                    # fill_size = min(entrust_size, self.participation_rate * volume)
                    fill_size = entrust_size

                if common.ORDER_ACTION.is_negative(order.entrust_action) and order.entrust_price >= low:
                    fill_price = min(entrust_price, high)
                    # fill_size = min(entrust_size, self.participation_rate * volume)
                    fill_size = entrust_size
            
            elif order.order_type == common.ORDER_TYPE.VWAP:
                fill_price = quote.vwap
                fill_size = entrust_size

            if not fill_size:
                continue
                
            trade_ind = Trade(order)
            trade_ind.set_fill_info(fill_price, fill_size,
                                    quote_date, quote_time,
                                    self._next_fill_no(),
                                    trade_date=quote.trade_date)
            
            order.fill_price = ((order.fill_price * order.fill_size + fill_size * fill_price)
                                / (order.fill_size + fill_size))
            order.fill_size += fill_size
            if order.fill_size == order.entrust_size:
                order.order_status = common.ORDER_STATUS.FILLED
            
            order_status_ind = OrderStatusInd(order)
            
            result.append((trade_ind, order_status_ind))
            
        self.orders = {k: v for k, v in self.orders.items() if not v.is_finished}

        return result
    
    def cancel_order(self, entrust_no):
        order = self.orders.pop(entrust_no)
        order.cancel_size = order.entrust_size - order.fill_size
        order.order_status = common.ORDER_STATUS.CANCELLED
        
        order_status_ind = OrderStatusInd(order)
        
        '''
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
    
        '''
        return order_status_ind
        
    '''
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

    '''


class BacktestTradeApi(BaseTradeApi):
    def __init__(self):
        super(BacktestTradeApi, self).__init__()
        
        self.ctx = None
        
        self._orderbook = OrderBook()
        self.seq_gen = SequenceGenerator()
        self.entrust_no_task_id_map = dict()
        
        self.commission_rate = 0.0
        
    def _get_next_num(self, key):
        """used to generate id for orders and trades."""
        return str(np.int64(self.ctx.trade_date) * 10000 + self.seq_gen.get_next(key))

    def _get_next_task_id(self):
        return np.int64(self.ctx.trade_date) * 10000 + self.seq_gen.get_next('task_id')
    
    def init_from_config(self, props):
        self.commission_rate = props.get('commission_rate', 0.0)
        
        self.set_order_status_callback(lambda ind: self.ctx.strategy.on_order_status(ind))
        self.set_trade_callback(lambda ind: self.ctx.strategy.on_trade(ind))
        self.set_task_status_callback(lambda ind: self.ctx.strategy.on_task_status(ind))

    def on_new_day(self, trade_date):
        self._orderbook = OrderBook()
    
    def use_strategy(self, strategy_id):
        pass
    
    def place_order(self, security, action, price, size, algo="", algo_param={}, userdata=""):
        if size <= 0:
            print("Invalid size {}".format(size))
            return
        
        # Generate Order
        if algo == 'vwap':
            order_type = common.ORDER_TYPE.VWAP
        else:
            order_type = common.ORDER_TYPE.LIMIT
        order = Order.new_order(security, action, price, size, self.ctx.trade_date, self.ctx.time,
                                order_type=order_type)

        # Generate Task
        task_id = self._get_next_task_id()
        order.task_id = task_id
        
        task = Task(task_id,
                    algo=algo, algo_param=algo_param, data=order,
                    function_name='place_order', trade_date=self.ctx.trade_date)
        # task.task_no = task_id
        
        # Send Order to Exchange
        entrust_no = self._orderbook.add_order(order)
        task.data.entrust_no = entrust_no
        
        self.ctx.pm.add_task(task)
        self.entrust_no_task_id_map[entrust_no] = task.task_id

        order_status_ind = OrderStatusInd(order)
        order_status_ind.order_status = common.ORDER_STATUS.ACCEPTED
        # order_status_ind.task_no = task_id
        self._order_status_callback(order_status_ind)


        '''
        # TODO: not necessary
        rsp = OrderRsp(entrust_no=entrust_no, task_id=task_id, msg="")
        self.ctx.instance.strategy.on_order_rsp(rsp)
        '''
        
        return task_id, ""

    def cancel_order(self, task_id):
        task = self.ctx.pm.get_task(task_id)
        if task.function_name == 'place_order':
            order = task.data
            if order.order_status in [common.ORDER_STATUS.NEW, common.ORDER_STATUS.ACCEPTED]:
                entrust_no = order.entrust_no
                order_status_ind = self._orderbook.cancel_order(entrust_no)
                task_id = self.entrust_no_task_id_map[entrust_no]
                order_status_ind.task_id = task_id
                # order_status_ind.task_no = task_id
                self._order_status_callback(order_status_ind)
            else:
                order_status_ind = OrderStatusInd(order)
                order_status_ind.task_id = task_id
                self._order_status_callback(order_status_ind)

        else:
            raise NotImplementedError("cancel task with function_name = {}".format(task.function_name))
    
    def _process_quote(self, df_quote, freq):
        return self._orderbook.make_trade(df_quote, freq)

    def _add_task_id(self, ind):
        no = ind.entrust_no
        task_id = self.entrust_no_task_id_map[no]
        ind.task_id = task_id
        # ind.task_no = task_id
    
    def _add_commission(self, ind):
        comm = calc_commission(ind, self.commission_rate)
        ind.commission = comm
        
    def match_and_callback(self, quote, freq):
        results = self._process_quote(quote, freq)
        
        for trade_ind, order_status_ind in results:
            self._add_commission(trade_ind)
            
            # self._add_task_id(trade_ind)
            # self._add_task_id(order_status_ind)
            task_id = self.entrust_no_task_id_map[trade_ind.entrust_no]

            self._order_status_callback(order_status_ind)
            self._trade_callback(trade_ind)

            task = self.ctx.pm.get_task(task_id)
            if task.is_finished:
                task_ind = TaskInd(task_id, task_status=task.task_status,
                                   task_algo='', task_msg="")
                self._task_status_callback(task_ind)
        
        return results

