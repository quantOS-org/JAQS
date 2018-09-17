# encoding: utf-8
"""
Classes defined in strategy module
"""

from __future__ import print_function
import abc
from abc import abstractmethod
from six import with_metaclass

import numpy as np
import pandas as pd

from jaqs.data.basic import GoalPosition
from jaqs.util.sequence import SequenceGenerator
from jaqs.data.basic import Bar, Quote
# import jaqs.util as jutil

from jaqs.trade import model
from jaqs.trade import common


class Strategy(with_metaclass(abc.ABCMeta)):
    """
    Abstract base class for strategies.

    Attributes
    ----------
    ctx : Context object
        Used to store relevant context of the strategy.
    run_mode : int
        Whether the trategy is under back-testing or live trading.
    pm : trade.PortfolioManger
        Responsible for managing orders, trades and positions.
    store : dict
        A dictionary to store variables that will be automatically saved.

    Methods
    -------

    """
    
    def __init__(self):
        super(Strategy, self).__init__()
        self.ctx = None
        # self.run_mode = common.RUN_MODE.BACKTEST
        
        # self.ctx.pm = PortfolioManager(strategy=self)
        # self.pm = self.ctx.pm

        # self.task_id_map = defaultdict(list)
        self.seq_gen = SequenceGenerator()

        self.init_balance = 0.0
        
    def init_from_config(self, props):
        pass
    
    def initialize(self):
        pass
    
    def _get_next_num(self, key):
        """used to generate id for orders and trades."""
        return str(np.int64(self.ctx.trade_date) * 10000 + self.seq_gen.get_next(key))
    
    '''
    # -------------------------------------------------------------------------------------------
    # Order

    def place_order(self, symbol, action, price, size, algo="", algo_param=None):
        """
        Send a request with an order to the system. Execution algorithm will be automatically chosen.
        Returns task_id which can be used to query execution and orders of this task.

        Parameters
        ----------
        symbol : str
            the symbol of symbol to be ordered, eg. "000001.SZ".
        action : str
        price : float.
            The price to be ordered at.
        size : int
            The quantity to be ordered at.
        algo : str, optional
            The algorithm to be used. If None then use default algorithm.
        algo_param : dict, optional
            Parameters of the algorithm. Default {}.

        Returns
        -------
        res : str
        msg : str.
            if res is None, message contains error information

        """
        pass

    def cancel_order(self, task_id):
        """Cancel all uncome orders of a task according to its task ID.

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
        pass

    # -------------------------------------------------------------------------------------------
    # Query

    def query_account(self):
        """
        
        Returns
        -------
        pd.DataFrame

        """
        pass

    def query_universe(self):
        """
        
        Returns
        -------
        pd.DataFrame

        """
        pass

    def query_position(self, mode="all", symbols=""):
        """
        Parameters
        ----------
        mode : str, optional
        symbols : str, optional
            Separated by ,
        
        Returns
        -------
        pd.DataFrame
        
        """
        pass

    def query_portfolio(self):
        """
        Return net positions of all securities in the strategy universe (including zero positions).

        Returns
        --------
        pd.DataFrame
            Current position of the strategy.

        """
        pass

    def query_task(self, task_id=-1):
        """
        Query order information of current day.

        Parameters
        ----------
        task_id : int, optional
            ID of the task. -1 by default (return all orders of the day; else return orders of this task).

        Returns
        -------
        pd.DataFrame

        """

    def query_order(self, task_id=-1):
        """
        Query order information of current day.

        Parameters
        ----------
        task_id : int
            ID of the task. -1 by default (return all orders of the day; else return orders of this task).

        Returns
        -------
        pd.DataFrame

        """
        pass

    def query_trade(self, task_id=-1):
        """
        Query trade information of current day.

        Parameters
        -----------
        task_id : int
            ID of the task. -1 by default (return all orders of the day; else return orders of this task).

        Returns
        --------
        pd.DataFrame

        """
        pass

    # -------------------------------------------------------------------------------------------
    # Portfolio Order

    def goal_portfolio(self, positions, algo="", algo_param=None):
        """
        Let the system automatically generate orders according to portfolio positions goal.
        If there are uncome orders of any symbol in the strategy universe, this order will be rejected. #TODO not impl

        Parameters
        -----------
        positions : list of GoalPosition
            This must include positions of all securities in the strategy universe.
            Use former value if there is no change.
        algo : str, optional
            The algorithm to be used. If None then use default algorithm.
        algo_param : dict, optional
            Parameters of the algorithm. Default {}.

        Returns
        --------
        result : bool
            Whether this command is accepted. True means the system's acceptance, instead of positions have changed.
        err_msg : str

        """
        pass

    def stop_portfolio(self):
        """
        Returns
        -------
        result : str
        message : str
            If result is None, message contains error information
        
        """
        pass

    def place_batch_order(self, orders, algo="", algo_param=None):
        """Send a batch of orders to the system together.

        Parameters
        -----------
        orders : list
            a list of trade.model.Order objects.
        algo : str, optional
            The algorithm to be used. If None then use default algorithm.
        algo_param : dict, optional
            Parameters of the algorithm. Default {}.

        Returns
        -------
        task_id : str
            Task ID generated by entrust_order.
        err_msg : str.

        """
        pass

    def basket_order(self, orders, algo="", algo_param=None):
        """
        Parameters
        ----------
        orders : list of dict
            [ {"security": "000001.SZ", "ref_price": 10.0, "inc_size" : 100}, ...]
        algo : str, optional
        algo_param : dict or None, optional
        
        Returns
        -------
        result : str
        message : str
            If result is None, message contains error information
        
        """
        pass
    '''
    
    # -------------------------------------------------------------------------------------------
    # Callback Indications & Responses
    
    def on_trade(self, ind):
        """

        Parameters
        ----------
        ind : TradeInd

        Returns
        -------

        """
        pass

    def on_order_status(self, ind):
        """

        Parameters
        ----------
        ind : OrderStatusInd

        Returns
        -------

        """
        pass
    
    def on_order_rsp(self, rsp):
        """
        
        Parameters
        ----------
        rsp

        """
        pass

    def on_task_rsp(self, rsp):
        """
        
        Parameters
        ----------
        rsp

        """
        pass

    def on_task_status(self, ind):
        """
        
        Parameters
        ----------
        rsp

        """
        pass


class AlphaStrategy(Strategy, model.FuncRegisterable):
    """
    Alpha strategy class.

    Attributes
    ----------
    period : str
        Interval between current and next. {'day', 'week', 'month'}
    days_delay : int
        n'th business day after next period.
    weights : np.array with the same shape with self.context.universe
    benchmark : str
        The benchmark symbol.
    risk_model : model.RiskModel
    signal_model : model.ReturnModel
    cost_model : model.CostModel

    Methods
    -------

    """
    # TODO register context
    def __init__(self, signal_model=None, stock_selector=None,
                 cost_model=None, risk_model=None,
                 pc_method="equal_weight",
                 match_method="vwap",
                 fc_selector=None,
                 fc_constructor=None,
                 fc_options=None
                 ):
        super(AlphaStrategy, self).__init__()
        
        self.period = ""
        self.n_periods = 1
        self.days_delay = 0
        self.cash = 0
        self.position_ratio = 0.98
        self.single_symbol_weight_limit = 1.0
        
        self.risk_model = risk_model
        self.signal_model = signal_model
        self.cost_model = cost_model
        self.stock_selector = stock_selector
        
        self.weights = None
        
        self.pc_method = pc_method

        self.goal_positions = None
        self.match_method = match_method

        self.portfolio_construction = self.forecast_portfolio_construction if pc_method=="forecast" else self.default_portfolio_construction
        self._fc_selector    = fc_selector if fc_selector else AlphaStrategy.default_forecast_selector
        self._fc_constructor = fc_constructor if fc_constructor else AlphaStrategy.default_forecast_constructor
        self._fc_options     = fc_options

    def init_from_config(self, props):
        Strategy.init_from_config(self, props)
        
        self.cash = props.get('init_balance', 100000000)
        self.period = props.get('period', 'month')
        self.days_delay = props.get('days_delay', 0)
        self.n_periods = props.get('n_periods', 1)
        self.position_ratio = props.get('position_ratio', 0.98)
        self.single_symbol_weight_limit = props.get('single_symbol_weight_limit', 1.0)

        self.use_pc_method(name='industry_neutral_equal_weight', func=self.industry_neutral_equal_weight, options=None)
        self.use_pc_method(name='industry_neutral_index_weight', func=self.industry_neutral_index_weight, options=None)
        self.use_pc_method(name='equal_weight', func=self.equal_weight, options=None)
        self.use_pc_method(name='mc', func=self.optimize_mc, options={'util_func': self.util_net_signal,
                                                                           'constraints': None,
                                                                           'initial_value': None})
        self.use_pc_method(name='factor_value_weight', func=self.factor_value_weight, options=None)
        self.use_pc_method(name='index_weight', func=self.index_weight, options=None)
        self.use_pc_method(name='market_value_weight', func=self.market_value_weight, options=None)
        self.use_pc_method(name='market_value_sqrt_weight', func=self.market_value_weight, options={'sqrt': True})
        self.use_pc_method(name='equal_index_weight', func=self.equal_index_weight, options=None)

        self._validate_parameters()
        print("AlphaStrategy Initialized.")
    
    def _validate_parameters(self):
        if self.pc_method in ['mc', 'quad_opt']:
            if self.signal_model is None and self.cost_model is None and self.risk_model is None:
                raise ValueError("At least one model of signal, cost and risk must be provided.")
        elif self.pc_method in ['factor_value_weight']:
            if self.signal_model is None:
                raise ValueError("signal_model must be provided when pc_method = 'factor_value_weight'")
        elif self.pc_method in ['equal_weight',
                                'index_weight',
                                'equal_index_weight',
                                'market_value_weight',
                                'market_value_sqrt_weight',
                                'industry_neutral_index_weight',
                                'industry_neutral_equal_weight']:
            pass
        elif self.pc_method in ['forecast']:
            pass
        else:
            raise NotImplementedError("pc_method = {:s}".format(self.pc_method))
    
    def on_trade(self, ind):
        """

        Parameters
        ----------
        ind : TradeInd

        Returns
        -------

        """
        pass
        
    def use_pc_method(self, name, func, options=None):
        self._register_func(name, func, options)
    
    def _get_weights_last(self):
        current_positions = self.query_portfolio()
        univ_pos_dic = {p.symbol: p.current_size for p in current_positions}
        for sec in self.ctx.universe:
            if sec not in univ_pos_dic:
                univ_pos_dic[sec] = 0
        return univ_pos_dic

    def util_net_signal(self, weights_target):
        """
        util = net_signal = signal - all costs.
        
        Parameters
        ----------
        weights_target : dict
        
        """
        weights_last = self._get_weights_last()
    
        signal = self.signal_model.forecast_signal(weights_target)
        cost = self.cost_model.calc_cost(weights_last, weights_target)
        # liquid = self.liquid_model.calc_liquid(weight_now)
        risk = self.risk_model.calc_risk(weights_target)
    
        risk_coef = 1.0
        cost_coef = 1.0
        net_signal = signal - risk_coef * risk - cost_coef * cost  # - liquid * liq_factor
        return net_signal


    def default_portfolio_construction(self, universe_list=None):
        """
        Calculate target weights of each symbol in the strategy universe.
        User should not modify this function arbitrarily.
        
        Attributes
        ----------
        universe_list : list of str
            Symbols that should be considered during this re-balance.

        Returns
        -------
        self.weights : weights / GoalPosition (without rounding)
            Weights of each symbol.

        """
        # Step.1 filter and narrow down universe to sub-universe
        if self.stock_selector is not None:
            selected_list = self.stock_selector.get_selection()
            if type(selected_list) is pd.DataFrame:
                self.ctx.forecast_selected_list = selected_list
                universe_list = [s for s in universe_list if s in selected_list['symbol']]
            else:
                universe_list = [s for s in universe_list if s in selected_list]

        sub_univ = sorted(universe_list)

        self.ctx.snapshot_sub = self.ctx.snapshot.loc[sub_univ, :]

        # Step.2 pick the registered portfolio construction method

        rf = self.func_table[self.pc_method]
        func, options = rf.func, rf.options

        # Step.3 use the registered method to calculate weights and get weights for all symbols in universe
        weights_sub_universe, msg = func(**options)

        # portfolio balance check
        weights_all_universe = {symbol: weights_sub_universe.get(symbol, 0.0) for symbol in self.ctx.universe}
        if msg:
            print(msg)

        # if nan assign zero
        weights_all_universe = {k: 0.0 if np.isnan(v) else v for k, v in weights_all_universe.items()}
        
        # normalize
        w_sum = np.sum(np.abs(list(weights_all_universe.values())))
        if w_sum > 1e-8:  # else all zeros weights
            weights_all_universe = {k: v / w_sum for k, v in weights_all_universe.items()}
        
        # single symbol weight limit process
        if self.single_symbol_weight_limit < 1:
            weights_all_universe = {k: v if v < self.single_symbol_weight_limit else self.single_symbol_weight_limit
                                    for k, v in weights_all_universe.items()}

        self.weights = weights_all_universe

    def forecast_portfolio_construction(self, universe_list=None):

        assert callable(self._fc_selector), "fc_selector should be function"
        assert callable(self._fc_constructor), "fc_constuctor should be function"

        forecast_list = self._fc_selector(self, universe=universe_list)

        options = {}
        if self._fc_options:
            options.update(self._fc_options)

        self.weights = self._fc_constructor(self,
                                            self.weights.copy() if self.weights else [],
                                            forecast_list,
                                            **options)

    @staticmethod
    def default_forecast_selector(self, forecast_field='close_adj', universe=None):
        forecast = self.ctx.dataview.get_snapshot(self.ctx.trade_date)[[forecast_field]].copy()
        forecast = forecast.rename( columns={forecast_field: "forecast"})
        forecast['symbol'] = forecast.index
        return forecast

    @staticmethod
    def default_forecast_constructor(self, cur_weights, forecast,
                                     max_turnover=1,
                                     alpha_threshold=0.0005,
                                     turnover_cost_rate=0.001,
                                     init_size=20):

        forecast = forecast.sort_values(['forecast'], ascending=False)
        forecast.index = forecast['symbol']
        if not cur_weights:
            new_weights = forecast[:init_size].copy()
            new_weights.loc[:, 'weight'] = 1.0 / len(forecast)
            new_weights.index = new_weights['symbol']
            return new_weights[['weight']].T.to_dict(orient='records')[0]

        cur_weights = pd.DataFrame({'symbol': list(cur_weights.keys()), 'weight': list(cur_weights.values())})
        zero_weights = cur_weights[cur_weights['weight'] == 0].copy()
        cur_weights = cur_weights[cur_weights['weight'] > 0].copy()
        cur_weights.index = cur_weights['symbol']

        cur_weights['forecast'] = forecast['forecast']
        cur_weights['forecast'] = cur_weights['forecast'].fillna(0.0)
        cur_weights = cur_weights.sort_values(['forecast'])
        cur_weights['handled'] = False

        turnover = 0.0
        new_weights = []
        for i in range(len(forecast)):
            fc = forecast.iloc[i]
            replaced = False
            if fc['symbol'] not in cur_weights['symbol']:
                for k in range(len(cur_weights)):
                    tmp = cur_weights.iloc[k]
                    if fc['forecast'] > tmp['forecast'] + turnover_cost_rate + alpha_threshold and \
                            tmp['weight'] + turnover <= max_turnover:
                        new_weights.append({'symbol': fc['symbol'], 'weight': tmp['weight']})
                        new_weights.append({'symbol': tmp['symbol'], 'weight': 0})
                        cur_weights['handled'][0] = True
                        turnover += tmp['weight']
                        replaced = True
                        break
                if not replaced:
                    break
            else:
                tmp = cur_weights.loc[fc['symbol']]
                cur_weights.loc[fc['symbol'], 'handled'] = True
                new_weights.append({'symbol': fc['symbol'], 'weight': tmp['weight']})

            cur_weights = cur_weights[cur_weights['handled'] != True].copy()
            if cur_weights.empty or turnover >= max_turnover:
                break

        if not cur_weights.empty:
            for i in range(len(cur_weights)):
                tmp = cur_weights.iloc[i]
                new_weights.append({'symbol': tmp['symbol'], 'weight': tmp['weight']})

        new_weights = pd.DataFrame(new_weights)
        w_sum = new_weights['weight'].sum()
        if w_sum > 1e-8:  # else all zeros weights
            new_weights.loc[:, 'weight'] /= w_sum

        # Keep all removed stocks in weight, so when stock can be sold after it is tradable again after suspended.
        tmp = zero_weights[ ~zero_weights['symbol'].isin(new_weights['symbol']) ]
        if not tmp.empty:
            new_weights = pd.concat([new_weights, tmp])

        new_weights.index = new_weights['symbol']
        del new_weights['symbol']

        return new_weights.T.to_dict(orient='records')[0]

    def equal_weight(self):
        # discrete
        weights = {k: 1.0 for k in self.ctx.snapshot_sub.index.values}
        return weights, ''

    def industry_neutral_equal_weight(self):
        snap = self.ctx.snapshot_sub
        snap['symbol'] = snap.index

        # calculate weight distribution of all industry
        # df_weight = self.ctx.dataview.get_snapshot(self.ctx.trade_date)[['total_mv', 'index_member', 'sw1']]
        # df_weight = df_weight[df_weight['index_member'] == 1]
        # df_weight['weight'] = df_weight['total_mv']/df_weight['total_mv'].sum()
        df_weight = self.ctx.dataview.get_snapshot(self.ctx.trade_date)[['index_weight', 'sw1']]
        df_weight.columns = ['weight', 'sw1']

        df_industry_weight = df_weight.groupby('sw1')['weight'].sum()

        # industries in portfolio
        industry_list = list(set(snap['sw1'].values.flatten()))
        df_industry_weight_sub = df_industry_weight.loc[industry_list]
        df_industry_weight_sub = pd.DataFrame(df_industry_weight_sub)
        df_industry_weight_sub.columns = ['weight']
        df_industry_weight_sub['equal_weight'] = pd.DataFrame(snap.groupby('sw1')['sw1'].count()/len(snap))

        df_industry_weight_sub['dif'] = df_industry_weight_sub['equal_weight'] - df_industry_weight_sub['weight']

        df_industry_weight_sub = df_industry_weight_sub.reset_index()

        count_industry = pd.DataFrame(snap.groupby('sw1')['close'].count()).reset_index()
        count_industry.columns = ['sw1', 'count']
        count_industry['internal_weight'] = 1.0/count_industry['count']

        snap = pd.merge(left = snap, right = count_industry[['sw1', 'internal_weight']], how = 'left', on = 'sw1')
        snap = pd.merge(left = snap, right = df_industry_weight_sub[['sw1', 'norm_weight']], how = 'left', on = 'sw1')
        snap['weight'] = snap['internal_weight'] * snap['norm_weight']
        df_weight = snap[['symbol','weight']]

        df_weight = df_weight.set_index('symbol')
        weights = df_weight['weight'].to_dict()

        return weights, ''

    def industry_neutral_index_weight(self):
        snap = self.ctx.snapshot_sub
        snap['symbol'] = snap.index

        # calculate weight distribution of all industry
        df_weight = self.ctx.dataview.get_snapshot(self.ctx.trade_date)[['total_mv', 'index_member', 'sw1']]
        df_weight = df_weight[df_weight['index_member'] == 1]
        df_weight['weight'] = df_weight['total_mv']/df_weight['total_mv'].sum()
        df_industry_weight = df_weight.groupby('sw1')['weight'].sum()

        # industries in portfolio
        industry_list = list(set(snap['sw1'].values.flatten()))
        df_industry_weight_sub = df_industry_weight.loc[industry_list]
        df_industry_weight_sub = pd.DataFrame(df_industry_weight_sub)
        df_industry_weight_sub.columns = ['weight']
        df_industry_weight_sub['norm_weight'] = df_industry_weight_sub['weight']/df_industry_weight_sub['weight'].sum()
        df_industry_weight_sub = df_industry_weight_sub.reset_index()

        count_industry = pd.DataFrame(snap.groupby('sw1')['index_weight'].sum()).reset_index()
        count_industry.columns = ['sw1', 'agg_index_weight']

        snap = pd.merge(left = snap, right = count_industry[['sw1', 'agg_index_weight']], how = 'left', on = 'sw1')
        snap['internal_weight'] = snap['index_weight']/snap['agg_index_weight']
        snap = pd.merge(left = snap, right = df_industry_weight_sub[['sw1', 'norm_weight']], how = 'left', on = 'sw1')
        snap['weight'] = snap['internal_weight'] * snap['norm_weight']
        df_weight = snap[['symbol','weight']]

        df_weight = df_weight.set_index('symbol')
        weights = df_weight['weight'].to_dict()

        return weights, ''

    def market_value_weight(self, sqrt=False):
        snap = self.ctx.snapshot_sub
        # TODO: pass options, instead of hard-code 'total_mv', 'float_mv'
        if 'total_mv' in snap.columns:
            mv = snap['total_mv']
        elif 'float_mv' in snap.columns:
            mv = snap['float_mv']
        else:
            raise ValueError("market_value_weight is chosen,"
                             "while no [float_mv] or [total_mv] field found in dataview.")
        mv = mv.fillna(0.0)
        if sqrt:
            print('sqrt')
            mv = np.sqrt(mv)
        weights = mv.to_dict()
        return weights, ""

    def index_weight(self):
        snap = self.ctx.snapshot_sub
        if 'index_weight' not in snap.columns:
            raise ValueError("index_weight is chosen,"
                             "while no [index_weight] field found in dataview.")
        ser_index_weight = snap['index_weight']
        ser_index_weight.fillna(0.0, inplace=True)
        weights = ser_index_weight.to_dict()
        return weights, ""

    def equal_index_weight(self):
        snap = self.ctx.snapshot_sub
        snap.fillna(0.0, inplace=True)

        wt_equal = snap['index_member'] / sum(snap['index_member'])
        wt_index = snap['index_weight'] / sum(snap['index_weight'])

        wt_final = (wt_equal + wt_index) / 2

        return wt_final.to_dict(), ""

    def factor_value_weight(self):
        def long_only_weight_adjust(w):
            """
            Adjust weights for long only constraints.
            
            Parameters
            ----------
            w : dict
    
            Returns
            -------
            res : dict
    
            """
            # TODO: we should not add a const
            if not len(w):
                return w
            w_min = np.min(list(w.values()))
            if w_min < 0:
                delta = 2 * abs(w_min)
            # if nan assign zero; else add const
                w = {k: v + delta for k, v in w.items()}
            return w
        
        dic_forecasts = self.signal_model.make_forecast()
        weights = {k: 0.0 if (np.isnan(v) or np.isinf(v)) else v for k, v in dic_forecasts.items()}
        weights = long_only_weight_adjust(weights)
        return weights, ""
        
    def optimize_mc(self, util_func):
        """
        Use naive search (Monte Carol) to find variable that maximize util_func.
        
        Parameters
        ----------
        util_func : callable
            Input variables, output the value of util function.

        Returns
        -------
        min_weights : dict
            best weights.
        msg : str
            error message.

        """
        n_exp = 5  # number of experiments of Monte Carol
        sub_univ = self.ctx.snapshot_sub.index.values
        n_var = len(sub_univ)
    
        weights_mat = np.random.rand(n_exp, n_var)
        weights_mat = weights_mat / weights_mat.sum(axis=1).reshape(-1, 1)
    
        min_f = 1e30
        min_weights = None
        for i in range(n_exp):
            weights = {sub_univ[j]: weights_mat[i, j] for j in range(n_var)}
            f = -util_func(weights)
            if f < min_f:
                min_weights = weights
                min_f = f
    
        if min_weights is None:
            msg = "No weights can make f > {:.2e} found in this search".format(min_f)
        else:
            msg = ""
        return min_weights, msg

    def re_weight_suspension(self, suspensions=None):
        """
        How we deal with weights when there are suspension securities.

        Parameters
        ----------
        suspensions : list of securities
            None if no suspension.

        """
        # TODO this can be refine: consider whether we increase or decrease shares on a suspended symbol.
        if not suspensions:
            return
        
        if len(suspensions) == len(self.ctx.universe):
            raise ValueError("All suspended")  # TODO custom error
        
        weights = {sec: w if sec not in suspensions else 0.0 for sec, w in self.weights.items()}
        weights_sum = np.sum(np.abs(list(weights.values())))
        if weights_sum > 0.0:
            weights = {sec: w / weights_sum for sec, w in weights.items()}
        
        self.weights = weights
    
    def on_after_rebalance(self, total):
        print("Before {} re-balance: available cash all = {:9.4e}".format(self.ctx.trade_date, total))  # DEBUG
        pass
    
    def send_bullets(self):
        # self.ctx.trade_api.goal_portfolio_by_batch_order(self.goal_positions)
        self.ctx.trade_api.goal_portfolio(self.goal_positions, algo=self.match_method)
    
    def generate_weights_order(self, weights_dic, turnover, prices, suspensions=None):
        """
        Send order according subject to total turnover and weights of different securities.

        Parameters
        ----------
        weights_dic : dict of {symbol: weight}
            Weight of each symbol.
        turnover : float
            Total turnover goal of all securities. (cash quota)
        prices : dict of {str: float}
            {symbol: price}
        suspensions : list of str

        Returns
        -------
        goals : list of GoalPosition
        cash_left : float

        """
        # cash_left = 0.0
        cash_used = 0.0
        goals = []
        for sec, w in weights_dic.items():
            goal_pos = dict()
            goal_pos['symbol'] = sec
            
            if sec in suspensions:
                current_pos = self.ctx.pm.get_position(sec)
                goal_pos['size'] = current_pos.current_size if current_pos is not None else 0
            elif abs(w) < 1e-8:
                # order.entrust_size = 0
                goal_pos['size'] = 0
            else:
                price = prices[sec]
                if not (np.isfinite(price) and np.isfinite(w)):
                    raise ValueError("NaN or Inf encountered! \n"
                                     "trade_date={}, symbol={}, price={}, weight={}".format(self.ctx.trade_date,
                                                                                            sec, price, w))
                shares_raw = w * turnover / price
                # shares unit 100
                shares = int(round(shares_raw / 100., 0)) * 100  # TODO cash may be not enough
                # shares_left = shares_raw - shares * 100  # may be negative
                # cash_left += shares_left * price
                cash_used += shares * price
                goal_pos['size'] = shares
            
            goals.append(goal_pos)
        
        cash_left = turnover - cash_used
        return goals, cash_left
    
    def query_portfolio(self):
        positions = []
        for sec in self.ctx.pm.holding_securities:
            positions.append(self.ctx.pm.get_position(sec))
        return positions


class EventDrivenStrategy(Strategy):
    def __init__(self):
        
        super(EventDrivenStrategy, self).__init__()
    
    def on_bar(self, quote):
        pass
    
    def on_tick(self, quote):
        pass
    
    def on_cycle(self):
        pass
    
    def initialize(self):
        pass
    
    def buy_or_sell_with_bar(self, action, bar, size, slippage=0.0):
        """
        Send a limit Buy order with quote.close + slippage.
        
        Parameters
        ----------
        action : {'Buy', 'Sell'}
        bar : Bar
        size : int or float
            Should be positive.
        slippage : float, optional
            Should be non-negative

        """
        if not isinstance(bar, Bar):
            raise TypeError("quote must be Bar type. You may have passed a Quote.")
        
        if action == common.ORDER_ACTION.SELL:
            slippage *= -1
        entrust_price = bar.close + slippage
        task_id, msg = self.ctx.trade_api.place_order(bar.symbol,
                                                      action,
                                                      entrust_price,
                                                      size)
        if (task_id is None) or (task_id == 0):
            print("place_order FAILED! msg = {}".format(msg))
        
    def buy(self, bar, size=1, slippage=0.0):
        """
        Send a limit Buy order with bar.close + slippage.
        
        Parameters
        ----------
        bar : Bar
        size : int or float
        slippage : float

        """
        self.buy_or_sell_with_bar(common.ORDER_ACTION.BUY, bar, size, slippage)

    def sell(self, bar, size=1, slippage=0.0):
        """
        Send a limit Sell order with bar.close + slippage.
        
        Parameters
        ----------
        bar : Bar
        size : int or float
        slippage : float

        """
        self.buy_or_sell_with_bar(common.ORDER_ACTION.SELL, bar, size, slippage)

    def cancel_all_orders(self):
        for task_id, task in self.ctx.pm.tasks.items():
            if task.trade_date == self.ctx.trade_date:
                if not task.is_finished:
                    self.ctx.trade_api.cancel_order(task_id)

    def liquidate(self, quote, n, tick_size=1.0, pos=0):
        self.cancel_all_orders()
        if pos == 0:
            return
    
        ref_price = quote.close
        if pos < 0:
            action = common.ORDER_ACTION.BUY
            price = ref_price + n * tick_size
        else:
            action = common.ORDER_ACTION.SELL
            price = ref_price - n * tick_size
        self.ctx.trade_api.place_order(quote.symbol, action, price, abs(pos))
