# encoding: utf-8

from __future__ import unicode_literals
try:
    basestring
except NameError:
    basestring = str
from collections import defaultdict

import numpy as np
import pandas as pd
import jaqs.util as jutil


class RegisteredFunction(object):
    def __init__(self, func, name="", options=None):
        self.func = func
        self.name = name
        if not options:
            options = dict()
        self.options = options


class Context(object):
    """
    Used to store relevant context of the strategy.

    Attributes
    ----------
    data_api : DataService
        Data provider for the strategy.
    dataview : DataView
    gateway : gateway.Gateway object
        Broker of the strategy.
    universe : list of str
        Securities that the strategy cares about.
    _calendar : Calendar
        A certain calendar that the strategy refers to.
    snapshot : pd.DataFrame
        Current snapshot of data.

    Methods
    -------
    init_universe(univ)
        Add new securities.

    """
    def __init__(self, data_api=None, trade_api=None, gateway=None,
                 dataview=None,
                 strategy=None, pm=None, instance=None):
        # TODO: should also support get calendar from dataview
        # self._calendar = None

        self.universe = []
        self._data_api = data_api
        self._trade_api = trade_api
        self._gateway = gateway
        
        self._dataview = dataview
        
        self.instance = instance

        self.strategy = strategy
        self.pm = pm
        
        self.trade_date = 0
        self.time = 0
        self.snapshot = None
        
        self.storage = dict()
        self.records = defaultdict(list)
        
        for member, obj in self.__dict__.items():
            if hasattr(obj, 'ctx'):
                if member in ['calendar', '_data_api', '_trade_api',
                              '_dataview', '_gateway', 'instance',
                              'pm', 'strategy']:
                    obj.ctx = self

    def save_store(self, path):
        jutil.save_pickle(self.storage, path)

    def load_store(self, path):
        s = jutil.load_pickle(path)
        if s is not None:
            self.storage = s
    
    def record(self, key, value):
        self.records[key].append((self.trade_date, self.time, value))
    
    def get_records(self):
        dic_df = dict()
        for key, list_of_entries in self.records.items():
            dic_df = pd.DataFrame(list_of_entries, columns=['trade_date', 'time', key])
        return dic_df
	
    '''
    @property
    def calendar(self):
        from jaqs.data import Calendar
        if self.data_api is not None:
            return self.data_api.calendar
        elif self._calendar is not None:
            return self._calendar
        else:
            self._calendar = Calendar()
            return self._calendar
        
    '''
    @property
    def data_api(self):
        return self._data_api

    @data_api.setter
    def data_api(self, value):
        if hasattr(value, 'register_context'):
            value.register_context(self)
        self._data_api = value

    @property
    def trade_api(self):
        return self._trade_api

    @trade_api.setter
    def trade_api(self, value):
        if hasattr(value, 'ctx'):
            value.ctx = self
        self._trade_api = value
    
    @property
    def gateway(self):
        return self._gateway

    @gateway.setter
    def gateway(self, value):
        if hasattr(value, 'register_context'):
            value.register_context(self)
        self._gateway = value

    @property
    def dataview(self):
        return self._dataview
    
    @dataview.setter
    def dataview(self, value):
        if hasattr(value, 'register_context'):
            value.register_context(self)
        self._dataview = value
        
    def init_universe(self, symbols):
        """
        univ could be single symbol or securities separated by ,
        
        Parameters
        ----------
        univ : str or list
        
        """
        if isinstance(symbols, list):
            self.universe = symbols
        elif isinstance(symbols, basestring):
            l = symbols.split(',')
            l = [x for x in l if x]
            self.universe = l
        else:
            raise NotImplementedError("type of univ is {}".format(type(symbols)))


class AlphaContext(Context):
    """
    Attributes
    ----------
    snapshot_sub : pd.DataFrame
        Current snapshot of the universe available to be traded.
        
    """
    def __init__(self, data_api=None, trade_api=None, gateway=None,
                 dataview=None,
                 strategy=None, pm=None, instance=None):
        super(AlphaContext, self).__init__(data_api=data_api, trade_api=trade_api, gateway=gateway,
                                           dataview=dataview,
                                           strategy=strategy, pm=pm, instance=instance)
        self.snapshot_sub = None
    
    
class FuncRegisterable(object):
    """
    A base class for all models. Support function register and context.
    
    Attributes
    ----------
    func_table : dict of {str: RegisteredFunction}
    context : Context
    
    """
    def __init__(self, context=None):
        super(FuncRegisterable, self).__init__()
        self.ctx = context
        self.func_table = dict()
        self.active_funcs = []
    
    def _register_func(self, name, func, options=None):
        if options is None:
            options = dict()
        
        rf = RegisteredFunction(func, name, options)
        self.func_table[name] = rf
        
        self.active_funcs.append(name)
    
    def register_context(self, context):
        self.ctx = context
    '''
    def activate_func(self, f_dict):
        """
        Activate key in self.func_table using kwargs in value.
        
        Parameters
        ----------
        f_dict : dict
            {str: dict}

        """
        self.active_funcs = []
        for f_name, options in f_dict.items():
            self.func_table[f_name].options = options
            self.active_funcs.append(f_name)
    
    '''


class StockSelector(FuncRegisterable):
    def __init__(self, context=None):
        super(StockSelector, self).__init__(context=context)
        pass

    def add_filter(self, name, func, options=None):
        self._register_func(name, func, options)

    def get_selection(self):
        """
        Return a list of stocks that are not selected.
        
        Returns
        -------
        selected : list

        """
        mask_selected = dict()
        for factor in self.active_funcs:
            rf = self.func_table[factor]
            res = rf.func(context=self.ctx, user_options=rf.options)
            res = convert_to_df(res)
            mask_selected[factor] = res
        
        merge = pd.concat(mask_selected.values(), axis=1).astype(float).astype(bool).fillna(False)
        symbol_arr = merge.index.values
        mask_arr = np.all(merge.values, axis=1)
        selected = symbol_arr[mask_arr].tolist()
        return selected


class BaseSignalModel(FuncRegisterable):
    def __init__(self, context=None):
        super(BaseSignalModel, self).__init__(context=context)
        pass

    def add_signal(self, name, func, options=None):
        self._register_func(name, func, options)

    def forecast_signal(self, weights):
        pass


'''
class FactorSignalModel(BaseSignalModel):
    """
    Forecast profit of target weight (portfolio), where:
    
    total_forecast = sum(individual_forecast)
    individual_forecast = combine(individual_forecast from different source)
    
    output: dict
    
    """
    def __init__(self):
        super(FactorSignalModel, self).__init__()
        
        self.total_forecast = None
    
    @staticmethod
    def order2z(order_arr):
        """Transform an array with order to z_score"""
        mean, std = np.mean(order_arr), np.std(order_arr)
        res = (np.asarray(order_arr, dtype=float) - mean) / std
        return res
    
    def forecast_individual(self, symbol):
        forecasts = dict()
        for factor in self.active_funcs:
            rf = self.func_table[factor]
            forecasts[factor] = rf.func(symbol, context=self.context, user_options=rf.options)
        return forecasts
    
    def combine_using_corr(self, forecasts):
        """
        Combine forecasts into one single forecast.
        
        Parameters
        ----------
        forecasts : dict

        Returns
        -------
        res : float

        """
        n = len(forecasts)
        forecast_corr = np.random.randn(n, n)
        forecasts_arr = np.asarray(forecasts.values(), dtype=float).reshape(-1, 1)
        return np.dot(forecast_corr, forecasts_arr).sum()
    
    def combine_sum(self, forecasts):
        res = np.sum(forecasts.values())
        return res
    
    def combine_custom_weight(self, forecasts, forecast_weights):
        res = 0.0
        for factor, f in forecasts.items():
            w = forecast_weights[factor]
            res += f * w
        return res
    
    def forecast_signal(self, weights):
        """
        Forecast total signal of the portfolio with weights.
        
        Parameters
        ----------
        weights : dict

        Returns
        -------
        res : float

        """
        total_signal = 0.0
        for sec, w in weights.items():
            forecasts = self.forecast_individual(sec)
            forecast = self.combine_sum(forecasts)
            
            total_signal += w * forecast
        
        return total_signal

'''


class FactorSignalModel(BaseSignalModel):
    """
    Forecast profit of target weight (portfolio), where:
    
    total_forecast = sum(individual_forecast)
    individual_forecast = combine(individual_forecast from different source)
    
    output: dict
    
    """
    def __init__(self, context=None):
        super(FactorSignalModel, self).__init__(context=context)
    
        self.total_forecast = None

    @staticmethod
    def order2z(order_arr):
        """Transform an array with order to z_score"""
        mean, std = np.mean(order_arr), np.std(order_arr)
        res = (np.asarray(order_arr, dtype=float) - mean) / std
        return res

    def combine_using_corr(self, forecasts):
        """
        Combine forecasts into one single forecast.
        
        Parameters
        ----------
        forecasts : dict

        Returns
        -------
        res : float

        """
        n = len(forecasts)
        forecast_corr = np.random.randn(n, n)
        forecasts_arr = np.asarray(list(forecasts.values()), dtype=float).reshape(-1, 1)
        return np.dot(forecast_corr, forecasts_arr).sum()

    def get_forecasts(self):
        """
        
        Returns
        -------
        forecasts : dict
            {str: pd.DataFrame}
            DataFrame index is symbol, column is field.

        """
        forecasts = dict()
        for factor in self.active_funcs:
            rf = self.func_table[factor]
            res = rf.func(context=self.ctx, user_options=rf.options)
            res = convert_to_df(res)
            forecasts[factor] = res
        return forecasts
    
    def combine_sum(self, forecasts):
        """
        
        Parameters
        ----------
        forecasts : dict
            {str: pd.DataFrame}
            DataFrame index is symbol, column is field.

        Returns
        -------
        res : dict

        """
        merge = pd.concat(forecasts.values(), axis=1)
        res = merge.sum(axis=1)
        res = res.to_dict()  # convert Series to {label -> value} dict
        return res
    
    def make_forecast(self):
        """
        Get and combine signals (forecasts). Return the combined signal.
        
        Returns
        -------
        forecast : dict of {str: float}
            {symbol_name: forecast_value}
        
        """
        forecasts = self.get_forecasts()  # {str: pd.DataFrame}
        # TODO NaN
        forecasts = {key: value.fillna(0) for key, value in forecasts.items()}
        forecast = self.combine_sum(forecasts)
        return forecast
        
    def forecast_signal(self, weights):
        """
        Forecast total signal of the portfolio with weights.
        
        Parameters
        ----------
        weights : dict

        Returns
        -------
        res : float

        """
        total_signal = 0.0
        forecast_dic = self.make_forecast()
        
        weighted_signal = {key: value * forecast_dic[key] for key, value in weights.items()}
        total_signal = np.sum(list(weighted_signal.values()))
        
        return total_signal


class FactorSignalModel_custom(FactorSignalModel):
    """
    Custom weight.

    """

    def __init__(self, context=None, signal_weights=None):
        super(FactorSignalModel_custom, self).__init__(context=context)

        self.signal_weights = signal_weights

    def combine_custom_weight(self, forecasts):
        td = self.ctx.trade_date

        res = 0.0
        for factor_name, factor_value in forecasts.items():
            weights_dic = self.signal_weights.loc[td, :].to_dict()

            weight = weights_dic[factor_name]
            res += factor_value * weight
        return res

    def make_forecast(self):
        """
        Get and combine signals (forecasts). Return the combined signal.

        Returns
        -------
        forecast : dict of {str: float}
            {symbol_name: forecast_value}

        """
        forecasts = self.get_forecasts()  # {str: pd.DataFrame}
        # TODO NaN
        forecasts = {key: value.fillna(0) for key, value in forecasts.items()}
        forecast = self.combine_custom_weight(forecasts)
        return forecast


class BaseCostModel(FuncRegisterable):
    """Transaction Cost = commission + spread + execution"""
    def __init__(self, context=None):
        super(BaseCostModel, self).__init__(context=context)
        pass
    
    def consider_cost(self, name, func, options=None):
        self._register_func(name, func, options=options)
    
    def add_cost(self, name, func, options=None):
        self._register_func(name, func, options=options)


class SimpleCostModel(BaseCostModel):
    def __init__(self, context=None):
        super(SimpleCostModel, self).__init__(context=context)
    
    def calc_individual_cost(self, symbol, turnover):
        # following data are fetched from the data server
        avg_bid_ask_spread = 1.0
        avg_daily_volume = 1e7
        yearly_volatility = 0.2
        price = 1e3
        trading_volume = turnover / price
        
        limit_reached = False
        if limit_reached:
            commission_rate = 1e9
        else:
            commission_rate = 1e-4
        
        cost_default = self._calc_individual_cost(trading_volume, commission_rate, price,
                                                  avg_bid_ask_spread, avg_daily_volume, yearly_volatility)

        cost_user_dic = dict()
        for cost_name in self.active_funcs:
            rf = self.func_table[cost_name]
            cost_user_dic[cost_name] = rf.func(symbol, trading_volume * price,
                                               context=self.ctx, user_options=rf.options)
        cost_user = sum(list(cost_user_dic.values()))
        
        return cost_default + cost_user
    
    @staticmethod
    def _calc_individual_cost(size, price, rate, avg_ba, adv, vol):
        """
        This function serves as a mathematical formula.
        
        Parameters
        ----------
        size : float
            Trading volume.
        rate : float
            Rate of commission.
        price : float
            Current price of the symbol.
        avg_ba : float
            Average bid-ask spread.
        adv : float
            Average daily trading volume.
        vol : float
            Volatility.

        Returns
        -------
        res : float

        """
        commission = rate * size * price
        
        spread_cost = avg_ba * 0.5
        
        kappa = 1e-6
        gamma = 1.5
        execution_cost = kappa * np.power(size, gamma)
        
        cost = commission + spread_cost + execution_cost  # 50bp
        
        return cost
    
    def calc_cost(self, weights_last, weights_now):
        """
        Calculate transaction cost from current position to target position.
        
        Parameters
        ----------
        weights_last : dict
        weights_now : dict
            Target positions.

        Returns
        -------
        total_cost : float

        """
        total_cost = 0.0
        for sec, w_last in weights_last.items():
            w_now = weights_now[sec]
            
            trading_turnover = abs(w_now - w_last)
            total_cost += self.calc_individual_cost(sec, trading_turnover)
        
        return total_cost


class BaseRiskModel(FuncRegisterable):
    def __init__(self, context=None):
        super(BaseRiskModel, self).__init__(context=context)
        pass
    
    def consider_risk(self, name, func, options=None):
        self._register_func(name, func, options=options)


class FactorRiskModel(BaseRiskModel):
    def __init__(self, context=None):
        super(FactorRiskModel, self).__init__(context=context)
        
        self.benchmark = ""
    
    def set_benchmark(self, benchmark):
        self.benchmark = benchmark
    
    def calc_risk(self, weights):
        """
        Calculate total risk of portfolio.
        
        Parameters
        ----------
        weights : dict
            {int: str}

        Returns
        -------
        total_risk : float

        """
        idiosyncratic_risk = self.calc_idiosyncratic_risk(weights)
        n = len(weights)
        corr_mat = np.random.randn(n, n)
        weights_arr = np.asarray(list(weights.values()), dtype=float)
        factor_risk = weights_arr.dot(corr_mat).dot(weights_arr)
        total_risk = factor_risk + idiosyncratic_risk
        return total_risk
    
    """
    def calc_factor_risk(weight):
        risk_factors = [10000.0, -20000.0]
        risk_factor = 100000.0  # beta risk
        return risk_factors, risk_factor
    """
    
    def _get_idiosyncratic_risk(self, sec):
        return 0.0
    
    def calc_idiosyncratic_risk(self, weights):
        """Calculate weighted sum of idiosyncratic risks of all securities."""
        res = 0.0
        for sec, w in weights.items():
            res += w * self._get_idiosyncratic_risk(sec)
        return res


def test_models():
    weight_last = {'symbol1': 0.2, 'symbolB': 0.8}
    weight_now = {'symbol1': 0.3, 'symbolB': 0.7}
    
    portfolio = 1e7
    weight_last = {k: v * portfolio for k, v in weight_last.items()}
    weight_now = {k: v * portfolio for k, v in weight_now.items()}
    
    signal = FactorSignalModel().forecast_signal(weight_now)
    cost = SimpleCostModel().calc_cost(weight_last, weight_now)
    # liquid = Liquidation().calc_liquid(weight_now)
    risk = FactorRiskModel().calc_risk(weight_now)
    
    risk_coef = 1.0
    cost_coef = 1.0
    util = signal - risk_coef * risk - cost_coef * cost  # - liquid * liq_factor
    
    """
    obj.: h(x) = argmax(util)
    s.t.: g1(x) = abs(risk_factors[3]) < 100000.0
    g2(x) = var(risk_factors) <= 0.0
    """


def convert_to_df(res):
    if isinstance(res, pd.DataFrame):
        pass
    elif isinstance(res, pd.Series):
        res = pd.DataFrame(index=res.index, data=res.values)
    else:
        raise ValueError("Return type of signal function must be DataFrame or Series or dict!"
                         + "\nWe got [{}] instead.".format(type(res)))
    '''
    elif isinstance(res, dict):
        res = pd.DataFrame(columns=[factor], data=pd.Series(data=res))
    '''
    return res

    
if __name__ == "__main__":
    test_models()
