# encoding: utf-8

import numpy as np
import pandas as pd
from jaqs.data.calendar import Calendar


class RegisteredFunction(object):
    def __init__(self, func, name="", options=None):
        self.func = func
        self.name = ""
        if not options:
            options = dict()
        self.options = options


class Context(object):
    """
    Used to store relevant context of the strategy.

    Attributes
    ----------
    data_api : trade.DataServer object
        Data provider for the strategy.
    gateway : gateway.Gateway object
        Broker of the strategy.
    universe : list of str
        Securities that the strategy cares about.
    calendar : trade.Calendar object
        A certain calendar that the strategy refers to.

    Methods
    -------
    add_universe(univ)
        Add new securities.

    """
    def __init__(self):
        # TODO: hard-code
        self.calendar = Calendar()
        
        self.pm = None
        
        self.data_api = None
        self.dataview = None
        
        self.gateway = None
        
        self.trade_api = None
        
        self.universe = []
        
        self.trade_date = 0

    def register_calendar(self, calendar):
        self.calendar = calendar

    def register_portfolio_manager(self, portfolio_manager):
        self.pm = portfolio_manager
        
    def register_data_api(self, data_api):
        self.data_api = data_api
    
    def register_trade_api(self, trade_api):
        self.trade_api = trade_api
        
    def register_gateway(self, gateway):
        self.gateway = gateway
    
    def register_dataview(self, dv):
        self.dataview = dv
        
    def add_universe(self, univ):
        """
        univ could be single symbol or securities separated by ,
        
        Parameters
        ----------
        univ : str or list
        
        """
        if isinstance(univ, list):
            self.universe = univ
        elif isinstance(univ, (str, unicode)):
            l = univ.split(',')
            l = [x for x in l if x]
            self.universe = l
        else:
            raise NotImplementedError("type of univ is {}".format(type(univ)))


class BaseModel(object):
    """
    A base class for all models. Support function register and context.
    
    Attributes
    ----------
    func_table : dict of {str: RegisteredFunction}
    context : Context
    
    """
    def __init__(self):
        self.context = None
        self.func_table = dict()
        self.active_funcs = []
    
    def register_func(self, name, func, options=None):
        rf = RegisteredFunction(func, name, options)
        self.func_table[name] = rf

    def activate_func(self, f_dict):
        """
        Activate key in self.func_table using kwargs in value.
        
        Parameters
        ----------
        f_dict : dict
            {str: dict}

        """
        self.active_funcs = []
        for f_name, options in f_dict.viewitems():
            self.func_table[f_name].options = options
            self.active_funcs.append(f_name)
    
    def register_context(self, context):
        self.context = context


class BaseRevenueModel(BaseModel):
    def __init__(self):
        super(BaseRevenueModel, self).__init__()
        pass
    
    def forecast_revenue(self, weights):
        pass


class FactorRevenueModel(BaseRevenueModel):
    """
    Forecast profit of target weight (portfolio), where:
    
    total_forecast = sum(individual_forecast)
    individual_forecast = combine(individual_forecast from different source)
    
    output: dict
    
    """
    def __init__(self):
        super(FactorRevenueModel, self).__init__()
        
        self.forecast_dic = None
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
        for factor, f in forecasts.viewitems():
            w = forecast_weights[factor]
            res += f * w
        return res
    
    def forecast_revenue(self, weights):
        """
        Forecast total revenue of the portfolio with weights.
        
        Parameters
        ----------
        weights : dict

        Returns
        -------
        res : float

        """
        total_revenue = 0.0
        for sec, w in weights.viewitems():
            forecasts = self.forecast_individual(sec)
            forecast = self.combine_sum(forecasts)
            
            total_revenue += w * forecast
        
        return total_revenue


class FactorRevenueModel_dv(FactorRevenueModel):
    """
    Forecast profit of target weight (portfolio), where:
    
    total_forecast = sum(individual_forecast)
    individual_forecast = combine(individual_forecast from different source)
    
    output: dict
    
    """
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
            forecasts[factor] = rf.func(context=self.context, user_options=rf.options)
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
        res = res.to_dict()
        return res
    
    def make_forecast(self):
        forecasts = self.get_forecasts()
        # TODO NaN
        forecasts = {key: value.fillna(0) for key, value in forecasts.items()}
        forecast = self.combine_sum(forecasts)
        self.forecast_dic = forecast
        
    def forecast_revenue(self, weights):
        """
        Forecast total revenue of the portfolio with weights.
        
        Parameters
        ----------
        weights : dict

        Returns
        -------
        res : float

        """
        total_revenue = 0.0
        self.make_forecast()
        
        weighted_revenue = {key: value * self.forecast_dic[key] for key, value in weights.viewitems()}
        total_revenue = np.sum(weighted_revenue.values())
        
        return total_revenue


class BaseCostModel(BaseModel):
    """Transaction Cost = commission + spread + execution"""
    def __init__(self):
        super(BaseCostModel, self).__init__()
        pass
    
    def calc_cost(self, symbol, size):
        pass


class SimpleCostModel(BaseRevenueModel):
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
                                               context=self.context, user_options=rf.options)
        cost_user = sum(cost_user_dic.values())
        
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
        for sec, w_last in weights_last.viewitems():
            w_now = weights_now[sec]
            
            trading_turnover = abs(w_now - w_last)
            total_cost += self.calc_individual_cost(sec, trading_turnover)
        
        return total_cost


class BaseRiskModel(BaseModel):
    def __init__(self):
        super(BaseRiskModel, self).__init__()
        pass


class FactorRiskModel(BaseRiskModel):
    def __init__(self):
        super(FactorRiskModel, self).__init__()
        
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
        weights_arr = np.asarray(weights.values(), dtype=float)
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
        for sec, w in weights.viewitems():
            res += w * self._get_idiosyncratic_risk(sec)
        return res


def test_models():
    weight_last = {'symbol1': 0.2, 'symbolB': 0.8}
    weight_now = {'symbol1': 0.3, 'symbolB': 0.7}
    
    portfolio = 1e7
    weight_last = {k: v * portfolio for k, v in weight_last.items()}
    weight_now = {k: v * portfolio for k, v in weight_now.items()}
    
    revenue = FactorRevenueModel().forecast_revenue(weight_now)
    cost = SimpleCostModel().calc_cost(weight_last, weight_now)
    # liquid = Liquidation().calc_liquid(weight_now)
    risk = FactorRiskModel().calc_risk(weight_now)
    
    risk_coef = 1.0
    cost_coef = 1.0
    util = revenue - risk_coef * risk - cost_coef * cost  # - liquid * liq_factor
    
    """
    obj.: h(x) = argmax(util)
    s.t.: g1(x) = abs(risk_factors[3]) < 100000.0
    g2(x) = var(risk_factors) <= 0.0
    """

    
if __name__ == "__main__":
    test_models()
