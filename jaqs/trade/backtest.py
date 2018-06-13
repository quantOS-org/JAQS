# encoding: utf-8
"""
Classes defined in backtest module are responsible to run backtests.

They follow a fix procedure, from loading data to looping through
data and finally save backtest results.

"""

from __future__ import print_function, unicode_literals
import six
import abc
from collections import defaultdict
import numpy as np
import pandas as pd
import datetime as dt

from jaqs.trade import common
from jaqs.data.basic import Bar
from jaqs.data.basic import Trade
import jaqs.util as jutil
from functools import reduce


def generate_cash_trade_ind(symbol, amount, date, time=200000):
    trade_ind = Trade()
    trade_ind.symbol = symbol
    trade_ind.task_id = 0
    trade_ind.entrust_no = "0"
    trade_ind.set_fill_info(price=0.0, size=abs(amount), date=date, time=time, no="0", trade_date=date)

    trade_ind2 = Trade()
    trade_ind2.symbol = symbol
    trade_ind2.task_id = 0
    trade_ind2.entrust_no = "0"
    trade_ind2.set_fill_info(price=1.0, size=abs(amount), date=date, time=time, no="0",trade_date=date)

    if amount > 0:
        trade_ind.entrust_action = common.ORDER_ACTION.BUY
        trade_ind2.entrust_action = common.ORDER_ACTION.SELL
    else:
        trade_ind.entrust_action = common.ORDER_ACTION.SELL
        trade_ind2.entrust_action = common.ORDER_ACTION.BUY
    return trade_ind, trade_ind2
    

class BacktestInstance(six.with_metaclass(abc.ABCMeta)):
    """
    BacktestInstance is an abstract base class. It can be derived to implement
    various backtest tasks.
    
    Attributes
    ----------
    
    start_date : int
        %YY%mm%dd, start date of the backtest.
    end_date : int
        %YY%mm%dd, end date of the backtest.
    ctx : Context
        Running context of the backtest.
    props : dict
        props store configurations (settings) of the backtest. Eg: start_date.
    
    """
    def __init__(self):
        super(BacktestInstance, self).__init__()
        
        self.start_date = 0
        self.end_date = 0

        self.props = None
        
        self.ctx = None

        self.commission_rate = 20E-4

        self.POSITION_ADJUST_NO = 101010
        self.POSITION_ADJUST_TIME = 200000
        self.DELIST_ADJUST_NO = 202020
        self.DELIST_ADJUST_TIME = 150000

    def init_from_config(self, props):
        """
        Initialize parameters values for all backtest components such as
        DataService, PortfolioManager, Strategy, etc.
        
        Parameters
        ----------
        props : dict

        """
        for name in ['start_date', 'end_date']:
            if name not in props:
                raise ValueError("{} must be provided in props.".format(name))
        
        self.props = props
        self.start_date = props.get("start_date")
        self.end_date = props.get("end_date")

        self.commission_rate = props.get('commission_rate', 20E-4)
        
        if 'symbol' in props:
            self.ctx.init_universe(props['symbol'])
        elif hasattr(self.ctx, 'dataview'):
            self.ctx.init_universe(self.ctx.dataview.symbol)
        else:
            raise ValueError("No dataview, no symbol either.")

        if 'init_balance' not in props:
            raise ValueError("No [init_balance] provided. Please specify it in props.")

        for obj in ['data_api', 'trade_api', 'pm', 'strategy']:
            obj = getattr(self.ctx, obj)
            if obj is not None:
                obj.init_from_config(props)



'''
class AlphaBacktestInstance_OLD_dataservice(BacktestInstance):
    def __init__(self):
        BacktestInstance.__init__(self)
        
        self.last_rebalance_date = 0
        self.current_rebalance_date = 0
        self.trade_days = None
    
    def _is_trade_date(self, start, end, date, data_server):
        if self.trade_days is None:
            df, msg = data_server.daily('000300.SH', start, end, fields="close")
            self.trade_days = df.loc[:, 'trade_date'].values
        return date in self.trade_days
    
    def go_next_rebalance_day(self):
        """update self.ctx.trade_date and last_date."""
        if self.ctx.gateway.match_finished:
            next_period_day = jutil.get_next_period_day(self.ctx.trade_date,
                                                         self.ctx.strategy.period, self.ctx.strategy.days_delay)
            # update current_date: next_period_day is a workday, but not necessarily a trade date
            if self.ctx.calendar.is_trade_date(next_period_day):
                self.ctx.trade_date = next_period_day
            else:
                self.ctx.trade_date = self.ctx.calendar.get_next_trade_date(next_period_day)
            self.ctx.trade_date = self.ctx.calendar.get_next_trade_date(next_period_day)

            # update re-balance date
            if self.current_rebalance_date > 0:
                self.last_rebalance_date = self.current_rebalance_date
            else:
                self.last_rebalance_date = self.start_date
            self.current_rebalance_date = self.ctx.trade_date
        else:
            # TODO here we must make sure the matching will not last to next period
            self.ctx.trade_date = self.ctx.calendar.get_next_trade_date(self.ctx.trade_date)

        self.last_date = self.ctx.calendar.get_last_trade_date(self.ctx.trade_date)

    def run_alpha(self):
        gateway = self.ctx.gateway
        
        self.ctx.trade_date = self.start_date
        while True:
            self.go_next_rebalance_day()
            if self.ctx.trade_date > self.end_date:
                break
            
            if gateway.match_finished:
                self.on_new_day(self.last_date)
                df_dic = self.ctx.strategy.get_univ_prices()  # access data
                self.ctx.strategy.re_balance_plan_before_open(df_dic, suspensions=[])
                
                self.on_new_day(self.ctx.trade_date)
                self.ctx.strategy.send_bullets()
            else:
                self.on_new_day(self.ctx.trade_date)
            
            df_dic = self.ctx.strategy.get_univ_prices()  # access data
            trade_indications = gateway.match(df_dic, self.ctx.trade_date)
            for trade_ind in trade_indications:
                self.ctx.strategy.on_trade(trade_ind)
        
        print "Backtest done. {:d} days, {:.2e} trades in total.".format(len(self.trade_days),
                                                                         len(self.ctx.strategy.pm.trades))
    
    def on_new_day(self, date):
        self.ctx.trade_date = date
        self.ctx.strategy.on_new_day(date)
        self.ctx.gateway.on_new_day(date)
    
    def save_results(self, folder='../output/'):
        import pandas as pd
        
        trades = self.ctx.strategy.pm.trades
        
        type_map = {'task_id': str,
                    'entrust_no': str,
                    'entrust_action': str,
                    'symbol': str,
                    'fill_price': float,
                    'fill_size': int,
                    'fill_date': np.integer,
                    'fill_time': np.integer,
                    'fill_no': str}
        # keys = trades[0].__dict__.keys()
        ser_list = dict()
        for key in type_map.keys():
            v = [t.__getattribute__(key) for t in trades]
            ser = pd.Series(data=v, index=None, dtype=type_map[key], name=key)
            ser_list[key] = ser
        df_trades = pd.DataFrame(ser_list)
        df_trades.index.name = 'index'
        
        from os.path import join
        trades_fn = join(folder, 'trades.csv')
        configs_fn = join(folder, 'configs.json')
        fileio.create_dir(trades_fn)
        
        df_trades.to_csv(trades_fn)
        fileio.save_json(self.props, configs_fn)

        print ("Backtest results has been successfully saved to:\n" + folder)


'''


class AlphaBacktestInstance(BacktestInstance):
    """
    Backtest alpha strategy using DataView.
    
    Attributes
    ----------
    last_date : int
        Last trade date before current trade date.
    last_rebalance_date : int
        Last re-balance date that we do re-balance.
    current_rebalance_date : int
        Current re-balance date that we do re-balance.
    univ_price_dic : dict
        Prices of symbols on current trade date.
    commission_rate : float
        Ratio of commission charged to turnover for each trade.

    """
    def __init__(self):
        super(AlphaBacktestInstance, self).__init__()
    
        self.last_date = 0
        self.last_rebalance_date = 0
        self.current_rebalance_date = 0

        self.univ_price_dic = {}
        self.tmp_univ_price_dic_map = {}

    def init_from_config(self, props):
        super(AlphaBacktestInstance, self).init_from_config(props)
        strategy = self.ctx.strategy

        # universe = props.get('universe', "")
        # symbol = props.get('symbol', "")
        # if symbol and universe or len(universe.split('.')) > 1:
        #     if strategy.pc_method in['index_weight', 'equal_index_weight']:
        #         raise ValueError("{} shouldn't be used if there are both symbol and universe in props", strategy.pc_method)


    def position_adjust(self):
        """
        adjust happens after market close
        Before each re-balance day, adjust for all dividend and cash paid actions during the last period.
        We assume all cash will be re-invested.
        Since we adjust our position at next re-balance day, PnL before that may be incorrect.

        """
        start = self.last_rebalance_date  # start will be one day later
        end = self.current_rebalance_date  # end is the same to ensure position adjusted for dividend on rebalance day
        df_adj = self.ctx.dataview.get_ts('_daily_adjust_factor', start_date=start, end_date=end)

        # FIXME: the first day should have been balanced before?
        df_adj = df_adj[1:]

        pm = self.ctx.pm

        # Find symbols which has adj_factor not equaling 1
        tmp = df_adj[df_adj!=1].fillna(0.0).sum()
        adj_symbols = set(tmp[tmp!=0].index).intersection(pm.holding_securities)

        #for symbol in pm.holding_securities:

        for symbol in adj_symbols:
            ser = df_adj.loc[:, symbol]
            ser_adj = ser.dropna()
            for date, ratio in ser_adj.iteritems():
                pos_old = pm.get_position(symbol).current_size
                # TODO pos will become float, original: int
                pos_new = pos_old * ratio
                pos_diff = pos_new - pos_old  # must be positive
                if pos_diff <= 0:
                    # TODO this is possible
                    # raise ValueError("pos_diff <= 0")
                    continue
                
                trade_ind = Trade()
                trade_ind.symbol = symbol
                trade_ind.task_id = self.POSITION_ADJUST_NO
                trade_ind.entrust_no = self.POSITION_ADJUST_NO
                trade_ind.entrust_action = common.ORDER_ACTION.BUY  # for now only BUY
                trade_ind.set_fill_info(price=0.0, size=pos_diff,
                                        date=date, time=200000,
                                        no=self.POSITION_ADJUST_NO,
                                        trade_date=date)
                
                self.ctx.strategy.on_trade(trade_ind)

    def delist_adjust(self):
        df_inst = self.ctx.dataview.data_inst

        start = self.last_rebalance_date  # start will be one day later
        end = self.current_rebalance_date  # end is the same to ensure position adjusted for dividend on rebalance day
        
        mask = np.logical_and(df_inst['delist_date'] >= start, df_inst['delist_date'] <= end)
        dic_inst = df_inst.loc[mask, :].to_dict(orient='index')
        
        if not dic_inst:
            return
        pm = self.ctx.pm
        for symbol in pm.holding_securities.copy():
            value_dic = dic_inst.get(symbol, None)
            if value_dic is None:
                continue
            pos = pm.get_position(symbol).current_size
            last_trade_date = self._get_last_trade_date(value_dic['delist_date'])
            last_close_price = self.ctx.dataview.get_snapshot(last_trade_date, symbol=symbol, fields='close')
            last_close_price = last_close_price.at[symbol, 'close']
            
            trade_ind = Trade()
            trade_ind.symbol = symbol
            trade_ind.task_id = self.DELIST_ADJUST_NO
            trade_ind.entrust_no = self.DELIST_ADJUST_NO
            trade_ind.entrust_action = common.ORDER_ACTION.SELL  # for now only BUY
            trade_ind.set_fill_info(price=last_close_price, size=pos,
                                    date=last_trade_date, time=150000,
                                    no=self.DELIST_ADJUST_NO,
                                    trade_date=last_trade_date)

            self.ctx.strategy.cash += trade_ind.fill_price * trade_ind.fill_size
            #self.ctx.pm.cash += trade_ind.fill_price * trade_ind.fill_size
            self.ctx.strategy.on_trade(trade_ind)

    def re_balance_plan_before_open(self):
        """
        Do portfolio re-balance before market open (not knowing suspensions) only calculate weights.
        For now, we stick to the same close price when calculate market value and do re-balance.
        
        Parameters
        ----------

        """
        # Step.1 set weights of those non-index-members to zero
        # only filter index members when universe is defined
        universe_list = self.ctx.universe
        if self.ctx.dataview.universe:
            df_is_member = self.ctx.dataview.get_snapshot(self.ctx.trade_date, fields='index_member')
            df_is_member = df_is_member.fillna(0).astype(bool)
            universe_list = df_is_member[df_is_member['index_member']].index.values

        # Step.2 filter out those not listed or already de-listed
        df_inst = self.ctx.dataview.data_inst
        mask = np.logical_and(self.ctx.trade_date > df_inst['list_date'],
                              self.ctx.trade_date < df_inst['delist_date'])
        listing_symbols = df_inst.loc[mask, :].index.values
        universe_list = np.intersect1d(universe_list, listing_symbols)
        
        # step.3 construct portfolio using models
        self.ctx.strategy.portfolio_construction(universe_list)
        
    def re_balance_plan_after_open(self):
        """
        Do portfolio re-balance after market open.
        With suspensions known, we re-calculate weights and generate orders.
        
        Notes
        -----
        Price here must not be adjusted.

        """
        prices = {k: v['vwap'] for k, v in self.univ_price_dic.items()}

        # suspensions & limit_reaches: list of str
        suspensions = self.get_suspensions()
        limit_reaches = self.get_limit_reaches()
        all_list = reduce(lambda s1, s2: s1.union(s2), [set(suspensions), set(limit_reaches)])

        # step1. weights of those suspended and limit will be remove, and weights of others will be re-normalized
        self.ctx.strategy.re_weight_suspension(all_list)
    
        # step2. calculate market value and cash
        # market value does not include those suspended
        market_value_float, market_value_frozen = self.ctx.pm.market_value(prices, all_list)
        #cash_available = self.ctx.pm.cash + market_value_float
        cash_available = self.ctx.strategy.cash + market_value_float
    
        cash_to_use = cash_available * self.ctx.strategy.position_ratio
        cash_unuse = cash_available - cash_to_use
    
        # step3. generate target positions
        # position of those suspended will remain the same (will not be traded)
        goals, cash_remain = self.ctx.strategy.generate_weights_order(self.ctx.strategy.weights, cash_to_use, prices,
                                                                      suspensions=all_list)
        self.ctx.strategy.goal_positions = goals
        
        #self.ctx.pm.cash = cash_remain + cash_unuse
        self.ctx.strategy.cash = cash_remain + cash_unuse
        #print("cash diff: ", self.ctx.pm.cash - self.ctx.strategy.cash)
        # self.liquidate_all()
        
        total = cash_available + market_value_frozen
        self.ctx.strategy.on_after_rebalance(total)
        self.ctx.record('total_cash', total)

    def run_alpha(self):
        print("Run alpha backtest from {0} to {1}".format(self.start_date, self.end_date))
        begin_time = dt.datetime.now()

        tapi = self.ctx.trade_api

        # Keep compatible, the original test starts with next day, not start_date
        if self.ctx.strategy.period in ['week', 'month']:
            self.ctx.trade_date = self._get_first_period_day()
        else:
            self.ctx.trade_date = self._get_next_trade_date(self.start_date)

        self.last_date = self._get_last_trade_date(self.ctx.trade_date)
        self.current_rebalance_date = self.ctx.trade_date
        while True:
            print("=======new day {}".format(self.ctx.trade_date))

            # match uncome orders or re-balance
            if tapi.match_finished:
                # Step1.
                # position adjust according to dividend, cash paid, de-list actions during the last period
                # two adjust must in order
                self.position_adjust()
                self.delist_adjust()

                # Step2.
                # plan re-balance before market open of the re-balance day:

                # use last trade date because strategy can only access data of last day
                self.on_new_day(self.last_date)

                # get index memebers, get signals, generate weights
                self.re_balance_plan_before_open()

                # Step3.
                # do re-balance on the re-balance day
                self.on_new_day(self.ctx.trade_date)

                # get suspensions, get up/down limits, generate goal positions and send orders.
                self.re_balance_plan_after_open()

                self.ctx.strategy.send_bullets()

            else:
                self.on_new_day(self.ctx.trade_date)

            # Deal with trade indications
            # results = gateway.match(self.univ_price_dic)
            results = tapi.match_and_callback(self.univ_price_dic)
            for trade_ind, order_status_ind in results:
                self.ctx.strategy.cash -= trade_ind.commission
                #self.ctx.pm.cash -= trade_ind.commission
                
            self.on_after_market_close()

            # switch trade date
            backtest_finish = self.go_next_rebalance_day()
            if backtest_finish:
                break

        used_time = (dt.datetime.now() - begin_time).total_seconds()
        print("Backtest done. {0:d} days, {1:.2e} trades in total. used time: {2}s".
              format(len(self.ctx.dataview.dates), len(self.ctx.pm.trades), used_time))

        jutil.prof_print()
    
    def on_after_market_close(self):
        self.ctx.trade_api.on_after_market_close()
        
    '''
    def get_univ_prices(self, field_name='close'):
        dv = self.ctx.dataview
        df = dv.get_snapshot(self.ctx.trade_date, fields=field_name)
        res = df.to_dict(orient='index')
        return res
    
    '''
    def _is_trade_date(self, date):
        if self.ctx.dataview is not None:
            return date in self.ctx.dataview.dates
        else:
            return self.ctx.data_api.is_trade_date(date)
    
    def _get_next_trade_date(self, date, n=1):
        if self.ctx.dataview is not None:
            dates = self.ctx.dataview.dates
            mask = dates > date
            return dates[mask][n-1]
        else:
            return self.ctx.data_api.query_next_trade_date(date, n)
    
    def _get_last_trade_date(self, date):
        if self.ctx.dataview is not None:
            dates = self.ctx.dataview.dates
            mask = dates < date
            return dates[mask][-1]
        else:
            return self.ctx.data_api.query_last_trade_date(date)

    def _get_first_period_day(self):

        current = self.start_date

        current_date = jutil.convert_int_to_datetime(current)
        period = self.ctx.strategy.period

        # set current date to first date of current period
        # set offset to first date of previous period
        if period == 'day':
            offset = pd.tseries.offsets.BDay()
        elif period == 'week':
            offset = pd.tseries.offsets.Week(weekday=0)
            current_date -= pd.tseries.offsets.Day(current_date.weekday())
        elif period == 'month':
            offset = pd.tseries.offsets.BMonthBegin()
            current_date -= pd.tseries.offsets.Day(current_date.day - 1)
        else:
            raise NotImplementedError("Frequency as {} not support".format(period))

        current_date -= offset * self.ctx.strategy.n_periods;

        current = jutil.convert_datetime_to_int(current_date)

        return self._get_next_period_day(current, self.ctx.strategy.period,
                                         n=self.ctx.strategy.n_periods,
                                         extra_offset=self.ctx.strategy.days_delay)

    def _get_next_period_day(self, current, period, n=1, extra_offset=0):
        """
        Get the n'th day in next period from current day.

        Parameters
        ----------
        current : int
            Current date in format "%Y%m%d".
        period : str
            Interval between current and next. {'day', 'week', 'month'}
        n : int
            n times period.
        extra_offset : int
            n'th business day after next period.

        Returns
        -------
        nxt : int

        """
        #while True:
        #for _ in range(4):

        current_date = jutil.convert_int_to_datetime(current)
        while current <= self.end_date:
            if period == 'day':
                offset = pd.tseries.offsets.BDay()  # move to next business day
                if extra_offset < 0:
                    raise ValueError("Wrong offset for day period")
            elif period == 'week':
                offset = pd.tseries.offsets.Week(weekday=0)  # move to next Monday
                if extra_offset < -5 :
                    raise ValueError("Wrong offset for week period")
            elif period == 'month':
                offset = pd.tseries.offsets.BMonthBegin()  # move to first business day of next month
                if extra_offset < -31:
                    raise  ValueError("Wrong offset for month period")
            else:
                raise NotImplementedError("Frequency as {} not support".format(period))

            offset = offset * n

            begin_date = current_date + offset
            if period == 'day':
                end_date = begin_date + pd.tseries.offsets.Day()*366
            elif period == 'week':
                end_date   = begin_date + pd.tseries.offsets.Day() * 6
            elif period == 'month':
                end_date = begin_date + pd.tseries.offsets.BMonthBegin() #- pd.tseries.offsets.BDay()


            if extra_offset > 0 :
                next_date = begin_date + extra_offset * pd.tseries.offsets.BDay()
                if next_date >= end_date:
                    next_date = end_date - pd.tseries.offsets.BDay()
            elif extra_offset < 0:
                next_date = end_date + extra_offset * pd.tseries.offsets.BDay()
                if next_date < begin_date:
                    next_date = begin_date
            else:
                next_date = begin_date

            date = next_date
            while date < end_date:
                nxt = jutil.convert_datetime_to_int(date)
                if self._is_trade_date(nxt):
                    return nxt
                date += pd.tseries.offsets.BDay()

            date = next_date
            while date >= begin_date:
                nxt = jutil.convert_datetime_to_int(date)
                if self._is_trade_date(nxt):
                    return nxt
                date -= pd.tseries.offsets.BDay()

            # no trading day in this period, try next period
            current_date = end_date - pd.tseries.offsets.BDay()
            current = jutil.convert_datetime_to_int( current_date)
            # Check if there are more trade dates
            self._get_next_trade_date(current)

        raise ValueError("no trading day after {0}".format(current))

    def go_next_rebalance_day(self):
        """
        update self.ctx.trade_date and last_date.
        
        Returns
        -------
        bool
            Whether the backtest is finished.

        """
        current_date = self.ctx.trade_date
        if self.ctx.trade_api.match_finished:
            if self.ctx.strategy.period == 'day':
                # use trade dates array
                try:
                    current_date = self._get_next_trade_date(current_date, self.ctx.strategy.n_periods)
                except IndexError:
                    return True
            else:
                # use natural week/month
                # next_period_day = jutil.get_next_period_day(current_date, self.ctx.strategy.period,
                #                                             n=self.ctx.strategy.n_periods,
                #                                             extra_offset=self.ctx.strategy.days_delay)
                # # update current_date: next_period_day is a workday, but not necessarily a trade date
                # if self._is_trade_date(next_period_day):
                #     current_date = next_period_day
                # else:
                #     try:
                #         current_date = self._get_next_trade_date(next_period_day)
                #     except IndexError:
                #         return True

                try:
                    current_date = self._get_next_period_day(current_date, self.ctx.strategy.period,
                                                            n=self.ctx.strategy.n_periods,
                                                            extra_offset=self.ctx.strategy.days_delay)
                except ValueError:
                    return True
                except IndexError:
                    return True

            if current_date > self.end_date:
                return True

            # update re-balance date
            if self.current_rebalance_date > 0:
                self.last_rebalance_date = self.current_rebalance_date
            else:
                self.last_rebalance_date = current_date
            self.current_rebalance_date = current_date
        else:
            # TODO here we must make sure the matching will not last to next period
            try:
                current_date = self._get_next_trade_date(current_date)
            except IndexError:
                return True
    
        self.ctx.trade_date = current_date
        self.last_date = self._get_last_trade_date(current_date)
        return False
    
    def get_suspensions(self):
        trade_status = self.ctx.dataview.get_snapshot(self.ctx.trade_date, fields='trade_status')
        trade_status = trade_status.loc[:, 'trade_status']
        # trade_status: {'N', 'XD', 'XR', 'DR', 'JiaoYi', 'TingPai', NUll (before 2003)}
        mask_sus = trade_status == '停牌'
        return list(trade_status.loc[mask_sus].index.values)

    def get_limit_reaches(self):
        # TODO: 10% is not the absolute value to check limit reach
        df = self.ctx.dataview.get_snapshot(self.ctx.trade_date, fields="_limit")
        df = df[df > 9.5E-2].dropna()
        return df.index.values
    
    def on_new_day(self, date):
        # self.ctx.strategy.on_new_day(date)
        self.ctx.trade_api.on_new_day(date)
        self.ctx.snapshot = self.ctx.dataview.get_snapshot(date)

        # temporary fix, tzxu
        # Can univ_price_dic has DataFrame format?
        if date in self.tmp_univ_price_dic_map:
            self.univ_price_dic = self.tmp_univ_price_dic_map[date]
        else:
            # self.univ_price_dic = self.ctx.snapshot.to_dict(orient='index')
            #self.univ_price_dic = self.ctx.snapshot.loc[:, ['close', 'vwap', 'open', 'high', 'low']].to_dict(orient='index')
            self.univ_price_dic = self.ctx.snapshot.to_dict(orient='index')
            self.tmp_univ_price_dic_map[date] = self.univ_price_dic

    def save_results(self, folder_path='.'):
        import os
        import pandas as pd
        folder_path = os.path.abspath(folder_path)
    
        trades = self.ctx.pm.trades
    
        type_map = {'task_id': str,
                    'entrust_no': str,
                    'entrust_action': str,
                    'symbol': str,
                    'fill_price': float,
                    'fill_size': float,
                    'fill_date': np.integer,
                    'fill_time': np.integer,
                    'fill_no': str,
                    'commission': float,
                    'trade_date': np.integer}
        # keys = trades[0].__dict__.keys()
        ser_list = dict()
        for key in type_map.keys():
            v = [t.__getattribute__(key) for t in trades]
            ser = pd.Series(data=v, index=None, dtype=type_map[key], name=key)
            ser_list[key] = ser
        df_trades = pd.DataFrame(ser_list)
        df_trades.index.name = 'index'
    
        trades_fn = os.path.join(folder_path, 'trades.csv')
        configs_fn = os.path.join(folder_path, 'configs.json')
        jutil.create_dir(trades_fn)
    
        df_trades.to_csv(trades_fn)
        jutil.save_json(self.props, configs_fn)
    
        print ("Backtest results has been successfully saved to:\n" + folder_path)
    
    def show_position_info(self):
        pm = self.ctx.pm
        
        prices = {k: v['open'] for k, v in self.univ_price_dic.items()}
        market_value_float, market_value_frozen = pm.market_value(prices)
        for symbol in pm.holding_securities:
            p = prices[symbol]
            size = pm.get_position(symbol).current_size
            print("{}  {:.2e}   {:.1f}@{:.2f}".format(symbol, p*size*100, p, size))
        print("float {:.2e}, frozen {:.2e}".format(market_value_float, market_value_frozen))


class EventBacktestInstance(BacktestInstance):
    """
    Backtest event-driven strategy using DataService.
    
    Attributes
    ----------
    bar_type : str
        {'1d', '1M', '5M', etc.}
    
    """
    def __init__(self):
        super(EventBacktestInstance, self).__init__()
        
        self.bar_type = ""
        self.df_dividend = None
        
    def init_from_config(self, props):
        super(EventBacktestInstance, self).init_from_config(props)
        
        self.bar_type = props.get("bar_type", "1d")
    
    def _get_dividend_info(self):
        """
        Query dividend information of stocks for use of daily settlement.
        
        """
        if self.ctx.data_api is not None:
            symbol_str = ','.join(self.ctx.universe)
            df, msg = self.ctx.data_api.query_dividend(symbol_str, start_date=self.start_date, end_date=self.end_date)
            df.loc[:, 'shares'] = (df['share_ratio'] + df['share_trans_ratio']) # / 10.0
            df.loc[:, 'cash_tax'] = df['cash_tax'] # / 10.0
            self.df_dividend = df
        else:
            # TODO
            pass
        
    def settle_for_stocks(self, last_date, date):
        if self.df_dividend is None:
            return
        
        df = self.df_dividend.loc[(self.df_dividend['exdiv_date'] > last_date) & (self.df_dividend['exdiv_date'] <= date)]
        if df.empty:
            return
        df2 = df.set_index('symbol')
        for symbol in df2.index:
            if symbol in self.ctx.pm.holding_securities:
                df_symbol = df2.loc[symbol]
                shares_ratio = df_symbol['shares']
                cash_ratio = df_symbol['cash_tax']
                pos = self.ctx.pm.get_position(symbol).current_size
                
                if cash_ratio > 0:
                    cash_added = cash_ratio * pos
                    #self.ctx.pm.cash += cash_added
                    trade_ind1, trade_ind2 = generate_cash_trade_ind(symbol, cash_added, date, 60000)
                    self.ctx.strategy.on_trade(trade_ind1)
                    self.ctx.strategy.on_trade(trade_ind2)
                    
                if shares_ratio > 0:
                    pos_diff = abs(pos * shares_ratio)
                    trade_ind = Trade()
                    trade_ind.symbol = symbol
                    trade_ind.task_id = self.POSITION_ADJUST_NO
                    trade_ind.entrust_no = self.POSITION_ADJUST_NO
                    if pos > 0:
                        trade_ind.entrust_action = common.ORDER_ACTION.BUY
                    else:
                        trade_ind.entrust_action = common.ORDER_ACTION.SELL
                    trade_ind.set_fill_info(price=0.0, size=pos_diff,
                                            date=date, time=60000,
                                            no=self.POSITION_ADJUST_NO,
                                            trade_date=date)
                    
                    self.ctx.strategy.on_trade(trade_ind)
            
    def on_new_day(self, date):
        self.ctx.trade_date = date
        self.ctx.time = 0
        if hasattr(self.ctx.trade_api, 'on_new_day'):
            self.ctx.trade_api.on_new_day(self.ctx.trade_date)
        if hasattr(self.ctx.trade_api, 'on_new_day'):
            self.ctx.trade_api.on_new_day(self.ctx.trade_date)
        self.ctx.strategy.initialize()
        print('on_new_day in trade {}'.format(self.ctx.trade_date))
    
    def on_after_market_close(self):
        pass
        
    def _get_df_bar(self, symbols, date):
        """
        Get bar DataFrame from DataApi or DataView.
        
        Parameters
        ----------
        symbols : str
        date : int

        Returns
        -------
        res : pd.DataFrame

        """
        if self.ctx.dataview is not None:
            df_quotes = self.ctx.dataview.get(symbol=symbols,
                                              start_date=date, end_date=date,
                                              fields='open,high,low,close,volume,oi,trade_date,date,time',
                                              data_format='long')
        elif self.ctx.data_api is not None:
            df_quotes, _ = self.ctx.data_api.bar(symbol=symbols,
                                                   start_time=200000, end_time=160000, trade_date=date,
                                                   freq=self.bar_type)
        else:
            raise ValueError()
        
        return df_quotes
            
    def _create_time_symbol_bars(self, date):
        """
        Given a trade date, query bars of all symbols on that day and return a nested dict.
        
        Parameters
        ----------
        date : int
            Trade date.

        Returns
        -------
        res : list of tuples
            Three-element tuple: (trade_date, time, dict of quote)

        """
        # query quotes data
        symbols_str = ','.join(self.ctx.universe)
        df_quotes = self._get_df_bar(symbols_str, date)
        if df_quotes is None or df_quotes.empty:
            return dict()
    
        # create nested dict
        df_quotes = df_quotes.sort_values(['date', 'time', 'symbol'])
        res = []
        for time, df in df_quotes.groupby(by=['time'], sort=False):
            quotes_list = Bar.create_from_df(df)
            dic = {quote.symbol: quote for quote in quotes_list}
            res.append((time, dic))
        return res
    
    def _run_bar(self):
        """Quotes of different symbols will be aligned into one dictionary."""
        trade_dates_arr = self.ctx.data_api.query_trade_dates(self.start_date, self.end_date)

        last_trade_date = trade_dates_arr[0]
        for trade_date in trade_dates_arr:
            self.settle_for_stocks(last_trade_date, trade_date)
            self.on_new_day(trade_date)
            
            list_of_quotes_tuples = self._create_time_symbol_bars(trade_date)
            for time, quotes_dic in list_of_quotes_tuples:
                self._process_quote_bar(quotes_dic)
            
            self.on_after_market_close()
            last_trade_date = trade_date

    def _get_df_daily(self, symbol, start_date, end_date):
        """
        Get bar DataFrame from DataApi or DataView.
        
        Parameters
        ----------
        symbol : str or unicode
        start_date : int
        end_date : int

        Returns
        -------
        res : pd.DataFrame

        """
        if self.ctx.dataview is not None:
            df_daily = self.ctx.dataview.get(symbol=symbol,
                                             start_date=start_date, end_date=end_date,
                                             fields='open,high,low,close,volume,oi,trade_date,time',
                                             data_format='long')
        elif self.ctx.data_api is not None:
            df_daily, _ = self.ctx.data_api.daily(symbol=symbol,
                                                  start_date=start_date, end_date=end_date,
                                                  adjust_mode=None)
        else:
            raise ValueError()
    
        return df_daily

    def _create_daily_symbol_bars(self, start_date, end_date):
        """
        Given a trade date range, query daily bars of all symbols in that range and return a list.
        
        Parameters
        ----------
        start_date : int
            Trade date.
        end_date : int
            Trade date.

        Returns
        -------
        res : list of tuples
            Two-element tuple: (trade_date, dict of quotes)

        """
        # query quotes data
        symbols_str = ','.join(self.ctx.universe)
        df_daily = self._get_df_daily(symbol=symbols_str, start_date=start_date, end_date=end_date)
        df_daily['date'] = df_daily['trade_date']
        if df_daily is None or df_daily.empty:
            return dict()

        # create nested dict
        df_daily = df_daily.sort_values(['trade_date', 'symbol'])
        res = []
        for date, df in df_daily.groupby(by='trade_date'):
            quotes_list = Bar.create_from_df(df)
            dic = {quote.symbol: quote for quote in quotes_list}
            res.append((date, dic))
    
        return res

    def _run_daily(self):
        """Quotes of different symbols will be aligned into one dictionary."""
        # create nested dict
        list_of_quotes_tuples = self._create_daily_symbol_bars(self.start_date, self.end_date)
        
        for i in range(len(list_of_quotes_tuples) - 1):
            date1, quotes_dic1 = list_of_quotes_tuples[i]
            date2, quotes_dic2 = list_of_quotes_tuples[i + 1]
            self.on_new_day(date2)
            
            self._process_quote_daily(quotes_dic1, quotes_dic2)
            
            self.on_after_market_close()
            self.settle_for_stocks(date1, date2)
    
    def _process_quote_daily(self, quote_yesterday, quote_today):
        # on_bar
        self.ctx.strategy.on_bar(quote_yesterday)
        
        self.ctx.trade_api.match_and_callback(quote_today, freq=self.bar_type)
  
        self.on_after_market_close()

    def _process_quote_bar(self, quotes_dic):
        results = self.ctx.trade_api.match_and_callback(quotes_dic, freq=self.bar_type)
    
        # on_bar
        self.ctx.strategy.on_bar(quotes_dic)

    def run(self):
        self._get_dividend_info()
        
        if self.bar_type == common.QUOTE_TYPE.DAILY:
            self._run_daily()
        
        elif (self.bar_type == common.QUOTE_TYPE.MIN
              or self.bar_type == common.QUOTE_TYPE.FIVEMIN
              or self.bar_type == common.QUOTE_TYPE.QUARTERMIN):
            self._run_bar()
        
        else:
            raise NotImplementedError("bar_type = {}".format(self.bar_type))
        
        print("Backtest done.")
        
    def save_results(self, folder_path='.'):
        import os
        import pandas as pd
        folder_path = os.path.abspath(folder_path)
    
        trades = self.ctx.pm.trades
    
        type_map = {'task_id': str,
                    'entrust_no': str,
                    'entrust_action': str,
                    'symbol': str,
                    'fill_price': float,
                    'fill_size': float,
                    'fill_date': np.integer,
                    'fill_time': np.integer,
                    'fill_no': str,
                    'commission': float,
                    'trade_date': np.integer}
        # keys = trades[0].__dict__.keys()
        ser_list = dict()
        for key in type_map.keys():
            v = [t.__getattribute__(key) for t in trades]
            ser = pd.Series(data=v, index=None, dtype=type_map[key], name=key)
            ser_list[key] = ser
        df_trades = pd.DataFrame(ser_list)
        df_trades.index.name = 'index'
    
        trades_fn = os.path.join(folder_path, 'trades.csv')
        configs_fn = os.path.join(folder_path, 'configs.json')
        jutil.create_dir(trades_fn)
    
        df_trades.to_csv(trades_fn)
        jutil.save_json(self.props, configs_fn)
    
        print ("Backtest results has been successfully saved to:\n" + folder_path)
