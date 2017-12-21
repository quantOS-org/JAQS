# encoding: utf-8

from __future__ import print_function, unicode_literals
import numpy as np
import pandas as pd

from jaqs.trade import common
from jaqs.data.basic import Bar
from jaqs.data.basic import Trade
import jaqs.util as jutil
from functools import reduce


class BacktestInstance(object):
    """
    Attributes
    ----------
    start_date : int
    end_date : int
    
    """
    def __init__(self):
        super(BacktestInstance, self).__init__()
        
        self.strategy = None
        self.start_date = 0
        self.end_date = 0

        self.props = None
        
        self.ctx = None
        
    def init_from_config(self, props):
        """
        
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
        
        if 'symbol' in props:
            self.ctx.init_universe(props['symbol'])
        elif hasattr(self.ctx, 'dataview'):
            self.ctx.init_universe(self.ctx.dataview.symbol)
        else:
            raise ValueError("No dataview, no symbol either.")

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
    last_rebalance_date : int
    current_rebalance_date : int
    univ_price_dic : dict
        Prices of symbols at current_date

    """
    def __init__(self):
        super(AlphaBacktestInstance, self).__init__()
    
        self.last_date = 0
        self.last_rebalance_date = 0
        self.current_rebalance_date = 0
        
        self.univ_price_dic = {}
        
        self.POSITION_ADJUST_NO = 101010
        self.POSITION_ADJUST_TIME = 200000
        self.DELIST_ADJUST_NO = 202020
        self.DELIST_ADJUST_TIME = 150000
        
        self.commission_rate = 20E-4
    
    def init_from_config(self, props):
        super(AlphaBacktestInstance, self).init_from_config(props)
        
        self.commission_rate = props.get('commission_rate', 20E-4)

    def position_adjust(self):
        """
        adjust happens after market close
        Before each re-balance day, adjust for all dividend and cash paid actions during the last period.
        We assume all cash will be re-invested.
        Since we adjust our position at next re-balance day, PnL before that may be incorrect.

        """
        start = self.last_rebalance_date  # start will be one day later
        end = self.current_rebalance_date  # end is the same to ensure position adjusted for dividend on rebalance day
        df_adj = self.ctx.dataview.get_ts('adjust_factor',
                                          start_date=start, end_date=end)
        pm = self.ctx.pm
        for symbol in pm.holding_securities:
            ser = df_adj.loc[:, symbol]
            ser_div = ser.div(ser.shift(1)).fillna(1.0)
            mask_diff = ser_div != 1
            ser_adj = ser_div.loc[mask_diff]
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
                trade_ind.set_fill_info(price=0.0, size=pos_diff, date=date, time=200000, no=self.POSITION_ADJUST_NO)
                
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
                                    date=last_trade_date, time=150000, no=self.DELIST_ADJUST_NO)

            self.ctx.strategy.cash += trade_ind.fill_price * trade_ind.fill_size
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
            col = 'index_member'
            df_is_member = self.ctx.dataview.get_snapshot(self.ctx.trade_date, fields=col)
            df_is_member = df_is_member.fillna(0).astype(bool)
            dic_index_member = df_is_member.loc[:, col].to_dict()
            universe_list = [symbol for symbol, value in dic_index_member.items() if value]

        # Step.2 filter out those not listed or already de-listed
        df_inst = self.ctx.dataview.data_inst
        mask = np.logical_and(self.ctx.trade_date > df_inst['list_date'],
                              self.ctx.trade_date < df_inst['delist_date'])
        listing_symbols = df_inst.loc[mask, :].index.values
        universe_list = [s for s in universe_list if s in listing_symbols]
        
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
        prices = {k: v['close'] for k, v in self.univ_price_dic.items()}
        # suspensions & limit_reaches: list of str
        suspensions = self.get_suspensions()
        limit_reaches = self.get_limit_reaches()
        all_list = reduce(lambda s1, s2: s1.union(s2), [set(suspensions), set(limit_reaches)])
    
        # step1. weights of those suspended and limit will be remove, and weights of others will be re-normalized
        self.ctx.strategy.re_weight_suspension(all_list)
    
        # step2. calculate market value and cash
        # market value does not include those suspended
        market_value_float, market_value_frozen = self.ctx.pm.market_value(prices, all_list)
        cash_available = self.ctx.strategy.cash + market_value_float
    
        cash_to_use = cash_available * self.ctx.strategy.position_ratio
        cash_unuse = cash_available - cash_to_use
    
        # step3. generate target positions
        # position of those suspended will remain the same (will not be traded)
        goals, cash_remain = self.ctx.strategy.generate_weights_order(self.ctx.strategy.weights, cash_to_use, prices,
                                                                  suspensions=all_list)
        self.ctx.strategy.goal_positions = goals
        self.ctx.strategy.cash = cash_remain + cash_unuse
        # self.liquidate_all()
        
        total = cash_available + market_value_frozen
        self.ctx.strategy.on_after_rebalance(total)
        self.ctx.record('total_cash', total)

    def run_alpha(self):
        tapi = self.ctx.trade_api
        
        self.ctx.trade_date = self._get_next_trade_date(self.start_date)
        self.last_date = self._get_last_trade_date(self.ctx.trade_date)
        self.current_rebalance_date = self.ctx.trade_date
        while True:
            print("\n=======new day {}".format(self.ctx.trade_date))

            # match uncome orders or re-balance
            if tapi.match_finished:
                # Step1.
                # position adjust according to dividend, cash paid, de-list actions during the last period
                # two adjust must in order
                self.position_adjust()
                self.delist_adjust()

                # Step2.
                # plan re-balance before market open of the re-balance day:
                self.on_new_day(self.last_date)  # use last trade date because strategy can only access data of last day
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
                
            self.on_after_market_close()
            
            # switch trade date
            backtest_finish = self.go_next_rebalance_day()
            if backtest_finish:
                break
        
        print("Backtest done. {:d} days, {:.2e} trades in total.".format(len(self.ctx.dataview.dates),
                                                                         len(self.ctx.pm.trades)))
    
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
    
    def _get_next_trade_date(self, date):
        if self.ctx.dataview is not None:
            dates = self.ctx.dataview.dates
            mask = dates > date
            return dates[mask][0]
        else:
            return self.ctx.data_api.get_next_trade_date(date)
    
    def _get_last_trade_date(self, date):
        if self.ctx.dataview is not None:
            dates = self.ctx.dataview.dates
            mask = dates < date
            return dates[mask][-1]
        else:
            return self.ctx.data_api.get_last_trade_date(date)
    
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
            next_period_day = jutil.get_next_period_day(current_date, self.ctx.strategy.period,
                                                        n=self.ctx.strategy.n_periods,
                                                        extra_offset=self.ctx.strategy.days_delay)
            if next_period_day > self.end_date:
                return True
            
            # update current_date: next_period_day is a workday, but not necessarily a trade date
            if self._is_trade_date(next_period_day):
                current_date = next_period_day
            else:
                try:
                    current_date = self._get_next_trade_date(next_period_day)
                except IndexError:
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
        df_open = self.ctx.dataview.get_snapshot(self.ctx.trade_date, fields='open')
        df_close = self.ctx.dataview.get_snapshot(self.last_date, fields='close')
        merge = pd.concat([df_close, df_open], axis=1)
        merge.loc[:, 'limit'] = np.abs((merge['open'] - merge['close']) / merge['close']) > 9.5E-2
        return list(merge.loc[merge.loc[:, 'limit'], :].index.values)
    
    def on_new_day(self, date):
        # self.ctx.strategy.on_new_day(date)
        self.ctx.trade_api.on_new_day(date)
        
        self.ctx.snapshot = self.ctx.dataview.get_snapshot(date)
        self.univ_price_dic = self.ctx.snapshot.loc[:, ['close', 'vwap', 'open', 'high', 'low']].to_dict(orient='index')
    
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
                    'commission': float}
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
    def __init__(self):
        super(EventBacktestInstance, self).__init__()
        
        self.bar_type = ""
        
    def init_from_config(self, props):
        super(EventBacktestInstance, self).init_from_config(props)
        
        self.bar_type = props.get("bar_type", "1d")
        
    def go_next_trade_date(self):
        next_dt = self.ctx.data_api.get_next_trade_date(self.ctx.trade_date)
        
        self.ctx.trade_date = next_dt
    
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

    def _create_time_symbol_bars(self, date):
        from collections import defaultdict
        
        # query quotes data
        symbols_str = ','.join(self.ctx.universe)
        df_quotes, msg = self.ctx.data_api.bar(symbol=symbols_str, start_time=200000, end_time=160000,
                                               trade_date=date, freq=self.bar_type)
        if msg != '0,':
            print(msg)
        if df_quotes is None or df_quotes.empty:
            return dict()
    
        # create nested dict
        quotes_list = Bar.create_from_df(df_quotes)
        
        dic = defaultdict(dict)
        for quote in quotes_list:
            dic[jutil.combine_date_time(quote.date, quote.time)][quote.symbol] = quote
        return dic
    
    def _run_bar(self):
        """Quotes of different symbols will be aligned into one dictionary."""
        trade_dates = self.ctx.data_api.get_trade_date_range(self.start_date, self.end_date)

        for trade_date in trade_dates:
            self.on_new_day(trade_date)
            
            quotes_dic = self._create_time_symbol_bars(trade_date)
            for dt in sorted(quotes_dic.keys()):
                _, time = jutil.split_date_time(dt)
                self.ctx.time = time
                
                quote_by_symbol = quotes_dic.get(dt)
                self._process_quote_bar(quote_by_symbol)
            
            self.on_after_market_close()
    
    def _run_daily(self):
        """Quotes of different symbols will be aligned into one dictionary."""
        from collections import defaultdict
        
        symbols_str = ','.join(self.ctx.universe)
        df_daily, msg = self.ctx.data_api.daily(symbol=symbols_str, start_date=self.start_date, end_date=self.end_date,
                                                adjust_mode='post')
        if msg != '0,':
            print(msg)
        if df_daily is None or df_daily.empty:
            return dict()
        
        # create nested dict
        quotes_list = Bar.create_from_df(df_daily)

        dic = defaultdict(dict)
        for quote in quotes_list:
            dic[quote.trade_date][quote.symbol] = quote
        
        dates = sorted(dic.keys())
        for i in range(len(dates) - 1):
            d1, d2 = dates[i], dates[i + 1]
            self.on_new_day(d2)
            
            quote1 = dic.get(d1)
            quote2 = dic.get(d2)
            self._process_quote_daily(quote1, quote2)
            
            self.on_after_market_close()
    
    def _process_quote_daily(self, quote_yesterday, quote_today):
        # on_bar
        self.ctx.strategy.on_bar(quote_yesterday)
        
        self.ctx.trade_api.match_and_callback(quote_today, freq=self.bar_type)
        
        '''
        # match
        trade_results = self.ctx.gateway._process_quote(quote_today, freq=self.bar_type)

        # trade indication
        for trade_ind, status_ind in trade_results:
            comm = self.calc_commission(trade_ind)
            trade_ind.commission = comm
            # self.ctx.strategy.cash -= comm
            
            self.ctx.strategy.on_trade(trade_ind)
            self.ctx.strategy.on_order_status(status_ind)
        '''
        
        self.on_after_market_close()

    def run(self):
        if self.bar_type == common.QUOTE_TYPE.DAILY:
            self._run_daily()
        
        elif (self.bar_type == common.QUOTE_TYPE.MIN
              or self.bar_type == common.QUOTE_TYPE.FIVEMIN
              or self.bar_type == common.QUOTE_TYPE.QUARTERMIN):
            self._run_bar()
        
        else:
            raise NotImplementedError("bar_type = {}".format(self.bar_type))
        
        print("Backtest done.")
        
    def _process_quote_bar(self, quotes_dic):
        self.ctx.trade_api.match_and_callback(quotes_dic, freq=self.bar_type)
        
        '''
        # match
        trade_results = self.ctx.trade_api._process_quote(quotes_dic, freq=self.bar_type)
        
        # trade indication
        for trade_ind, status_ind in trade_results:
            comm = self.calc_commission(trade_ind)
            trade_ind.commission = comm
            # self.ctx.strategy.cash -= comm
    
            self.ctx.strategy.on_trade(trade_ind)
            self.ctx.strategy.on_order_status(status_ind)
        '''
        
        # on_bar
        self.ctx.strategy.on_bar(quotes_dic)

    '''
    def generate_report(self, output_format=""):
        return self.pnlmgr.generateReport(output_format)
    '''
    
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
                    'commission': float}
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
    
    '''
    def run_event(self):
        data_api = self.ctx.data_api
        universe = self.ctx.universe
        
        data_api.add_batch_subscribe(self, universe)
        
        self.ctx.trade_date = self.start_date
        
        def __extract(func):
            return lambda event: func(event.data, **event.kwargs)
        
        ee = self.ctx.strategy.eventEngine  # TODO event-driven way of lopping, is it proper?
        ee.register(EVENT.CALENDAR_NEW_TRADE_DATE, __extract(self.ctx.strategy.on_new_day))
        ee.register(EVENT.MD_QUOTE, __extract(self._process_quote_bar))
        ee.register(EVENT.MARKET_CLOSE, __extract(self.close_day))
        
        while self.ctx.trade_date <= self.end_date:  # each loop is a new trading day
            quotes = data_api.get_daily_quotes(self.ctx.trade_date)
            if quotes is not None:
                # gateway.oneNewDay()
                e_newday = Event(EVENT.CALENDAR_NEW_TRADE_DATE)
                e_newday.data = self.ctx.trade_date
                ee.put(e_newday)
                ee.process_once()  # this line should be done on another thread
                
                # self.ctx.strategy.onNewday(self.ctx.trade_date)
                self.ctx.strategy.pm.on_new_day(self.ctx.trade_date, self.last_date)
                self.ctx.strategy.ctx.trade_date = self.ctx.trade_date
                
                for quote in quotes:
                    # self.processQuote(quote)
                    e_quote = Event(EVENT.MD_QUOTE)
                    e_quote.data = quote
                    ee.put(e_quote)
                    ee.process_once()
                
                # self.ctx.strategy.onMarketClose()
                # self.closeDay(self.ctx.trade_date)
                e_close = Event(EVENT.MARKET_CLOSE)
                e_close.data = self.ctx.trade_date
                ee.put(e_close)
                ee.process_once()
                # self.ctx.strategy.onSettle()
                
            else:
                # no quotes because of holiday or other issues. We don't update last_date
                print "in trade.py: function run(): {} quotes is None, continue.".format(self.last_date)
            
            self.ctx.trade_date = self.go_next_trade_date(self.ctx.trade_date)
            
            # self.ctx.strategy.onTradingEnd()

    '''
