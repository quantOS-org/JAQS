# encoding: utf-8

from __future__ import absolute_import, print_function, unicode_literals, division
import datetime

import numpy as np
import pandas as pd

from jaqs.trade.event import EventEngine, Event, EVENT_TYPE
from jaqs.data.basic import Quote
import jaqs.util as jutil
from functools import reduce


class AlphaLiveTradeInstance(object):
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
        super(AlphaLiveTradeInstance, self).__init__()
        
        self.ctx = None
        
        self.last_date = 0
        self.last_rebalance_date = 0
        self.current_rebalance_date = 0
        
        self.univ_price_dic = {}

    def init_from_config(self, props):
        """
        
        Parameters
        ----------
        props : dict

        """
        self.props = props
    
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
        
        self.ctx.strategy.initialize()
    
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
            
            # we are able to know information about index component before market open
            #col = 'index_member'
            #df_is_member = self.ctx.dataview.get_snapshot(self.ctx.trade_date, fields=col)
            #df_is_member = df_is_member.fillna(0).astype(bool)
            #dic_index_member = df_is_member.loc[:, col].to_dict()
            ser_is_member = self.ctx.dataview.get_ts('index_member').iloc[-1, :]
            dic_index_member = ser_is_member.to_dict()
            universe_list = [symbol for symbol, value in dic_index_member.items() if value]
        
        # Step.2 filter out those not listed or already de-listed
        # we are able to know information about list/delist before market open
        df_inst = self.ctx.dataview.data_inst
        mask = np.logical_and(self.ctx.trade_date > df_inst['list_date'],
                              self.ctx.trade_date < df_inst['delist_date'])
        listing_symbols = df_inst.loc[mask, :].index.values
        universe_list = [s for s in universe_list if s in listing_symbols]
        
        # step.3 construct portfolio using models
        self.ctx.strategy.portfolio_construction(universe_list)

    def get_suspensions(self):
        # trade_status = self.ctx.dataview.get_snapshot(self.ctx.trade_date, fields='trade_status')
        # trade_status = trade_status.loc[:, 'trade_status']
        # trade_status: {'N', 'XD', 'XR', 'DR', 'JiaoYi', 'TingPai', NUll (before 2003)}
        # mask_sus = trade_status == u'停牌'
        res = [k for k, v in self.univ_price_dic.items() if v['volume'] == 0]
        return res

    def get_limit_reaches(self):
        res = [k for k, v in self.univ_price_dic.items() if (v['last'] >= v['limit_up'] or v['last'] <= v['limit_down'])]
        return res
    
    def _get_latest_prices(self):
        symbols_str = ','.join(self.ctx.universe)
        cols = ','.join(['last', 'open', 'high', 'low', 'limit_up', 'limit_down', 'askprice1', 'bidprice1', 'volume', 'turnover'])
        res, _ = self.ctx.data_api.quote(symbol=symbols_str, fields=cols)
        res = res.to_dict(orient='index')
        return res
    
    def re_balance_plan_after_open(self):
        """
        Do portfolio re-balance after market open.
        With suspensions known, we re-calculate weights and generate orders.
        
        Notes
        -----
        Price here must not be adjusted.

        """
        self.univ_price_dic = self._get_latest_prices()
        prices_dic = {k: v['last'] for k, v in self.univ_price_dic.items()}
        # suspensions & limit_reaches: list of str
        suspensions = self.get_suspensions()
        limit_reaches = self.get_limit_reaches()
        all_list = reduce(lambda s1, s2: s1.union(s2), [set(suspensions), set(limit_reaches)])
        
        # step1. weights of those suspended and limit will be remove, and weights of others will be re-normalized
        self.ctx.strategy.re_weight_suspension(all_list)
        
        # step2. calculate market value and cash
        # market value does not include those suspended
        market_value_float, market_value_frozen = self.ctx.pm.market_value(prices_dic, all_list)
        cash_available = self.ctx.strategy.cash + market_value_float
        
        cash_to_use = cash_available * self.ctx.strategy.position_ratio
        cash_unuse = cash_available - cash_to_use
        
        # step3. generate target positions
        # position of those suspended will remain the same (will not be traded)
        goals, cash_remain = self.ctx.strategy.generate_weights_order(self.ctx.strategy.weights,
                                                                      cash_to_use, prices_dic,
                                                                      suspensions=all_list)
        
        self.ctx.strategy.goal_positions = self._to_valide_goals(goals)
        self.ctx.strategy.cash = cash_remain + cash_unuse
        # self.liquidate_all()
        
        self.ctx.strategy.on_after_rebalance(cash_available + market_value_frozen)
    
    def _to_valide_goals(self, goals_raw):
        univ, msg = self.ctx.trade_api.query_universe()
        univ.loc[:, 'size'] = 0.0
        univ.loc[:, 'ref_price'] = 0.0
        univ = univ.set_index('security').sort_index(axis=0)
        univ = univ[['size', 'ref_price']]
        for d in goals_raw:
            symbol = d['symbol']
            size = d['size']
            # TODO: better method needed
            if symbol in univ.index and symbol in self.univ_price_dic:
                univ.loc[symbol, 'ref_price'] = self.univ_price_dic[symbol]['last']
                if size > 0:
                    univ.loc[symbol, 'size'] = size
        goals_valid = list(univ.reset_index().to_dict(orient='index').values())
        return goals_valid
    
    @staticmethod
    def _get_current_date():
        now_utc = datetime.datetime.utcnow()
        now_utc8 = now_utc + datetime.timedelta(hours=8)
        now_date = jutil.convert_datetime_to_int(now_utc8)
        return now_date
    
    def run_alpha(self):
        tapi = self.ctx.trade_api
        
        self.ctx.trade_date = self._get_current_date()
        self.last_date = self._get_last_trade_date(self.ctx.trade_date)
        print("\n=======Re-balance starts, trade_date = {}".format(self.ctx.trade_date))
        
        # Step1.
        # we do not need adjust
        
        # Step2.
        # plan re-balance before market open of the re-balance day:
        self.ctx.snapshot = self.ctx.dataview.get_snapshot(self.last_date)
        # get index memebers, get signals, generate weights
        self.re_balance_plan_before_open()
        
        # Step3.
        # do re-balance on the re-balance day
        # get suspensions, get up/down limits, generate goal positions and send orders.
        self.re_balance_plan_after_open()
        # self.ctx.strategy.send_bullets()
        
        print("Re-balance done.")
    
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
            return self.ctx.data_api.query_next_trade_date(date)
    
    def _get_last_trade_date(self, date):
        if self.ctx.dataview is not None:
            dates = self.ctx.dataview.dates
            mask = dates < date
            return dates[mask][-1]
        else:
            return self.ctx.data_api.query_last_trade_date(date)
    
    '''
    def on_new_day(self, date):
        self.ctx.strategy.initialize()
        self.ctx.snapshot = self.ctx.dataview.get_snapshot(date)
        self.univ_price_dic = self.ctx.snapshot.loc[:, ['close', 'vwap', 'open', 'high', 'low']].to_dict(orient='index')
    
    '''
    def show_position_info(self):
        pm = self.ctx.pm
        
        prices = {k: v['open'] for k, v in self.univ_price_dic.items()}
        market_value_float, market_value_frozen = pm.market_value(prices)
        for symbol in pm.holding_securities:
            p = prices[symbol]
            size = pm.get_position(symbol).current_size
            print("{}  {:.2e}   {:.1f}@{:.2f}".format(symbol, p*size*100, p, size))
        print("float {:.2e}, frozen {:.2e}".format(market_value_float, market_value_frozen))


class EventLiveTradeInstance(EventEngine):
    """
    Attributes
    ----------
    start_date : int
    end_date : int
    
    """
    def __init__(self):
        super(EventLiveTradeInstance, self).__init__()
        
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
                pass
                # raise ValueError("{} must be provided in props.".format(name))
        
        self.props = props
        self.start_date = props.get("start_date")
        self.end_date = props.get("end_date")

        for obj in ['data_api', 'trade_api', 'pm', 'strategy']:
            if hasattr(self.ctx, obj):
                getattr(self.ctx, obj).init_from_config(props)

    def register_context(self, context=None):
        self.ctx = context

    # -------------------------------------------------------------------------------------------
    # Run
    def run(self):
        """
        Listen to certain events and run the EventEngine.
        Events include:
            1. market_data are from DataService
            2. trades & orders indications are from TradeApi.
            3. etc.

        """

        self.register(EVENT_TYPE.MARKET_DATA, self.on_bar)
        
        self.register(EVENT_TYPE.TASK_STATUS_IND, self.on_task_status)
        self.register(EVENT_TYPE.ORDER_RSP, self.on_order_rsp)
        self.register(EVENT_TYPE.TASK_RSP, self.on_task_rsp)
        self.register(EVENT_TYPE.TRADE_IND, self.on_trade)
        self.register(EVENT_TYPE.ORDER_STATUS_IND, self.on_order_status)
        
        self.start(timer=False)
    
    def on_bar(self, event):
        quote_dic = event.dic['quote']
        quote = Quote.create_from_dict(quote_dic)
        self.ctx.strategy.on_tick(quote)
    
    def on_order_rsp(self, event):
        rsp = event.dic['rsp']
        self.ctx.strategy.on_order_rsp(rsp)

    def on_task_rsp(self, event):
        rsp = event.dic['rsp']
        self.ctx.strategy.on_task_rsp(rsp)
    
    def on_trade(self, event):
        ind = event.dic['ind']
        self.ctx.strategy.on_trade(ind)

    def on_order_status(self, event):
        ind = event.dic['ind']
        self.ctx.strategy.on_order_status(ind)

    def on_task_status(self, event):
        ind = event.dic['ind']
        self.ctx.strategy.on_task_status(ind)
    
    # ---------------------------------------------------------
    # Save Results
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
