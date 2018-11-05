# encoding: utf-8

"""
Analyze module defines classes to analyze trading results, including I/O,
calculation, plot.

# TODO:
each year metrics, Max DD;
psotion during each re-balance period

"""
from __future__ import print_function
import os
import codecs
import json
from collections import OrderedDict
try:
    basestring
except NameError:
    basestring = str

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
# import seaborn as sns
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter

from jaqs.trade.analyze.report import Report
from jaqs.data import RemoteDataService
from jaqs.data.basic.instrument import InstManager
from jaqs.data.py_expression_eval import Parser
from jaqs.trade import common

import jaqs.util as jutil
from jaqs.util.profile import prof_sample_begin, prof_sample_end

STATIC_FOLDER = jutil.join_relative_path("trade/analyze/static")
TO_PCT = 100.0
MPL_RCPARAMS = {'figure.facecolor': '#F6F6F6',
                'axes.facecolor': '#F6F6F6',
                'axes.edgecolor': '#D3D3D3',
                'text.color': '#555555',
                'grid.color': '#B1B1B1',
                'grid.alpha': 0.3,
                # scale
                'axes.linewidth': 2.0,
                'axes.titlepad': 12,
                'grid.linewidth': 1.0,
                'grid.linestyle': '-',
                # font size
                'font.size': 13,
                'axes.titlesize': 18,
                'axes.labelsize': 14,
                'legend.fontsize': 'small',
                'lines.linewidth': 2.5,
                }

class AnalyzeView(object):
    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    def get_ts(self, field, symbol="", start_date=0, end_date=0, keep_level=False):
        """
        Get time series data of single field.

        Parameters
        ----------
        field : str or unicode
            Single field.
        symbol : str, optional
            Separated by ',' default "" (all securities).
        start_date : int, optional
            Default 0 (self.start_date).
        end_date : int, optional
            Default 0 (self.start_date).

        Returns
        -------
        res : pd.DataFrame
            Index is int date, column is symbol.

        """
        res = self.get(symbol, start_date=start_date, end_date=end_date, fields=field)
        if res is None:
            print("No data. for start_date={}, end_date={}, field={}, symbol={}".format(start_date,
                                                                                        end_date, field, symbol))
            raise ValueError
            return

        #if not keep_level and len(res.columns) and len(field.split(',')) == 1:
        if not keep_level and len(field.split(',')) == 1:
            res.columns = res.columns.droplevel(level='field')
            # XXX Save field name for ResReturnFunc
            #res.columns.name = field

        return res

    def get_snapshot(self, snapshot_date, symbol="", fields=""):
        """
        Get snapshot of given fields and symbol at snapshot_date.

        Parameters
        ----------
        snapshot_date : int
            Date of snapshot.
        symbol : str, optional
            Separated by ',' default "" (all securities).
        fields : str, optional
            Separated by ',' default "" (all fields).

        Returns
        -------
        res : pd.DataFrame
            symbol as index, field as columns

        """

        # if self._snapshot is not None:
        #     if snapshot_date not in self._snapshot:
        #         return
        #
        #     df = self._snapshot[snapshot_date]
        #     if fields:
        #         return df[fields.split(',')]
        #     else:
        #         return df

        res = self.get(symbol=symbol, start_date=snapshot_date, end_date=snapshot_date, fields=fields)
        if res is None:
            print("No data. for date={}, fields={}, symbol={}".format(snapshot_date, fields, symbol))
            return

        res = res.stack(level='symbol', dropna=False)
        res.index = res.index.droplevel(level='trade_date')

        return res

    def get(self, symbol="", start_date=0, end_date=0, fields=""):
        """
        Basic API to get arbitrary data. If nothing fetched, return None.

        Parameters
        ----------
        symbol : str, optional
            Separated by ',' default "" (all securities).
        start_date : int, optional
            Default 0 (self.start_date).
        end_date : int, optional
            Default 0 (self.start_date).
        fields : str, optional
            Separated by ',' default "" (all fields).

        Returns
        -------
        res : pd.DataFrame or None
            index is datetimeindex, columns are (symbol, fields) MultiIndex

        """
        sep = ','

        if self._data is None:
            return None

        if len(self._data.index) == 0:
            return None

        if not fields:
            fields = slice(None)  # self.fields
        else:
            fields = fields.split(sep)

        if not symbol:
            symbol = slice(None)  # this is 3X faster than symbol = self.symbol
        else:
            symbol = symbol.split(sep)

        if not start_date:
            start_date = self._data.index[0]
        if not end_date:
            end_date = self._data.index[-1]

        res = self._data.loc[pd.IndexSlice[start_date: end_date], pd.IndexSlice[symbol, fields]]
        return res

    def add_formula(self, field_name, formula, overwrite=True,
                    formula_func_name_style='camel'):

        parser = Parser()
        parser.set_capital(formula_func_name_style)

        expr = parser.parse(formula)

        var_df_dic = dict()
        var_list = expr.variables()

        for var in var_list:
            df_var = self.get_ts(var)
            var_df_dic[var] = df_var

        df_eval = parser.evaluate(var_df_dic)
        self.add_field(df_eval, field_name)

    def add_field(self, df, field_name=None):
        """
        Add fields to self._data.
        :param df:
        :param field_names: format:   'open', or 'open,close,high'
        :return:
        """
        if not isinstance(df.columns, pd.core.indexes.multi.MultiIndex):
            if not field_name:
                raise ValueError("no field_name provided.")

            df = df.copy()
            #exist_symbols = self._data.columns.levels[0]
            exist_symbols = df.columns
            df.columns = pd.MultiIndex.from_product([exist_symbols, [field_name]])

        self._data = pd.concat([self._data, df], axis=1).sort_index(axis=1)

class TradeRecordEmptyError(Exception):
    def __init__(self, *args):
        super(TradeRecordEmptyError, self).__init__(*args)


class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y%m'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        """Return the label for time x at position pos"""
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''

        # return self.dates[ind].strftime(self.fmt)
        return pd.to_datetime(self.dates[ind], format="%Y%m%d").strftime(self.fmt)


class BaseAnalyzer(object):
    """
    BaseAnalyzer is a concrete class. It defines what should be analyzed for all types
    of strategies.
    
    Attributes
    ----------
    _trades : pd.DataFrame
        Raw trading records, inluding fill_price, fill_time, fill_size, etc.
    _configs : dict
        Configuration read from file.
    data_api : BaseDataServer
    dataview : DataView
    _universe : set
        All securities that have been traded.
    _closes : pd.DataFrame
        Daily close prices of all symbols.
    _closes_adj : pd.DataFrame
        Daily adjusted close prices of all symbols.
    returns : pd.DataFrame
        Daily return and cumulative return of strategy, benchmark and excess.
    daily : pd.DataFrame
        Essential daily statistics of trading.
    df_pnl : pd.DataFrame
        Daily trading, holding and total PnL.
    adjust_mode : {'pre', 'post', None}
        Adjust_mode for adjusted price.
    inst_map : dict
        Keys are symbols, values are dict of symbol attributes like multiplier.
    performance_metrics : dict
        Names and values of strategy performance indicator.
    risk_metrics : dict
        
    """
    def __init__(self, is_alpha_analyzer=True):
        self.file_folder = ""

        self._is_alpha_analyzer = is_alpha_analyzer

        self._raw_trades = None
        self._trades = None
        self._configs = None
        self.data_api = None
        self.dataview = None
        
        self._universe = []
        self._closes = None
        self._closes_adj = None
        self.daily_position = None
        self.rebalance_positions = None
        self.returns = None
        self.position_change = None
        self.account = None
        self.daily = None
        self.df_pnl = None

        self.adjust_mode = None
        
        self.inst_map = dict()
        
        self.performance_metrics = dict()
        self.risk_metrics = dict()
        
        self.report_dic = dict()

        self._holding_data   = None
        self._portfolio_data = None
        # self._order_data   = None
        # self._trade_data   = None

        self._alpha_decay_image = None
        self._industry_overweight_images = None
        self._alpha_decomposition_images = None
        self._alpha_decomposition_industry_images = None

        self._average_industry_overweight = None
        self._alpha_decay_weight_image = None
        self._cum_alpha_weight_image = None
        self._alpha_decomposition = None
        self._industry_agg = None

        self._sold_alpha_decay = None
        self._sold_alpha_decay_image = None

    @property
    def trades(self):
        """Read-only attribute"""
        return self._trades
    
    @property
    def universe(self):
        """Read-only attribute"""
        return self._universe
    
    @property
    def configs(self):
        """Read-only attribute"""
        return self._configs
    
    @property
    def closes(self):
        """Read-only attribute, close prices of securities in the universe"""
        return self._closes

    @property
    def closes_adj(self):
        """Read-only attribute, close prices of securities in the universe"""
        return self._closes_adj

    @property
    def holding_data(self):
        """Read-only attribute, holding data of securities in each day"""
        return self._holding_data

    @property
    def portfolio_data(self):
        return self._portfolio_data

    def initialize(self, data_api=None, dataview=None, file_folder='.'):
        """
        Read trading records and configurations from file.
        Initialized various data for analysis, including:
            - Daily close price
            - Basic instrument information

        Parameters
        ----------
        data_api : RemoteDataService
        dataview : DataView
        file_folder : str or list of str
            Directory path where trades and configs are stored.

        """
        if isinstance(file_folder, basestring):
            file_folder = [file_folder]
        
        self.data_api = data_api
        self.dataview = dataview
        
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

        abs_path_list = [os.path.abspath(folder) for folder in file_folder]
        self.file_folder = abs_path_list
        trades_list = [pd.read_csv(os.path.join(folder, 'trades.csv'), ',', dtype=type_map)
                       for folder in self.file_folder]
        if any([trades.empty for trades in trades_list]):
            raise TradeRecordEmptyError("No trade records found in your 'trades.csv' file. Analysis stopped.")
        
        # combine trades
        trades = pd.concat(trades_list, axis=0)
        trades = trades.sort_values(['trade_date', 'fill_time'])
        
        self._init_universe(trades.loc[:, 'symbol'].values)
        self._init_configs(self.file_folder)
        self._init_trades(trades)
        self._init_symbol_price()
        self._init_inst_data()
    
    def _init_inst_data(self):
        """
        Query instrument info from DataService or DataView.
        
        """
        symbol_str = ','.join(self.universe)
        if self.dataview is not None:
            data_inst = self.dataview.data_inst
            self.inst_map = data_inst.to_dict(orient='index')
        elif self.data_api is not None:
            inst_mgr = InstManager(data_api=self.data_api, symbol=symbol_str)
            self.inst_map = {k: v.__dict__ for k, v in inst_mgr._inst_map.items()}
            del inst_mgr
        else:
            raise ValueError("no dataview or dataapi provided.")
        
    def _init_trades(self, df):
        """
        Modify trading records dataframe. (Add datetime column)
        
        Parameters
        ----------
        df : pd.DataFrame
            Each row represents a single trading record.
        
        """
        self._raw_trades = df.copy()
        
        df.loc[:, 'fill_dt'] = jutil.combine_date_time(df.loc[:, 'trade_date'], df.loc[:, 'fill_time'])
        
        df = df.set_index(['symbol', 'fill_dt']).sort_index(axis=0)
        
        # self._trades = jutil.group_df_to_dict(df, by='symbol')
        self._trades = df
    
    def _init_symbol_price(self):
        """
        Get close price of securities in the universe from DataService or DataView.
        Both raw close prices and adjusted close prices are stored.

        """
        if self.dataview is not None:
            df_close = self.dataview.get_ts('close', start_date=self.start_date, end_date=self.end_date)
            df_close_adj = self.dataview.get_ts('close_adj', start_date=self.start_date, end_date=self.end_date)
        else:
            df, msg = self.data_api.daily(symbol=','.join(self.universe), fields='trade_date,symbol,close',
                                          start_date=self.start_date, end_date=self.end_date)
            if msg != '0,':
                print(msg)
            df_close = df.pivot(index='trade_date', columns='symbol', values='close')

            df_adj, msg = self.data_api.daily(symbol=','.join(self.universe), fields='trade_date,symbol,close',
                                          start_date=self.start_date, end_date=self.end_date)
            if msg != '0,':
                print(msg)
            df_close_adj = df_adj.pivot(index='trade_date', columns='symbol', values='close')
        self._closes = df_close
        self._closes_adj = df_close_adj

    def _init_universe(self, securities):
        """Return a set of securities."""
        self._universe = set(securities)
    
    def _init_configs(self, folder_list):
        """
        Read configs from file and get some important items.
        
        Parameters
        ----------
        folder_list : list of str
            Directory path where configs.json is under.

        """
        configs_list = []
        # TODO: support weight
        for folder in folder_list:
            with codecs.open(os.path.join(folder, 'configs.json'), 'r', encoding='utf-8') as f:
                configs_list.append(json.load(f))
                
        self._configs = configs_list[0]
        
        self._configs['start_date'] = min([c['start_date'] for c in configs_list])
        self._configs['end_date'] = max([c['end_date'] for c in configs_list])
        self._configs['init_balance'] = sum([c['init_balance'] for c in configs_list])
        
        self.benchmark = self.configs.get('benchmark', "")
        self.init_balance = self.configs['init_balance']
        self.start_date = self.configs['start_date']
        self.end_date = self.configs['end_date']
        
    #@staticmethod
    def _process_trades(self,df):
        """
        Add various statistics to trades DataFrame.
        
        Parameters
        ----------
        df : pd.DataFrame
            Index is int datetime, columns are different terms.

        Returns
        -------

        """
        from jaqs.trade import common
        
        # pre-process
        cols_to_drop = ['task_id', 'entrust_no', 'fill_no']
        df = df.drop(cols_to_drop, axis=1)

        perf_tmp = prof_sample_begin("_process_trades")

        def _apply(gp_df,inst_map):
            # calculation of non-cumulative fields
            direction = gp_df['entrust_action'].apply(lambda s: 1 if common.ORDER_ACTION.is_positive(s) else -1)
            fill_size, fill_price = gp_df['fill_size'], gp_df['fill_price']
            symbol = gp_df.index.levels[0][0]
            mult = self.inst_map.get(symbol).get("multiplier")
            turnover = fill_size * fill_price * mult
            gp_df.loc[:, 'commission'] = gp_df.loc[:, 'commission'] * mult
            gp_df.loc[:, 'BuyVolume'] = (direction + 1) / 2 * fill_size
            gp_df.loc[:, 'SellVolume'] = (direction - 1) / -2 * fill_size
            
            # Calculation of cumulative fields
            gp_df.loc[:, 'TurnOver'] = turnover
            gp_df.loc[:, 'CumVolume'] = fill_size.cumsum()
            gp_df.loc[:, 'CumTurnOver'] = turnover.cumsum()
            gp_df.loc[:, 'CumNetTurnOver'] = (turnover * -direction).cumsum()
    
            gp_df.loc[:, 'position'] = (fill_size * direction).cumsum()
    
            gp_df.loc[:, 'AvgPosPrice'] = calc_avg_pos_price(gp_df.loc[:, 'position'].values, fill_price.values)
    
            gp_df.loc[:, 'CumProfit'] = (gp_df.loc[:, 'CumNetTurnOver'] + gp_df.loc[:, 'position'] * fill_price * mult)
            return gp_df
        gp = df.groupby(by='symbol')
        res = gp.apply(_apply,self.inst_map)

        prof_sample_end(perf_tmp)
        return res
    
    def process_trades(self):
        self._trades = self._process_trades(self._trades)
    
    def get_pos_change_info(self):
        """
        Calculate daily position and average holding price.
        Definition of Average Holding price:
            If abs(position) increases, avg_price will be weighted average of fill_price. Weights are fill_size.
            If abs(position) decreases, avg_price will NOT change.

        """
        trades = pd.concat(self.trades.values(), axis=0)
        gp = trades.groupby(by=['trade_date'], as_index=False)
        res = OrderedDict()
        account = OrderedDict()
        
        for date, df in gp:
            df_mod = df.loc[:, ['symbol', 'entrust_action', 'fill_size', 'fill_price',
                                'position', 'AvgPosPrice']]
            df_mod.columns = ['symbol', 'action', 'size', 'price',
                              'position', 'cost price']
        
            res[str(date)] = df_mod
        
            mv = sum(df_mod.loc[:, 'price'] * df.loc[:, 'position'])
            current_profit = sum(df.loc[:, 'CumProfit'])
            cash = self.init_balance + current_profit - mv
        
            account[str(date)] = {'market_value': mv, 'cash': cash}
        self.position_change = res
        self.account = account

    def get_minute(self):
        freq = 'fill_time'
        #td = 20180118  #
        td = self.trades['trade_date'].iat[0]
        df, msg = self.data_api.bar(symbol=','.join(self.universe), fields='trade_date,symbol,close',
                                    trade_date=td)
        df_close = df.pivot(index='time', columns='symbol', values='close')
        
        close = df_close
        trade = self.trades
    
        # pro-process
        trade_cols = [freq, 'BuyVolume', 'SellVolume', 'commission', 'position', 'AvgPosPrice', 'CumNetTurnOver']
    
        trade = trade.loc[:, trade_cols]
        trade.loc[:, 'fill_minute'] = trade['fill_time'] // 100 * 100
        gp = trade.reset_index().groupby(by=['symbol', 'fill_minute'])
        func_last = lambda ser: ser.iat[-1]
        trade = gp.agg({'BuyVolume': np.sum, 'SellVolume': np.sum, 'commission': np.sum,
                        'position': func_last, 'AvgPosPrice': func_last, 'CumNetTurnOver': func_last})
        trade.index.names = ['symbol', freq]
    
        # get daily position
        df_position = trade['position'].unstack('symbol').fillna(method='ffill').fillna(0.0)
        daily_position = df_position.reindex(close.index)
        daily_position = daily_position.fillna(method='ffill').fillna(0)
        self.daily_position = daily_position
    
        # calculate statistics
        close = pd.DataFrame(close.T.stack())
        close.columns = ['close']
        close.index.names = ['symbol', freq]
        merge = pd.concat([close, trade], axis=1, join='outer')
    
        def _apply(gp_df):
            cols_nan_to_zero = ['BuyVolume', 'SellVolume', 'commission']
            cols_nan_fill = ['close', 'position', 'AvgPosPrice', 'CumNetTurnOver']
            # merge: pd.DataFrame
            gp_df.loc[:, cols_nan_fill] = gp_df.loc[:, cols_nan_fill].fillna(method='ffill')
            gp_df.loc[:, cols_nan_fill] = gp_df.loc[:, cols_nan_fill].fillna(0)
        
            gp_df.loc[:, cols_nan_to_zero] = gp_df.loc[:, cols_nan_to_zero].fillna(0)
        
            mask = gp_df.loc[:, 'AvgPosPrice'] < 1e-5
            gp_df.loc[mask, 'AvgPosPrice'] = gp_df.loc[mask, 'close']
        
            gp_df.loc[:, 'CumProfit'] = gp_df.loc[:, 'CumNetTurnOver'] + gp_df.loc[:, 'position'] * gp_df.loc[:, 'close']
            gp_df.loc[:, 'CumProfitComm'] = gp_df['CumProfit'] - gp_df['commission'].cumsum()
        
            daily_net_turnover = gp_df['CumNetTurnOver'].diff(1).fillna(gp_df['CumNetTurnOver'].iat[0])
            daily_position_change = gp_df['position'].diff(1).fillna(gp_df['position'].iat[0])
            gp_df['trading_pnl'] = (daily_net_turnover + gp_df['close'] * daily_position_change )
            gp_df['holding_pnl'] = (gp_df['close'].diff(1) * gp_df['position'].shift(1)).fillna(0.0)
            gp_df.loc[:, 'total_pnl'] = gp_df['trading_pnl'] + gp_df['holding_pnl']
        
            return gp_df
    
        gp = merge.groupby(by='symbol')
        res = gp.apply(_apply)
    
        self.daily = res

    @staticmethod
    def _pivot_and_sort(df):
        data = df.unstack(level=0)
        data.columns = data.columns.swaplevel()
        col_names = ['symbol', 'field']
        data.columns.names = col_names
        data = data.sort_index(axis=1, level=col_names)
        #data.sortlevel(axis = 1, inplace=True)
        data.sort_index(level=0, axis=1, inplace=True)
        return data
        
    def get_daily(self):
        """
        Calculate daily trading statistics, including:
        - daily Buy/Sell volume
        - daily position
        - daily average holding price
        - daily trading, holding and total PnL
        
        """
        close = self.closes
        trade = self.trades
        
        # pro-process
        trade_cols = ['trade_date', 'BuyVolume', 'SellVolume', 'commission', 'position', 'AvgPosPrice', 'CumNetTurnOver','TurnOver']
    
        trade = trade.loc[:, trade_cols]
        gp = trade.reset_index().groupby(by=['symbol', 'trade_date'])
        func_last = lambda ser: ser.iat[-1]
        trade = gp.agg({'BuyVolume': np.sum, 'SellVolume': np.sum, 'commission': np.sum, 'TurnOver': np.sum,
                        'position': func_last, 'AvgPosPrice': func_last, 'CumNetTurnOver': func_last})
        trade.index.names = ['symbol', 'trade_date']

        # get daily position
        df_position = trade['position'].unstack('symbol').fillna(method='ffill').fillna(0.0)
        daily_position = df_position.reindex(close.index)
        daily_position = daily_position.fillna(method='ffill').fillna(0)
        self.daily_position = daily_position
        
        # calculate statistics
        close = pd.DataFrame(close.T.stack())
        close.columns = ['close']     
        close.index.names = ['symbol', 'trade_date']   
        adj_close = pd.DataFrame(self.closes_adj.T.stack())
        adj_close.columns = ['close_adj']
        adj_close.index.names = ['symbol', 'trade_date']         
        merge = pd.concat([close, adj_close, trade], axis=1, join='outer')
        
        # def _apply(gp_df,inst_map):
        #     symbol = gp_df.index.levels[0][0]
        #     mult = self.inst_map.get(symbol).get("multiplier")
        #     cols_nan_to_zero = ['BuyVolume', 'SellVolume', 'commission']
        #     cols_nan_fill = ['close', 'position', 'AvgPosPrice', 'CumNetTurnOver']
        #     # merge: pd.DataFrame
        #     gp_df.loc[:, cols_nan_fill] = gp_df.loc[:, cols_nan_fill].fillna(method='ffill')
        #     gp_df.loc[:, cols_nan_fill] = gp_df.loc[:, cols_nan_fill].fillna(0)
        #
        #     gp_df.loc[:, cols_nan_to_zero] = gp_df.loc[:, cols_nan_to_zero].fillna(0)
        #
        #     mask = gp_df.loc[:, 'AvgPosPrice'] < 1e-5
        #     gp_df.loc[mask, 'AvgPosPrice'] = gp_df.loc[mask, 'close']
        #
        #     gp_df.loc[:, 'CumProfit']     = gp_df.loc[:, 'CumNetTurnOver'] + mult * gp_df.loc[:, 'position'] * gp_df.loc[:, 'close']
        #     gp_df.loc[:, 'CumProfitComm'] = gp_df['CumProfit'] - gp_df['commission'].cumsum()
        #
        #     daily_net_turnover = gp_df['CumNetTurnOver'].diff(1).fillna(gp_df['CumNetTurnOver'].iat[0])
        #     daily_position_change = gp_df['position'].diff(1).fillna(gp_df['position'].iat[0])
        #     gp_df['trading_pnl']      = (daily_net_turnover + mult * gp_df['close'] * daily_position_change - gp_df['commission'])
        #     gp_df['holding_pnl']      = (mult * gp_df['close'].diff(1) * gp_df['position'].shift(1)).fillna(0.0)
        #     gp_df.loc[:, 'total_pnl'] = gp_df['trading_pnl'] + gp_df['holding_pnl']
        #     gp_df['trade_shares']     = daily_position_change
        #
        #     return gp_df

        # gp = merge.groupby(by='symbol')
        # res = gp.apply(_apply,self.inst_map)a.columns
        # self.daily = res


        prof_tmp = prof_sample_begin("get_daily")

        def get_ts(dv, field):
            df = dv.loc[:, pd.IndexSlice[slice(None), field]]
            df.columns = df.columns.droplevel(1)
            return df

        def set_ts(dv, field, df):
            df.columns = pd.MultiIndex.from_product([[field], df.columns])
            # df = df.sort_index(axis=1)

            dv.columns = dv.columns.swaplevel()
            #dv = dv.sort_index(axis=1)

            if field not in dv.columns.levels[0]:
                new_cols = dv.columns.append(df.columns)
                dv = dv.reindex(columns=new_cols)
            dv[field] = df[field]
            dv.columns = dv.columns.swaplevel()
            #the_data = the_data.sort_index(axis=1)

            return dv

        def build_daily(dv, inst_map):

            if False:
                mult = get_ts(dv, 'close')
                for symbol in mult.columns:
                    mult[symbol] = 1.0#inst_map.get(symbol).get("multiplier")
            else:
                mult = 1.0

            cols_nan_to_zero = ['BuyVolume', 'SellVolume', 'commission']
            cols_nan_fill = ['close', 'position', 'AvgPosPrice', 'CumNetTurnOver']

            for col in cols_nan_fill:
                df = get_ts(dv, col)
                df = df.fillna(method='ffill').fillna(0)
                dv = set_ts(dv, col, df)

            # # merge: pd.DataFrame
            # gp_df.loc[:, cols_nan_fill] = gp_df.loc[:, cols_nan_fill].fillna(method='ffill')
            # gp_df.loc[:, cols_nan_fill] = gp_df.loc[:, cols_nan_fill].fillna(0)

            for col in cols_nan_to_zero:
                df = get_ts(dv, col)
                df = df.fillna(0)
                dv = set_ts(dv, col, df)
            #gp_df.loc[:, cols_nan_to_zero] = gp_df.loc[:, cols_nan_to_zero].fillna(0)

            df = get_ts(dv, 'AvgPosPrice').copy()
            mask = df < 1e-5
            df_close = get_ts(dv, 'close')
            df[mask] = df_close[mask]
            dv = set_ts(dv, 'AvgPosPrice', df)

            # mask = gp_df.loc[:, 'AvgPosPrice'] < 1e-5
            # gp_df.loc[mask, 'AvgPosPrice'] = gp_df.loc[mask, 'close']

            df_turnover    = get_ts(dv, 'CumNetTurnOver')
            df_position    = get_ts(dv, 'position')
            df_profit      = df_turnover + mult * df_position * df_close
            df_commisson   = get_ts(dv, 'commission')
            df_profit_comm = df_profit - df_commisson.cumsum()

            dv = set_ts(dv, 'CumProfit', df_profit)
            dv = set_ts(dv, 'CumProfitComm', df_profit_comm)

            # gp_df.loc[:, 'CumProfit'] = gp_df.loc[:, 'CumNetTurnOver'] + mult * gp_df.loc[:, 'position'] * gp_df.loc[:,'close']
            # gp_df.loc[:, 'CumProfitComm'] = gp_df['CumProfit'] - gp_df['commission'].cumsum()

            daily_net_turnover    = df_turnover.diff(1)#.fillna(df_turnover.iat[0])
            daily_net_turnover.iloc[0] = df_turnover.iloc[0]

            daily_position_change = df_position.diff(1)#.fillna(gp_df['position'].iat[0])
            daily_position_change.iloc[0] = df_position.iloc[0]

            # daily_net_turnover = gp_df['CumNetTurnOver'].diff(1).fillna(gp_df['CumNetTurnOver'].iat[0])
            # daily_position_change = gp_df['position'].diff(1).fillna(gp_df['position'].iat[0])

            df_trading_pnl = (daily_net_turnover + mult * df_close * daily_position_change - df_commisson)
            df_holding_pnl = (mult * df_close.diff(1) * df_position.shift(1)).fillna(0.0)
            df_total_pnl = df_trading_pnl + df_holding_pnl

            dv = set_ts(dv, 'trading_pnl',  df_trading_pnl)
            dv = set_ts(dv, 'holding_pnl',  df_holding_pnl)
            dv = set_ts(dv, 'total_pnl',    df_total_pnl)
            dv = set_ts(dv, 'trade_shares', daily_position_change)


            # gp_df['trading_pnl'] = (daily_net_turnover + mult * gp_df['close'] * daily_position_change - gp_df['commission'])
            # gp_df['holding_pnl'] = (mult * gp_df['close'].diff(1) * gp_df['position'].shift(1)).fillna(0.0)
            # gp_df.loc[:, 'total_pnl'] = gp_df['trading_pnl'] + gp_df['holding_pnl']
            # gp_df['trade_shares'] = daily_position_change
            #
            # return gp_df

            return dv

        dv = merge.copy()
        dv.index =dv.index.swaplevel()
        dv = dv.unstack()
        dv.columns = dv.columns.swaplevel()

        dv = build_daily(dv, self.inst_map)

        # Index(date) : Column(code/field) -> Index(code/date) : Column(field)
        a = dv.copy()
        a.columns = a.columns.swaplevel()
        a = a.stack()
        a.index = a.index.swaplevel()
        print(a.columns)

        self.daily = a

        if self._is_alpha_analyzer and self.dataview:
            self._build_holding_data()
            self._build_portfolio_data()

        prof_tmp.end()

        self.save_data()

        print(" finished! ")

    def save_data(self):
        file_path = self.file_folder[0] + "/analyze_data.h5"
        if self.portfolio_data is not None:
            self.portfolio_data.to_hdf(file_path, "portfolio_data")
        if self.holding_data  is not None:
            self.holding_data._data.to_hdf(file_path, "holding_data")

    def load_data(self):

        file_path = self.file_folder[0] + "/analyze_data.h5"
        if not os.path.exists(file_path):
            print("load data error: no analyze_data.h5")
            return

        self._portfolio_data = pd.read_hdf(file_path, "portfolio_data")
        self._holding_data = AnalyzeView(pd.read_hdf(file_path, "holding_data"))


    def _build_holding_data(self):
        assert self.dataview, "Should have dataview"

        cols = ['trading_pnl', 'holding_pnl', 'total_pnl', 'commission', 'close','close_adj','position','trade_shares','AvgPosPrice']
        daily = self.daily.loc[:, cols]
        daily = daily.rename ( columns={'position': 'holding_shares'})

        tmp = BaseAnalyzer._pivot_and_sort(daily)
        if len(tmp.index) == 0:
            return

        holding_data = AnalyzeView(tmp)
        self._holding_data = holding_data

        start_date = holding_data._data.index[0]
        end_date   = holding_data._data.index[-1]

        #df_pos = self.get_ts("position")
        #df_pos_chg = df_pos.diff(1).fillna(0.0)
        #self.add_analyze_field(df_pos_chg, "trade_shares")

        # Copy base data from dataview, such as OHLC, vwap, volumn.
        #base_fields = "open,high,low,vwap,turnover,index_weight"
        # FIXME: tzxu 20180419 no turnover in dataview!
        base_fields = "open,high,low,vwap,index_weight,turnover2"
        tmp = self.dataview.get_ts(base_fields, keep_level=True)
        tmp = tmp.rename (columns={ 'turnover' : 'market_turnover'})
        holding_data.add_field(tmp)#, base_fields.replace("turnover", "market_turnover"))

        df_close_adj = holding_data.get_ts("close_adj")

        df_rtn = (df_close_adj.diff(1) / df_close_adj).fillna(0.0)

        s_bench_return = (self.data_benchmark.diff(1) / self.data_benchmark).fillna(0.0)
        s_bench_return = s_bench_return.rename ( columns={'close' : 'benchmark_return'} )

        df_active_return = df_rtn.sub( s_bench_return['benchmark_return'], axis=0 )

        tmp = np.where(holding_data.get_ts("holding_shares") >= 1, 1, 0)
        df_active_return = df_active_return * tmp
        df_rtn = df_rtn * tmp

        df_bench_return = df_active_return - df_active_return
        df_bench_return = df_bench_return.add(s_bench_return['benchmark_return'], axis=0 )
		
        holding_data.add_field (df_rtn,           "holding_return")
        holding_data.add_field (df_active_return, "active_holding_return")
        holding_data.add_field (df_bench_return,  "benchmark_return")

        df_fillprice   = holding_data.get_ts("AvgPosPrice")
        df_tradeshares = holding_data.get_ts("trade_shares")
        df_turnover    =  df_fillprice * df_tradeshares
        holding_data.add_field(df_turnover, "strategy_turnover")
        
        df_mktvalue = holding_data.get_ts("holding_shares") * holding_data.get_ts("close")
        df_mktvalue = df_mktvalue.abs()
        total_mktvalue = df_mktvalue.apply(lambda x: x.sum(), axis=1)
        df_weight = df_mktvalue.div(total_mktvalue, axis=0)
        holding_data.add_field(df_weight, "weight")

        for i in range(5):
            day = i + 1
            field_name = 'T+{0}'.format(day )

            field_value = df_weight * df_active_return.rolling(day).sum().shift(-day)
            #field_value = df_weight * df_active_return.shift(-day)
            holding_data.add_field(field_value, field_name)

        holding_data._data = holding_data._data.drop(
            ['AvgPosPrice', 'BuyVolume', 'CumNetTurnOver', 'CumProfit', 'CumProfitComm', 'SellVolume'], axis=1, level=1)

    def _build_portfolio_data(self):
        assert self.dataview, "Should have dataview"
        assert self._holding_data != None, "Should build holding data firstly"

        df_weight = self._holding_data.get_ts("weight").shift()
        df = pd.DataFrame()

        df['holding_return']          = (self._holding_data.get_ts("holding_return")        * df_weight).apply(lambda x: x.sum(), axis=1)
        df['active_holding_return']   = (self._holding_data.get_ts("active_holding_return") * df_weight).apply(lambda x: x.sum(), axis=1)
        df['holding_pnl']             = (self._holding_data.get_ts("holding_pnl") ).apply(lambda x: x.sum(), axis=1)
        df['trading_pnl']             = (self._holding_data.get_ts("trading_pnl") ).apply(lambda x: x.sum(), axis=1)

        self._portfolio_data = df

    def get_returns(self, compound_return=False, consider_commission=True, show_turnover_ratio=True):
        """
        Calculate strategy daily return and various metrics indicating strategy's performance.
        
        Parameters
        ----------
        compound_return : bool
            If True, we will calculate compound return. Otherwise non-compound (just sum).
        consider_commission : bool
            If True, we will consider commission when calculating PnL curve. Otherwise no commission.
            Note: commission is stored in a column in self.trades.

        """
        # only get columns that we need
        cols = ['trading_pnl', 'holding_pnl', 'total_pnl', 'commission', 'CumProfitComm', 'CumProfit', 'TurnOver']
        '''
        dic_symbol = {sec: self.inst_map[sec]['multiplier'] * df_daily.loc[:, cols]
                      for sec, df_daily in self.daily.items()}
                
        df_profit = pd.concat(dic_symbol, axis=1)  # this is cumulative profit
        df_profit = df_profit.fillna(method='ffill').fillna(0.0)
        df_pnl = df_profit.stack(level=1)
        df_pnl = df_pnl.sum(axis=1)
        df_pnl = df_pnl.unstack(level=1)
        '''
        
        daily = self.daily.loc[:, cols]
        daily = daily.stack().unstack('symbol')
        
        # calculate daily PnL DataFrame
        df_pnl = daily.sum(axis=1)
        df_pnl = df_pnl.unstack(level=1)
        self.df_pnl = df_pnl
    
        # TODO temperary solution
       
        strategy_value = df_pnl['total_pnl'].cumsum() + self.init_balance
            
        # get strategy & benchmark NAV (Net Asset Value)
        market_values = pd.concat([strategy_value, self.data_benchmark], axis=1).fillna(method='ffill')
        market_values.columns = ['strat', 'bench']
    
        # get strategy & benchmark daily return, cumulative return
        df_returns = market_values.pct_change(periods=1).fillna(0.0)
        df_cum_returns = (df_returns.loc[:, ['strat', 'bench']] + 1.0).cumprod()
        df_returns = df_returns.join(df_cum_returns, rsuffix='_cum')
        df_returns.loc[:,'turnover_ratio'] = df_pnl.loc[:,"TurnOver"] / strategy_value
        
        if compound_return:
            df_returns.loc[:, 'active_cum'] = df_returns['strat_cum'] - df_returns['bench_cum'] + 1
            df_returns.loc[:, 'active'] = df_returns['active_cum'].pct_change(1).fillna(0.0)
        else:
            df_returns.loc[:, 'active'] = df_returns['strat'] - df_returns['bench']
            df_returns.loc[:, 'active_cum'] = df_returns['active'].cumsum() + 1.0
    
        start = pd.to_datetime(self.start_date, format="%Y%m%d")
        end = pd.to_datetime(self.end_date, format="%Y%m%d")
        years = (end - start).days / 365.0
        
        active_cum = df_returns['active_cum'].values
        cum_peak = np.maximum.accumulate(active_cum)
        dd_to_cum_peak = (cum_peak - active_cum) / cum_peak
        max_dd_end = np.argmax(dd_to_cum_peak)  # end of the period
        max_dd_start = np.argmax(active_cum[:max_dd_end])  # start of period
        max_dd = dd_to_cum_peak[max_dd_end]
        
        win_count = len(df_pnl[df_pnl.total_pnl > 0.0].index)
        lose_count = len(df_pnl[df_pnl.total_pnl < 0.0].index)
        total_count = len(df_pnl.index)
        win_rate = win_count * 1.0 / total_count
        lose_rate = lose_count * 1.0 / total_count
        
        max_pnl   = df_pnl.loc[:,'total_pnl'].nlargest(1)
        min_pnl   = df_pnl.loc[:,'total_pnl'].nsmallest(1)
        up5pct    = df_pnl.loc[:,'total_pnl'].quantile(0.95)
        low5pct   = df_pnl.loc[:,'total_pnl'].quantile(0.05)
        top5_pnl  = df_pnl.loc[:,'total_pnl'].nlargest(5)
        tail5_pnl = df_pnl.loc[:,'total_pnl'].nsmallest(5)
    
        if compound_return:
            self.performance_metrics['Annual Return (%)'] = \
                100 * (np.power(df_returns.loc[:, 'active_cum'].values[-1], 1. / years) - 1)
        else:
            self.performance_metrics['Annual Return (%)'] = \
                100 * (df_returns.loc[:, 'active_cum'].values[-1] - 1.0) / years
        self.performance_metrics['Annual Volatility (%)'] =\
            100 * (df_returns.loc[:, 'active'].std() * np.sqrt(common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR))
        self.performance_metrics['Sharpe Ratio'] = (self.performance_metrics['Annual Return (%)']
                                                    / self.performance_metrics['Annual Volatility (%)'])

        # self.performance_metrics['Annual Holding Alpha (%)'] = np.mean(self._alpha_decomposition['total_holding_alpha']) * 242/100
        # self.performance_metrics['Annual Volatility of Holding Alpha (%)'] = np.std(self._alpha_decomposition['total_holding_alpha']) * sqrt(242)/100
        # self.performance_metrics['Sharpe Ratio of Holding Alpha'] = self.performance_metrics['Annual Holding Alpha (%)']/self.performance_metrics['Annual Volatility of Holding Alpha (%)']
        #
        # self.performance_metrics['Annual Industry Holding Alpha (%)'] = np.mean(self._alpha_decomposition['industry_holding_alpha']) * 242/100
        # self.performance_metrics['Annual Volatility of Industry Holding Alpha (%)'] = np.std(self._alpha_decomposition['industry_holding_alpha']) * sqrt(242)/100
        # self.performance_metrics['Sharpe Ratio of Industry Holding Alpha'] = self.performance_metrics['Annual Industry Holding Alpha (%)']/self.performance_metrics['Annual Volatility of Industry Holding Alpha (%)']
        #
        # self.performance_metrics['Annual Stock Specific Holding Alpha (%)'] = np.mean(self._alpha_decomposition['stock_specific_holding_alpha']) * 242/100
        # self.performance_metrics['Annual Volatility of Stock Specific Holding Alpha (%)'] = np.std(self._alpha_decomposition['stock_specific_holding_alpha']) * sqrt(242)/100
        # self.performance_metrics['Sharpe Ratio of Stock Specific Holding Alpha'] = self.performance_metrics['Annual Stock Specific Holding Alpha (%)']/self.performance_metrics['Annual Volatility of Stock Specific Holding Alpha (%)']

        self.performance_metrics['Number of Trades']   = len(self.trades.index)
        self.performance_metrics['Total PNL']          = df_pnl.loc[:,'total_pnl'].sum()
        self.performance_metrics['Daily Win Rate(%)']  = win_rate*100
        self.performance_metrics['Daily Lose Rate(%)'] = lose_rate*100
        self.performance_metrics['Commission']         = df_pnl.loc[:,'commission'].sum()
        self.performance_metrics['Turnover Ratio']         = df_returns.loc[:,'turnover_ratio'].sum() / years
        
        self.risk_metrics['Beta'] = np.corrcoef(df_returns.loc[:, 'bench'], df_returns.loc[:, 'strat'])[0, 1]
        self.risk_metrics['Maximum Drawdown (%)']   = max_dd * TO_PCT
        self.risk_metrics['Maximum Drawdown start'] = df_returns.index[max_dd_start]
        self.risk_metrics['Maximum Drawdown end']   = df_returns.index[max_dd_end]

        #self.performance_metrics_report = sorted([(k,v) for (k,v) in self.performance_metrics.items()])
        self.performance_metrics_report = []
        self.performance_metrics_report.append(('Annual Return (%)',        "{:,.2f}".format( self.performance_metrics['Annual Return (%)']))     )
        self.performance_metrics_report.append(('Annual Volatility (%)',    "{:,.2f}".format( self.performance_metrics['Annual Volatility (%)'])) )
        self.performance_metrics_report.append(('Sharpe Ratio',             "{:,.2f}".format( self.performance_metrics['Sharpe Ratio']))          )

        # self.performance_metrics_report.append(('Annual Holding Alpha (%)',        "{:,.2f}".format( self.performance_metrics['Annual Holding Alpha (%)']))     )
        # self.performance_metrics_report.append(('Annual Volatility of Holding Alpha (%)',    "{:,.2f}".format( self.performance_metrics['Annual Volatility of Holding Alpha (%)'])) )
        # self.performance_metrics_report.append(('Sharpe Ratio of Holding Alpha',    "{:,.2f}".format( self.performance_metrics['Sharpe Ratio of Holding Alpha'])) )
        #
        # self.performance_metrics_report.append(('Annual Industry Holding Alpha (%)',        "{:,.2f}".format( self.performance_metrics['Annual Industry Holding Alpha (%)']))     )
        # self.performance_metrics_report.append(('Annual Volatility of Industry Holding Alpha (%)',    "{:,.2f}".format( self.performance_metrics['Annual Volatility of Industry Holding Alpha (%)'])) )
        # self.performance_metrics_report.append(('Sharpe Ratio of Industry Holding Alpha',    "{:,.2f}".format( self.performance_metrics['Sharpe Ratio of Industry Holding Alpha'])) )
        #
        # self.performance_metrics_report.append(('Annual Stock Specific Holding Alpha (%)',        "{:,.2f}".format( self.performance_metrics['Annual Stock Specific Holding Alpha (%)']))     )
        # self.performance_metrics_report.append(('Annual Volatility of Stock Specific Holding Alpha (%)',    "{:,.2f}".format( self.performance_metrics['Annual Volatility of Stock Specific Holding Alpha (%)'])) )
        # self.performance_metrics_report.append(('Sharpe Ratio of Stock Specific Holding Alpha',    "{:,.2f}".format( self.performance_metrics['Sharpe Ratio of Stock Specific Holding Alpha'])) )

        if show_turnover_ratio:
            self.performance_metrics_report.append(('Annual Turnover Ratio',               "{:,.2f}".format( self.performance_metrics['Turnover Ratio']))    )
        self.performance_metrics_report.append(('Total PNL',                "{:,.2f}".format( self.performance_metrics['Total PNL']))             )
        self.performance_metrics_report.append(('Commission',               "{:,.2f}".format( self.performance_metrics['Commission']))            )
        self.performance_metrics_report.append(('Number of Trades',         self.performance_metrics['Number of Trades']))
        self.performance_metrics_report.append(('Daily Win Rate(%)',        "{:,.2f}".format( self.performance_metrics['Daily Win Rate(%)']))     )
        self.performance_metrics_report.append(('Daily Lose Rate(%)',       "{:,.2f}".format( self.performance_metrics['Daily Lose Rate(%)']))    )
        
        self.dailypnl_metrics_report = []
        self.dailypnl_metrics_report.append(('Daily PNL Max Time', max_pnl.index[0]))
        self.dailypnl_metrics_report.append(('Daily PNL Max',      "{:,.2f}".format(max_pnl.values[0])))
        self.dailypnl_metrics_report.append(('Daily PNL Min Time', min_pnl.index[0]))
        self.dailypnl_metrics_report.append(('Daily PNL Min',      "{:,.2f}".format(min_pnl.values[0])))
        self.dailypnl_metrics_report.append(('Daily PNL Up  5%',   "{:,.2f}".format(up5pct)))
        self.dailypnl_metrics_report.append(('Daily PNL Low 5%',   "{:,.2f}".format(low5pct)))
        
        self.dailypnl_tail5_metrics_report = []
        for k,v in tail5_pnl.iteritems():
            self.dailypnl_tail5_metrics_report.append((k, "{:,.2f}".format(v)))
            
        self.dailypnl_top5_metrics_report = []
        for k,v in top5_pnl.iteritems():
            self.dailypnl_top5_metrics_report.append((k,"{:,.2f}".format(v)))
                                               
        #self.risk_metrics_report = sorted([(k, v) for (k, v) in self.risk_metrics.items()])
        self.risk_metrics_report = []
        self.risk_metrics_report.append(("Beta",                   "{:,.3f}".format( self.risk_metrics["Beta"])) )
        self.risk_metrics_report.append(("Maximum Drawdown (%)",   "{:,.2f}".format( self.risk_metrics["Maximum Drawdown (%)"])) )
        self.risk_metrics_report.append(("Maximum Drawdown start", self.risk_metrics["Maximum Drawdown start"]))
        self.risk_metrics_report.append(("Maximum Drawdown end",   self.risk_metrics["Maximum Drawdown end"]))

        # bt_strat_mv = pd.read_csv('bt_strat_mv.csv').set_index('trade_date')
        # df_returns = df_returns.join(bt_strat_mv, how='right')
        self.returns = df_returns
    
    def plot_alpha_decay(self, df, output_folder):
        fig, ax = plt.subplots(figsize=(16, 8))
        plt.bar(df.index, df.values)
        fig.savefig(os.path.join(output_folder, 'alpha_decay.png'), facecolor=fig.get_facecolor(), dpi=fig.get_dpi())

        plt.close(fig)

    def plot_alpha_decay_weight(self, df, output_folder):
        fig, ax = plt.subplots(figsize=(16, 8))
        plt.bar(df.index, df.values)
        fig.savefig(os.path.join(output_folder, 'alpha_decay_weight.png'), facecolor=fig.get_facecolor(), dpi=fig.get_dpi())

        plt.close(fig)

    def plot_cum_alpha_weight(self, df, output_folder):
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(range(len(df)), df.mean_weight.cumsum(), lw=1, color='dodgerblue',
                label='Cumulative weight')
        ax.plot(range(len(df)), df.alpha_ratio.cumsum(), lw=1, color='darkorchid',
                label='Cumulative Alpha')
        ax.legend()
        ax.grid()
        fig.savefig(os.path.join(output_folder, 'cum_alpha_weight.png'), facecolor=fig.get_facecolor(), dpi=fig.get_dpi())

        plt.close(fig)

    def plot_pnl(self, output_folder):
        """
        Plot 2 graphs:
            1. Percentage return of strategy, benchmark and strategy's excess part.
            2. Daily trading, holding and total PnL.

        Parameters
        ----------
        output_folder : str
            Output folder of the PnL image.

        """
        old_mpl_rcparams = {k: v for k, v in mpl.rcParams.items()}
        mpl.rcParams.update(MPL_RCPARAMS)

        fig1 = plot_portfolio_bench_pnl(self.returns.loc[:, 'strat_cum'],
                                        self.returns.loc[:, 'bench_cum'],
                                        self.returns.loc[:, 'active_cum'],
                                        self.risk_metrics['Maximum Drawdown start'],
                                        self.risk_metrics['Maximum Drawdown end'])
        fig1.savefig(os.path.join(output_folder, 'pnl_img.png'), facecolor=fig1.get_facecolor(), dpi=fig1.get_dpi())
        plt.close(fig1)

        fig2 = plot_daily_trading_holding_pnl(self.df_pnl['trading_pnl'],
                                              self.df_pnl['holding_pnl'],
                                              self.df_pnl['total_pnl'],
                                              self.df_pnl['total_pnl'].cumsum())
        fig2.savefig(os.path.join(output_folder, 'pnl_img_trading_holding.png'), facecolor=fig2.get_facecolor(),
                     dpi=fig2.get_dpi())
        plt.close(fig2)

        mpl.rcParams.update(old_mpl_rcparams)

    def plot_sold_alpha_decay(self, df, output_folder):
        fig, ax = plt.subplots(figsize=(16, 8))
        fig, ax = plt.subplots(figsize=(16, 8))
        plt.bar(df.index, df['alpha'].values)

        fig.savefig(os.path.join(output_folder, self._sold_alpha_decay_image), facecolor=fig.get_facecolor(), dpi=fig.get_dpi())

        plt.close(fig)

    def analyze_alpha_decay(self,result_dir):

        # 个股平均持仓
        df_weight = self.holding_data.get_ts('weight')
        df_weight = pd.DataFrame(df_weight.mean(axis=0))
        df_weight.columns = ['mean_weight']
        df_weight = pd.merge(left=df_weight, right=self.dataview.data_inst[['name']], left_index=True, right_index=True,
                             how='left')
        df_weight = df_weight[df_weight['mean_weight'] > 0]
        #df_weight.sort_values('mean_weight', ascending=False)

        # 个股Alpha合计
        raw_alpha = self.holding_data.get_ts('active_holding_return')
        raw_weight = self.holding_data.get_ts('weight')

        df_alpha = raw_alpha.mul(raw_weight)
        df_alpha = pd.DataFrame(df_alpha.sum(axis=0))
        df_alpha.columns = ['alpha']
        df_alpha = pd.merge(left=df_alpha, right=self.dataview.data_inst[['name']], left_index=True, right_index=True, how='left')
        df_alpha = df_alpha[df_alpha['alpha'] != 0]
        df_alpha.sort_values('alpha', ascending=False, inplace=True)
        df_alpha_weight = pd.concat([df_weight, df_alpha], axis=1)

        # Alpha在个股中分布
        n_group = 5
        df_alpha_weight = df_alpha_weight.sort_values('mean_weight', ascending=False)
        df_alpha_weight['rank'] = range(len(df_alpha_weight))
        df_alpha_weight['group'] = df_alpha_weight['rank'].apply(lambda x: int(x / n_group))
        df_alpha_weight['alpha_ratio'] = df_alpha_weight['alpha'] / df_alpha_weight['alpha'].sum()

        df_alpha_group = df_alpha_weight.groupby('group')['alpha_ratio'].sum()
        df_weight_group = df_alpha_weight.groupby('group')['mean_weight'].sum()
        df_alpha_weight_group = df_alpha_group / df_weight_group

        #df_alpha_weight_group.plot.bar(figsize=(16, 8), grid=True)
        self.plot_alpha_decay(df_alpha_group,result_dir)
        self._alpha_decay_image = "alpha_decay.png"

        self.plot_alpha_decay_weight(df_alpha_weight_group,result_dir)
        self._alpha_decay_weight_image = "alpha_decay_weight.png"

        self.plot_cum_alpha_weight(df_alpha_weight,result_dir)
        self._cum_alpha_weight_image = "cum_alpha_weight.png"

    def analyze_alpha_contribution(self, result_dir):
        """
        Calculate alpha contribution from industry and specific stocks
        """
        if 'sw1' not in self.dataview.data_d.columns.levels[1]:
            print("Ignore industry overwight analysis for missing sw1 in dataview")
            return

        if not self.data_api:
            print("Ignore industry overwight analysis for missing data_api in dataview")
            return

        # Get the start and end date
        START_DATE, END_DATE = self.configs['start_date'], self.configs['end_date']
        # Get stock weight in the portfolio
        df_weight = self.holding_data.get_ts('weight').loc[START_DATE:END_DATE, :]
        df_weight = pd.DataFrame(df_weight.mean(axis=0))
        df_weight.columns = ['mean_weight']
        df_weight = pd.merge(left=df_weight, right=self.dataview.data_inst[['name']], left_index=True, right_index=True,
                             how='left')
        df_weight = df_weight[df_weight['mean_weight'] > 0]

        # Calculate industry weight in the portfolio
        raw_weight = self.holding_data.get_ts('weight')[df_weight.index].loc[START_DATE:END_DATE, :]
        index = self.dataview.get_ts('sw1')[df_weight.index].loc[START_DATE:END_DATE, :]
        index = index.loc[raw_weight.index]

        matching = {
            '110000': 'NLMY',
            '210000': 'Digging',
            '220000': 'Chemistry',
            '230000': 'Metal',
            '240000': 'Nonferrous Metal',
            '270000': 'Electronic',
            '280000': 'Car',
            '330000': 'Appliance',
            '340000': 'Food',
            '350000': 'Clothing',
            '360000': 'Light Industrials',
            '370000': 'Health Care',
            '410000': 'Utility',
            '420000': 'Transportation',
            '430000': 'Housing',
            '450000': 'Commercial',
            '460000': 'Service',
            '480000': 'Bank',
            '490000': 'Non bank',
            '510000': 'Others',
            '610000': 'Construction Material',
            '620000': 'Construction decoration',
            '630000': 'Electronic equipment',
            '640000': 'Mechenical equipment',
            '650000': 'Army',
            '710000': 'IT',
            '720000': 'Media',
            '730000': 'Telecom',
            'nan': 'Unclassified'
        }

        for key, value in matching.items():
            index = index.replace(key, value)

        def group_sum(df, group_daily):
            groups = np.unique(group_daily.values.flatten())
            mask = pd.isnull(groups)
            groups = groups[np.logical_not(mask)]
            res = pd.DataFrame(index=df.index, columns=groups, data=np.nan)
            for g in groups:
                mask = group_daily == g
                tmp = df[mask]
                res.loc[:, g] = tmp.sum(axis=1)
            return res

        weight_portfolio = group_sum(raw_weight, index)

        # Calculate index weight in the portfolio
        raw_index_weight = self.dataview.get_ts('index_weight').loc[START_DATE:END_DATE, :]
        raw_industry = self.dataview.get_ts('sw1').loc[START_DATE:END_DATE, :]

        for key, value in matching.items():
            raw_industry = raw_industry.replace(key, value)

        weight_index = group_sum(raw_index_weight, raw_industry)
        weight_overweight = weight_portfolio - weight_index

        # Get the SW industry daily returns
        df_industry, msg = self.data_api.query(view="jz.industryIndexDaily",
                                         filter="start_date=%d&end_date=%d" % (START_DATE, END_DATE), fields='preclose')

        mapping = {
            '801010.SI': 'NLMY',
            '801020.SI': 'Digging',
            '801030.SI': 'Chemistry',
            '801040.SI': 'Metal',
            '801050.SI': 'Nonferrous Metal',
            '801080.SI': 'Electronic',
            '801880.SI': 'Car',
            '801110.SI': 'Appliance',
            '801120.SI': 'Food',
            '801130.SI': 'Clothing',
            '801140.SI': 'Light Industrials',
            '801150.SI': 'Health Care',
            '801160.SI': 'Utility',
            '801170.SI': 'Transportation',
            '801180.SI': 'Housing',
            '801200.SI': 'Commercial',
            '801210.SI': 'Service',
            '801780.SI': 'Bank',
            '801790.SI': 'Non bank',
            '801230.SI': 'Others',
            '801710.SI': 'Construction Material',
            '801720.SI': 'Construction decoration',
            '801730.SI': 'Electronic equipment',
            '801890.SI': 'Mechenical equipment',
            '801740.SI': 'Army',
            '801750.SI': 'IT',
            '801760.SI': 'Media',
            '801770.SI': 'Telecom'
        }

        for key, value in mapping.items():
            df_industry = df_industry.replace(key, value)

        # Calculate industry daily return
        df_industry['ret'] = (df_industry['close'] - df_industry['preclose']) / df_industry['preclose']
        df_industry['trade_date'] = df_industry['trade_date'].apply(lambda x: int(x))
        df_industry_ret = df_industry.pivot_table(index='trade_date', columns='symbol', values='ret')

        # Calculate index daily return
        index_ret = self.dataview.data_benchmark.pct_change()
        index_ret.columns = ['index_ret']
        index_ret = index_ret.loc[df_industry_ret.index]

        START_DATE, END_DATE = weight_overweight.index[0], weight_overweight.index[-1]
        
        # Calculate index and industry period return from START_DATE to END_DATE
        df_industry_period_start = df_industry[df_industry['trade_date'] == START_DATE][['close', 'symbol']]
        df_industry_period_end   = df_industry[df_industry['trade_date'] == END_DATE][['close', 'symbol']]
        df_industry_period       = pd.merge(left = df_industry_period_start, right = df_industry_period_end,
                                            how = 'left', on = 'symbol', suffixes=('_start', '_end'))
        df_industry_period['period_ret'] = (df_industry_period['close_end'] - df_industry_period['close_start'])/df_industry_period['close_start']

        df_index_period = (self.dataview.data_benchmark.loc[END_DATE] - self.dataview.data_benchmark.loc[START_DATE])/self.dataview.data_benchmark.loc[START_DATE]
        df_industry_period['index_ret'] = df_index_period[0]
        df_industry_period['period_alpha'] = df_industry_period['period_ret'] - df_industry_period['index_ret']
        df_industry_period['period_alpha'] *= 100
        df_industry_period = df_industry_period.set_index('symbol')

        # Calculate industry daily alpha
        df_industry_alpha = pd.concat([df_industry_ret, index_ret], axis=1)
        df_industry_alpha = df_industry_alpha.sub(df_industry_alpha['index_ret'], axis=0)
        del df_industry_alpha['index_ret']

        weight_overweight_shift = weight_overweight.shift(1)
        df_industry_alpha_stock = weight_overweight_shift.mul(df_industry_alpha)
        df_industry_alpha_daily = df_industry_alpha_stock.sum(axis=1)

        # Get the total alpha on each day
        raw_alpha = self.holding_data.get_ts('active_holding_return').loc[START_DATE:END_DATE, :]
        raw_weight = self.holding_data.get_ts('weight').loc[START_DATE:END_DATE, :]
        df_alpha = raw_alpha.mul(raw_weight)
        total_alpha = df_alpha.sum(axis=1)

        df_alpha_all = pd.concat([total_alpha, df_industry_alpha_daily], axis=1)
        df_alpha_all.columns = ['total_holding_alpha', 'industry_holding_alpha']
        df_alpha_all['stock_specific_holding_alpha'] = df_alpha_all['total_holding_alpha'] - df_alpha_all['industry_holding_alpha']
        df_alpha_all *= 10000

        df_alpha_sum = df_alpha_all.sum()/100
        df_alpha_mean = df_alpha_all.mean() * 242 / 100
        df_alpha_std  = df_alpha_all.std() * np.sqrt(242) / 100
        df_alpha_sharpe = df_alpha_mean/df_alpha_std
        df_alpha_agg = pd.concat([df_alpha_mean, df_alpha_std, df_alpha_sharpe], axis = 1)
        df_alpha_agg.columns = ['Annual Alpha(%)', 'Annual Alpha Volatility(%)', 'IR']
        df_alpha_agg = df_alpha_agg.T
        df_alpha_agg.columns = ['Holding Alpha', 'Industry Holding Alpha', 'Stock Specific Holding Alpha']

        df_industry_alpha_byindustry = df_industry_alpha_stock.sum(axis = 0) * 100
        df_industry_agg = pd.concat([df_industry_alpha_byindustry, self._average_industry_overweight], axis = 1)
        df_industry_agg.columns = ['total alpha', 'portfolio', 'index', 'overweight']
        df_industry_agg = df_industry_agg.sort_values('total alpha', ascending = False)
        df_industry_agg = pd.concat([df_industry_agg, df_industry_period['period_alpha']], axis = 1)
        df_industry_agg['selection_alpha'] = df_industry_agg['overweight'] * df_industry_agg['period_alpha']
        df_industry_agg['timing_alpha'] = df_industry_agg['total alpha'] - df_industry_agg['selection_alpha']
        df_industry_agg = df_industry_agg[['total alpha', 'selection_alpha', 'timing_alpha', 'period_alpha', 'portfolio', 'index', 'overweight']]
        df_industry_agg.columns = ['total alpha', 'selection alpha', 'timing alpha', 'period alpha', 'portfolio', 'index', 'overweight']

        # Aggregate the alpha by month
        df_alpha_all_month_total = df_alpha_all.copy()
        df_alpha_all_month_total/= 100
        df_alpha_all_month_total['date'] = df_alpha_all_month_total.index
        df_alpha_all_month_total['year_month'] = df_alpha_all_month_total['date'].apply(lambda x: str(int(x))[:6])
        del df_alpha_all_month_total['date']
        df_alpha_all_month_total = df_alpha_all_month_total.groupby('year_month').sum()
        df_alpha_all_month_total.columns = ['Total', 'Industry', 'Stock specific']

        df_alpha_all_month_industry = df_industry_alpha_stock.copy()
        df_alpha_all_month_industry *= 100
        df_alpha_all_month_industry['date'] = df_alpha_all_month_industry.index
        df_alpha_all_month_industry['year_month'] = df_alpha_all_month_industry['date'].apply(lambda x: str(int(x))[:6])
        del df_alpha_all_month_industry['date']
        df_alpha_all_month_industry = df_alpha_all_month_industry.groupby('year_month').sum()

        self._alpha_decomposition_images = []
        self._alpha_decomposition_industry_images = []

        image_name1 = 'total_alpha_m'
        self._alpha_decomposition_images = 'total_alpha_m.png'
        self.plot_return_heatmap(df_alpha_all_month_total, image_name1, result_dir, (6,12))

        image_name2 = 'industry_alpha_m'
        self._alpha_decomposition_industry_images = 'industry_alpha_m.png'
        self.plot_return_heatmap(df_alpha_all_month_industry, image_name2, result_dir, (20,12))

        self._industry_agg = df_industry_agg
        self._alpha_decomposition = df_alpha_agg

    def analyze_alpha_weight_contribution(self, result_dir):
        """
        Get top 40% weights of stocks and calculate its alpaha contribution
        :return:
        """

        # Get Top 40% wight stocks
        mv = self.holding_data.get_ts('holding_shares') * self.holding_data.get_ts('close')
        total_mv = mv.sum(axis=1)

        for col in mv.columns:
            mv[col] /= total_mv

        weight_in_day = mv

        weight_in_cycle = weight_in_day.sum(axis=0).sort_values(ascending=False)
        tmp = weight_in_cycle.cumsum()
        weight_in_cycle = weight_in_cycle[tmp < tmp[-1] * 0.4]
        weight_in_cycle /= tmp[-1]
        symbols = weight_in_cycle.index

        # Get alpha contribution of each stock on whole test cycle
        active_return = self.holding_data.get_ts('active_holding_return') * weight_in_day
        alpha_contribution = active_return.loc[:, symbols].sum()

        df_contrib = pd.DataFrame(index=weight_in_cycle.index)
        df_contrib['symbol'] = weight_in_cycle.index
        df_contrib['name']   = df_contrib['symbol'].apply(lambda x: self.inst_map[x]['name'])
        df_contrib['weight']             = weight_in_cycle.apply(lambda x: str(np.round(x * 100, 2)) + "%")
        df_contrib['alpha_contribution'] = alpha_contribution.apply(lambda x: str(np.round(x * 100, 2)) + "%")
        df_contrib.reset_index(drop=True, inplace=True)

        holding_shares = self.holding_data.get_ts('holding_shares').sum(axis=0)

        self._alpha_weight_traded_stocks = holding_shares[holding_shares>0].shape[0]
        self._alpha_wegith_contribution = df_contrib

    def analyze_sold_alpha_decay(self, result_dir):
        """
        Take sold stocks as a portfolio and calculate next 10 days' pnl.
        """

        # Get Top 40% wight stocks
        sold_shares = self.holding_data.get_ts('holding_shares').diff()
        sold_shares *= np.where(sold_shares < 0, -1, 0)

        mv = sold_shares * self.holding_data.get_ts('close')
        total_mv = mv.sum(axis=1)

        for col in mv.columns:
            mv[col] /= total_mv

        weight_in_day = mv

        active_return = self.holding_data.get_ts('active_holding_return')

        sold_pnl = [0 for i in range(10)]
        for day in range(10):
            sold_pnl[day] = (active_return.shift(day + 1) * weight_in_day).sum(axis=1).mean()

        df = pd.DataFrame(sold_pnl)
        df.columns = ['alpha']
        df.index = [i + 1 for i in range(10)]

        self._sold_alpha_decay_image = "sold_alpha_decay.png"
        self._sold_alpha_decay = df
        self.plot_sold_alpha_decay(df, result_dir)

    def plot_return_heatmap(self, df, image_name, output_folder, figsize):

        # plot
        # df_factor_heatmap_pivot = df_factor_heatmap.pivot('year', 'month', single_factor_name)
        import seaborn as sns
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = sns.heatmap(df, ax=ax, annot=True, center=0,
                         annot_kws={"size": 10},
                         fmt="0.2f", linewidths=0.5,
                         square=False, cbar=True, cmap=cmap)
        fig.subplots_adjust(hspace=0)
        plt.yticks(rotation=0)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.tight_layout()

        fig.savefig(os.path.join(output_folder, image_name),
                    facecolor=fig.get_facecolor(),
                    dpi=fig.get_dpi())
        plt.close(fig)

    def analyze_industry_overweight(self, result_dir):
        if 'sw1' not in self.dataview.data_d.columns.levels[1]:
            print("Ignore industry overwight analysis for missing sw1 in dataview")
            return

        START_DATE, END_DATE = self.configs['start_date'], self.configs['end_date']
        # 个股平均持仓
        df_weight = self.holding_data.get_ts('weight').loc[START_DATE:END_DATE, :]
        df_weight = pd.DataFrame(df_weight.mean(axis=0))
        df_weight.columns = ['mean_weight']
        df_weight = pd.merge(left=df_weight, right=self.dataview.data_inst[['name']], left_index=True, right_index=True,
                             how='left')
        df_weight = df_weight[df_weight['mean_weight'] > 0]

        raw_weight = self.holding_data.get_ts('weight')[df_weight.index].loc[START_DATE:END_DATE, :]
        index = self.dataview.get_ts('sw1')[df_weight.index].loc[START_DATE:END_DATE, :]
        index = index.loc[raw_weight.index]

        matching = {
            '110000': 'NLMY',
            '210000': 'Digging',
            '220000': 'Chemistry',
            '230000': 'Metal',
            '240000': 'Nonferrous Metal',
            '270000': 'Electronic',
            '280000': 'Car',
            '330000': 'Appliance',
            '340000': 'Food',
            '350000': 'Clothing',
            '360000': 'Light Industrials',
            '370000': 'Health Care',
            '410000': 'Utility',
            '420000': 'Transportation',
            '430000': 'Housing',
            '450000': 'Commercial',
            '460000': 'Service',
            '480000': 'Bank',
            '490000': 'Non bank',
            '510000': 'Others',
            '610000': 'Construction Material',
            '620000': 'Construction decoration',
            '630000': 'Electronic equipment',
            '640000': 'Mechenical equipment',
            '650000': 'Army',
            '710000': 'IT',
            '720000': 'Media',
            '730000': 'Telecom',
            'nan': 'Unclassified'
        }

        for key, value in matching.items():
            index = index.replace(key, value)

        def group_sum(df, group_daily):
            groups = np.unique(group_daily.values.flatten())
            mask = pd.isnull(groups)
            groups = groups[np.logical_not(mask)]
            res = pd.DataFrame(index=df.index, columns=groups, data=np.nan)
            for g in groups:
                mask = group_daily == g
                tmp = df[mask]
                res.loc[:, g] = tmp.sum(axis=1)
            return res

        weight_industry = group_sum(raw_weight, index)

        ## 指数行业分析
        raw_index_weight = self.dataview.get_ts('index_weight').loc[START_DATE:END_DATE, :]
        raw_industry = self.dataview.get_ts('sw1').loc[START_DATE:END_DATE, :]


        for key, value in matching.items():
            raw_industry = raw_industry.replace(key, value)


        weight_index = group_sum(raw_index_weight, raw_industry)

        ## 行业超配分析
        # weight_industry['Army'] = 0.0
        # weight_industry['Clothing'] = 0.0
        # weight_industry['Others'] = 0.0

        weight_dif = weight_industry - weight_index
        index_weight_industry = pd.DataFrame(weight_index.mean(axis=0).sort_values(ascending=False))
        index_weight_industry.columns = ['index']
        #print('相对指数超配')
        self._industry_overweight_images = []
        for single_industry in weight_dif.columns:
            image_name = ('iw_' + single_industry + '.png').replace(' ', '_').lower()
            self._industry_overweight_images.append(image_name)
            self.plot_industry_weight(weight_dif, single_industry, image_name, result_dir)

        # 行业平均持仓及相对指数超配
        portfolio_weight_industry = pd.DataFrame(weight_industry.mean(axis=0).sort_values(ascending=False))
        portfolio_weight_industry.columns = ['portfolio']
        weight_industry_compare = pd.concat([portfolio_weight_industry, index_weight_industry], axis=1)
        weight_industry_compare['overweight'] = weight_industry_compare['portfolio'] - weight_industry_compare['index']
        self._average_industry_overweight = weight_industry_compare.sort_values('overweight', ascending=False)


    def plot_industry_weight(self, df, industry_name, image_name, output_folder):
        idx0 = df.index.astype(str)
        n = len(idx0)
        idx = np.arange(n)
        idx2016 = np.where(idx0 == '20160701')
        fig, ax3 = plt.subplots(figsize=(16, 8))
        # ax1 = ax0.twinx()

        bar_width = 0.2
        profit_color, lose_color = '#D63434', '#2DB635'
        curve_color = '#174F67'
        y_label = 'Daily Return'
        color_arr_raw = np.array([profit_color] * n)

        color_arr = color_arr_raw.copy()
        # color_arr[holding_pnl_tot.PnL < 0] = lose_color
        # ax3.bar(idx, df_volume_cyb.turnover_300, width = 1, color='green', label = '300', alpha = 0.4)
        # ax3.bar(idx, df_volume_cyb.turnover_cyb, width = 1, color='yellow', label = 'cyb', alpha = 0.4)

        # ax3.set(title='Cumulative and daily holding PnL', ylabel=y_label)
        ax3.xaxis.set_major_formatter(MyFormatter(idx0, '%Y%m%d'))
        ax3.legend()
        ax3.axhline(0, color='red', lw=1, ls='--')
        ax3.plot(idx, df[industry_name], color='blue', label='%s' % industry_name, linewidth=1.5, alpha=0.8)

        # ax3.plot(idx, stock_alpha['active_holding_return'].cumsum(), lw=1, color='red', label = 'alpha')
        # ax6 = ax3.twinx()
        # ax3.plot(idx, stock_alpha['weight'], lw = 1, color = 'blue', label = 'weight')
        # ax3.yaxis.label.set_color(curve_color)
        ax3.grid()
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.legend(fontsize=14)
        plt.title('%s' % industry_name, fontsize=16)
        plt.tight_layout()

        fig.savefig(os.path.join(output_folder, image_name),
                    facecolor=fig.get_facecolor(),
                    dpi=fig.get_dpi())
        plt.close(fig)

    def gen_report(self, source_dir, template_fn, out_folder='.', selected=None):
        """
        Generate HTML (and PDF) report of the trade analysis.

        Parameters
        ----------
        source_dir : str
            path of directory where HTML template and css files are stored.
        template_fn : str
            File name of HTML template.
        out_folder : str
            Output folder of report.
        selected : list of str or None
            List of symbols whose detailed PnL curve and position will be plotted.
            # TODO: this parameter should not belong to function


        """
        dic = dict()
        dic['html_title'] = "Alpha Strategy Backtest Result"
        dic['selected_securities'] = selected
        # we do not want to show username / password in report
        dic['props'] = {k: v for k, v in self.configs.items() if ('username' not in k and 'password' not in k)}
        dic['performance_metrics_report'] = self.performance_metrics_report
        dic['risk_metrics_report'] = self.risk_metrics_report
        dic['position_change'] = self.position_change
        dic['account'] = self.account
        dic['df_daily'] = jutil.group_df_to_dict(self.daily, by='symbol')
        dic['daily_position'] = None # self.daily_position
        dic['rebalance_positions'] = self.rebalance_positions
        dic['dailypnl_metrics_report'] = self.dailypnl_metrics_report
        dic['dailypnl_top5_metrics_report'] = self.dailypnl_top5_metrics_report
        dic['dailypnl_tail5_metrics_report'] = self.dailypnl_tail5_metrics_report
        dic['industry_overweight_images'] = self._industry_overweight_images
        dic['alpha_decomposition_images'] = self._alpha_decomposition_images
        dic['alpha_decomposition_industry_images'] = self._alpha_decomposition_industry_images
        dic['alpha_decay_image'] = self._alpha_decay_image
        dic['alpha_decay_weight_image'] = self._alpha_decay_weight_image
        dic['cum_alpha_weight_image'] = self._cum_alpha_weight_image
        dic['average_industry_overweight'] = self._average_industry_overweight
        dic['alpha_decomposition'] = self._alpha_decomposition
        dic['industry_agg'] = self._industry_agg

        dic['alpha_weight_contribution']  = self._alpha_wegith_contribution
        dic['alpha_weight_traded_stocks'] = self._alpha_weight_traded_stocks

        dic['sold_alpha_decay_image'] = self._sold_alpha_decay_image


        self.report_dic.update(dic)
        
        r = Report(self.report_dic, source_dir=source_dir, template_fn=template_fn, out_folder=out_folder)
        
        r.generate_html()
        r.output_html('report.html')

    def do_analyze(self, result_dir, selected_sec=None, compound_rtn = False):
        """
        Convenient function to do a series of analysis.
        The reason why define these separate steps and put them in one function is
        this function is convenient for common users but advanced users can still customize.
        
        Parameters
        ----------
        result_dir
        selected_sec

        Returns
        -------

        """
        
        jutil.create_dir(os.path.join(os.path.abspath(result_dir), 'dummy.dummy'))
        
        if selected_sec is None:
            selected_sec = []
            
        print("process trades...")
        self.process_trades()
        print("get daily stats...")

        self.get_daily()

        print("calc strategy return...")

        prof_tmp = prof_sample_begin("get_returns")
        self.get_returns(compound_return = compound_rtn, consider_commission=True)
        prof_tmp.end()

        if len(selected_sec) > 0:
            print("Plot single securities PnL")
            for symbol in selected_sec:
                df_daily = self.daily.loc[pd.IndexSlice[symbol, :], :]
                df_daily.index = df_daily.index.droplevel(0)
                if df_daily is not None:
                    plot_trades(df_daily, symbol=symbol, output_folder=result_dir)

        print("Plot strategy PnL...")
        self.plot_pnl(result_dir)
        
        print("generate report...")
        self.gen_report(source_dir=STATIC_FOLDER, template_fn='report_template.html',
                        out_folder=result_dir,
                        selected=selected_sec)

        return self.performance_metrics_report

class EventAnalyzer(BaseAnalyzer):
    def __init__(self):
        super(EventAnalyzer, self).__init__(is_alpha_analyzer = False)
        
        self.metrics = dict()
        self.daily = None
        self.data_benchmark = None
        
        self.returns = None  # OrderedDict
        self.position_change = None  # OrderedDict
        self.account = None  # OrderedDict
        
    def initialize(self, data_server_=None, dataview=None, file_folder='.'):
        super(EventAnalyzer, self).initialize(data_api=data_server_, dataview=dataview,
                                              file_folder=file_folder)
        if self.dataview is not None and self.dataview.data_benchmark is not None:
            self.data_benchmark = self.dataview.data_benchmark.loc[(self.dataview.data_benchmark.index >= self.start_date)
                                                                   &(self.dataview.data_benchmark.index <= self.end_date)]
        else:
            if self.benchmark and data_server_:
                df, msg = data_server_.daily(self.benchmark, start_date=self.closes.index[0], end_date=self.closes.index[-1])
                self.data_benchmark = df.set_index('trade_date').loc[:, ['close']]
                self.data_benchmark.columns = ['bench']
            else:
                self.data_benchmark = pd.DataFrame(index=self.closes.index, columns=['bench'], data=np.ones(len(self.closes), dtype=float))
            

class AlphaAnalyzer(BaseAnalyzer):
    def __init__(self):
        super(AlphaAnalyzer, self).__init__()
        
        self.metrics = dict()
        self.daily = None
        self.returns = None  # OrderedDict
        self.position_change = None  # OrderedDict
        self.account = None  # OrderedDict
        
        self.df_brinson = None
        
        self.data_benchmark = None

    def initialize(self, data_api=None, dataview=None, file_folder='.'):
        super(AlphaAnalyzer, self).initialize(data_api=data_api, dataview=dataview,
                                              file_folder=file_folder)
        if self.dataview is not None and self.dataview.data_benchmark is not None:
            self.data_benchmark = self.dataview.data_benchmark.loc[(self.dataview.data_benchmark.index >= self.start_date)
                                                                   &(self.dataview.data_benchmark.index <= self.end_date)]
        else:
            if self.benchmark and data_api:
                df, msg = data_api.daily(self.benchmark, start_date=self.closes.index[0], end_date=self.closes.index[-1])
                self.data_benchmark = df.set_index('trade_date').loc[:, ['close']]
                self.data_benchmark.columns = ['bench']
            else:
                self.data_benchmark = pd.DataFrame(index=self.closes.index, columns=['bench'], data=np.ones(len(self.closes), dtype=float))
    
    @staticmethod
    def _to_pct_return(arr, cumulative=False):
        """Convert portfolio value to portfolio (linear) return."""
        r = np.empty_like(arr)
        r[0] = 0.0
        if cumulative:
            r[1:] = arr[1:] / arr[0] - 1
        else:
            r[1:] = arr[1:] / arr[:-1] - 1
        return r

    def _get_index_weight(self):
        if self.dataview is not None:
            res = self.dataview.get_ts('index_weight', start_date=self.start_date, end_date=self.end_date)
        else:
            res = self.data_api.query_index_weights_daily(self.universe, self.start_date, self.end_date)
        return res
    
    def _brinson(self, close, pos, index_weight, group):
        """
        Brinson Attribution.
        
        Parameters
        ----------
        close : pd.DataFrame
            Index is date, columns are symbols.
        pos : pd.DataFrame
            Index is date, columns are symbols.
        index_weight : pd.DataFrame
            Index is date, columns are symbols.
        group : pd.DataFrame
            Index is date, columns are symbols.

        Returns
        -------
        dict

        """
        def group_sum(df, group_daily):
            groups = np.unique(group_daily.values.flatten())
            mask = np.isnan(groups.astype(float))
            groups = groups[np.logical_not(mask)]
            res = pd.DataFrame(index=df.index, columns=groups, data=np.nan)
            for g in groups:
                mask = group_daily == g
                tmp = df[mask]
                res.loc[:, g] = tmp.sum(axis=1)
            return res

        ret = close.pct_change(1)

        pos_sum = pos.sum(axis=1)
        pf_weight = pos.div(pos_sum, axis=0)
        pf_weight.loc[pos_sum == 0, :] = 0.0
        assert pf_weight.isnull().sum().sum() == 0
        pf_weight = pf_weight.reindex(index=ret.index, columns=ret.columns)
        pf_weight = pf_weight.fillna(0.0)

        weighted_ret_pf = ret.mul(pf_weight)
        weighted_ret_index = ret.mul(index_weight)

        index_group_weight = group_sum(index_weight, group)
        pf_group_weight = group_sum(pf_weight, group)

        pf_group_ret = group_sum(weighted_ret_pf, group).div(pf_group_weight)
        index_group_ret = group_sum(weighted_ret_index, group).div(index_group_weight)

        allo_ret_group = (pf_group_weight - index_group_weight).mul(index_group_ret)
        allo_ret = allo_ret_group.sum(axis=1)

        selection_ret_group = (pf_group_ret - index_group_ret).mul(index_group_weight)
        selection_ret = selection_ret_group.sum(axis=1)

        active_ret = (weighted_ret_pf.sum(axis=1) - weighted_ret_index.sum(axis=1))
        inter_ret = active_ret - selection_ret - allo_ret
    
        df_brinson = pd.DataFrame(index=allo_ret.index,
                                  data={'allocation': allo_ret,
                                        'selection': selection_ret,
                                        'interaction': inter_ret,
                                        'total_active': active_ret})
        
        return {'df_brinson': df_brinson, 'allocation': allo_ret_group, 'selection': selection_ret_group}
    
    def brinson(self, group, output_folder):
        """
        
        Parameters
        ----------
        group : str or pd.DataFrame
            If group is string, this function will try to fetch the corresponding DataFrame from DataView.
            If group is pd.DataFrame, it will be used as-is.

        Returns
        -------

        """
        if isinstance(group, basestring):
            group = self.dataview.get_ts(group, start_date=self.start_date, end_date=self.end_date)
        elif isinstance(group, pd.DataFrame):
            pass
        else:
            raise ValueError("Group must be string or DataFrame. But {} is provided.".format(group))
        
        if group is None or group.empty:
            raise ValueError("group is None or group is empty")
        
        close = self.closes_adj
        pos = self.daily_position
        index_weight = self._get_index_weight()
        
        res_dic = self._brinson(close, pos, index_weight, group)
        
        df_brinson = res_dic['df_brinson']
        self.df_brinson = df_brinson
        self.report_dic['df_brinson'] = df_brinson
        plot_brinson(df_brinson, output_folder=output_folder)

    def get_rebalance_position(self):
        mask = (self._raw_trades['fill_no'] != '101010') & (self._raw_trades['fill_no'] != '202020')
        trades_rebalance = self._raw_trades.loc[mask]
        rebalance_dates = trades_rebalance['trade_date'].unique()

        tmp = self.holding_data.get_ts('holding_shares')
        for col in tmp.columns:
            tmp[col] = self.inst_map[col]['name']

        self.holding_data.add_field(tmp, 'name')

        dic_pos = OrderedDict()
        for date in rebalance_dates:
            fields = ['name', 'holding_shares','weight', 'T+1','T+2','T+3','T+4','T+5']
            daily = self.holding_data.data.loc[[date], pd.IndexSlice[:, fields]].T.unstack()
            daily.columns = daily.columns.droplevel(level='trade_date')
            daily.columns.name = ""
            daily['symbol'] = daily.index

            daily.reset_index(drop=True, inplace=True)
            daily.index.name = ""
            daily.rename(inplace=True, columns= {'holding_shares': 'position'})

            daily = daily.sort_values('weight', ascending=False)
            for col in ['weight', 'T+1', 'T+2', 'T+3', 'T+4', 'T+5']:
                daily[col] = daily[col].apply(lambda x: str(np.round(x * 100, 2)) + "%")

            daily = daily[['symbol', 'name', 'position', 'weight', 'T+1','T+2','T+3','T+4','T+5']]

            dic_pos[date] = daily[(daily['weight'] != "0.0%") & (daily['weight'] != "nan%")]
            dic_pos[date].index = range(1, len(dic_pos[date]) + 1)

        self.rebalance_positions = dic_pos

        # daily_pos_name = self.daily_position.T.copy()
        # daily_pos_name.loc[:, 0] = u'               '
        # for idx, _ in daily_pos_name.iterrows():
        #     daily_pos_name.loc[idx, 0] = self.inst_map[idx]['name']
        #
        # dic_pos = OrderedDict()
        # for date in rebalance_dates:
        #     daily = daily_pos_name.loc[:, [0, date]]
        #     daily = daily.loc[daily[date] >= 1]
        #     daily = daily.reset_index()
        #     daily.index.name = date
        #     daily.columns = ['symbol', 'name', 'position']
        #     daily.loc[:, 'position'] = daily['position'].astype(np.integer)
        #     dic_pos[date] = daily
        # self.rebalance_positions = dic_pos
        
    @staticmethod
    def calc_win_ratio(ret_arr):
        n_total = len(ret_arr)
        n_win = (ret_arr > 0).sum()
        n_lose = (ret_arr < 0).sum()
        n_equal = n_total - n_win - n_lose
        win_ratio = n_win / n_total
        return n_win, n_total, win_ratio
        
    def get_stats(self):
        df_return = self.returns.copy()
        idx = df_return.index
        df_return.loc[:, 'daily_active'] = df_return['strat'] - df_return['bench']
        df_return.loc[:, 'month'] = jutil.date_to_month(idx)
        df_return.loc[:, 'year'] = jutil.date_to_year(idx)
        def calc_cum(df):
            return np.prod(df.add(1.0)) - 1.0
        stats_monthly = df_return.groupby('month')['daily_active'].apply(calc_cum)
        stats_yearly = df_return.groupby('year')['daily_active'].apply(calc_cum)
        '''
        plt.figure(figsize=(16, 10))
        sns.distplot(df_return['daily_active'], bins=np.arange(-5e-2, 5e-2, 1e-3), kde=False)
        plt.savefig('a.png')
        plt.close()
        '''
        print
        
    def do_analyze(self, result_dir, selected_sec=None, brinson_group=None, compound_rtn = False, show_turnover_ratio = True):
        if selected_sec is None:
            selected_sec = []
    
        print("process trades...")
        self.process_trades()
        print("get daily stats...")
        self.get_daily()
        print("calc strategy return...")
        self.get_returns(compound_return = compound_rtn, consider_commission=True, show_turnover_ratio = show_turnover_ratio)
        print("calc re-balance position")
        self.get_rebalance_position()
        print("Get stats")
        self.get_stats()
    
        not_none_sec = []
        if len(selected_sec) > 0:
            print("Plot single securities PnL")
            for symbol in selected_sec:
                df_daily = self.daily.loc[pd.IndexSlice[symbol, :], :]
                df_daily.index = df_daily.index.droplevel(0)
                if df_daily is not None:
                    not_none_sec.append(symbol)
                    plot_trades(df_daily, symbol=symbol, output_folder=result_dir)
    
        print("Plot strategy PnL...")
        self.plot_pnl(result_dir)
        
        if brinson_group is not None:
            print("Do brinson attribution.")
            group = self.dataview.get_ts(brinson_group)
            if group is None:
                raise ValueError("group data is None.")
            self.brinson(group, output_folder=result_dir)

        test_name = self.configs['Name'] if 'Name' in self.configs else "default"
        self.daily_position.to_csv(os.path.join(result_dir, 'daily_position_%s.csv' % test_name))
        self.returns.to_csv(os.path.join(result_dir, 'returns_%s.csv' % test_name))

        if self.dataview:
            print("Analyze alpha data...")
            self.analyze_alpha_decay(result_dir)
            self.analyze_industry_overweight(result_dir)
            self.analyze_alpha_contribution(result_dir)
            self.analyze_alpha_weight_contribution(result_dir)
            self.analyze_sold_alpha_decay(result_dir)
        else:
            print("Ignore analyzing alpha data")

        print("generate report...")
        self.gen_report(source_dir=STATIC_FOLDER, template_fn='report_template.html',
                        out_folder=result_dir,
                        selected=not_none_sec)


def plot_daily_trading_holding_pnl(trading, holding, total, total_cum):
    """
    Parameters
    ----------
    trading : pd.Series
    holding : pd.Series
    total : pd.Series
    total_cum : pd.Series
    
    """
    idx0 = total.index
    n = len(idx0)
    idx = np.arange(n)
    
    fig, (ax0, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 13.5), sharex=True)
    ax1 = ax0.twinx()
    
    bar_width = 0.4
    profit_color, lose_color = '#D63434', '#2DB635'
    curve_color = '#174F67'
    y_label = 'Profit / Loss ($)'
    color_arr_raw = np.array([profit_color] * n)
    
    color_arr = color_arr_raw.copy()
    color_arr[total < 0] = lose_color
    ax0.bar(idx, total, width=bar_width, color=color_arr)
    ax0.set(title='Daily PnL', ylabel=y_label, xlim=[-2, n+2],)
    ax0.xaxis.set_major_formatter(MyFormatter(idx0, '%y-%m-%d'))
    
    ax1.plot(idx, total_cum, lw=1.5, color=curve_color)
    ax1.set(ylabel='Cum. ' + y_label)
    ax1.yaxis.label.set_color(curve_color)

    color_arr = color_arr_raw.copy()
    color_arr[trading < 0] = lose_color
    ax2.bar(idx-bar_width/2, trading, width=bar_width, color=color_arr)
    ax2.set(title='Daily Trading PnL', ylabel=y_label)
    
    color_arr = color_arr_raw.copy()
    color_arr[holding < 0] = lose_color
    ax3.bar(idx+bar_width/2, holding, width=bar_width, color=color_arr)
    ax3.set(title='Daily Holding PnL', ylabel=y_label, xticks=idx[: : n//10 + 1])
    return fig
    
    
def plot_portfolio_bench_pnl(portfolio_cum_ret, benchmark_cum_ret, excess_cum_ret,
                             max_dd_start, max_dd_end):
    """
    Parameters
    ----------
    Series
    
    """
    n_subplots = 3
    fig, (ax1, ax2, ax3) = plt.subplots(n_subplots, 1, figsize=(16, 4.5 * n_subplots), sharex=True)
    idx_dt = portfolio_cum_ret.index
    idx = np.arange(len(idx_dt))
    
    y_label_ret = "Cumulative Return (%)"
    
    ax1.plot(idx, (benchmark_cum_ret-1) * TO_PCT, label='Benchmark', color='#174F67')
    ax1.plot(idx, (portfolio_cum_ret-1) * TO_PCT, label='Strategy', color='#198DD6')
    ax1.legend(loc='upper left')
    ax1.set(title="Absolute Return of Portfolio and Benchmark", 
            #xlabel="Date", 
            ylabel=y_label_ret)
    ax1.grid(axis='y')
    
    ax2.plot(idx, (excess_cum_ret-1) * TO_PCT, label='Extra Return', color='#C37051')
    ax2.axvspan(idx_dt.get_loc(max_dd_start), idx_dt.get_loc(max_dd_end), color='lightgreen', alpha=0.5, label='Maximum Drawdown')
    ax2.legend(loc='upper left')
    ax2.set(title="Excess Return Compared to Benchmark", ylabel=y_label_ret
            #xlabel="Date", 
            )
    ax2.grid(axis='y')
    ax2.xaxis.set_major_formatter(MyFormatter(idx_dt, '%y-%m-%d'))  # 17-09-31
    
    ax3.plot(idx, (portfolio_cum_ret ) / (benchmark_cum_ret ), label='Ratio of NAV', color='#C37051')
    ax3.legend(loc='upper left')
    ax3.set(title="NaV of Portfolio / NaV of Benchmark", ylabel=y_label_ret
           #xlabel="Date",
           )
    ax3.grid(axis='y')
    ax3.xaxis.set_major_formatter(MyFormatter(idx_dt, '%y-%m-%d'))  # 17-09-31
    
    fig.tight_layout()  
    return fig
    
    
def plot_brinson(df, output_folder):
    """
    
    Parameters
    ----------
    df : pd.DataFrame

    """
    allo, selec, inter, total = df['allocation'], df['selection'], df['interaction'], df['total_active']
    fig, ax1 = plt.subplots(1, 1, figsize=(21, 8))
    
    idx0 = df.index
    idx = range(len(idx0))

    ax1.plot(idx, selec, lw=1.5, color='indianred', label='Selection Return')
    ax1.plot(idx, allo, lw=1.5, color='royalblue', label='Allocation Return')
    ax1.plot(idx, inter, lw=1.5, color='purple', label='Interaction Return')
    # ax1.plot(idx, total, lw=1.5, ls='--', color='k', label='Total Active Return')
    
    ax1.axhline(0.0, color='k', lw=0.5, ls='--')
    
    ax1.legend(loc='upper left')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Return")
    ax1.xaxis.set_major_formatter(MyFormatter(idx0, '%Y-%m-%d'))

    plt.tight_layout()
    fig.savefig(os.path.join(output_folder, 'brinson_attribution.png'))
    plt.close()
    

def calc_avg_pos_price(pos_arr, price_arr):
    """
    Calculate average cost price using position and fill price.
    When position = 0, cost price = symbol price.
    """
    assert len(pos_arr) == len(price_arr)

    avg_price = np.zeros_like(pos_arr, dtype=float)
    avg_price[0] = price_arr[0]
    for i in range(pos_arr.shape[0] - 1):
        if pos_arr[i+1] == 0:
            avg_price[i+1] = 0.0
        else:
            pos_diff = pos_arr[i+1] - pos_arr[i]
            if pos_arr[i] == 0 or pos_diff * pos_arr[i] > 0:
                count = True
            else:
                count = False
        
            if count:
                avg_price[i+1] = (avg_price[i] * pos_arr[i] + pos_diff * price_arr[i+1]) * 1. / pos_arr[i+1]
            else:
                avg_price[i+1] = avg_price[i]
    return avg_price


def plot_trades(df, symbol="", output_folder='.', marker_size_adjust_ratio=0.1):
    old_mpl_rcparams = {k: v for k, v in mpl.rcParams.items()}
    mpl.rcParams.update(MPL_RCPARAMS)

    idx0 = df.index
    idx = range(len(idx0))
    price = df.loc[:, 'close']
    bv, sv = df.loc[:, 'BuyVolume'].values, df.loc[:, 'SellVolume'].values
    profit = df.loc[:, 'CumProfit'].values
    avgpx = df.loc[:, 'AvgPosPrice']
    
    bv_m = np.max(bv)
    sv_m = np.max(sv)
    if bv_m > 0:
        bv = bv / bv_m * 100
    if sv_m > 0:
        sv = sv / sv_m * 100
    
    fig = plt.figure(figsize=(14, 10))
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
    
    ax2 = ax1.twinx()
    
    ax1.plot(idx, price, label='Price', linestyle='-', lw=1, marker='', color='yellow')
    ax1.scatter(idx, price, label='buy', marker='o', s=bv, color='indianred')
    ax1.scatter(idx, price, label='sell', marker='o', s=sv, color='forestgreen')
    ax1.plot(idx, avgpx, lw=1, marker='', color='green')
    ax1.legend(loc='upper left')
    ax1.set(title="Price, Trades and PnL for {:s}".format(symbol), ylabel="Price ($)")
    ax1.xaxis.set_major_formatter(MyFormatter(idx0, '%Y-%m'))
    
    ax2.plot(idx, profit, label='PnL', color='k', lw=1, ls='--', alpha=.4)
    ax2.legend(loc='upper right')
    ax2.set(ylabel="Profit / Loss ($)")
    
    # ax1.xaxis.set_major_formatter(MyFormatter(df.index))#, '%H:%M'))
    
    ax3.plot(idx, df.loc[:, 'position'], marker='D', markersize=3, lw=2)
    ax3.axhline(0, color='k', lw=1, ls='--', alpha=0.8)
    ax3.set(title="Position of {:s}".format(symbol))
    
    fig.tight_layout()
    fig.savefig(output_folder + '/' + "{}.png".format(symbol), facecolor=fig.get_facecolor(), dpi=fig.get_dpi())
    plt.close(fig)

    mpl.rcParams.update(old_mpl_rcparams)

