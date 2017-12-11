# encoding: utf-8

from __future__ import print_function
import os
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter

from jaqs.trade.analyze.report import Report
from jaqs.data import RemoteDataService
from jaqs.data.basic.instrument import InstManager
from jaqs.trade import common
import jaqs.util as jutil

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
    Attributes
    ----------
    _trades : pd.DataFrame
    _configs : dict
    data_api : BaseDataServer
    _universe : set
        All securities that have been traded.
        
    """
    def __init__(self):
        self.file_folder = ""
        
        self._trades = None
        self._configs = None
        self.data_api = None
        self.dataview = None
        
        self._universe = []
        self._closes = None
        self._closes_adj = None
        self.daily_position = None
        
        self.adjust_mode = None
        
        self.inst_map = dict()
        
        self.performance_metrics = dict()
        self.risk_metrics = dict()
        
        self.report_dic = dict()
        
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

    def initialize(self, data_api=None, dataview=None, file_folder='.'):
        """
        Read trades from csv file to DataFrame of given data type.

        Parameters
        ----------
        data_api : RemoteDataService
        dataview : DataView
        file_folder : str
            Directory path where trades and configs are stored.

        """
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
                    'commission': float}
        abs_path = os.path.abspath(file_folder)
        self.file_folder = abs_path
        trades = pd.read_csv(os.path.join(self.file_folder, 'trades.csv'), ',', dtype=type_map)
        if trades.empty:
            raise TradeRecordEmptyError("No trade records found in your 'trades.csv' file. Analysis stopped.")
        
        self._init_universe(trades.loc[:, 'symbol'].values)
        self._init_configs(self.file_folder)
        self._init_trades(trades)
        self._init_symbol_price()
        self._init_inst_data()
    
    def _init_inst_data(self):
        symbol_str = ','.join(self.universe)
        if self.dataview is not None:
            data_inst = self.dataview.data_inst
            self.inst_map = data_inst.to_dict(orient='index')
        elif self.data_api is not None:
            inst_mgr = InstManager(data_api=self.data_api, symbol=symbol_str)
            self.inst_map = {k: v.__dict__ for k, v in inst_mgr.inst_map.items()}
            del inst_mgr
        else:
            raise ValueError("no dataview or dataapi provided.")
        
    def _init_trades(self, df):
        """Add datetime column. """
        df.loc[:, 'fill_dt'] = jutil.combine_date_time(df.loc[:, 'fill_date'], df.loc[:, 'fill_time'])
        
        df = df.set_index(['symbol', 'fill_dt']).sort_index(axis=0)
        
        # self._trades = jutil.group_df_to_dict(df, by='symbol')
        self._trades = df
    
    def _init_symbol_price(self):
        """Get close price of securities in the universe from data server."""
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
    
    def _init_configs(self, folder):
        import codecs
        with codecs.open(os.path.join(folder, 'configs.json'), 'r', encoding='utf-8') as f:
            configs = json.load(f)
        self._configs = configs
        self.init_balance = self.configs['init_balance']
        self.start_date = self.configs['start_date']
        self.end_date = self.configs['end_date']
        
    @staticmethod
    def _process_trades(df):
        """Add various statistics to trades DataFrame."""
        from jaqs.trade import common
        
        # df = df.set_index('fill_date')
    
        # pre-process
        cols_to_drop = ['task_id', 'entrust_no', 'fill_no']
        df = df.drop(cols_to_drop, axis=1)
        
        def _apply(gp_df):
            # calculation of non-cumulative fields
            direction = gp_df['entrust_action'].apply(lambda s: 1 if common.ORDER_ACTION.is_positive(s) else -1)
            fill_size, fill_price = gp_df['fill_size'], gp_df['fill_price']
            turnover = fill_size * fill_price

            gp_df.loc[:, 'BuyVolume'] = (direction + 1) / 2 * fill_size
            gp_df.loc[:, 'SellVolume'] = (direction - 1) / -2 * fill_size
            
            # Calculation of cumulative fields
            gp_df.loc[:, 'CumVolume'] = fill_size.cumsum()
            gp_df.loc[:, 'CumTurnOver'] = turnover.cumsum()
            gp_df.loc[:, 'CumNetTurnOver'] = (turnover * -direction).cumsum()
    
            gp_df.loc[:, 'position'] = (fill_size * direction).cumsum()
    
            gp_df.loc[:, 'AvgPosPrice'] = calc_avg_pos_price(gp_df.loc[:, 'position'].values, fill_price.values)
    
            gp_df.loc[:, 'CumProfit'] = (gp_df.loc[:, 'CumNetTurnOver'] + gp_df.loc[:, 'position'] * fill_price)
            return gp_df
        gp = df.groupby(by='symbol')
        res = gp.apply(_apply)
        
        return res
    
    def process_trades(self):
        # self._trades = {k: self._process_trades(v) for k, v in self.trades.items()}
        self._trades = self._process_trades(self._trades)
    
    def get_pos_change_info(self):
        trades = pd.concat(self.trades.values(), axis=0)
        gp = trades.groupby(by=['fill_date'], as_index=False)
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
            cash = self.configs['init_balance'] + current_profit - mv
        
            account[str(date)] = {'market_value': mv, 'cash': cash}
        self.position_change = res
        self.account = account

    def get_daily(self):
        close = self.closes
        trade = self.trades
        
        # pro-process
        trade_cols = ['fill_date', 'BuyVolume', 'SellVolume', 'commission', 'position', 'AvgPosPrice', 'CumNetTurnOver']
    
        trade = trade.loc[:, trade_cols]
        gp = trade.groupby(by=['symbol', 'fill_date'])
        func_last = lambda ser: ser.iat[-1]
        trade = gp.agg({'BuyVolume': np.sum, 'SellVolume': np.sum, 'commission': np.sum,
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
            gp_df['trading_pnl'] = (daily_net_turnover + gp_df['close'] * daily_position_change)
            gp_df['holding_pnl'] = (gp_df['close'].diff(1) * gp_df['position'].shift(1)).fillna(0.0)
            gp_df.loc[:, 'total_pnl'] = gp_df['trading_pnl'] + gp_df['holding_pnl']
            
            return gp_df

        gp = merge.groupby(by='symbol')
        res = gp.apply(_apply)
        
        self.daily = res

    '''
    def get_daily(self):
        """Add various statistics to daily DataFrame."""
        self.daily = self._get_daily(self.closes, self.trades)
        daily_dic = dict()
        for sec, df_trade in self.trades.items():
            df_close = self.closes[sec].rename('close')
        
            res = self._get_daily(df_close, df_trade)
            daily_dic[sec] = res
    
        self.daily = daily_dic
    '''

    def get_returns(self, compound_return=True, consider_commission=True):
        cols = ['trading_pnl', 'holding_pnl', 'total_pnl', 'commission', 'CumProfitComm', 'CumProfit']
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
        
        df_pnl = daily.sum(axis=1)
        df_pnl = df_pnl.unstack(level=1)

        self.df_pnl = df_pnl
    
        # TODO temperary solution
        if consider_commission:
            strategy_value = (df_pnl['total_pnl'] - df_pnl['commission']).cumsum() + self.init_balance
        else:
            strategy_value = df_pnl['total_pnl'].cumsum() + self.init_balance
    
        market_values = pd.concat([strategy_value, self.data_benchmark], axis=1).fillna(method='ffill')
        market_values.columns = ['strat', 'bench']
    
        df_returns = market_values.pct_change(periods=1).fillna(0.0)
    
        df_returns = df_returns.join((df_returns.loc[:, ['strat', 'bench']] + 1.0).cumprod(), rsuffix='_cum')
        if compound_return:
            df_returns.loc[:, 'active_cum'] = df_returns['strat_cum'] - df_returns['bench_cum'] + 1
            df_returns.loc[:, 'active'] = df_returns['active_cum'].pct_change(1).fillna(0.0)
        else:
            df_returns.loc[:, 'active'] = df_returns['strat'] - df_returns['bench']
            df_returns.loc[:, 'active_cum'] = df_returns['active'].add(1.0).cumprod(axis=0)
    
        start = pd.to_datetime(self.configs['start_date'], format="%Y%m%d")
        end = pd.to_datetime(self.configs['end_date'], format="%Y%m%d")
        years = (end - start).days / 365.0
    
        self.performance_metrics['Annual Return (%)'] =\
            100 * (np.power(df_returns.loc[:, 'active_cum'].values[-1], 1. / years) - 1)
        self.performance_metrics['Annual Volatility (%)'] =\
            100 * (df_returns.loc[:, 'active'].std() * np.sqrt(common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR))
        self.performance_metrics['Sharpe Ratio'] = (self.performance_metrics['Annual Return (%)']
                                                    / self.performance_metrics['Annual Volatility (%)'])
        
        self.risk_metrics['Beta'] = np.corrcoef(df_returns.loc[:, 'bench'], df_returns.loc[:, 'strat'])[0, 1]
    
        # bt_strat_mv = pd.read_csv('bt_strat_mv.csv').set_index('trade_date')
        # df_returns = df_returns.join(bt_strat_mv, how='right')
        self.returns = df_returns
    
    def plot_pnl(self, save_folder=None):
        old_mpl_rcparams = {k: v for k, v in mpl.rcParams.items()}
        mpl.rcParams.update(MPL_RCPARAMS)
        
        if save_folder is None:
            save_folder = self.file_folder
        fig1 = plot_portfolio_bench_pnl(self.returns.loc[:, 'strat_cum'],
                                        self.returns.loc[:, 'bench_cum'],
                                        self.returns.loc[:, 'active_cum'])
        fig1.savefig(os.path.join(save_folder,'pnl_img.png'), facecolor=fig1.get_facecolor(), dpi=fig1.get_dpi())
        
        fig2 = plot_daily_trading_holding_pnl(self.df_pnl['trading_pnl'],
                                              self.df_pnl['holding_pnl'],
                                              self.df_pnl['total_pnl'],
                                              self.df_pnl['total_pnl'].cumsum())
        fig2.savefig(os.path.join(save_folder,'pnl_img_trading_holding.png'), facecolor=fig2.get_facecolor(), dpi=fig2.get_dpi())
        
        mpl.rcParams.update(old_mpl_rcparams)

    def plot_pnl_OLD(self, save_folder=None):
        if save_folder is None:
            save_folder = self.file_folder
        
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(21, 8), dpi=300, sharex=True)
        idx0 = self.returns.index
        idx = np.arange(len(idx0))
        
        bar_width = 0.3
        ax0.bar(idx-bar_width/2, self.df_pnl['trading_pnl'], width=bar_width, color='indianred', label='Trading PnL',)
        ax0.bar(idx+bar_width/2, self.df_pnl['holding_pnl'], width=bar_width, color='royalblue', label='Holding PnL')
        ax0.axhline(0.0, color='k', lw=1, ls='--')
        # ax0.plot(idx, self.pnl['total_pnl'], lw=1.5, color='violet', label='Total PnL')
        ax0.legend(loc='upper left')
        
        ax1.plot(idx, self.returns.loc[:, 'bench_cum'], label='Benchmark')
        ax1.plot(idx, self.returns.loc[:, 'strat_cum'], label='Strategy')
        ax1.legend(loc='upper left')
        
        ax2.plot(idx, self.returns.loc[:, 'active_cum'], label='Extra Return')
        ax2.legend(loc='upper left')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Net Value")
        ax1.set_ylabel("Net Value")
        ax2.xaxis.set_major_formatter(MyFormatter(idx0, '%Y-%m-%d'))
    
        plt.tight_layout()
        fig.savefig(os.path.join(save_folder, 'pnl_img.png'))
        plt.close()

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
        dic['performance_metrics'] = self.performance_metrics
        dic['risk_metrics'] = self.risk_metrics
        dic['position_change'] = self.position_change
        dic['account'] = self.account
        dic['df_daily'] = jutil.group_df_to_dict(self.daily, by='symbol')
        dic['daily_position'] = self.daily_position
        
        self.report_dic.update(dic)
        
        self.returns.to_csv(os.path.join(out_folder, 'returns.csv'))
    
        r = Report(self.report_dic, source_dir=source_dir, template_fn=template_fn, out_folder=out_folder)
        
        r.generate_html()
        r.output_html('report.html')

    def do_analyze(self, result_dir, selected_sec=None):
        if selected_sec is None:
            selected_sec = []
            
        print("process trades...")
        self.process_trades()
        print("get daily stats...")
        self.get_daily()
        print("calc strategy return...")
        self.get_returns(consider_commission=False)

        if len(selected_sec) > 0:
            print("Plot single securities PnL")
            for symbol in selected_sec:
                df_daily = self.daily.loc[pd.IndexSlice[symbol, :], :]
                df_daily.index = df_daily.index.droplevel(0)
                if df_daily is not None:
                    plot_trades(df_daily, symbol=symbol, save_folder=self.file_folder)

        print("Plot strategy PnL...")
        self.plot_pnl(result_dir)
        
        print("generate report...")
        self.gen_report(source_dir=STATIC_FOLDER, template_fn='report_template.html',
                        out_folder=result_dir,
                        selected=selected_sec)


class EventAnalyzer(BaseAnalyzer):
    def __init__(self):
        super(EventAnalyzer, self).__init__()
        
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
            benchmark = self.configs.get('benchmark', "")
            if benchmark and data_server_:
                df, msg = data_server_.daily(benchmark, start_date=self.closes.index[0], end_date=self.closes.index[-1])
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

    '''
    def get_returns_OLD(self, compound_return=True, consider_commission=True):
        profit_col_name = 'CumProfitComm' if consider_commission else 'CumProfit'
        vp_list = {sec: df_profit.loc[:, profit_col_name] for sec, df_profit in self.daily.items()}
        df_profit = pd.concat(vp_list, axis=1)  # this is cumulative profit
        # TODO temperary solution
        df_profit = df_profit.fillna(method='ffill').fillna(0.0)
        strategy_value = df_profit.sum(axis=1) + self.configs['init_balance']
        
        market_values = pd.concat([strategy_value, self.data_benchmark], axis=1).fillna(method='ffill')
        market_values.columns = ['strat', 'bench']
        
        df_returns = market_values.pct_change(periods=1).fillna(0.0)
        
        df_returns = df_returns.join((df_returns.loc[:, ['strat', 'bench']] + 1.0).cumprod(), rsuffix='_cum')
        if compound_return:
            df_returns.loc[:, 'active_cum'] = df_returns['strat_cum'] - df_returns['bench_cum'] + 1
            df_returns.loc[:, 'active'] = df_returns['active_cum'].pct_change(1).fillna(0.0)
        else:
            df_returns.loc[:, 'active'] = df_returns['strat'] - df_returns['bench']
            df_returns.loc[:, 'active_cum'] = df_returns['active'].add(1.0).cumprod(axis=0)
        
        start = pd.to_datetime(self.configs['start_date'], format="%Y%m%d")
        end = pd.to_datetime(self.configs['end_date'], format="%Y%m%d")
        years = (end - start).days / 365.0
        
        self.metrics['yearly_return'] = np.power(df_returns.loc[:, 'active_cum'].values[-1], 1. / years) - 1
        self.metrics['yearly_vol'] = df_returns.loc[:, 'active'].std() * np.sqrt(225.)
        self.metrics['beta'] = np.corrcoef(df_returns.loc[:, 'bench'], df_returns.loc[:, 'strat'])[0, 1]
        self.metrics['sharpe'] = self.metrics['yearly_return'] / self.metrics['yearly_vol']
        
        # bt_strat_mv = pd.read_csv('bt_strat_mv.csv').set_index('trade_date')
        # df_returns = df_returns.join(bt_strat_mv, how='right')
        self.returns = df_returns

    '''
    
    def _get_index_weight(self):
        if self.dataview is not None:
            res = self.dataview.get_ts('index_weight', start_date=self.start_date, end_date=self.end_date)
        else:
            res = self.data_api.get_index_weights_daily(self.universe, self.start_date, self.end_date)
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
    
    def brinson(self, group):
        """
        
        Parameters
        ----------
        group : str or pd.DataFrame
            If group is string, this function will try to fetch the corresponding DataFrame from DataView.
            If group is pd.DataFrame, it will be used as-is.

        Returns
        -------

        """
        if isinstance(group, str):
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
        plot_brinson(df_brinson, save_folder=self.file_folder)

    def do_analyze(self, result_dir, selected_sec=None, brinson_group=None):
        if selected_sec is None:
            selected_sec = []
    
        print("process trades...")
        self.process_trades()
        print("get daily stats...")
        self.get_daily()
        print("calc strategy return...")
        self.get_returns(consider_commission=False)
    
        not_none_sec = []
        if len(selected_sec) > 0:
            print("Plot single securities PnL")
            for symbol in selected_sec:
                df_daily = self.daily.loc[pd.IndexSlice[symbol, :], :]
                df_daily.index = df_daily.index.droplevel(0)
                if df_daily is not None:
                    not_none_sec.append(symbol)
                    plot_trades(df_daily, symbol=symbol, save_folder=self.file_folder)
    
        print("Plot strategy PnL...")
        self.plot_pnl(result_dir)
        
        if brinson_group is not None:
            print("Do brinson attribution.")
            group = self.dataview.get_ts(brinson_group)
            if group is None:
                raise ValueError("group data is None.")
            self.brinson(group)
    
        print("generate report...")
        self.gen_report(source_dir=STATIC_FOLDER, template_fn='report_template.html',
                        out_folder=result_dir,
                        selected=not_none_sec)


def plot_daily_trading_holding_pnl(trading, holding, total, total_cum):
    """
    Parameters
    ----------
    Series
    
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
    ax3.set(title='Daily Holding PnL', ylabel=y_label, xticks=idx[: : n//10])
    return fig
    
    
def plot_portfolio_bench_pnl(portfolio_cum_ret, benchmark_cum_ret, excess_cum_ret):
    """
    Parameters
    ----------
    Series
    
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
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
    ax2.set(title="Excess Return Compared to Benchmark", ylabel=y_label_ret
            #xlabel="Date", 
            )
    ax2.grid(axis='y')
    ax2.xaxis.set_major_formatter(MyFormatter(idx_dt, '%y-%m-%d'))  # 17-09-31
    
    fig.tight_layout()  
    return fig
    
def plot_brinson(df, save_folder):
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
    fig.savefig(os.path.join(save_folder, 'brinson_attribution.png'))
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


def plot_trades(df, symbol="", save_folder='.', marker_size_adjust_ratio=0.1):
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
    fig.savefig(save_folder + '/' + "{}.png".format(symbol), facecolor=fig.get_facecolor(), dpi=fig.get_dpi())

    mpl.rcParams.update(old_mpl_rcparams)

