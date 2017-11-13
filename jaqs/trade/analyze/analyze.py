# encoding: utf-8

import os
import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import Formatter

from jaqs.trade.analyze.report import Report
from jaqs.data.dataservice import RemoteDataService
from jaqs.data.basic.instrument import InstManager
import jaqs.util as jutil


# sys.path.append(os.path.abspath(".."))


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
        self._trades = None
        self._configs = None
        self.data_api = None
        self.dataview = None
        
        self._universe = []
        self._closes = None
        
        self.adjust_mode = None
        
        self.inst_mgr = None
        
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
    
    def initialize(self, data_server_=None, dataview=None, file_folder='.'):
        """
        Read trades from csv file to DataFrame of given data type.

        Parameters
        ----------
        data_server_ : RemoteDataService
        dataview : DataView
        file_folder : str
            Directory path where trades and configs are stored.

        """
        self.data_api = data_server_
        self.dataview = dataview
        
        type_map = {'task_id': str,
                    'entrust_no': str,
                    'entrust_action': str,
                    'symbol': str,
                    'fill_price': float,
                    'fill_size': float,
                    'fill_date': int,
                    'fill_time': int,
                    'fill_no': str,
                    'commission': float}
        abs_path = os.path.abspath(file_folder)
        trades = pd.read_csv(os.path.join(abs_path, 'trades.csv'), ',', dtype=type_map)
        
        self._init_universe(trades.loc[:, 'symbol'].values)
        self._init_configs(file_folder)
        symbol_str = ','.join(self.universe)
        self.inst_mgr = InstManager(symbol=symbol_str)
        self._init_trades(trades)
        self._init_symbol_price()
    
    def _init_trades(self, df):
        """Add datetime column. """
        df.loc[:, 'fill_dt'] = jutil.combine_date_time(df.loc[:, 'fill_date'], df.loc[:, 'fill_time'])
        
        self._trades = jutil.group_df_to_dict(df, by='symbol')
    
    def _init_symbol_price(self):
        """Get close price of securities in the universe from data server."""
        if self.dataview is not None:
            df_close = self.dataview.get_ts('close', start_date=self.start_date, end_date=self.end_date)
        else:
            df, msg = self.data_api.daily(symbol=','.join(self.universe), fields='trade_date,symbol,close',
                                          start_date=self.start_date, end_date=self.end_date)
            if msg != '0,':
                print(msg)
            df_close = df.pivot(index='trade_date', columns='symbol', values='close')
        self._closes = df_close

    def _init_universe(self, securities):
        """Return a set of securities."""
        self._universe = set(securities)
    
    def _init_configs(self, file_folder):
        configs = json.load(open(os.path.join(file_folder, 'configs.json'), 'r'))
        self._configs = configs
        self.init_balance = self.configs['init_balance']
        self.start_date = self.configs['start_date']
        self.end_date = self.configs['end_date']
        
    @staticmethod
    def _get_avg_pos_price(pos_arr, price_arr):
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
    
    @staticmethod
    def _process_trades(df):
        """Add various statistics to trades DataFrame."""
        # df = df.set_index('fill_date')
    
        cols_to_drop = ['task_id', 'entrust_no', 'fill_no']
        df = df.drop(cols_to_drop, axis=1)
    
        fs, fp = df.loc[:, 'fill_size'], df.loc[:, 'fill_price']
        turnover = fs * fp
    
        df.loc[:, 'CumTurnOver'] = turnover.cumsum()
    
        direction = df.loc[:, 'entrust_action'].apply(lambda s: 1 if s == 'buy' else -1)
    
        df.loc[:, 'BuyVolume'] = (direction + 1) / 2 * fs
        df.loc[:, 'SellVolume'] = (direction - 1) / -2 * fs
        df.loc[:, 'CumVolume'] = fs.cumsum()
        df.loc[:, 'CumNetTurnOver'] = (turnover * -direction).cumsum()
        df.loc[:, 'position'] = (fs * direction).cumsum()
    
        df.loc[:, 'AvgPosPrice'] = AlphaAnalyzer._get_avg_pos_price(df.loc[:, 'position'].values, fp.values)
    
        df.loc[:, 'CumProfit'] = (df.loc[:, 'CumNetTurnOver'] + df.loc[:, 'position'] * fp)
    
        return df
    
    def process_trades(self):
        self._trades = {k: self._process_trades(v) for k, v in self.trades.viewitems()}
    
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

    @staticmethod
    def _get_daily(close, trade):
        trade_cols = ['fill_date', 'BuyVolume', 'SellVolume', 'commission', 'position', 'AvgPosPrice', 'CumNetTurnOver']
    
        trade = trade.loc[:, trade_cols]
        gp = trade.groupby(by='fill_date')
        func_last = lambda ser: ser.iat[-1]
        trade = gp.agg({'BuyVolume': np.sum, 'SellVolume': np.sum, 'commission': np.sum,
                        'position': func_last, 'AvgPosPrice': func_last, 'CumNetTurnOver': func_last})
        merge = pd.concat([close, trade], axis=1, join='outer')
    
        cols_nan_to_zero = ['BuyVolume', 'SellVolume', 'commission']
        cols_nan_fill = ['close', 'position', 'AvgPosPrice', 'CumNetTurnOver']
        # merge: pd.DataFrame
        merge.loc[:, cols_nan_fill] = merge.loc[:, cols_nan_fill].fillna(method='ffill')
        merge.loc[:, cols_nan_fill] = merge.loc[:, cols_nan_fill].fillna(0)
    
        merge.loc[:, cols_nan_to_zero] = merge.loc[:, cols_nan_to_zero].fillna(0)
    
        merge.loc[merge.loc[:, 'AvgPosPrice'] < 1e-5, 'AvgPosPrice'] = merge.loc[:, 'close']
    
        merge.loc[:, 'CumProfit'] = merge.loc[:, 'CumNetTurnOver'] + merge.loc[:, 'position'] * merge.loc[:, 'close']
        merge.loc[:, 'CumProfitComm'] = merge['CumProfit'] - merge['commission'].cumsum()
    
        daily_net_turnover = merge['CumNetTurnOver'].diff(1).fillna(merge['CumNetTurnOver'].iat[0])
        daily_position_change = merge['position'].diff(1).fillna(merge['position'].iat[0])
        merge['trading_pnl'] = (daily_net_turnover + merge['close'] * daily_position_change)
        merge['holding_pnl'] = (merge['close'].diff(1) * merge['position'].shift(1)).fillna(0.0)
        merge.loc[:, 'total_pnl'] = merge['trading_pnl'] + merge['holding_pnl']
    
        return merge

    def get_daily(self):
        """Add various statistics to daily DataFrame."""
        daily_dic = dict()
        for sec, df_trade in self.trades.viewitems():
            df_close = self.closes[sec].rename('close')
        
            res = self._get_daily(df_close, df_trade)
            daily_dic[sec] = res
    
        self.daily = daily_dic

    def get_returns(self, compound_return=True, consider_commission=True):
        cols = ['trading_pnl', 'holding_pnl', 'total_pnl', 'commission', 'CumProfitComm', 'CumProfit']
        dic_symbol = {sec: self.inst_mgr.get_instrument(sec).multiplier * df_daily.loc[:, cols]
                      for sec, df_daily in self.daily.items()}
        df_profit = pd.concat(dic_symbol, axis=1)  # this is cumulative profit
        df_profit = df_profit.fillna(method='ffill').fillna(0.0)
        df_pnl = df_profit.stack(level=1)
        df_pnl = df_pnl.sum(axis=1)
        df_pnl = df_pnl.unstack(level=1)
        self.pnl = df_pnl
    
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
    
        self.metrics['yearly_return'] = np.power(df_returns.loc[:, 'active_cum'].values[-1], 1. / years) - 1
        self.metrics['yearly_vol'] = df_returns.loc[:, 'active'].std() * np.sqrt(225.)
        self.metrics['beta'] = np.corrcoef(df_returns.loc[:, 'bench'], df_returns.loc[:, 'strat'])[0, 1]
        self.metrics['sharpe'] = self.metrics['yearly_return'] / self.metrics['yearly_vol']
    
        # bt_strat_mv = pd.read_csv('bt_strat_mv.csv').set_index('trade_date')
        # df_returns = df_returns.join(bt_strat_mv, how='right')
        self.returns = df_returns
    
    def plot_pnl(self, save_folder="."):
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(21, 8), dpi=300, sharex=True)
        idx0 = self.returns.index
        idx = range(len(idx0))
        
        ax0.plot(idx, self.pnl['trading_pnl'], lw=1, color='indianred', label='Trading PnL')
        ax0.plot(idx, self.pnl['holding_pnl'], lw=1, color='royalblue', label='Holding PnL')
        ax0.plot(idx, self.pnl['total_pnl'], lw=1.5, color='violet', label='Total PnL')
        ax0.legend(loc='upper left')
        
        ax1.plot(idx, self.returns.loc[:, 'bench_cum'], label='Benchmark')
        ax1.plot(idx, self.returns.loc[:, 'strat_cum'], label='Strategy')
        ax1.legend(loc='upper left')
        
        ax2.plot(idx, self.returns.loc[:, 'active_cum'], label='Extra Return')
        ax2.legend(loc='upper left')
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Percent")
        ax1.set_ylabel("Percent")
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
        dic['props'] = self.configs
        dic['metrics'] = self.metrics
        dic['position_change'] = self.position_change
        dic['account'] = self.account
        dic['df_daily'] = self.daily
        self.returns.to_csv(os.path.join(out_folder, 'returns.csv'))
    
        r = Report(dic, source_dir=source_dir, template_fn=template_fn, out_folder=out_folder)
    
        r.generate_html()
        r.output_html('report.html')
        # r.output_pdf('report.pdf')


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
        super(EventAnalyzer, self).initialize(data_server_=data_server_, dataview=dataview,
                                              file_folder=file_folder)
        if self.dataview is not None and self.dataview.data_benchmark is not None:
            self.data_benchmark = self.dataview.data_benchmark.loc[(self.dataview.data_benchmark.index >= self.start_date)
                                                                   &(self.dataview.data_benchmark.index <= self.end_date)]
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
        
        self.data_benchmark = None

    def initialize(self, data_server_=None, dataview=None, file_folder='.'):
        super(AlphaAnalyzer, self).initialize(data_server_=data_server_, dataview=dataview,
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

    def get_returns_OLD(self, compound_return=True, consider_commission=True):
        profit_col_name = 'CumProfitComm' if consider_commission else 'CumProfit'
        vp_list = {sec: df_profit.loc[:, profit_col_name] for sec, df_profit in self.daily.viewitems()}
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

def calc_uat_metrics(t1, symbol):
    cump1 = t1.loc[:, 'CumProfit'].values
    profit1 = cump1[-1]
    
    n_trades = t1.loc[:, 'CumVolume'].values[-1] / 2.  # signle
    avg_trade = profit1 / n_trades
    print "profit without commission = {} \nprofit with commission {}".format(profit1, profit1)
    print "avg_trade = {:.3f}".format(avg_trade)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8))
    ax1.plot(cump1, label='inst1')
    ax1.set_title("{} PnL in price".format(symbol))
    ax1.legend(loc='upper left')
    ax1.axhline(0, color='k', lw=1, ls='--')
    ax2.plot(t1.loc[:, 'position'].values)
    ax2.set_title("Position")
    ax2.axhline(0, color='k', lw=1, ls='--')
    
    plt.show()
    return


def plot_trades(df, symbol="", save_folder="."):
    idx0 = df.index
    idx = range(len(idx0))
    price = df.loc[:, 'close']
    bv, sv = df.loc[:, 'BuyVolume'].values, df.loc[:, 'SellVolume'].values
    profit = df.loc[:, 'CumProfit'].values
    avgpx = df.loc[:, 'AvgPosPrice']
    bv *= .1
    sv *= .1
    
    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
    ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)
    
    # fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 18), sharex=True)
    # fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(idx, price, label='Price', linestyle='-', lw=1, marker='', color='yellow')
    ax1.scatter(idx, price, label='buy', marker='o', s=bv, color='red')
    ax1.scatter(idx, price, label='sell', marker='o', s=sv, color='green')
    ax1.plot(idx, avgpx, lw=1, marker='', color='green')
    ax1.legend(loc='upper left')
    
    ax2.plot(idx, profit, label='PnL', color='k', lw=1, ls='--', alpha=.4)
    ax2.legend(loc='upper right')
    
    # ax1.xaxis.set_major_formatter(MyFormatter(df.index))#, '%H:%M'))
    
    ax3.plot(idx, df.loc[:, 'position'], marker='D', markersize=3, lw=2)
    ax3.axhline(0, color='k', lw=1)
    
    ax1.set_title(symbol)
    
    ax1.xaxis.set_major_formatter(MyFormatter(idx0, '%Y-%m'))
    
    fig.savefig(save_folder + '/' + "{}.png".format(symbol))
    plt.tight_layout()
    return

