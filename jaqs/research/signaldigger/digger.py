# encoding: utf-8

from __future__ import print_function
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import performance as pfm
from . import plotting

import jaqs.util as jutil
from jaqs.trade import common


class SignalDigger(object):
    """
    
    Attributes
    ----------
    signal_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['signal', 'return', 'quantile']
    period : int
        Horizon used to calculate return.
    n_quantiles : int
    output_format : str
    output_folder : str
        
    """
    def __init__(self, output_folder=".", output_format='pdf'):
        self.output_format = output_format
        self.output_folder = os.path.abspath(output_folder)
        
        self.signal_data = None
        self.period = None
        self.n_quantiles = 5
        self.benchmark_ret = None
        
        self.returns_report_data = dict()
        self.ic_report_data = dict()
        self.fig_data = dict()
        self.fig_objs = dict()

    def process_signal_before_analysis(self,
                                       signal, price=None, ret=None, benchmark_price=None,
                                       period=5, n_quantiles=5,
                                       mask=None,
                                       forward=False):
        """
        Prepare for signal analysis.

        Parameters
        ----------
        signal : pd.DataFrame
            Index is date, columns are stocks.
        price : pd.DataFrame
            Index is date, columns are stocks.
        ret : pd.DataFrame
            Index is date, columns are stocks.
        benchmark_price : pd.DataFrame or pd.Series or None
            Price of benchmark.
        mask : pd.DataFrame
            Data cells that should NOT be used.
        n_quantiles : int
        period : int
            periods to compute forward returns on.

        Returns
        -------
        res : pd.DataFrame
            Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['signal', 'return', 'quantile']
            
        """
        """
        Deal with suspensions:
            If the period of calculating return is d (from T to T+d), then
            we do not use signal values of those suspended on T,
            we do not calculate return for those suspended on T+d.
        """
        # ----------------------------------------------------------------------
        # parameter validation
        if price is None and ret is None:
            raise ValueError("One of price / ret must be provided.")
        if price is not None and ret is not None:
            raise ValueError("Only one of price / ret should be provided.")
        if ret is not None and benchmark_price is not None:
            raise ValueError("You choose 'return' mode but benchmark_price is given.")
        if not (n_quantiles > 0 and isinstance(n_quantiles, int)):
            raise ValueError("n_quantiles must be a positive integer. Input is: {}".format(n_quantiles))
        
        # ensure inputs are aligned
        data = price if price is not None else ret
        assert np.all(signal.index == data.index)
        assert np.all(signal.columns == data.columns)
        if mask is not None:
            assert np.all(signal.index == mask.index)
            assert np.all(signal.columns == mask.columns)
            mask = jutil.fillinf(mask)
            mask = mask.astype(int).fillna(0).astype(bool)  # dtype of mask could be float. So we need to convert.
        else:
            mask = pd.DataFrame(index=signal.index, columns=signal.columns, data=False)
        signal = jutil.fillinf(signal)
        data = jutil.fillinf(data)

        # ----------------------------------------------------------------------
        # save data
        self.n_quantiles = n_quantiles
        self.period = period

        # ----------------------------------------------------------------------
        # Get dependent variables
        if price is not None:
            df_ret = pfm.price2ret(price, period=self.period, axis=0)
            if benchmark_price is not None:
                benchmark_price = benchmark_price.loc[signal.index]
                bench_ret = pfm.price2ret(benchmark_price, self.period, axis=0)
                self.benchmark_ret = bench_ret
                residual_ret = df_ret.sub(bench_ret.values.flatten(), axis=0)
            else:
                residual_ret = df_ret
        else:
            residual_ret = ret
        
        # Get independent varibale
        signal = signal.shift(1)  # avoid forward-looking bias

        # forward or not
        if forward:
            # point-in-time signal and forward return
            residual_ret = residual_ret.shift(-self.period)
        else:
            # past signal and point-in-time return
            signal = signal.shift(self.period)

        # ----------------------------------------------------------------------
        # get masks
        # mask_prices = data.isnull()
        # Because we use FORWARD return, if one day's price is broken, the day that is <period> days ago is also broken.
        # mask_prices = np.logical_or(mask_prices, mask_prices.shift(self.period))
        mask_price_return = residual_ret.isnull()
        mask_signal = signal.isnull()

        mask_tmp = np.logical_or(mask_signal, mask_price_return)
        mask_all = np.logical_or(mask, mask_tmp)

        # if price is not None:
        #     mask_forward = np.logical_or(mask, mask.shift(self.period).fillna(True))
        #     mask = np.logical_or(mask, mask_forward)

        # ----------------------------------------------------------------------
        # calculate quantile
        signal_masked = signal.copy()
        signal_masked = signal_masked[~mask_all]
        if n_quantiles == 1:
            df_quantile = signal_masked.copy()
            df_quantile.loc[:, :] = 1.0
        else:
            df_quantile = jutil.to_quantile(signal_masked, n_quantiles=n_quantiles)

        # ----------------------------------------------------------------------
        # stack
        def stack_td_symbol(df):
            df = pd.DataFrame(df.stack(dropna=False))  # do not dropna
            df.index.names = ['trade_date', 'symbol']
            df.sort_index(axis=0, level=['trade_date', 'symbol'], inplace=True)
            return df

        mask_all = stack_td_symbol(mask_all)
        df_quantile = stack_td_symbol(df_quantile)
        residual_ret = stack_td_symbol(residual_ret)

        # ----------------------------------------------------------------------
        # concat signal value
        res = stack_td_symbol(signal)
        res.columns = ['signal']
        res['return'] = residual_ret
        res['quantile'] = df_quantile
        res = res.loc[~(mask_all.iloc[:, 0]), :]
        
        print("Nan Data Count (should be zero) : {:d};  " \
              "Percentage of effective data: {:.0f}%".format(res.isnull().sum(axis=0).sum(),
                                                             len(res) * 100. / signal.size))
        res = res.astype({'signal': float, 'return': float, 'quantile': int})
        self.signal_data = res
    
    def show_fig(self, fig, file_name):
        """
        Save fig object to self.output_folder/filename.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
        file_name : str

        """
        
        self.fig_objs[file_name] = fig
        
        if self.output_format in ['pdf', 'png', 'jpg']:
            fp = os.path.join(self.output_folder, '.'.join([file_name, self.output_format]))
            jutil.create_dir(fp)
            fig.savefig(fp)
            print("Figure saved: {}".format(fp))
        elif self.output_format == 'base64':
            fig_b64 = jutil.fig2base64(fig, 'png')
            self.fig_data[file_name] = fig_b64
            print("Base64 data of figure {} will be stored in dictionary.".format(file_name))
        elif self.output_format == 'plot':
            fig.show()
        else:
            raise NotImplementedError("output_format = {}".format(self.output_format))
    
    @plotting.customize
    def create_returns_report(self):
        """
        Creates a tear sheet for returns analysis of a signal.

        """
        n_quantiles = self.signal_data['quantile'].max()
        
        # ----------------------------------------------------------------------------------
        # Daily Signal Return Time Series
        # Use regression or weighted average to calculate.
        period_wise_long_ret =\
            pfm.calc_period_wise_weighted_signal_return(self.signal_data, weight_method='long_only')
        period_wise_short_ret = \
            pfm.calc_period_wise_weighted_signal_return(self.signal_data, weight_method='short_only')
        cum_long_ret = pfm.period_wise_ret_to_cum(period_wise_long_ret, period=self.period, compound=False)
        cum_short_ret = pfm.period_wise_ret_to_cum(period_wise_short_ret, period=self.period, compound=False)
        # period_wise_ret_by_regression = perf.regress_period_wise_signal_return(signal_data)
        # period_wise_ls_signal_ret = \
        #     pfm.calc_period_wise_weighted_signal_return(signal_data, weight_method='long_short')
        # daily_ls_signal_ret = pfm.period2daily(period_wise_ls_signal_ret, period=period)
        # ls_signal_ret_cum = pfm.daily_ret_to_cum(daily_ls_signal_ret)

        # ----------------------------------------------------------------------------------
        # Period-wise Quantile Return Time Series
        # We calculate quantile return using equal weight or market value weight.
        # Quantile is already obtained according to signal values.
        
        # quantile return
        period_wise_quantile_ret_stats = pfm.calc_quantile_return_mean_std(self.signal_data, time_series=True)
        cum_quantile_ret = pd.concat({k: pfm.period_wise_ret_to_cum(v['mean'], period=self.period, compound=False)
                                      for k, v in period_wise_quantile_ret_stats.items()},
                                     axis=1)
        
        # top quantile minus bottom quantile return
        period_wise_tmb_ret = pfm.calc_return_diff_mean_std(period_wise_quantile_ret_stats[n_quantiles],
                                                            period_wise_quantile_ret_stats[1])
        cum_tmb_ret = pfm.period_wise_ret_to_cum(period_wise_tmb_ret['mean_diff'], period=self.period, compound=False)

        # ----------------------------------------------------------------------------------
        # Alpha and Beta
        # Calculate using regression.
        '''
        weighted_portfolio_alpha_beta
        tmb_alpha_beta =
        '''
        
        # start plotting
        if self.output_format:
            vertical_sections = 6
            gf = plotting.GridFigure(rows=vertical_sections, cols=1)
            gf.fig.suptitle("Returns Tear Sheet\n\n(no compound)\n (period length = {:d} days)".format(self.period))
    
            plotting.plot_quantile_returns_ts(period_wise_quantile_ret_stats,
                                              ax=gf.next_row())

            plotting.plot_cumulative_returns_by_quantile(cum_quantile_ret,
                                                         ax=gf.next_row())

            plotting.plot_cumulative_return(cum_long_ret,
                                            title="Signal Weighted Long Only Portfolio Cumulative Return",
                                            ax=gf.next_row())
            
            plotting.plot_cumulative_return(cum_short_ret,
                                            title="Signal Weighted Short Only Portfolio Cumulative Return",
                                            ax=gf.next_row())

            plotting.plot_mean_quantile_returns_spread_time_series(period_wise_tmb_ret, self.period,
                                                                   bandwidth=0.5,
                                                                   ax=gf.next_row())
            
            plotting.plot_cumulative_return(cum_tmb_ret,
                                            title="Top Minus Bottom (long top, short bottom)"
                                                  "Portfolio Cumulative Return",
                                            ax=gf.next_row())

            self.show_fig(gf.fig, 'returns_report')
        
        self.returns_report_data = {'period_wise_quantile_ret': period_wise_quantile_ret_stats,
                                    'cum_quantile_ret': cum_quantile_ret,
                                    'cum_long_ret': cum_long_ret,
                                    'cum_short_ret': cum_short_ret,
                                    'period_wise_tmb_ret': period_wise_tmb_ret,
                                    'cum_tmb_ret': cum_tmb_ret}

    @plotting.customize
    def create_information_report(self):
        """
        Creates a tear sheet for information analysis of a signal.
        
        """
        ic = pfm.calc_signal_ic(self.signal_data)
        ic.index = pd.to_datetime(ic.index, format="%Y%m%d")
        monthly_ic = pfm.mean_information_coefficient(ic, "M")

        if self.output_format:
            ic_summary_table = pfm.calc_ic_stats_table(ic)
            plotting.plot_information_table(ic_summary_table)
            
            columns_wide = 2
            fr_cols = len(ic.columns)
            rows_when_wide = (((fr_cols - 1) // columns_wide) + 1)
            vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
            gf = plotting.GridFigure(rows=vertical_sections, cols=columns_wide)
            gf.fig.suptitle("Information Coefficient Report\n\n(period length = {:d} days)"
                            "\ndaily IC = rank_corr(period-wise forward return, signal value)".format(self.period))

            plotting.plot_ic_ts(ic, self.period, ax=gf.next_row())
            plotting.plot_ic_hist(ic, self.period, ax=gf.next_row())
            # plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

            plotting.plot_monthly_ic_heatmap(monthly_ic, period=self.period, ax=gf.next_row())
        
            self.show_fig(gf.fig, 'information_report')
        
        self.ic_report_data = {'daily_ic': ic,
                               'monthly_ic': monthly_ic}
    
    def create_binary_event_report(self, signal, price, mask, benchmark_price, periods,
                                   join_method_periods='inner', group_by=None):
        """
        
        Parameters
        ----------
        signal : pd.DataFrame
        price : pd.DataFrame
        mask : pd.DataFrame
        benchmark_price : pd.DataFrame
        periods : list of int
        join_method_periods : {'inner', 'outer'}.
            Whether to take intersection or union of data of different periods.
        group_by : {'year', 'month', None}
            Calculate various statistics within each year/month/whole sample.

        Returns
        -------
        res : dict

        """
        import scipy.stats as scst

        # Raw Data
        dic_signal_data = OrderedDict()
        for my_period in periods:
            self.process_signal_before_analysis(signal, price=price, mask=mask,
                                                n_quantiles=1, period=my_period,
                                                benchmark_price=benchmark_price,
                                                forward=True)
            dic_signal_data[my_period] = self.signal_data
        
        # Processed Data
        dic_events = OrderedDict()
        dic_all = OrderedDict()
        for period, df in dic_signal_data.items():
            ser_ret = df['return']
            ser_sig = df['signal'].astype(bool)
            events_ret = ser_ret.loc[ser_sig]
            dic_events[period] = events_ret
            dic_all[period] = ser_ret
        df_events = pd.concat(dic_events, axis=1, join=join_method_periods)
        df_all = pd.concat(dic_all, axis=1, join=join_method_periods)

        # Data Statistics
        def _calc_statistics(df):
            df_res = pd.DataFrame(index=periods,
                                  columns=['Annu. Ret.', 'Annu. Vol.',
                                           #'Annual Return (all sample)', 'Annual Volatility (all sample)',
                                           't-stat', 'p-value', 'skewness', 'kurtosis', 'occurance'],
                                  data=np.nan)
            df_res.index.name = 'Period'
            
            ser_periods = pd.Series(index=df.columns, data=df.columns.values)
            ratio = (1.0 * common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR / ser_periods)
            mean = df.mean(axis=0)
            std = df.std(axis=0)
            annual_ret, annual_vol = mean * ratio, std * np.sqrt(ratio)
            
            t_stats, p_values = scst.ttest_1samp(df.values, np.zeros(df.shape[1]), axis=0)
            df_res.loc[:, 't-stat'] = t_stats
            df_res.loc[:, 'p-value'] = np.round(p_values, 5)
            df_res.loc[:, "skewness"] = scst.skew(df, axis=0)
            df_res.loc[:, "kurtosis"] = scst.kurtosis(df, axis=0)
            df_res.loc[:, 'Annu. Ret.'] = annual_ret
            df_res.loc[:, 'Annu. Vol.'] = annual_vol
            df_res.loc[:, 'occurance'] = len(df)
            # dic_res[period] = df
            return df_res

        if group_by == 'year':
            grouper_func = jutil.date_to_year
        elif group_by == 'month':
            grouper_func = jutil.date_to_month
        else:
            grouper_func = get_dummy_grouper

        idx_group = grouper_func(df_events.index.get_level_values('trade_date'))
        df_stats = df_events.groupby(idx_group).apply(_calc_statistics)
        idx_group_all = grouper_func(df_all.index.get_level_values('trade_date'))
        df_all_stats = df_all.groupby(idx_group_all).apply(_calc_statistics)
        df_all_stats = df_all_stats.loc[df_stats.index, ['Annu. Ret.', 'Annu. Vol.']]
        df_all_stats.columns = ['Annu. Ret. (all samp)', 'Annu. Vol. (all samp)']
        df_stats = pd.concat([df_stats, df_all_stats], axis=1)

        # return df_all, df_events, df_stats
        ser_signal_raw, monthly_signal, yearly_signal = calc_calendar_distribution(signal)

        # return
        # plot
        gf = plotting.GridFigure(rows=len(np.unique(idx_group)) * len(periods) + 3, cols=2, height_ratio=1.2)
        gf.fig.suptitle("Event Return Analysis (annualized)")

        plotting.plot_calendar_distribution(ser_signal_raw,
                                            monthly_signal=monthly_signal, yearly_signal=yearly_signal,
                                            ax1=gf.next_row(), ax2=gf.next_row())
        plotting.plot_event_bar(df_stats.reset_index(), x='Period', y='Annu. Ret.', hue='trade_date', ax=gf.next_row())
        # plotting.plot_event_pvalue(df_stats['p-value'], ax=gf.next_subrow())
        
        def _plot_dist(df):
            date = grouper_func(df.index.get_level_values('trade_date'))[0]
            plotting.plot_event_dist(df, group_by.title() + ' ' + str(date), axs=[gf.next_cell() for _ in periods])
        if group_by is not None:
            df_events.groupby(idx_group).apply(_plot_dist)
        else:
            plotting.plot_event_dist(df_events, "", axs=[gf.next_cell() for _ in periods])
        
        self.show_fig(gf.fig, 'event_report')

        # dic_res['df_res'] = df_res
        return df_all, df_events, df_stats
        
    @plotting.customize
    def create_full_report(self):
        """
        Creates a full tear sheet for analysis and evaluating single
        return predicting (alpha) signal.
        
        """
        # signal quantile description statistics
        qstb = calc_quantile_stats_table(self.signal_data)
        if self.output_format:
            plotting.plot_quantile_statistics_table(qstb)
            
        self.create_returns_report()
        self.create_information_report()
        # we do not do turnover analysis for now
        # self.create_turnover_report(signal_data)
        
        res = dict()
        res.update(self.returns_report_data)
        res.update(self.ic_report_data)
        res.update(self.fig_data)
        return res

    def create_single_signal_report(self, signal, price, periods, n_quantiles, mask=None, trade_condition=None):
        """
        
        Parameters
        ----------
        signal : pd.Series
        index is integer date, values are signals
        price : pd.Series
        index is integer date, values are prices
        mask : pd.Series or None, optional
        index is integer date, values are bool
        periods : list of int
        trade_condition : dict , optional
            {'cond_name1': {'col_name': str, 'hold': int, 'filter': func, 'direction': 1},
             'cond_name2': {'col_name': str, 'hold': int, 'filter': func, 'direction': -1},
            }
        
        Returns
        -------
        res : dict
        
        """
        if isinstance(signal, pd.DataFrame):
            signal = signal.iloc[:, 0]
        if isinstance(price, pd.DataFrame):
            price = price.iloc[:, 0]
            
        # calc return
        ret_l = {period: pfm.price2ret(price, period=period, axis=0) for period in periods}
        df_ret = pd.concat(ret_l, axis=1)

        # ----------------------------------------------------------------------
        # calculate quantile
        if n_quantiles == 1:
            df_quantile = signal.copy()
            df_quantile.loc[:] = 1.0
        else:
            df_quantile = jutil.to_quantile(signal, n_quantiles=n_quantiles, axis=0)

        # ----------------------------------------------------------------------
        # concat signal value
        res = pd.DataFrame(signal.shift(1))
        res.columns = ['signal']
        res['quantile'] = df_quantile
        res = pd.concat([res, df_ret], axis=1)
        res = res.dropna()

        print("Nan Data Count (should be zero) : {:d};  " \
              "Percentage of effective data: {:.0f}%".format(res.isnull().sum(axis=0).sum(),
                                                             len(res) * 100. / signal.size))
        
        # calc quantile stats
        gp = res.groupby(by='quantile')
        dic_raw = {k: v for k, v in gp}
        dic_stats = OrderedDict()
        for q, df in gp:
            df_stat = pd.DataFrame(index=['mean', 'std'], columns=df_ret.columns, data=np.nan)
            df_stat.loc['mean', :] = df.loc[:, df_ret.columns].mean(axis=0)
            df_stat.loc['std', :] = df.loc[:, df_ret.columns].std(axis=0)
            dic_stats[q] = df_stat
        
        # calculate IC
        ics = calc_various_ic(res, ret_cols=df_ret.columns)
        
        # backtest
        if trade_condition is not None:
            def sim_backtest(df, dic_of_cond):
                dic_cum_ret = dict()
                for key, dic in dic_of_cond.items():
                    col_name = dic['column']
                    func = dic['filter']
                    n_hold = dic['hold']
                    direction = dic['direction']
                    mask = df[col_name].apply(func).astype(int)
                    dic_cum_ret[key] = (df[n_hold] * mask).cumsum() * direction
                df_cumret = pd.concat(dic_cum_ret, axis=1)
                return df_cumret
            df_backtest = sim_backtest(res, trade_condition)
            
        # plot
        gf = plotting.GridFigure(rows=3, cols=1, height_ratio=1.2)
        gf.fig.suptitle("Event Return Analysis (annualized)")
        
        plotting.plot_ic_decay(ics, ax=gf.next_row())
        
        plotting.plot_quantile_return_mean_std(dic_stats, ax=gf.next_row())
        
        if trade_condition is not None:
            plotting.plot_batch_backtest(df_backtest, ax=gf.next_row())
        
        self.show_fig(gf.fig, 'single_inst.pdf')


def calc_ic(x, y, method='rank'):
    """
    Calculate IC between x and y.
    
    Parameters
    ----------
    x : np.ndarray
    y : np.ndarray
    method : {'rank', 'normal'}

    Returns
    -------
    corr : float

    """
    import scipy.stats as scst
    if method == 'rank':
        corr = scst.spearmanr(x, y)[0]
    elif method == 'normal':
        corr = np.corrcoef(x, y)[0, 1]
    else:
        raise NotImplementedError("method = {}".format(method))
    return corr


def calc_various_ic(df, ret_cols):
    res_dic = dict()
    
    # signal normal IC: signal value v.s. return
    res_dic['normal'] = [calc_ic(df['signal'], df[col], method='normal') for col in ret_cols]
    
    # signal rank IC: signal value v.s. return
    res_dic['rank'] = [calc_ic(df['signal'], df[col], method='rank') for col in ret_cols]
    
    # quantile normal IC: signal quantile v.s. return
    res_dic['normal_q'] = [calc_ic(df['quantile'], df[col], method='normal') for col in ret_cols]
    
    # quantile rank IC: signal quantile v.s. return
    res_dic['rank_q'] = [calc_ic(df['quantile'], df[col], method='rank') for col in ret_cols]
    
    res = pd.DataFrame(index=ret_cols, data=res_dic)
    return res
    

def calc_quantile_stats_table(signal_data):
    quantile_stats = signal_data.groupby('quantile').agg(['min', 'max', 'mean', 'std', 'count'])['signal']
    quantile_stats['count %'] = quantile_stats['count'] / quantile_stats['count'].sum() * 100.
    return quantile_stats


def get_dummy_grouper(ser):
    res = pd.Index(np.array(['all_sample'] * len(ser)), name=ser.name)
    return res
    
    
def calc_calendar_distribution(df_signal):
    daily_signal = df_signal.sum(axis=1)
    daily_signal = daily_signal.fillna(0).astype(int)
    idx = daily_signal.index.values
    month = jutil.date_to_month(idx)
    year = jutil.date_to_year(idx)
    
    monthly_signal = daily_signal.groupby(by=month).sum()
    yearly_signal = daily_signal.groupby(by=year).sum()
    
    monthly_signal = pd.DataFrame(monthly_signal, columns=['Time'])
    yearly_signal = pd.DataFrame(yearly_signal, columns=['Time'])
    monthly_signal.index.name = 'Month'
    yearly_signal.index.name = 'Year'
    
    return daily_signal, monthly_signal, yearly_signal
