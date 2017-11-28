# encoding: utf-8

import os
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
                                       mask=None):
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
        benchmark_price : pd.DataFrame or pd.Series
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
        
        data = price if price is not None else ret
        assert np.all(signal.index == data.index)
        assert np.all(signal.columns == data.columns)
        if mask is not None:
            assert np.all(signal.index == mask.index)
            assert np.all(signal.columns == mask.columns)
            mask = jutil.fillinf(mask)
            mask = mask.astype(int).fillna(0).astype(bool)
        else:
            mask = pd.DataFrame(index=signal.index, columns=signal.columns, data=False)
        signal = jutil.fillinf(signal)
        data = jutil.fillinf(data)

        # ----------------------------------------------------------------------
        # save data
        self.n_quantiles = n_quantiles
        self.period = period

        # ----------------------------------------------------------------------
        # fwd_returns are processed forward returns
        signal = signal.shift(self.period)
        
        if price is not None:
            df_return = pfm.price2ret(price, self.period)
            if benchmark_price is not None:
                benchmark_price = benchmark_price.loc[signal.index]
                bench_ret = pfm.price2ret(benchmark_price, self.period, axis=0)
                self.benchmark_ret = bench_ret
                df_return = df_return.sub(bench_ret.values.flatten(), axis=0)
        else:
            df_return = ret

        # ----------------------------------------------------------------------
        # get masks
        mask_prices = data.isnull()
        # Because we use FORWARD return, if one day's price is broken, the day that is <period> days ago is also broken.
        mask_prices = np.logical_or(mask_prices, mask_prices.shift(self.period).fillna(True))
        mask_signal = signal.isnull()

        mask = np.logical_or(mask, mask_prices)
        mask = np.logical_or(mask, mask_signal)

        if price is not None:
            mask_forward = np.logical_or(mask, mask.shift(self.period).fillna(True))
            mask = np.logical_or(mask, mask_forward)

        # ----------------------------------------------------------------------
        # calculate quantile
        signal_masked = signal.copy()
        signal_masked = signal_masked[~mask]
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
        
        mask = stack_td_symbol(mask)
        df_quantile = stack_td_symbol(df_quantile)
        df_return = stack_td_symbol(df_return)

        # ----------------------------------------------------------------------
        # concat signal value
        res = stack_td_symbol(signal)
        res.columns = ['signal']
        res['return'] = df_return
        res['quantile'] = df_quantile
        res = res.loc[~(mask.iloc[:, 0]), :]
        
        print "Nan Data Count (should be zero) : {:d};  " \
              "Percentage of effective data: {:.0f}%".format(res.isnull().sum(axis=0).sum(),
                                                             len(res) * 100. / signal.size)
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
            print "Figure saved: {}".format(fp)
        elif self.output_format == 'base64':
            fig_b64 = jutil.fig2base64(fig, 'png')
            self.fig_data[file_name] = fig_b64
            print "Base64 data of figure {} will be stored in dictionary.".format(file_name)
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
    
    '''
    def OLS_create_binary_event_report(self, before, after):
        """
        
        Parameters
        ----------
        before : int
        after : int

        Returns
        -------

        """
        event = self.signal_data['signal']
        ret = self.signal_data['return']
        
        event = event.unstack('symbol')
        ret = ret.unstack('symbol')
        event = event.astype(bool)
        
        index_len = event.shape[0]
        l = []
        idx = range(-before, after+1)
        for day_zero_index in range(before, index_len - after):
            date = event.index[day_zero_index]
            row = event.iloc[day_zero_index, :]
            
            equities_slice = row.values
            if not any(equities_slice):
                continue
            
            starting_index = day_zero_index - before
            ending_index = day_zero_index + after
            
            ser_events = ret.iloc[starting_index: ending_index + 1, equities_slice]
            ser_events.index = idx
    
            l.append(ser_events)
        
        res = pd.concat(l, axis=1)
        res = pfm.ret2cum(res, compound=False, axis=0)
        res = res - res.loc[0, :]
        
        return res.mean(axis=1), res.std(axis=1)
    
    '''
    
    def create_binary_event_report(self, signal, price, mask, benchmark_price, periods):
        import scipy.stats as scst
        
        dic_signal_data = dict()
        for my_period in periods:
            self.process_signal_before_analysis(signal, price=price, mask=mask,
                                                n_quantiles=1, period=my_period,
                                                benchmark_price=benchmark_price)
            dic_signal_data[my_period] = self.signal_data

        # analyze ret: annualized
        '''
        dic_ret = {k: v['return'] * (1.0 * common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR / k) for k, v in dic_signal_data.items()}
        res = pd.concat(dic_ret, axis=1, join='outer')

        mask = dic_signal_data[20]['signal'].astype(bool)
        res = res.loc[mask[mask].index, :]
        '''
        df_res = pd.DataFrame(index=periods,
                              columns=['Annual Return', 'Annual Volatility',
                                       'Annual Return (all sample)', 'Annual Volatility (all sample)',
                                       't-stat', 'p-value', 'skewness', 'kurtosis'],
                              data=np.nan)
        dic_res = dict()
        for period, df in dic_signal_data.items():
            ser_ret = df['return']
            ser_sig = df['signal'].astype(bool)
            events_ret = ser_ret.loc[ser_sig]
            
            ratio = (1.0 * common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR / period)
            annual_ret, annual_vol = events_ret.mean() * ratio, events_ret.std() * np.sqrt(ratio)
            annual_ret_allsamp, annual_vol_allsamp = ser_ret.mean() * ratio, ser_ret.std() * np.sqrt(ratio)
            
            n_all = len(ser_ret)
            n_events = len(events_ret)
            print("For period={}, Probability of event = {:.1f}%".format(period, n_events * 100. / n_all))
            
            t_stat, p_value = scst.ttest_1samp(events_ret, 0)
            df_res.loc[period, ['t-stat']] = t_stat
            df_res.loc[period, ['p-value']] = round(p_value, 5)
            df_res.loc[period, "skewness"] = scst.skew(events_ret)
            df_res.loc[period, "kurtosis"] = scst.kurtosis(events_ret)
            df_res.loc[period, ['Annual Return']] = annual_ret
            df_res.loc[period, ['Annual Volatility']] = annual_vol
            df_res.loc[period, ['Annual Return (all sample)']] = annual_ret_allsamp
            df_res.loc[period, ['Annual Volatility (all sample)']] = annual_vol_allsamp
            dic_res[period] = events_ret
            
            # print(events_ret.sort_values().tail())

        '''
        dic_ret = {k: v['return'] * (1.0 * common.CALENDAR_CONST.TRADE_DAYS_PER_YEAR / k) for k, v in dic_signal_data.items()}
        dic_ret['signal'] = dic_signal_data[my_period]['signal'].astype(bool)
        res = pd.concat(dic_ret, axis=1, join='inner')
        mask = res['signal']
        res = res.loc[mask[mask].index, :]
        print(res.shape)
        mean, std = res.loc[:, periods].mean(axis=0), res.loc[:, periods].std(axis=0)
        
        '''
        print(df_res.applymap(lambda x: round(x, 5)))

        # return
        # plot
        gf = plotting.GridFigure(rows=len(periods) + 1, cols=2, height_ratio=1.2)
        gf.fig.suptitle("Event Return Analysis (annualized)")

        plotting.plot_event_bar(df_res['Annual Return'], ax=gf.next_row())
        plotting.plot_event_dist(dic_res, axs=[gf.next_cell() for _ in periods])
        
        self.show_fig(gf.fig, 'event_report')

        dic_res['df_res'] = df_res
        return dic_res
        
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


def calc_quantile_stats_table(signal_data):
    quantile_stats = signal_data.groupby('quantile').agg(['min', 'max', 'mean', 'std', 'count'])['signal']
    quantile_stats['count %'] = quantile_stats['count'] / quantile_stats['count'].sum() * 100.
    return quantile_stats
