# encoding: utf-8

import os
import numpy as np
import pandas as pd

from . import performance as pfm
from . import plotting

from jaqs.util import fileio, pdutil


class SignalDigger(object):
    """
    
    Attributes
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['factor', 'return', 'quantile']
    period : int
        Horizon used to calculate return.
    n_quantiles : int
    output_format : str
    output_folder : str
        
    """
    def __init__(self, output_folder=".", output_format='pdf'):
        self.output_format = output_format
        self.output_folder = os.path.abspath(output_folder)
        
        self.factor_data = None
        self.period = None
        self.n_quantiles = 5
        self.benchmark_ret = None
        
        self.returns_report_data = dict()
        self.ic_report_data = dict()
        self.fig_data = dict()

    def process_factor_before_analysis(self,
                                       factor, price=None, ret=None, benchmark_price=None,
                                       period=5, n_quantiles=5,
                                       mask=None):
        """
        Prepare for signal analysis.

        Parameters
        ----------
        factor : pd.DataFrame
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
            Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['factor', 'return', 'quantile']
            
        """
        """
        Deal with suspensions:
            If the period of calculating return is d (from T to T+d), then
            we do not use factor values of those suspended on T,
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
        
        data = price if price is not None else ret
        assert np.all(factor.index == data.index)
        assert np.all(factor.columns == data.columns)
        if mask is not None:
            assert np.all(factor.index == mask.index)
            assert np.all(factor.columns == mask.columns)
            mask = pdutil.fillinf(mask)
            mask = mask.astype(int).fillna(0).astype(bool)
        else:
            mask = pd.DataFrame(index=factor.index, columns=factor.columns, data=False)
        factor = pdutil.fillinf(factor)
        data = pdutil.fillinf(data)

        # ----------------------------------------------------------------------
        # save data
        self.n_quantiles = n_quantiles
        self.period = period
        
        # fwd_returns are processed forward returns
        if price is not None:
            df_return = pfm.calc_forward_return(price, self.period)
            if benchmark_price is not None:
                benchmark_price = benchmark_price.loc[factor.index]
                bench_ret = pfm.calc_forward_return(benchmark_price, self.period)
                self.benchmark_ret = bench_ret
                df_return = df_return.sub(bench_ret.values.flatten(), axis=0)
        else:
            df_return = ret

        # ----------------------------------------------------------------------
        # get masks
        mask_prices = data.isnull()
        # Because we use FORWARD return, if one day's price is broken, the day that is <period> days ago is also broken.
        mask_prices = np.logical_or(mask_prices, mask_prices.shift(-self.period).fillna(True))
        mask_factor = factor.isnull()

        mask = np.logical_or(mask, mask_prices)
        mask = np.logical_or(mask, mask_factor)

        if price is not None:
            mask_forward = np.logical_or(mask, mask.shift(-self.period).fillna(True))
            mask = np.logical_or(mask, mask_forward)

        # ----------------------------------------------------------------------
        # calculate quantile
        factor_masked = factor.copy()
        factor_masked = factor_masked[~mask]
        df_quantile = pdutil.to_quantile(factor_masked, n_quantiles=n_quantiles)

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
        # concat factor value
        res = stack_td_symbol(factor)
        res.columns = ['factor']
        res['return'] = df_return
        res['quantile'] = df_quantile
        res = res.loc[~(mask.iloc[:, 0]), :]
        
        print "Nan Data Count (should be zero) : {:d};  " \
              "Percentage of effective data: {:.0f}%".format(res.isnull().sum(axis=0).sum(),
                                                             len(res) * 100. / factor.size)
        res = res.astype({'factor': float, 'return': float, 'quantile': int})
        self.factor_data = res
    
    def show_fig(self, fig, file_name):
        """
        Save fig object to self.output_folder/filename.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
        file_name : str

        """
        
        if self.output_format in ['pdf', 'png', 'jpg']:
            fp = os.path.join(self.output_folder, '.'.join([file_name, self.output_format]))
            fileio.create_dir(fp)
            fig.savefig(fp)
            print "Figure saved: {}".format(fp)
        elif self.output_format == 'base64':
            fig_b64 = fileio.fig2base64(fig, 'png')
            self.fig_data[file_name] = fig_b64
            print "Base64 data of figure {} will be stored in dictionary.".format(file_name)
        elif self.output_format == 'plot':
            fig.show()
        else:
            raise NotImplementedError("output_format = {}".format(self.output_format))
    
    @plotting.customize
    def create_returns_report(self):
        """
        Creates a tear sheet for returns analysis of a factor.

        """
        n_quantiles = self.factor_data['quantile'].max()
        
        # ----------------------------------------------------------------------------------
        # Daily Factor Return Time Series
        # Use regression or weighted average to calculate.
        period_wise_weighted_factor_ret =\
            pfm.calc_period_wise_weighted_factor_return(self.factor_data, weight_method='long_only')
        daily_weighted_factor_ret = pfm.period2daily(period_wise_weighted_factor_ret, period=self.period)
        # period_wise_ret_by_regression = perf.regress_period_wise_factor_return(factor_data)
        # period_wise_ls_factor_ret = \
        #     pfm.calc_period_wise_weighted_factor_return(factor_data, weight_method='long_short')
        # daily_ls_factor_ret = pfm.period2daily(period_wise_ls_factor_ret, period=period)
        # ls_factor_ret_cum = pfm.daily_ret_to_cum(daily_ls_factor_ret)

        # ----------------------------------------------------------------------------------
        # Daily Quantile Return Time Series
        # We calculate quantile return using equal weight or market value weight.
        # Quantile is already obtained according to factor values.
        
        # quantile return
        period_wise_quantile_ret_stats = pfm.calc_quantile_return_mean_std(self.factor_data, time_series=True)
        daily_quantile_ret_stats = {k: pd.concat([pfm.period2daily(v['mean'], period=self.period),
                                                  v['std'] / np.sqrt(self.period)], axis=1)
                                    for k, v in period_wise_quantile_ret_stats.items()}
        
        # top quantile minus bottom quantile return ( note: (1 + x)^n ~= 1 + nx )
        daily_tmb_ret_mean_std = pfm.calc_return_diff_mean_std(daily_quantile_ret_stats[n_quantiles],
                                                               daily_quantile_ret_stats[1])
        daily_tmb_ret = daily_tmb_ret_mean_std['mean_diff']

        # ----------------------------------------------------------------------------------
        # Alpha and Beta
        # Calculate using regression.
        '''
        weighted_portfolio_alpha_beta
        tmb_alpha_beta =
        '''
        
        # start plotting
        if self.output_format:
            vertical_sections = 5
            gf = plotting.GridFigure(rows=vertical_sections, cols=1)
            gf.fig.suptitle("Returns Tear Sheet\n (period length = {:d} days)".format(self.period))
    
            plotting.plot_quantile_returns_ts(period_wise_quantile_ret_stats,
                                              ax=gf.next_row())

            plotting.plot_cumulative_return(daily_weighted_factor_ret,
                                            title="Factor Weighted (long only) Portfolio Cumulative Return",
                                            ax=gf.next_row())

            plotting.plot_cumulative_return(daily_tmb_ret,
                                            title="Top Minus Bottom (long top short bottom)"
                                                  "Portfolio Cumulative Return",
                                            ax=gf.next_row())
            
            plotting.plot_cumulative_returns_by_quantile(daily_quantile_ret_stats,
                                                         ax=gf.next_row())
        
            plotting.plot_mean_quantile_returns_spread_time_series(daily_tmb_ret_mean_std, self.period,
                                                                   bandwidth=0.5,
                                                                   ax=gf.next_row())
            self.show_fig(gf.fig, 'returns_report')
        
        self.returns_report_data = {'quantile_active_ret_correct': daily_quantile_ret_stats}

    @plotting.customize
    def create_information_report(self):
        """
        Creates a tear sheet for information analysis of a factor.
        
        """
        ic = pfm.calc_factor_ic(self.factor_data)
        ic.index = pd.to_datetime(ic.index, format="%Y%m%d")
        mean_monthly_ic = pfm.mean_information_coefficient(ic, "M")

        if self.output_format:
            plotting.plot_information_table(ic)
            
            columns_wide = 2
            fr_cols = len(ic.columns)
            rows_when_wide = (((fr_cols - 1) // columns_wide) + 1)
            vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
            gf = plotting.GridFigure(rows=vertical_sections, cols=columns_wide)
            gf.fig.suptitle("Information Coefficient Report\n(period length = {:d} days)"
                            "\ndaily IC = spearman_rank_corr(period-wise forward return, factor value)".format(self.period))

            plotting.plot_ic_ts(ic, self.period, ax=gf.next_row())
            plotting.plot_ic_hist(ic, self.period, ax=gf.next_row())
            # plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

            plotting.plot_monthly_ic_heatmap(mean_monthly_ic, period=self.period, ax=gf.next_row())
        
            self.show_fig(gf.fig, 'information_report')
        
        self.ic_report_data = {'daily_ic': ic}

    @plotting.customize
    def create_full_report(self):
        """
        Creates a full tear sheet for analysis and evaluating single
        return predicting (alpha) factor.
        
        """
        # factor quantile description statistics
        qstb = calc_quantile_stats_table(self.factor_data)
        if self.output_format:
            plotting.plot_quantile_statistics_table(qstb)
            
        self.create_returns_report()
        self.create_information_report()
        # we do not do turnover analysis for now
        # self.create_turnover_report(factor_data)
        
        res = dict()
        res.update(self.returns_report_data)
        res.update(self.ic_report_data)
        res.update(self.fig_data)
        return res


def calc_quantile_stats_table(factor_data):
    quantile_stats = factor_data.groupby('quantile').agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] / quantile_stats['count'].sum() * 100.
    return quantile_stats
