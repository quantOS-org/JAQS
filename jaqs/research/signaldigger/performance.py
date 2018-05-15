# encoding: utf-8

import numpy as np
import pandas as pd
import scipy.stats as scst
import statsmodels.api as sm

from jaqs.trade.common import CALENDAR_CONST


def calc_signal_ic(signal_data):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between signal values and N period forward returns for each period in
    the signal index.

    Parameters
    ----------
    signal_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['signal', 'return', 'quantile']

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between signal and provided forward returns.
        
    """
    def src_ic(df):
        _ic = scst.spearmanr(df['signal'], df['return'])[0]
        return _ic

    signal_data = signal_data.copy()

    grouper = ['trade_date']

    ic = signal_data.groupby(grouper).apply(src_ic)
    ic = pd.DataFrame(ic)
    ic.columns = ['ic']

    return ic


def calc_ic_stats_table(ic_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    t_stat, p_value = scst.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = scst.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = scst.kurtosis(ic_data)
    ic_summary_table["Ann. IR"] = ic_data.mean() / ic_data.std()
    return ic_summary_table


def mean_information_coefficient(ic, by_time=None):
    """
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each group for our whole timerange?
    What is the mean IC for for each group, each week?

    Parameters
    ----------
    by_time : str (pd time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.

    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between signal and provided
        forward price movement windows.
    """

    grouper = []
    if by_time is not None:
        grouper.append(pd.TimeGrouper(by_time))

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic.index = pd.to_datetime(ic.index, format="%Y%m%d")
        ic = (ic.reset_index().set_index('trade_date').groupby(grouper).mean())

    return ic


def calc_period_wise_weighted_signal_return(signal_data, weight_method):
    """
    Computes period wise period_wise_returns for portfolio weighted by signal
    values. Weights are computed by demeaning signals and dividing
    by the sum of their absolute value (achieving gross leverage of 1).

    Parameters
    ----------
    signal_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['signal', 'return', 'quantile']
    weight_method : {'equal_weight', 'long_only', 'long_short'}

    Returns
    -------
    res : pd.DataFrame
        Period wise period_wise_returns of dollar neutral portfolio weighted by signal value.
        
    """
    def calc_norm_weights(ser, method):
        if method == 'equal_weight':
            ser.loc[:] = 1.0 / len(ser)
        elif method == 'long_short':
            # TODO: do we need to de-mean?
            ser = ser - ser.mean()
        elif method == 'long_only':
            ser = (ser + ser.abs()) / 2.0
        elif method == 'short_only':
            ser = (ser - ser.abs()) / 2.0
        else:
            raise ValueError("method can only be equal_weight, long_only or long_short,"
                             "but [{}] is provided".format(method))
        return ser / ser.abs().sum()
    
    grouper = ['trade_date']
    
    weights = signal_data.groupby(grouper)['signal'].apply(calc_norm_weights, weight_method)
    # df_sig = signal_data['signal'].unstack(level='symbol')
    # weights = df_sig.apply(calc_norm_weights, axis=1, args=(weight_method, ))
    
    weighted_returns = signal_data['return'].multiply(weights, axis=0)
    
    period_wise_returns = weighted_returns.groupby(level='trade_date').sum()
    
    res = pd.DataFrame(period_wise_returns)
    res.columns = ['return']
    return res


def regress_period_wise_signal_return(signal_data, group=False):
    """
    Computes period wise period_wise_returns for portfolio weighted by signal
    values. Weights are computed by demeaning signals and dividing
    by the sum of their absolute value (achieving gross leverage of 1).

    Parameters
    ----------
    signal_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['signal', 'return', 'quantile']

    Returns
    -------
    period_wise_returns : pd.DataFrame
        Period wise period_wise_returns of dollar neutral portfolio weighted by signal
        value.
    """
    
    def regress(df):
        x = df['signal'].values
        y = df['return'].values
        x = sm.add_constant(x)

        mod = sm.OLS(y, x).fit()
        idiosyncractic, signal_return = mod.params
        # return pd.Series(index=['idio', 'signal_return'], data=[idiosyncractic, signal_return])
        return signal_return
    
    grouper = [signal_data.index.get_level_values('trade_date')]
    if group:
        grouper.append('group')
    
    regress_res = signal_data.groupby(grouper).apply(regress)
    
    return pd.DataFrame(regress_res)


'''
def calc_alpha_beta(active_return, period, benchmark_return=None):
    if isinstance(active_return, pd.Series):
        active_return = pd.DataFrame(active_return)
    if isinstance(benchmark_return, pd.Series):
        benchmark_return = pd.DataFrame(benchmark_return)
    benchmark_return = benchmark_return.loc[active_return.index, :]

    alpha_beta = pd.DataFrame()
    
    x = benchmark_return.values
    y = active_return.values
    x = sm.add_constant(x)

    reg_fit = sm.OLS(y, x).fit()
    alpha, beta = reg_fit.params

    alpha_beta.loc['Ann. alpha', period] = \
        (1 + alpha) ** (252.0 / period) - 1
    alpha_beta.loc['beta', period] = beta

    return alpha_beta


'''


def calc_quantile_return_mean_std(signal_data, time_series=False):
    """
    Computes mean returns for signal quantiles across
    provided forward returns columns.

    Parameters
    ----------
    signal_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['signal', 'return', 'quantile']

    Returns
    -------
    res : pd.DataFrame of dict
    
    """
    signal_data = signal_data.copy()
    
    grouper = ['quantile']
    if time_series:
        grouper.append('trade_date')
    
    group_mean_std = signal_data.groupby(grouper)['return'].agg(['mean', 'std', 'count'])

    # TODO: why?
    '''
    std_error_ret = group_mean_std.loc[:, 'std'].copy() / np.sqrt(group_mean_std.loc[:, 'count'].copy())
    '''

    if time_series:
        quantile_daily_mean_std_dic = dict()
        quantiles = np.unique(group_mean_std.index.get_level_values(level='quantile'))
        for q in quantiles:  # loop for different quantiles
            df_q = group_mean_std.loc[pd.IndexSlice[q, :], :]  # bug
            df_q.index = df_q.index.droplevel(level='quantile')
            quantile_daily_mean_std_dic[q] = df_q
        return quantile_daily_mean_std_dic
    else:
        return group_mean_std


def calc_return_diff_mean_std(q1, q2):
    """
    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard error
    of this difference.

    Parameters
    ----------
    q1, q2 : pd.DataFrame
        DataFrame of mean period wise returns by quantile.
        Index is datet, columns = ['mean', 'std', 'count']

    Returns
    -------
    res : pd.DataFrame
        Difference of mean return and corresponding std.
        
    """
    res_raw = pd.merge(q1, q2, how='outer', suffixes=('_1','_2'), left_index=True, right_index=True).fillna(0)
    res_raw['mean_diff'] = res_raw['mean_1'] - res_raw['mean_2']
    res_raw['std'] = np.sqrt(res_raw['mean_1'] **2 + res_raw['mean_2']**2)
    res = res_raw[['mean_diff','std']]
    return res

'''
def period2daily(ser, period, do_roll_mean=False):
    if not period > 1:
        return ser
    
    if do_roll_mean:
        ser = ser.rolling(window=period, min_periods=1, axis=1).mean()
    
    ser_daily_pow = (ser + 1) ** (1. / period)
    return ser_daily_pow - 1.0
'''


def calc_active_cum_return_way2(portfolio_ret, benchmark_ret):
    benchmark_ret = benchmark_ret.loc[portfolio_ret.index]
    
    portfolio_cum = portfolio_ret.add(1.0).cumprod(axis=0)
    benchmark_cum = benchmark_ret.add(1.0).cumprod(axis=0)
    active_cum = portfolio_cum.sub(benchmark_cum.values.flatten(), axis=0) + 1.0
    return active_cum


def calc_active_cum_return(portfolio_ret, benchmark_ret):
    benchmark_ret = benchmark_ret.loc[portfolio_ret.index]
    
    active_ret = portfolio_ret.sub(benchmark_ret.values.flatten(), axis=0)
    active_cum = active_ret.add(1.0).cumprod()
    return active_cum


def price2ret(prices, period=5, axis=None):
    """

    Parameters
    ----------
    prices : pd.DataFrame or pd.Series
        Index is datetime.
    period : int
    axis : {0, 1, None}
    
    Returns
    -------
    ret : pd.DataFrame or pd.Series
    
    """
    ret = prices.pct_change(periods=period, axis=axis)
    return ret


def cum2ret(cum, period=1, axis=None, compound=False):
    """
    
    Parameters
    ----------
    cum : pd.Series
        Starts from zero.
    period : int
    axis : {0, 1, None}
    compound : bool

    Returns
    -------
    ret : pd.Series

    """
    if axis is None:
        kwargs = dict()
    else:
        kwargs = {'axis': axis}
    
    if np.any(cum.min(**kwargs)) < 0:
        raise ValueError("Minimum value of cumulative return is less than zero.")
    cum = cum.add(1.0)
    if compound:
        ret = cum.pct_change(periods=period, **kwargs)
    else:
        ret = cum.diff(periods=period, **kwargs)
    return ret


def ret2cum(ret, compound=False, axis=None):
    """
    
    Parameters
    ----------
    ret : pd.Series
        Starts from zero.
    compound : bool
    axis : {0, 1, None}

    Returns
    -------
    cum : pd.Series

    """
    if axis is None:
        kwargs = dict()
    else:
        kwargs = {'axis': axis}
    if compound:
        # use log to avoid numerical problems
        log_sum = np.log(ret.add(1.0)).cumsum(**kwargs)
        cum = np.exp(log_sum).sub(1.0)
    else:
        cum = ret.cumsum(**kwargs)
    return cum


def calc_performance_metrics(ser, cum_return=False, compound=False):
    """
    Calculate annualized return, volatility and sharpe.
    We assumed data frequency to be day.

    Parameters
    ----------
    ser : pd.DataFrame or pd.Series
        Index is int date, values are floats.
        ser should start from 0.
    cum_return : bool
        Whether ser is cumulative or daily return.
    compound
        Whether calculation of return is compound.

    Returns
    -------
    res : dict

    """
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0]
    
    idx = ser.index
    
    if cum_return:
        cum_ret = ser
        ret = cum2ret(cum_ret, period=1, compound=compound)
    else:
        ret = ser
        cum_ret = ret2cum(ret, compound=compound)
    
    n_trade_days = len(idx)
    n_years = n_trade_days * 1. / CALENDAR_CONST.TRADE_DAYS_PER_YEAR
    
    total_ret = cum_ret.iat[-1]
    if compound:
        ann_ret = np.power(cum_ret.iat[-1] + 1.0, 1. / n_years) - 1
    else:
        ann_ret = total_ret / n_years
    std = np.std(ret)  # use std instead of np.sqrt( (ret**2).sum() / len(ret) )
    ann_vol = std * np.sqrt(CALENDAR_CONST.TRADE_DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol
    # print "ann. ret = {:.1f}%; ann. vol = {:.1f}%, sharpe = {:.2f}".format(ann_ret * 100, ann_vol * 100, sharpe)
    res = {'ann_ret': ann_ret,
           'ann_vol': ann_vol,
           'sharpe': sharpe}
    return res


def period_wise_ret_to_cum(ret, period, compound=False):
    """
    Calculate cumulative returns from N-periods returns, no compounding.
    When 'period' N is greater than 1 the cumulative returns plot is computed
    building and averaging the cumulative returns of N interleaved portfolios
    (started at subsequent periods 1,2,3,...,N) each one rebalancing every N
    periods.
    
    Parameters
    ----------
    ret: pd.Series or pd.DataFrame
        pd.Series containing N-periods returns
    period: integer
        Period for which the returns are computed
    compound : bool
        Whether calculate using compound return.
    
    Returns
    -------
    pd.Series
        Cumulative returns series starting from zero.
        
    """
    if isinstance(ret, pd.DataFrame):
        # deal with each column recursively
        return ret.apply(period_wise_ret_to_cum, axis=0, args=(period,))
    elif isinstance(ret, pd.Series):
        if period == 1:
            return ret.add(1).cumprod().sub(1.0)
        
        # invest in each portfolio separately
        periods_index = np.arange(len(ret.index)) // period
        period_portfolios = ret.groupby(by=periods_index, axis=0).apply(lambda ser: pd.DataFrame(np.diag(ser)))
        period_portfolios.index = ret.index
        
        # cumulate returns separately
        if compound:
            cum_returns = period_portfolios.add(1).cumprod().sub(1.0)
        else:
            cum_returns = period_portfolios.cumsum()
        
        # since capital of all portfolios are the same, return in all equals average return
        res = cum_returns.mean(axis=1)
        
        return res
    else:
        raise NotImplementedError("ret must be Series or DataFrame.")

