# encoding: utf-8

import numpy as np
import pandas as pd
import scipy.stats as scst
import statsmodels.api as sm


def calc_factor_ic(factor_data):
    """
    Computes the Spearman Rank Correlation based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['factor', 'return', 'quantile']

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and provided forward returns.
        
    """
    def src_ic(df):
        _ic = scst.spearmanr(df['factor'], df['return'])[0]
        return _ic

    factor_data = factor_data.copy()

    grouper = ['trade_date']

    ic = factor_data.groupby(grouper).apply(src_ic)
    ic = pd.DataFrame(ic)
    ic.columns = ['ic']

    return ic


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
        Mean Spearman Rank correlation between factor and provided
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


def calc_period_wise_weighted_factor_return(factor_data, weight_method):
    """
    Computes period wise period_wise_returns for portfolio weighted by factor
    values. Weights are computed by demeaning factors and dividing
    by the sum of their absolute value (achieving gross leverage of 1).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['factor', 'return', 'quantile']
    weight_method : {'equal_weight', 'long_only', 'long_short'}

    Returns
    -------
    res : pd.DataFrame
        Period wise period_wise_returns of dollar neutral portfolio weighted by factor value.
        
    """
    def norm_factor(ser, method):
        if method == 'equal_weight':
            ser.loc[:] = 1.0 / len(ser)
        elif method == 'long_short':
            ser = ser - ser.mean()
        elif method == 'long_only':
            if ser.min() <= 0:
                ser = ser + ser.min()
        else:
            raise ValueError("method can only be equal_weight, long_only or long_short,"
                             "but [{}] is provided".format(method))
        return ser / ser.abs().sum()
    
    grouper = [factor_data.index.get_level_values('trade_date')]
    
    weights = factor_data.groupby(grouper)['factor'].apply(norm_factor, weight_method)
    
    weighted_returns = factor_data['return'].multiply(weights, axis=0)
    
    period_wise_returns = weighted_returns.groupby(level='trade_date').sum()
    
    res = pd.DataFrame(period_wise_returns)
    res.columns = ['return']
    return res


def regress_period_wise_factor_return(factor_data, group=False):
    """
    Computes period wise period_wise_returns for portfolio weighted by factor
    values. Weights are computed by demeaning factors and dividing
    by the sum of their absolute value (achieving gross leverage of 1).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['factor', 'return', 'quantile']

    Returns
    -------
    period_wise_returns : pd.DataFrame
        Period wise period_wise_returns of dollar neutral portfolio weighted by factor
        value.
    """
    
    def regress(df):
        x = df['factor'].values
        y = df['return'].values
        x = sm.add_constant(x)

        mod = sm.OLS(y, x).fit()
        idiosyncractic, factor_return = mod.params
        # return pd.Series(index=['idio', 'factor_return'], data=[idiosyncractic, factor_return])
        return factor_return
    
    grouper = [factor_data.index.get_level_values('trade_date')]
    if group:
        grouper.append('group')
    
    regress_res = factor_data.groupby(grouper).apply(regress)
    
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


def calc_quantile_return_mean_std(factor_data, time_series=False):
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Index is pd.MultiIndex ['trade_date', 'symbol'], columns = ['factor', 'return', 'quantile']

    Returns
    -------
    res : pd.DataFrame of dict
    
    """
    factor_data = factor_data.copy()
    
    grouper = ['quantile']
    if time_series:
        grouper.append('trade_date')
    
    group_mean_std = factor_data.groupby(grouper)['return'].agg(['mean', 'std', 'count'])

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
    assert np.all(q1.index == q2.index)
    assert np.all(q1.columns == q2.columns)
    
    res = pd.DataFrame(index=q1.index, columns=['mean_diff', 'std'])
    res.loc[:, 'mean_diff'] = q1['mean'] - q2['mean']
    res.loc[:, 'std'] = np.sqrt(q1['std']**2 - q2['std']**2)
    return res


def price_to_return(price, period, cumulative=False, axis=0):
    """
    Convert prices to net value starting from 1.0
    For DataFrame, operation is on axis=0.
    
    Parameters
    ----------
    price : pd.DataFrame or pd.Series
    period : int
    cumulative : bool
        If true convert to cumulative return instead of period-wise return.

    Returns
    -------
    res : pd.DataFrame

    """
    if isinstance(price, pd.Series):
        price_df = pd.DataFrame(price)
    else:
        price_df = price
    ret = price_df.pct_change(periods=period, axis=axis)
    if cumulative:
        ret = ret.add(1.0).cumprod()
    return ret


def period2daily(ser, period, do_roll_mean=False):
    if not period > 1:
        return ser
    
    if do_roll_mean:
        ser = ser.rolling(window=period, min_periods=1, axis=1).mean()
    
    ser_daily_pow = (ser + 1) ** (1. / period)
    return ser_daily_pow - 1.0


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


def daily_ret_to_cum(df_ret, axis=0):
    cum = df_ret.add(1.0).cumprod(axis=axis)
    return cum


def calc_forward_return(prices, period=5, axis=0):
    """

    Parameters
    ----------
    prices : pd.DataFrame
        Index is datetime.
    period : int
    axis : {0, 1}

    Returns
    -------
    forward_returns : pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    """
    ret = price_to_return(prices, period=period, axis=axis)
    fwd_ret = ret.shift(-period)
    return fwd_ret


