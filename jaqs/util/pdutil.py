# encoding: utf-8

import numpy as np
import pandas as pd

from jaqs.util import numeric


def to_quantile(df, n_quantiles=5, axis=1):
    """
    Convert cross-section values to the quantile number they belong.
    Small values get small quantile numbers.
    
    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        index date, column symbols
    n_quantiles : int
        The number of quantile to be divided to.
    axis : int
        Axis to apply quantilize.

    Returns
    -------
    res : DataFrame
        index date, column symbols

    """
    # TODO: unnecesssary warnings
    # import warnings
    # warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='py_exp')
    res_arr = numeric.quantilize_without_nan(df.values, n_quantiles=n_quantiles, axis=axis)
    if isinstance(df, pd.DataFrame):
        res = pd.DataFrame(index=df.index, columns=df.columns, data=res_arr)
    elif isinstance(df, pd.Series):
        res = pd.Series(index=df.index, data=res_arr)
    else:
        raise ValueError
    return res


def fillinf(df):
    return df.replace([np.inf, -np.inf], np.nan)


def group_df_to_dict(df, by):
    gp = df.groupby(by=by)
    res = {key: value for key, value in gp}
    return res


def rank_with_mask(df, axis=1, mask=None, normalize=False):
    not_nan_mask = (~df.isnull())
    
    if mask is None:
        mask = not_nan_mask
    else:
        mask = np.logical_and(not_nan_mask, mask)
    
    rank = df[mask].rank(axis=axis, na_option='keep')
    
    if normalize:
        rank = rank.div(mask.sum(axis=axis), axis=(1 - axis))
    return rank

