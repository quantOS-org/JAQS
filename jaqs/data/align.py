# encoding: utf-8
"""
Function align is used by DataView, to expand and re-arrange data in a DataFrame
according to their available time, which is stored in another DataFrame.

"""
from __future__ import print_function
import numpy as np
import pandas as pd
from jaqs.util import is_numeric


def _get_neareast(df_ann, df_value, date):
    """
    Get the value whose ann_date is earlier and nearest to date.
    
    Parameters
    ----------
    df_ann : np.ndarray
        announcement dates. shape = (n_quarters, n_securities)
    df_value : np.ndarray
        announcement values. shape = (n_quarters, n_securities)
    date : np.ndarray
        shape = (1,)

    Returns
    -------
    res : np.array
        The value whose ann_date is earlier and nearest to date. shape (n_securities)

    """
    """
    df_ann.fillna(99999999, inplace=True)  # IMPORTANT: At cells where no quarterly data is available,
                                           # we know nothing, thus it will be filled nan in the next step
    """
    if is_numeric(df_value):
        df_value = df_value.astype(float)
    mask = date[0] >= df_ann
    # res = np.where(mask, df_value, np.nan)
    n = df_value.shape[1]
    res = np.empty(n, dtype=df_value.dtype)
    
    # for each column, get the last True value
    for i in range(n):
        v = df_value[:, i]
        m = mask[:, i]
        r = v[m]
        res[i] = r[-1] if len(r) else np.nan
    
    return res
    

def align(df_value, df_ann, date_arr):
    """
    Expand low frequency DataFrame df_value to frequency of data_arr using announcement date from df_ann.
    
    Parameters
    ----------
    df_ann : pd.DataFrame
        DataFrame of announcement dates. shape = (n_quarters, n_securities)
    df_value : pd.DataFrame
        DataFrame of announcement values. shape = (n_quarters, n_securities)
    date_arr : list or np.array
        Target date array. dtype = int

    Returns
    -------
    df_res : pd.DataFrame
        Expanded DataFrame. shape = (n_days, n_securities)

    """
    df_ann = df_ann.fillna(99999999).astype(int)
    
    date_arr = np.asarray(date_arr, dtype=int)
    
    res = np.apply_along_axis(lambda date: _get_neareast(df_ann.values, df_value.values, date), 1, date_arr.reshape(-1, 1))

    df_res = pd.DataFrame(index=date_arr, columns=df_value.columns, data=res)
    return df_res
