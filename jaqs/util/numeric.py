# encoding: utf-8
import numpy as np


def quantilize_without_nan(mat, n_quantiles=5, axis=-1):
    mask = np.isnan(mat)
    
    rank = mat.argsort(axis=axis).argsort(axis=axis)  # int
    
    count = np.sum(~mask, axis=axis)  # int
    divisor = count * 1. / n_quantiles  # float
    shape = list(mat.shape)
    shape[axis] = 1
    divisor = divisor.reshape(*shape)
    
    res = np.floor(rank / divisor) + 1.0
    res[mask] = np.nan
    
    return res
