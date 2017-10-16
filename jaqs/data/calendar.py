# encoding: utf-8

import datetime

import numpy as np
import pandas as pd

from jaqs.data.dataservice import RemoteDataService
from jaqs.util import dtutil


class Calendar(object):
    """
    A calendar for manage trade date.
    
    Attributes
    ----------
    data_api :

    """
    
    def __init__(self, data_api=None):
        if data_api is None:
            self.data_api = RemoteDataService()
        else:
            self.data_api = data_api

    def get_trade_date_range(self, begin, end):
        """
        Get array of trade dates within given range.
        Return zero size array if no trade dates within range.
        
        Parameters
        ----------
        begin : int
            YYmmdd
        end : int

        Returns
        -------
        trade_dates_arr : np.ndarray
            dtype = int

        """
        filter_argument = self.data_api._dic2url({'start_date': begin,
                                                  'end_date': end})

        df_raw, msg = self.data_api.query("jz.secTradeCal", fields="trade_date",
                                          filter=filter_argument, orderby="")
        if df_raw.empty:
            return np.array([], dtype=int)

        trade_dates_arr = df_raw['trade_date'].values.astype(int)
        return trade_dates_arr

    def get_last_trade_date(self, date):
        """
        
        Parameters
        ----------
        date : int

        Returns
        -------
        res : int

        """
        dt = dtutil.convert_int_to_datetime(date)
        delta = pd.Timedelta(weeks=2)
        dt_old = dt - delta
        date_old = dtutil.convert_datetime_to_int(dt_old)
        
        dates = self.get_trade_date_range(date_old, date)
        mask = dates < date
        res = dates[mask][-1]
        
        return res

    def is_trade_date(self, date):
        """
        Check whether date is a trade date.

        Parameters
        ----------
        date : int

        Returns
        -------
        bool

        """
        dates = self.get_trade_date_range(date, date)
        return len(dates) > 0

    def get_next_trade_date(self, date):
        """
        
        Parameters
        ----------
        date : int

        Returns
        -------
        res : int

        """
        dt = dtutil.convert_int_to_datetime(date)
        delta = pd.Timedelta(weeks=2)
        dt_new = dt + delta
        date_new = dtutil.convert_datetime_to_int(dt_new)
    
        dates = self.get_trade_date_range(date, date_new)
        mask = dates > date
        res = dates[mask][0]
    
        return res
