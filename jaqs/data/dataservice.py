# encoding: UTF-8
"""
Module dataservice defines DataService and RemoteDataService.

DataService is just an interface. RemoteDataService is a wrapper class for DataApi.
It inherits all methods of DataApi and implements several convenient methods making
query data more natural and easy.

"""

from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from abc import abstractmethod
from six import with_metaclass

try:
    basestring
except NameError:
    basestring = str

import numpy as np
import pandas as pd

from jaqs.trade.event import EVENT_TYPE, Event
from jaqs.data import DataApi
from jaqs.data import align
import jaqs.util as jutil


class InitializeError(Exception):
    def __init__(self, *args):
        super(InitializeError, self).__init__(*args)


class NotLoginError(Exception):
    def __init__(self, *args):
        super(NotLoginError, self).__init__(*args)


class QueryDataError(Exception):
    def __init__(self, *args):
        super(QueryDataError, self).__init__(*args)


class Singleton(type):
    """
    Metaclass that can make a class to be singleton.
    
    Usage:
        class Foo(with_metaclass(Singleton, OtherMetaClass)):
            pass
        
    """
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DataService(object):
    """
    DataService is an abstract base class providing both historic and live data
    from various data sources.

    Derived classes of DataService may use different data source, but use the same API.

    Attributes
    ----------
    ctx : Context
        Running context.

    Methods
    -------
    daily
    quote
    bar
    bar_quote
    query
    subscribe

    """

    def __init__(self):
        self.ctx = None
    
    def register_context(self, context):
        self.ctx = context
    
    @abstractmethod
    def quote(self, symbol, fields=""):
        """
        Query latest market data in DataFrame.
        
        Parameters
        ----------
        symbol : str
        fields : str, optional
            default ""

        Returns
        -------
        df : pd.DataFrame
        err_msg : str
            error code and error message joined by comma

        """
        pass

    @abstractmethod
    def bar_quote(self, symbol, start_time=200000, end_time=160000,
                  trade_date=0, freq="1M", fields="", data_format="", **kwargs):
        """
        Query minute bars with latest quote, return DataFrame.

        Parameters
        ----------
        symbol : str
            support multiple securities, separated by comma.
        start_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market open time.
        end_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market close time.
        trade_date : int (YYYMMDD) or str ('YYYY-MM-DD')
            Default is current trade_date.
        fields : str, optional
            separated by comma ',', default "" (all fields included).
        freq : trade.common.MINBAR_TYPE, optional
            {'1m', '5m', '15m'}, Minute bar type, default is '1m'

        Returns
        -------
        df : pd.DataFrame
            columns:
                symbol, code, date, time, trade_date, freq, open, high, low, close, volume, turnover, vwap, oi
        err_msg : str
            error code and error message joined by comma

        Examples
        --------
        df, err_msg = api.bar("000001.SH,cu1709.SHF", start_time="09:56:00", end_time="13:56:00",
                          trade_date="20170823", fields="open,high,low,last,volume", freq="5m")

        """
        pass
        
    @abstractmethod
    def daily(self, symbol, start_date, end_date, fields="", adjust_mode=None):
        """
        Query dar bar,
        support auto-fill suspended securities data,
        support auto-adjust for splits, dividends and distributions.

        Parameters
        ----------
        symbol : str
            support multiple securities, separated by comma.
        start_date : int or str
            YYYMMDD or 'YYYY-MM-DD'
        end_date : int or str
            YYYMMDD or 'YYYY-MM-DD'
        fields : str, optional
            separated by comma ',', default "" (all fields included).
        adjust_mode : str or None, optional
            None for no adjust;
            'pre' for forward adjust;
            'post' for backward adjust.

        Returns
        -------
        df : pd.DataFrame
            columns:
                symbol, code, trade_date, open, high, low, close, volume, turnover, vwap, oi, suspended
        err_msg : str
            error code and error message joined by comma

        Examples
        --------
        df, err_msg = api.daily("00001.SH,cu1709.SHF",start_date=20170503, end_date=20170708,
                            fields="open,high,low,last,volume", fq=None, skip_suspended=True)

        """
        pass
    
    @abstractmethod
    def bar(self, symbol, start_time=200000, end_time=160000, trade_date=None, freq='1M', fields=""):
        """
        Query minute bars of various type, return DataFrame.

        Parameters
        ----------
        symbol : str
            support multiple securities, separated by comma.
        start_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market open time.
        end_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market close time.
        trade_date : int (YYYMMDD) or str ('YYYY-MM-DD')
            Default is current trade_date.
        fields : str, optional
            separated by comma ',', default "" (all fields included).
        freq : trade.common.MINBAR_TYPE, optional
            {'1m', '5m', '15m'}, Minute bar type, default is '1m'

        Returns
        -------
        df : pd.DataFrame
            columns:
                symbol, code, date, time, trade_date, freq, open, high, low, close, volume, turnover, vwap, oi
        err_msg : str
            error code and error message joined by comma

        Examples
        --------
        df, err_msg = api.bar("000001.SH,cu1709.SHF", start_time="09:56:00", end_time="13:56:00",
                          trade_date="20170823", fields="open,high,low,last,volume", freq="5m")

        """
        # TODO data_server DOES NOT know "current date".
        pass
    
    @abstractmethod
    def query(self, view, filter, fields):
        """
        Query reference data.
        Input query type and parameters, return DataFrame.
        
        Parameters
        ----------
        view : str
            Type of reference data. See doc for details.
        filter : str
            Query conditions, separated by '&'.
        fields : str
            Fields to return, separated by ','.

        Returns
        -------
        df : pd.DataFrame
        err_msg : str
            error code and error message joined by comma

        """
        pass
    

class RemoteDataService(with_metaclass(Singleton, DataService)):
    """
    RemoteDataService is a concrete class using data from remote server's database.
    It wraps DataApi and simplify usage.

    """

    def __init__(self):
        # print("Init RemoteDataService DEBUG")
        super(RemoteDataService, self).__init__()
        
        self.data_api = None

        self._address = ""
        self._username = ""
        self._password = ""
        self._timeout = 60
        self._trade_dates_df = None
        
        self._REPORT_DATE_FIELD_NAME = 'report_date'
        
    '''
    def __del__(self):
        self.data_api.close()

    '''

    def init_from_config(self, props):
        """
        
        Parameters
        ----------
        props : dict
            Configurations used for initialization.

        Example
        -------
        {"remote.data.address": "tcp://Address:Port",
        "remote.data.username": "your username",
        "remote.data.password": "your password"}

        """

        def get_from_list_of_dict(l, key, default=None):
            res = None
            for dic in l:
                res = dic.get(key, None)
                if res is not None:
                    break
            if res is None:
                res = default
            return res
        
        props_default = dict()  # jutil.read_json(jutil.join_relative_path('etc/data_config.json'))
        dic_list = [props, props_default]
        
        address = get_from_list_of_dict(dic_list, "remote.data.address", "")
        username = get_from_list_of_dict(dic_list, "remote.data.username", "")
        password = get_from_list_of_dict(dic_list, "remote.data.password", "")
        time_out = get_from_list_of_dict(dic_list, "timeout", 60)

        print("\nBegin: DataApi login {}@{}".format(username, address))
        INDENT = ' ' * 4
        
        if self.data_api_loginned:
            if (address == "") or (username == "") or (password == ""):
                raise InitializeError("no address, username or password available!")
            elif ((address == self._address) and (time_out == self._timeout)
                and (username == self._username) and (password == self._password)):
                print(INDENT + "Already login as {:s}, skip init_from_config".format(username))
                return '0,'  # do not login with the same props again
            else:
                self.data_api.close()
                self.data_api = None

        self._address = address
        self._username = username
        self._password = password
        self._timeout = time_out
        
        data_api = DataApi(self._address, use_jrpc=False)
        data_api.set_timeout(timeout=self._timeout)
        r, err_msg = data_api.login(username=self._username, password=self._password)
        if not r:
            print(INDENT + "login failed: err_msg = '{}'\n".format(err_msg))
        else:
            self.data_api = data_api
            print(INDENT + "login success \n")
        
        trade_dates_df, err_msg1 = self.query("jz.secTradeCal", fields="trade_date", filter="", orderby="")
        if not trade_dates_df.empty:
            self._trade_dates_df = trade_dates_df
        else:
            print("No trade date.\n".format(err_msg1))

        return err_msg
        
    @property
    def data_api_loginned(self):
        return (self.data_api is not None) and (self.data_api._loggined) and (self.data_api._connected)
    
    def _raise_error_if_no_data_api(self):
        if not self.data_api_loginned:
            raise NotLoginError("Please first login using init_from_config.")
    
    @staticmethod
    def _raise_error_if_msg(err_msg):
        splited = err_msg.split(',')
        if not (splited and (splited[0] == '0')):
            raise QueryDataError(err_msg)
    
    # -----------------------------------------------------------------------------------
    # Basic APIs
    def daily(self, symbol, start_date, end_date,
              fields="", adjust_mode=None):
        """
        Query dar bar,
        support auto-fill suspended securities data,
        support auto-adjust for splits, dividends and distributions.

        Parameters
        ----------
        symbol : str
            support multiple securities, separated by comma.
        start_date : int or str
            YYYMMDD or 'YYYY-MM-DD'
        end_date : int or str
            YYYMMDD or 'YYYY-MM-DD'
        fields : str, optional
            separated by comma ',', default "" (all fields included).
        adjust_mode : str or None, optional
            None for no adjust;
            'pre' for forward adjust;
            'post' for backward adjust.

        Returns
        -------
        df : pd.DataFrame
            columns:
                symbol, code, trade_date, open, high, low, close, volume, turnover, vwap, oi, suspended
        err_msg : str
            error code and error message joined by comma

        Examples
        --------
        df, err_msg = api.daily("00001.SH,cu1709.SHF",start_date=20170503, end_date=20170708,
                            fields="open,high,low,last,volume", fq=None, skip_suspended=True)

        """
        self._raise_error_if_no_data_api()
        
        df, err_msg = self.data_api.daily(symbol=symbol, start_date=start_date, end_date=end_date,
                                          fields=fields, adjust_mode=adjust_mode, data_format="")

        self._raise_error_if_msg(err_msg)
        
        # TODO there will be duplicate entries when on stocks' IPO day
        df = df.drop_duplicates()
        return df, err_msg

    def bar(self, symbol,
            start_time=200000, end_time=160000, trade_date=None,
            freq='1M', fields=""):
        """
        Query minute bars of various type, return DataFrame.

        Parameters
        ----------
        symbol : str
            support multiple securities, separated by comma.
        start_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market open time.
        end_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market close time.
        trade_date : int (YYYMMDD) or str ('YYYY-MM-DD')
            Default is current trade_date.
        fields : str, optional
            separated by comma ',', default "" (all fields included).
        freq : trade.common.MINBAR_TYPE, optional
            {'1m', '5m', '15m'}, Minute bar type, default is '1m'

        Returns
        -------
        df : pd.DataFrame
            columns:
                symbol, code, date, time, trade_date, freq, open, high, low, close, volume, turnover, vwap, oi
        err_msg : str
            error code and error message joined by comma

        Examples
        --------
        df, err_msg = api.bar("000001.SH,cu1709.SHF", start_time="09:56:00", end_time="13:56:00",
                          trade_date="20170823", fields="open,high,low,last,volume", freq="5m")

        """
        self._raise_error_if_no_data_api()
        
        df, err_msg = self.data_api.bar(symbol=symbol, fields=fields,
                                        start_time=start_time, end_time=end_time, trade_date=trade_date,
                                        freq=freq, data_format="")
        
        self._raise_error_if_msg(err_msg)
        return df, err_msg
    
    def quote(self, symbol, fields=""):
        """
        Query latest market data in DataFrame.
        
        Parameters
        ----------
        symbol : str
        fields : str, optional
            default ""

        Returns
        -------
        df : pd.DataFrame
        err_msg : str
            error code and error message joined by comma

        """
        self._raise_error_if_no_data_api()
        
        df, err_msg = self.data_api.quote(symbol=symbol, fields=fields)
        
        self._raise_error_if_msg(err_msg)
        
        return df, err_msg
    
    def bar_quote(self, symbol, start_time=200000, end_time=160000,
                  trade_date=0, freq="1M", fields="", data_format="", **kwargs):
        """
        Query minute bars with latest quote, return DataFrame.

        Parameters
        ----------
        symbol : str
            support multiple securities, separated by comma.
        start_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market open time.
        end_time : int (HHMMSS) or str ('HH:MM:SS')
            Default is market close time.
        trade_date : int (YYYMMDD) or str ('YYYY-MM-DD')
            Default is current trade_date.
        fields : str, optional
            separated by comma ',', default "" (all fields included).
        freq : trade.common.MINBAR_TYPE, optional
            {'1m', '5m', '15m'}, Minute bar type, default is '1m'

        Returns
        -------
        df : pd.DataFrame
            columns:
                symbol, code, date, time, trade_date, freq, open, high, low, close, volume, turnover, vwap, oi
        err_msg : str
            error code and error message joined by comma

        Examples
        --------
        df, err_msg = api.bar("000001.SH,cu1709.SHF", start_time="09:56:00", end_time="13:56:00",
                          trade_date="20170823", fields="open,high,low,last,volume", freq="5m")

        """
        self._raise_error_if_no_data_api()

        df, err_msg = self.data_api.bar_quote(symbol=symbol, start_time=start_time, end_time=end_time,
                                              trade_date=trade_date, freq=freq, fields=fields)

        self._raise_error_if_msg(err_msg)
        return df, err_msg
    
    def query(self, view, filter="", fields="", **kwargs):
        """
        Get various reference data.
        
        Parameters
        ----------
        view : str or unicode
            data source.
        fields : str or unicode
            Separated by ','
        filter : str or unicode
            filter expressions.
        kwargs

        Returns
        -------
        df : pd.DataFrame
        err_msg : str
            error code and error message, joined by ','
        
        Examples
        --------
        res3, err_msg3 = ds.query("lb.secDailyIndicator", fields="price_level,high_52w_adj,low_52w_adj",\
                              filter="start_date=20170907&end_date=20170907",\
                              orderby="trade_date",\
                              data_format='pandas')
            view does not change. fileds can be any field predefined in reference data api.

        """
        self._raise_error_if_no_data_api()
        
        df, err_msg = self.data_api.query(view, fields=fields, filter=filter, **kwargs)
        
        self._raise_error_if_msg(err_msg)
        return df, err_msg

    # -----------------------------------------------------------------------------------
    # Convenient Functions
    
    @staticmethod
    def _dic2url(d):
        """
        Convert a dict to str like 'k1=v1&k2=v2'
        
        Parameters
        ----------
        d : dict

        Returns
        -------
        str

        """
        l = ['='.join([key, str(value)]) for key, value in d.items()]
        return '&'.join(l)

    def query_lb_fin_stat(self, type_, symbol, start_date, end_date, fields="", drop_dup_cols=None):
        """
        Helper function to call data_api.query with 'lb.income' more conveniently.
        
        Parameters
        ----------
        type_ : {'income', 'balance_sheet', 'cash_flow'}
        symbol : str
            separated by ','
        start_date : int
            Annoucement date in results will be no earlier than start_date
        end_date : int
            Annoucement date in results will be no later than start_date
        fields : str, optional
            separated by ',', default ""
        drop_dup_cols : list or tuple
            Whether drop duplicate entries according to drop_dup_cols.

        Returns
        -------
        df : pd.DataFrame
            index date, columns fields
        err_msg : str

        """
        view_map = {'income': 'lb.income', 'cash_flow': 'lb.cashFlow', 'balance_sheet': 'lb.balanceSheet',
                    'fin_indicator': 'lb.finIndicator'}
        view_name = view_map.get(type_, None)
        if view_name is None:
            raise NotImplementedError("type_ = {:s}".format(type_))
        
        dic_argument = {'symbol': symbol,
                        'start_date': start_date,
                        'end_date': end_date,
                        # 'update_flag': '0'
                       }
        if view_name != 'lb.finIndicator':
            dic_argument.update({'report_type': '408001000'})  # we do not use single quarter single there are zeros
            """
            408001000: joint
            408002000: joint (single quarter)
            """
        
        filter_argument = self._dic2url(dic_argument)  # 0 means first time, not update
        
        res, err_msg = self.query(view_name, fields=fields, filter=filter_argument,
                                  order_by=self._REPORT_DATE_FIELD_NAME)
        self._raise_error_if_msg(err_msg)
        
        # change data type
        try:
            cols = list(set.intersection({'ann_date', 'report_date'}, set(res.columns)))
            dic_dtype = {col: np.integer for col in cols}
            res = res.astype(dtype=dic_dtype)
        except:
            pass
        
        if drop_dup_cols is not None:
            res = res.sort_values(by=drop_dup_cols, axis=0)
            res = res.drop_duplicates(subset=drop_dup_cols, keep='first')
        
        return res, err_msg

    def query_lb_dailyindicator(self, symbol, start_date, end_date, fields=""):
        """
        Helper function to call data_api.query with 'lb.secDailyIndicator' more conveniently.
        
        Parameters
        ----------
        symbol : str
            separated by ','
        start_date : int
        end_date : int
        fields : str, optional
            separated by ',', default ""

        Returns
        -------
        df : pd.DataFrame
            index date, columns fields
        err_msg : str
        
        """
        filter_argument = self._dic2url({'symbol': symbol,
                                         'start_date': start_date,
                                         'end_date': end_date})
    
        res, err_msg = self.query("lb.secDailyIndicator", fields=fields,
                                  filter=filter_argument, orderby="trade_date")
        self._raise_error_if_msg(err_msg)
        return res, err_msg

    def query_index_weights_raw(self, index, trade_date):
        """
        Return all securities that have been in index during start_date and end_date.
        
        Parameters
        ----------
        index : str
            separated by ','
        trade_date : int

        Returns
        -------
        pd.DataFrame

        """
        if index == '000300.SH':
            index = '399300.SZ'
            
        filter_argument = self._dic2url({'index_code': index,
                                         'trade_date': trade_date})
    
        df_io, err_msg = self.query("lb.indexWeight", fields="", filter=filter_argument)
        self._raise_error_if_msg(err_msg)
        
        df_io = df_io.set_index('symbol')
        df_io = df_io.astype({'weight': float, 'trade_date': np.integer})
        df_io.loc[:, 'weight'] = df_io['weight'] / 100.
        return df_io

    def query_index_weights_range(self, index, start_date, end_date):
        """
        Return all securities that have been in index during start_date and end_date.
        
        Parameters
        ----------
        index : str
            separated by ','
        trade_date : int

        Returns
        -------
        pd.DataFrame

        """
        if index == '000300.SH':
            index = '399300.SZ'
    
        filter_argument = self._dic2url({'index_code': index,
                                         'start_date': start_date,
                                         'end_date': end_date})
    
        df_io, msg = self.query("lb.indexWeightRange", fields="",
                                filter=filter_argument)
        if msg != '0,':
            print(msg)
        # df_io = df_io.set_index('symbol')
        df_io = df_io.astype({'weight': float, 'trade_date': np.integer})
        df_io.loc[:, 'weight'] = df_io['weight'] / 100.
        df_io = df_io.pivot(index='trade_date', columns='symbol', values='weight')
        df_io = df_io.fillna(0.0)
        return df_io

    def query_index_weights_daily(self, index, start_date, end_date):
        """
        Return all securities that have been in index during start_date and end_date.
        
        Parameters
        ----------
        index : str
        start_date : int
        end_date : int

        Returns
        -------
        res : pd.DataFrame
            Index is trade_date, columns are symbols.

        """
        
        start_dt = jutil.convert_int_to_datetime(start_date)
        start_dt_extended = start_dt - pd.Timedelta(days=45)
        start_date_extended = jutil.convert_datetime_to_int(start_dt_extended)
        trade_dates = self.query_trade_dates(start_date_extended, end_date)
        
        df_weight_raw = self.query_index_weights_range(index, start_date=start_date_extended, end_date=end_date)
        res = df_weight_raw.reindex(index=trade_dates)
        res = res.fillna(method='ffill')
        res = res.loc[res.index >= start_date]
        res = res.loc[res.index <= end_date]
        
        mask_col = res.sum(axis=0) > 0
        res = res.loc[:, mask_col]
        
        return res

    def _get_index_comp(self, index, start_date, end_date):
        """
        Return all securities that have been in index during start_date and end_date.
        
        Parameters
        ----------
        index : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        list

        """
        filter_argument = self._dic2url({'index_code': index,
                                         'start_date': start_date,
                                         'end_date': end_date})
    
        df_io, err_msg = self.query("lb.indexCons", fields="",
                                    filter=filter_argument, orderby="symbol")
        self._raise_error_if_msg(err_msg)
        return df_io, err_msg
    
    def query_index_member(self, index, start_date, end_date):
        """
        Return list of symbols that have been in index during start_date and end_date.
        
        Parameters
        ----------
        index : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        list

        """
        df_io, err_msg = self._get_index_comp(index, start_date, end_date)
        return list(np.unique(df_io.loc[:, 'symbol']))
    
    def query_index_member_daily(self, index, start_date, end_date):
        """
        Get index components on each day during start_date and end_date.
        
        Parameters
        ----------
        index : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        res : pd.DataFrame
            index dates, columns all securities that have ever been components,
            values are 0 (not in) or 1 (in)

        """
        df_io, err_msg = self._get_index_comp(index, start_date, end_date)
        if err_msg != '0,':
            print(err_msg)
        
        def str2int(s):
            if isinstance(s, basestring):
                return int(s) if s else 99999999
            elif isinstance(s, (int, np.integer, float, np.float)):
                return s
            else:
                raise NotImplementedError("type s = {}".format(type(s)))

        df_io.loc[:, 'in_date'] = df_io.loc[:, 'in_date'].apply(str2int)
        df_io.loc[:, 'out_date'] = df_io.loc[:, 'out_date'].apply(str2int)
        
        # df_io.set_index('symbol', inplace=True)
        dates = self.query_trade_dates(start_date=start_date, end_date=end_date)

        dic = dict()
        gp = df_io.groupby(by='symbol')
        for sec, df in gp:
            mask = np.zeros_like(dates, dtype=np.integer)
            for idx, row in df.iterrows():
                bool_index = np.logical_and(dates > row['in_date'], dates < row['out_date'])
                mask[bool_index] = 1
            dic[sec] = mask
            
        res = pd.DataFrame(index=dates, data=dic)
        res.index.name = 'trade_date'
        
        return res

    def _get_universe_comp(self, universe, start_date, end_date):
        """
        Return all securities that have been in index during start_date and end_date.

        Parameters
        ----------
        universe : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        list

        """
        # Remove mkt suffix

        filter_argument = self._dic2url({'univ_id'   : universe,
                                         'start_date': start_date,
                                         'end_date'  : end_date})

        df_io, err_msg = self.query("jz.univMember", fields="",
                                    filter=filter_argument, orderby="univ_member")
        self._raise_error_if_msg(err_msg)

        return df_io, err_msg

    def query_universe_member(self, universe, start_date, end_date):
        """
        Return list of symbols that have been in index during start_date and end_date.

        Parameters
        ----------
        index : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        list

        """
        df_io, err_msg = self._get_universe_comp(universe, start_date, end_date)
        return list(np.unique(df_io.loc[:, 'univ_member']))

    def query_universe_member_daily(self, universe, start_date, end_date):
        """
        Get universe's members on each day during start_date and end_date.

        Parameters
        ----------
        universe : str
        start_date : int
        end_date : int

        Returns
        -------
        res : pd.DataFrame
            index dates, columns all securities that have ever been components,
            values are 0 (not in) or 1 (in)

        """
        df_io, err_msg = self._get_universe_comp(universe, start_date, end_date)
        if err_msg != '0,':
            print(err_msg)

        def str2int(s):
            if isinstance(s, basestring):
                return int(s) if s else 99999999
            elif isinstance(s, (int, np.integer, float, np.float)):
                return s
            else:
                raise NotImplementedError("type s = {}".format(type(s)))

        # df_io.loc[:, 'in_date'] = df_io.loc[:, 'in_date'].apply(str2int)
        # df_io.loc[:, 'out_date'] = df_io.loc[:, 'out_date'].apply(str2int)

        # df_io.set_index('symbol', inplace=True)
        dates = self.query_trade_dates(start_date=start_date, end_date=end_date)
        dates = pd.Series(dates)

        df_io = df_io.rename(columns={'univ_weight': 'weight', 'univ_member':'symbol'})
        df_io = df_io[['trade_date', 'symbol','weight']]

        dic = dict()
        gp = df_io.groupby(by='symbol')
        for sec, df in gp:
            mask = np.zeros_like(dates, dtype=np.integer)
            for idx, row in df.iterrows():
                bool_index = dates.isin(df['trade_date'])
                mask[bool_index] = 1
            dic[sec] = mask

        res = pd.DataFrame(index=dates, data=dic)
        res.index.name = 'trade_date'

        return res

    def query_universe_weights_range(self, universe, start_date, end_date):
        """
        Return all securities that have been in index during start_date and end_date.

        Parameters
        ----------
        index : str
            separated by ','
        trade_date : int

        Returns
        -------
        pd.DataFrame

        """
        df_io, msg = self._get_universe_comp(universe, start_date, end_date)
        if msg != '0,':
            print(msg)

        df_io = df_io.rename(columns={'univ_weight': 'weight', 'univ_member':'symbol'})
        df_io = df_io[['trade_date', 'symbol','weight']]
        df_io = df_io.astype({'weight': float, 'trade_date': np.integer})
        df_io.loc[:, 'weight'] = df_io['weight'] # / 100.
        df_io = df_io.pivot(index='trade_date', columns='symbol', values='weight')
        df_io = df_io.fillna(0.0)
        return df_io

    def query_universe_weights_daily(self, universe, start_date, end_date):
        """
        Return all securities that have been in index during start_date and end_date.

        Parameters
        ----------
        universe : str
        start_date : int
        end_date : int

        Returns
        -------
        res : pd.DataFrame
            Index is trade_date, columns are symbols.

        """

        start_dt = jutil.convert_int_to_datetime(start_date)
        start_dt_extended = start_dt - pd.Timedelta(days=45)
        start_date_extended = jutil.convert_datetime_to_int(start_dt_extended)
        trade_dates = self.query_trade_dates(start_date_extended, end_date)

        df_weight_raw = self.query_universe_weights_range(universe, start_date=start_date_extended, end_date=end_date)
        res = df_weight_raw.reindex(index=trade_dates)
        res = res.fillna(method='ffill')
        res = res.loc[res.index >= start_date]
        res = res.loc[res.index <= end_date]

        mask_col = res.sum(axis=0) > 0
        res = res.loc[:, mask_col]

        return res

    def query_industry_daily(self, symbol, start_date, end_date, type_='SW', level=1):
        """
        Get index components on each day during start_date and end_date.
        
        Parameters
        ----------
        symbol : str
            separated by ','
        start_date : int
        end_date : int
        type_ : {'SW', 'ZZ'}

        Returns
        -------
        res : pd.DataFrame
            index dates, columns symbols
            values are industry code

        """
        df_raw = self.query_industry_raw(symbol, type_=type_, level=level)
        
        dic_sec = jutil.group_df_to_dict(df_raw, by='symbol')
        dic_sec = {sec: df.sort_values(by='in_date', axis=0).reset_index()
                   for sec, df in dic_sec.items()}

        df_ann_tmp = pd.concat({sec: df.loc[:, 'in_date'] for sec, df in dic_sec.items()}, axis=1)
        df_value_tmp = pd.concat({sec: df.loc[:, 'industry{:d}_code'.format(level)]
                                  for sec, df in dic_sec.items()},
                                 axis=1)
        
        idx = np.unique(np.concatenate([df.index.values for df in dic_sec.values()]))
        symbol_arr = np.sort(symbol.split(','))
        df_ann = pd.DataFrame(index=idx, columns=symbol_arr, data=np.nan)
        df_ann.loc[df_ann_tmp.index, df_ann_tmp.columns] = df_ann_tmp
        df_value = pd.DataFrame(index=idx, columns=symbol_arr, data=np.nan)
        df_value.loc[df_value_tmp.index, df_value_tmp.columns] = df_value_tmp

        dates_arr = self.query_trade_dates(start_date, end_date)
        df_industry = align.align(df_value, df_ann, dates_arr)
        
        # TODO before industry classification is available, we assume they belong to their first group.
        df_industry = df_industry.fillna(method='bfill')
        df_industry = df_industry.astype(str)
        
        return df_industry
        
    def query_industry_raw(self, symbol, type_='ZZ', level=1):
        """
        Get daily industry of securities from ShenWanZhiShu or ZhongZhengZhiShu.
        
        Parameters
        ----------
        symbol : str
            separated by ','
        type_ : {'SW', 'ZZ'}
        level : {1, 2, 3, 4}
            Use which level of industry index classification.

        Returns
        -------
        df : pd.DataFrame

        """
        if type_ == 'SW':
            src = 'sw'
            if level not in [1, 2, 3, 4]:
                raise ValueError("For [SW], level must be one of {1, 2, 3, 4}")
        elif type_ == 'ZZ':
            src = 'zz'
            if level not in [1, 2]:
                raise ValueError("For [ZZ], level must be one of {1, 2}")
        elif type_ == 'ZJH':
            src = 'zjh'
            if level not in [1, 2]:
                raise ValueError("For [ZJH], level must be one of {1, 2}")
        else:
            raise ValueError("type_ must be one of SW of ZZ")
        
        filter_argument = self._dic2url({'symbol': symbol,
                                         'industry_src': src})
        fields_list = ['symbol', 'industry{:d}_code'.format(level), 'industry{:d}_name'.format(level)]
    
        df_raw, err_msg = self.query("lb.secIndustry", fields=','.join(fields_list),
                                     filter=filter_argument, orderby="symbol")
        self._raise_error_if_msg(err_msg)
        
        df_raw = df_raw.astype(dtype={'in_date': np.integer,
                                      # 'out_date': np.integer
                                     })
        return df_raw.drop_duplicates()

    def query_adj_factor_daily(self, symbol, start_date, end_date, div=False):
        """
        Get index components on each day during start_date and end_date.
        
        Parameters
        ----------
        symbol : str
            separated by ','
        start_date : int
        end_date : int
        div : bool
            False for normal adjust factor, True for diff.

        Returns
        -------
        res : pd.DataFrame
            index dates, columns symbols
            values are industry code

        """
        df_raw = self.query_adj_factor_raw(symbol, start_date=start_date, end_date=end_date)
    
        dic_sec = jutil.group_df_to_dict(df_raw, by='symbol')
        dic_sec = {sec: df.set_index('trade_date').loc[:, 'adjust_factor']
                   for sec, df in dic_sec.items()}
        
        # TODO: duplicate codes with dataview.py: line 512
        res = pd.concat(dic_sec, axis=1)  # TODO: fillna ?
        
        idx = np.unique(np.concatenate([df.index.values for df in dic_sec.values()]))
        symbol_arr = np.sort(symbol.split(','))
        res_final = pd.DataFrame(index=idx, columns=symbol_arr, data=np.nan)
        res_final.loc[res.index, res.columns] = res

        # align to every trade date
        s, e = df_raw.loc[:, 'trade_date'].min(), df_raw.loc[:, 'trade_date'].max()
        dates_arr = self.query_trade_dates(s, e)
        if not len(dates_arr) == len(res_final.index):
            res_final = res_final.reindex(dates_arr)
            
            res_final = res_final.fillna(method='ffill').fillna(method='bfill')

        if div:
            res_final = res_final.div(res_final.shift(1, axis=0)).fillna(1.0)
            
        # res = res.loc[start_date: end_date, :]

        return res_final

    def query_adj_factor_raw(self, symbol, start_date=None, end_date=None):
        """
        Query adjust factor for symbols.
        
        Parameters
        ----------
        symbol : str
            separated by ','
        start_date : int
        end_date : int

        Returns
        -------
        df : pd.DataFrame

        """
        if start_date is None:
            start_date = ""
        if end_date is None:
            end_date = ""
        
        filter_argument = self._dic2url({'symbol': symbol,
                                         'start_date': start_date, 'end_date': end_date})
        fields_list = ['symbol', 'trade_date', 'adjust_factor']

        df_raw, err_msg = self.query("lb.secAdjFactor",
                                     fields=','.join(fields_list),
                                     filter=filter_argument,
                                     orderby="symbol")
        self._raise_error_if_msg(err_msg)
        
        df_raw = df_raw.astype(dtype={'symbol': str,
                                      'trade_date': np.integer,
                                      'adjust_factor': float
                                      })
        return df_raw.drop_duplicates()
    
    def query_dividend(self, symbol, start_date, end_date):
        filter_argument = self._dic2url({'symbol': symbol,
                                         'start_date': start_date,
                                         'end_date': end_date})
        df, err_msg = self.query(view="lb.secDividend",
                                 fields="",
                                 filter=filter_argument,
                                 data_format='pandas')
        
        # df = df.set_index('exdiv_date').sort_index(axis=0)
        df = df.astype({'cash': float, 'cash_tax': float,
                        # 'bonus_list_date': np.integer,
                        # 'cashpay_date': np.integer,
                        'exdiv_date': np.integer,
                        'publish_date': np.integer,
                        'record_date': np.integer})
        self._raise_error_if_msg(err_msg)
        
        return df, err_msg
    
    def query_inst_info(self, symbol, inst_type="", fields=""):
        if inst_type == "":
            inst_type = "1,2,3,4,5,100,101,102,103,104"
        
        filter_argument = self._dic2url({'symbol': symbol,
                                         'inst_type': inst_type})
    
        df_raw, err_msg = self.query("jz.instrumentInfo", fields=fields,
                                     filter=filter_argument, orderby="symbol")
        self._raise_error_if_msg(err_msg)

        dtype_map = {'symbol': str, 'list_date': np.integer, 'delist_date': np.integer, 'inst_type': np.integer}
        cols = set(df_raw.columns)
        dtype_map = {k: v for k, v in dtype_map.items() if k in cols}
        
        df_raw = df_raw.astype(dtype=dtype_map)
        
        res = df_raw.set_index('symbol')
        return res
    
    # -----------------------------------------------------------------------------------
    # subscribe for real time trading
    def subscribe(self, symbols):
        """
        
        Parameters
        ----------
        symbols : str
            Separated by ,

        """
        self.data_api.subscribe(symbols, func=self.mkt_data_callback)

    def mkt_data_callback(self, key, quote):
        e = Event(EVENT_TYPE.MARKET_DATA)
        # print quote
        e.dic = {'quote': quote}
        if (self.ctx is not None) and (self.ctx.instance is not None):
            self.ctx.instance.put(e)
    
    # ---------------------------------------------------------------------
    # Calendar
    
    def query_trade_dates(self, start_date, end_date):
        """
        Get array of trade dates within given range.
        Return zero size array if no trade dates within range.
        
        Parameters
        ----------
        start_date : int
            YYmmdd
        end_date : int

        Returns
        -------
        trade_dates_arr : np.ndarray
            dtype = int

        """
        filter_argument = self._dic2url({'start_date': start_date,
                                         'end_date': end_date})
    
        # df_raw, err_msg = self.query("jz.secTradeCal", fields="trade_date",
        #                              filter=filter_argument, orderby="")
        # self._raise_error_if_msg(err_msg)
        df_raw = self._trade_dates_df[self._trade_dates_df['trade_date'] >= str(start_date)]
        df_raw = df_raw[df_raw['trade_date'] <= str(end_date)]
        
        if df_raw.empty:
            return np.array([], dtype=int)
    
        trade_dates_arr = df_raw['trade_date'].values.astype(np.integer)
        return trade_dates_arr

    def query_last_trade_date(self, date):
        """
        
        Parameters
        ----------
        date : int

        Returns
        -------
        res : int

        """
        dt = jutil.convert_int_to_datetime(date)
        delta = pd.Timedelta(weeks=2)
        dt_old = dt - delta
        date_old = jutil.convert_datetime_to_int(dt_old)
    
        dates = self.query_trade_dates(date_old, date)
        mask = dates < date
        res = dates[mask][-1]
    
        return int(res)

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
        dates = self.query_trade_dates(date, date)
        return len(dates) > 0

    def query_next_trade_date(self, date, n=1):
        """
        
        Parameters
        ----------
        date : int
        n : int, optional
            Next n trade dates, default 0 (next trade date).

        Returns
        -------
        res : int

        """
        dt = jutil.convert_int_to_datetime(date)
        delta = pd.Timedelta(weeks=(n // 7 + 2))
        dt_new = dt + delta
        date_new = jutil.convert_datetime_to_int(dt_new)
    
        dates = self.query_trade_dates(date, date_new)
        mask = dates > date
        res = dates[mask][n - 1]
    
        return int(res)

#
# # test code
# if __name__ == "__main__":
#     data_config = jutil.read_json("D:\\whuang_git\\JAQs\\JAQS\\config\\data_config.json")
#     ds = RemoteDataService()
#     ds.init_from_config(data_config)
#     while True:
#         ds.query_trade_dates("20180501","20180507")
