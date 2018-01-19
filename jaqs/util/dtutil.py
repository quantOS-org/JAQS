# encoding: utf-8
import datetime
import numpy as np
import pandas as pd


def get_next_period_day(current, period, n=1, extra_offset=0):
    """
    Get the n'th day in next period from current day.

    Parameters
    ----------
    current : int
        Current date in format "%Y%m%d".
    period : str
        Interval between current and next. {'day', 'week', 'month'}
    n : int
        n times period.
    extra_offset : int
        n'th business day after next period.

    Returns
    -------
    nxt : int

    """
    current_dt = convert_int_to_datetime(current)
    if period == 'day':
        offset = pd.tseries.offsets.BDay()  # move to next business day
        # offset = offsets.Day
    elif period == 'week':
        offset = pd.tseries.offsets.Week(weekday=0)  # move to next Monday
    elif period == 'month':
        offset = pd.tseries.offsets.BMonthBegin()  # move to first business day of next month
        # offset = offsets.MonthBegin
    else:
        raise NotImplementedError("Frequency as {} not support".format(period))
    offset = offset * n
    
    next_dt = current_dt + offset
    if extra_offset:
        next_dt = next_dt + extra_offset * pd.tseries.offsets.BDay()
    nxt = convert_datetime_to_int(next_dt)
    return nxt


def convert_int_to_datetime(dt):
    """Convert int date (%Y%m%d) to datetime.datetime object."""
    if isinstance(dt, pd.Series):
        dt = dt.astype(str)
    elif isinstance(dt, int):
        dt = str(dt)
    return pd.to_datetime(dt, format="%Y%m%d")


def convert_datetime_to_int(dt):
    f = lambda x: x.year * 10000 + x.month * 100 + x.day
    if isinstance(dt, (datetime.datetime, datetime.date)):
        dt = pd.Timestamp(dt)
        res = f(dt)
    elif isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt)
        res = f(dt)
    else:
        dt = pd.Series(dt)
        res = dt.apply(f)
    return res


def shift(date, n_weeks=0):
    """Shift date backward or forward for n weeks.
    
    Parameters
    ----------
    date : int or datetime
        The date to be shifted.
    n_weeks : int, optional
        Positive for increasing date, negative for decreasing date.
        Default 0 (no shift).
    
    Returns
    -------
    res : int or datetime
    
    """
    delta = pd.Timedelta(weeks=n_weeks)
    
    is_int = isinstance(date, (int, np.integer))
    if is_int:
        dt = convert_int_to_datetime(date)
    else:
        dt = date
    res = dt + delta
    if is_int:
        res = convert_datetime_to_int(res)
    return res


def combine_date_time(date, time):
    return np.int64(date) * 1000000 + np.int64(time)


def split_date_time(dt):
    date = dt // 1000000
    time = dt % 1000000
    return date, time


def date_to_month(ser):
    # ser = pd.Series(ser)
    res = ser % 10000 // 100
    MONTH_MAP = {1: 'Jan',
                 2: 'Feb',
                 3: 'Mar',
                 4: 'Apr',
                 5: 'May',
                 6: 'Jun',
                 7: 'Jul',
                 8: 'Aug',
                 9: 'Sep',
                 10: 'Oct',
                 11: 'Nov',
                 12: 'Dec'}
    # res = res.replace(MONTH_MAP)
    return res


def date_to_year(ser):
    return ser // 10000
