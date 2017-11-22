# encoding: utf-8

import datetime

from jaqs.data import RemoteDataService
from jaqs.util import dtutil


def test_calendar():
    calendar = RemoteDataService()
    calendar.init_from_config()

    res1 = calendar.get_trade_date_range(20121224, 20130201)
    assert len(res1) == 27

    res2 = calendar.get_next_trade_date(20170102)
    assert res2 == 20170103

    res3 = calendar.get_next_trade_date(20170104)
    assert res3 == 20170105

    res11 = calendar.get_trade_date_range(20161224, 20170201)
    assert len(res11) == 23

    assert not calendar.is_trade_date(20150101)
    assert not calendar.is_trade_date(20130501)


def test_dtutil():
    date = 20170808
    assert dtutil.get_next_period_day(20170831, 'day', extra_offset=1) == 20170904
    assert dtutil.get_next_period_day(20170831, 'day', n=2, extra_offset=0) == 20170904
    assert dtutil.get_next_period_day(20170831, 'day', n=7, extra_offset=0) == 20170911
    assert dtutil.get_next_period_day(20170831, 'week', extra_offset=1) == 20170905
    assert dtutil.get_next_period_day(20170831, 'month', extra_offset=0) == 20170901
    
    monthly = 20170101
    while monthly < 20180301:
        monthly = dtutil.get_next_period_day(monthly, 'month', extra_offset=0)
        assert datetime.datetime.strptime(str(monthly), "%Y%m%d").weekday() < 5


if __name__ == "__main__":
    test_calendar()
    test_dtutil()
