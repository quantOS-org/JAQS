# encoding: utf-8

from __future__ import print_function
import numpy as np
import pandas as pd
import jaqs.util as jutil


def test_read_save_pickle():
    fp = '../../output/tests/test_read_save_pickle.pic'
    d = {'a': 1.0, 'b': 2, 'c': True, 'd': list()}
    jutil.save_pickle(d, fp)
    
    d2 = jutil.load_pickle(fp)
    assert d2['b'] == 2
    
    d3 = jutil.load_pickle('a_non_exits_file_blabla.pic')
    assert d3 is None


def test_read_save_json():
    fp = '../../output/tests/test_read_save_pickle.pic'
    d = {'a': 1.0, 'b': 2, 'c': True, 'd': list()}
    jutil.save_json(d, fp)
    
    d2 = jutil.read_json(fp)
    assert d2['b'] == 2
    
    d3 = jutil.read_json('a_non_exits_file_blabla.pic')
    assert d3 == dict()


def test_combine_date_time():
    date = 20170101
    time = 145930
    assert jutil.combine_date_time(date, time) == 20170101145930

    a = np.arange(20170101, 20170101+500, dtype=np.int64)
    b = np.arange(93000, 93000+500, dtype=np.int64)
    assert np.all(jutil.combine_date_time(a, b) == a * 1000000 + b)


def test_seq_gen():
    seq_gen = jutil.SequenceGenerator()
    for i in range(1, 100):
        assert seq_gen.get_next('a') == i


def test_timer():
    import time
    timer = jutil.SimpleTimer()
    timer.tick('start')
    time.sleep(1.5)
    timer.tick('e1')
    time.sleep(0.5)
    timer.tick('e2')


def test_pdutil():
    df = pd.DataFrame(np.random.rand(4, 20))
    df.iloc[1, 2] = np.nan
    df.iloc[3, 4] = np.nan
    df.iloc[1, 4] = np.nan
    assert df.isnull().sum().sum() == 3
    df.iloc[2, 11] = np.inf
    df.iloc[2, 12] = -np.inf
    assert df.isnull().sum().sum() == 3
    df2 = jutil.fillinf(df)
    assert df2.isnull().sum().sum() == 5

    res_q = jutil.to_quantile(df, 5, axis=1)
    
    df3 = df.copy()
    df3['group'] = ['a', 'a', 'b', 'a']
    
    dic = jutil.group_df_to_dict(df3, by='group')
    assert set(list(dic.keys())) == {'a', 'b'}
    

def test_dtutil():
    import datetime
    date = 20170105
    dt = jutil.convert_int_to_datetime(date)
    
    assert jutil.shift(date, 2) == 20170119
    assert jutil.convert_datetime_to_int(jutil.shift(dt, 2)) == 20170119
    
    t = 145539
    dt = jutil.combine_date_time(date, t)
    assert jutil.split_date_time(dt) == (date, t)
    
    assert jutil.get_next_period_day(date, 'day', 1, 1) == 20170109
    assert jutil.get_next_period_day(date, 'week', 2, 0) == 20170116
    assert jutil.get_next_period_day(date, 'month', 1, 0) == 20170201
    
    date = 20170808
    assert jutil.get_next_period_day(20170831, 'day', extra_offset=1) == 20170904
    assert jutil.get_next_period_day(20170831, 'day', n=2, extra_offset=0) == 20170904
    assert jutil.get_next_period_day(20170831, 'day', n=7, extra_offset=0) == 20170911
    assert jutil.get_next_period_day(20170831, 'week', extra_offset=1) == 20170905
    assert jutil.get_next_period_day(20170831, 'month', extra_offset=0) == 20170901
    
    monthly = 20170101
    while monthly < 20180301:
        monthly = jutil.get_next_period_day(monthly, 'month', extra_offset=0)
        assert datetime.datetime.strptime(str(monthly), "%Y%m%d").weekday() < 5


def test_rank_percentile():
    df = pd.DataFrame(np.random.rand(500, 3000))
    res1 = jutil.rank_with_mask(df, axis=1, mask=None, normalize=False)
    res2 = jutil.rank_with_mask(df, axis=1, mask=None, normalize=True)
    print
    
    #assert np.nanmean(val[res == 1].values.flatten()) < 0.11
    #
    #val = pd.DataFrame(np.random.rand(1000, 100))
    #expr = parser.parse('Ts_Quantile(val, 500, 12)')
    #res = parser.evaluate({'val': val})
    #assert np.nanmean(val[res == 1].values.flatten()) < 0.11


def test_io():
    folder_relative = '../output/test/test_file_io'
    folder = jutil.join_relative_path(folder_relative)
    fp = jutil.join_relative_path(folder_relative+'/file.postfix')
    
    jutil.create_dir(fp)
    jutil.create_dir(folder)
    

def test_base64():
    import matplotlib
    matplotlib.use('Agg')
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(range(10))
    res = jutil.fig2base64(fig, 'png')
    res = jutil.fig2base64(fig, 'pdf')


def test_is_numeric():
    from jaqs.util import is_numeric
    
    NUMERIC = [True, 1, -1, 1.0, 1+1j]
    NOT_NUMERIC = [object(), 'string', u'unicode', None]
    
    def test_is_numeric(self):
        for x in self.NUMERIC:
            for y in (x, [x], [x] * 2):
                for z in (y, np.array(y)):
                    self.assertTrue(is_numeric(z))
        for x in self.NOT_NUMERIC:
            for y in (x, [x], [x] * 2):
                for z in (y, np.array(y)):
                    self.assertFalse(is_numeric(z))
        for kind, dtypes in np.sctypes.items():
            if kind != 'others':
                for dtype in dtypes:
                    self.assertTrue(is_numeric(np.array([0], dtype=dtype)))


if __name__ == "__main__":
    import time
    t_start = time.time()
    
    g = globals()
    g = {k: v for k, v in g.items() if k.startswith('test_') and callable(v)}
    
    for test_name, test_func in g.items():
        print("\n==========\nTesting {:s}...".format(test_name))
        test_func()
    print("Test Complete.")
    
    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))
