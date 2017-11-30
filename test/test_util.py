# encoding: utf-8

from __future__ import print_function
import jaqs.util as jutil


def test_read_save_pickle():
    fp = '../../output/tests/test_read_save_pickle.pic'
    d = {'a': 1.0, 'b': 2, 'c': True, 'd': list()}
    jutil.save_pickle(d, fp)
    
    d2 = jutil.load_pickle(fp)
    assert d2['b'] == 2
    
    d3 = jutil.load_pickle('a_non_exits_file_blabla.pic')
    assert d3 is None


def test_combine_date_time():
    date = 20170101
    time = 145930
    assert jutil.combine_date_time(date, time) == 20170101145930

    import numpy as np
    a = np.arange(20170101, 20170101+500, dtype=np.int64)
    b = np.arange(93000, 93000+500, dtype=np.int64)
    assert np.all(jutil.combine_date_time(a, b) == a * 1000000 + b)


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
