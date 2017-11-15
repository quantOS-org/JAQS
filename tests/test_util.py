# encoding: utf-8

import jaqs.util as jutil


def test_read_save_pickle():
    fp = jutil.join_relative_path('../output/tests/test_read_save_pickle.pic')
    d = {'a': 1.0, 'b': 2, 'c': True, 'd': list()}
    jutil.save_pickle(d, fp)
    
    d2 = jutil.load_pickle(fp)
    assert d2['b'] == 2
    
    d3 = jutil.load_pickle('a_non_exits_file_blabla.pic')
    assert d3 is None


if __name__ == "__main__":
    import time
    t_start = time.time()
    
    g = globals()
    g = {k: v for k, v in g.viewitems() if k.startswith('test_') and callable(v)}
    
    for test_name, test_func in g.viewitems():
        print "\nTesting {:s}...".format(test_name)
        test_func()
    print "Test Complete."
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
