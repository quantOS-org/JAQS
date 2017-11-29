# encoding: utf-8

from __future__ import print_function
from jaqs.trade import model
import jaqs.util as jutil
import random


def test_context():
    r = random.random()
    path = '../../output/tests/storage{:.6f}.pic'.format(r)
    context = model.Context()
    context.load_store(path)
    assert len(context.storage) == 0
    context.storage['me'] = 1.0
    context.save_store(path)
    
    context = model.Context()
    context.load_store(path)
    assert context.storage['me'] == 1.0


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
