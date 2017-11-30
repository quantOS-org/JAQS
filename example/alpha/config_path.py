# encoding: UTF-8

from __future__ import print_function
import os


_test_dir = os.path.dirname(os.path.abspath(__file__))
DATA_CONFIG_PATH = os.path.abspath(os.path.join(_test_dir, '../../config/data_config.json'))
TRADE_CONFIG_PATH = os.path.abspath(os.path.join(_test_dir, '../../config/trade_config.json'))

print("Current data config file path: {}".format(DATA_CONFIG_PATH))
print("Current trade config file path: {}".format(TRADE_CONFIG_PATH))
