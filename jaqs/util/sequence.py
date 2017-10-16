# encoding: utf-8

from collections import defaultdict


class SequenceGenerator(object):
    def __init__(self):
        self.__d = defaultdict(int)
    
    def get_next(self, key):
        self.__d[key] += 1
        return self.__d[key]
