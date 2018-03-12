# encoding: utf-8

from __future__ import print_function
import time


class SimpleTimer(object):
    """All time are counted relative to start."""
    def __init__(self):
        self.start = 0
        self.events = []
    
    def tick(self, event_name="Untitled"):
        now = time.time()
        if not self.start:
            self.start = now
            last = now
        else:
            _, last = self.events[-1]
        total = now - self.start
        delta = now - last
        
        self.events.append((event_name, now))
        
        print("Total {:3.1f}    | Delta {:3.1f}    | {:s}".format(total, delta, event_name))



_prof_data = {}

class ProfData :
    def __init__(self, name):
        self.name = name
        self.begin_time = time.time()
        self.used_time = 0

    def end(self):
        self.end_time = time.time()
        self.used_time =  self.end_time - self.begin_time



def prof_sample_begin(name):
    data = ProfData(name)
    if name in _prof_data:
        _prof_data[name].append(data)
    else:
        _prof_data[name] = [data]
    return data

def prof_sample_end(data):
    data.end()

def prof_sample(name, func):
    d = prof_sample_begin(name)
    v = func()
    d.end()
    return v

def prof_print():

    print_data = []
    for data in _prof_data.values():
        name = data[0].name
        max_time = 0.0
        min_time = 10000000000
        total_time = 0
        for d in data:
            total_time += d.used_time
            if d.used_time > max_time: max_time = d.used_time
            if d.used_time < min_time: min_time = d.used_time

        print_data.append( { 'name' : name,
                             'max': max_time,
                             'min': min_time,
                             'total': total_time,
                             'count':len(data),
                             'avg': total_time / len(data)
                             })

    print_data = sorted(print_data, key=lambda x : -x['total'])

    for d in print_data:
        print ("prof: {:<30} count {:8d} total {:8.4f}s, min {:8.4f}s, max {:8.4f}s, avg {:8.4f}"
               .format(d['name'], d['count'], d['total'], d['min'], d['max'], d['avg']))


