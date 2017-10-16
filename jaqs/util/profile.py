# encoding: utf-8

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
        
        print "Total {:3.1f}    | Delta {:3.1f}    | {:s}".format(total, delta, event_name)


if __name__ == "__main__":
    timer = SimpleTimer()
    timer.tick('start')
    time.sleep(3)
    timer.tick('e1')
    time.sleep(2)
    timer.tick('e2')
