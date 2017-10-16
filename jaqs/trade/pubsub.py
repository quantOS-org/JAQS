# encoding: UTF-8

from abc import abstractmethod
from collections import defaultdict


class EventTemp(object):
    def __init__(self, topic="unknown", data=None):
        self.data = data
        self.topic = topic


class Publisher(object):
    def __init__(self):
        self.subscribers = defaultdict(list)  # use empty list as default-factory
    
    def add_subscriber(self, subscriber, topic):
        self.subscribers[topic].append(subscriber)
    
    def publish(self, event):
        """
        Publish an event to all its subscribers.

        Parameters
        ----------
        event : Event object

        Returns
        -------

        """
        sub_list = self.subscribers.get(event.topic, None)
        if sub_list is None:
            return
        
        for subscriber in sub_list:
            subscriber.on_event(event)
    
    def get_topics(self):
        return self.subscribers.keys()


class Subscriber(object):
    def __init__(self):
        pass
    
    @abstractmethod
    def subscribe(self, publisher, topic):
        pass
    
    @abstractmethod
    def on_event(self, event):
        """
        Process an event.

        Parameters
        ----------
        event : Event object

        """
        pass


if __name__ == "__main__":
    p = Publisher()
    p.add_subscriber(lambda x: x, 'topic1')
