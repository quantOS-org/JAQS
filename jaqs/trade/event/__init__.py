# encoding: UTF-8
"""
The event processing engine is where the event is identified,
and the appropriate reaction is selected and executed.

Our framework utilizes event engine to run in an efficient way.

"""
from .engine import EventEngine, Event
from .eventtype import EVENT_TYPE
