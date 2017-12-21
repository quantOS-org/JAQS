# encoding: utf-8
"""
jaqs.data.basic module includes basic definitions of basic data types
that are used across jaqs package.

Basic data types:
- `Instrument`: An Instrument represents a specific financial contract.
- `InstManager`: InstManager query information of instruments from data server and store them.
- `Bar`: A Bar is a summary of information of price and volume during a certain length of time span.
- `Quote`: Quote represents a snapshot of price and volume information.
- `Order`: Basic order class.
- `OrderStatusInd`: OrderStatusInd is a indication of status change of an order.
- `Task`: Task is a high-level trading target, which may contain many orders.
- `Position`: Basic position class.
- `Trade`: Trade represents fill/partial fill of an order.
- `TradeInd`: TaskInd is a indication of status change of a task.
- `TradeStat`: TradeStat stores statistics of trading of a certain symbol.

"""
from .marketdata import *
from .order import *
from .trade import *
from .position import *
from .instrument import *
