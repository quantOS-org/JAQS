# encoding: utf-8

from jaqs.trade.event import EventEngine, Event, EVENT_TYPE
from jaqs.data.basic import Quote


class RealInstance(EventEngine):
    """
    Attributes
    ----------
    start_date : int
    end_date : int
    
    """
    def __init__(self):
        super(RealInstance, self).__init__()
        
        self.start_date = 0
        self.end_date = 0
        
        self.props = None
        
        self.ctx = None
    
    def init_from_config(self, props):
        """
        
        Parameters
        ----------
        props : dict

        """
        for name in ['start_date', 'end_date']:
            if name not in props:
                pass
                # raise ValueError("{} must be provided in props.".format(name))
        
        self.props = props
        self.start_date = props.get("start_date")
        self.end_date = props.get("end_date")

        for obj in ['data_api', 'trade_api', 'pm', 'strategy']:
            if hasattr(self.ctx, obj):
                getattr(self.ctx, obj).init_from_config(props)

    def register_context(self, context=None):
        self.ctx = context

    # -------------------------------------------------------------------------------------------
    # Run
    def run(self):
        """
        Listen to certain events and run the EventEngine.
        Events include:
            1. market_data are from DataService
            2. trades & orders indications are from TradeApi.
            3. etc.

        """

        self.register(EVENT_TYPE.MARKET_DATA, self.on_quote)
        
        self.register(EVENT_TYPE.TASK_STATUS_IND, self.on_task_status)
        self.register(EVENT_TYPE.ORDER_RSP, self.on_order_rsp)
        self.register(EVENT_TYPE.TASK_RSP, self.on_task_rsp)
        self.register(EVENT_TYPE.TRADE_IND, self.on_trade)
        self.register(EVENT_TYPE.ORDER_STATUS_IND, self.on_order_status)
        
        self.start(timer=False)
    
    def on_quote(self, event):
        quote_dic = event.dic['quote']
        quote = Quote.create_from_dict(quote_dic)
        self.ctx.strategy.on_tick(quote)
    
    def on_order_rsp(self, event):
        rsp = event.dic['rsp']
        self.ctx.strategy.on_order_rsp(rsp)

    def on_task_rsp(self, event):
        rsp = event.dic['rsp']
        self.ctx.strategy.on_task_rsp(rsp)
    
    def on_trade(self, event):
        ind = event.dic['ind']
        self.ctx.strategy.on_trade(ind)

    def on_order_status(self, event):
        ind = event.dic['ind']
        self.ctx.strategy.on_order_status(ind)

    def on_task_status(self, event):
        ind = event.dic['ind']
        self.ctx.strategy.on_task_status(ind)


    
    
    
    
    
    
    
    
    
    
    
    
    
