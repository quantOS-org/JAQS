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
        
        self.strategy = None
        self.start_date = 0
        self.end_date = 0
        
        self.props = None
        
        self.ctx = None
    
    def init_from_config(self, props, strategy):
        """
        
        Parameters
        ----------
        props : dict
        strategy : Strategy
        context : Context

        """
        for name in ['start_date', 'end_date']:
            if name not in props:
                pass
                # raise ValueError("{} must be provided in props.".format(name))
        
        self.props = props
        self.start_date = props.get("start_date")
        self.end_date = props.get("end_date")
        self.strategy = strategy
        
        strategy.ctx = self.ctx
        strategy.init_from_config(props)

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
        self.strategy.on_quote(quote)
    
    def on_order_rsp(self, event):
        rsp = event.dic['rsp']
        self.strategy.on_order_rsp(rsp)

    def on_task_rsp(self, event):
        rsp = event.dic['rsp']
        self.strategy.on_task_rsp(rsp)
    
    def on_trade(self, event):
        ind = event.dic['ind']
        self.strategy.on_trade(ind)

    def on_order_status(self, event):
        ind = event.dic['ind']
        self.strategy.on_order_status(ind)

    def on_task_status(self, event):
        ind = event.dic['ind']
        self.strategy.on_task_status(ind)


def remote_data_service_mkt_data_callback():
    import time
    from jaqs.data.dataservice import RemoteDataService
    from jaqs.trade import model
    from jaqs.example.eventdriven.realtrade import RealStrategy
    from jaqs.trade.gateway import RealGateway, RealTimeTradeApi
    
    ds = RemoteDataService()
    tapi = RealTimeTradeApi()

    strat = RealStrategy()
    from jaqs.trade.portfoliomanager import PortfolioManager
    pm = PortfolioManager(strategy=strat)
    
    gateway = RealGateway()
    
    ins = RealInstance()
    
    gateway.trade_api.use_strategy(3)
    df_univ, msg = gateway.trade_api.query_universe()
    print("Universe:")
    print(df_univ)
    
    context = model.Context(data_api=ds, gateway=gateway, instance=ins)
    context.pm = pm

    # order dependent is not good! we should make it not dependent or hard-code the order
    props = {'symbol': 'hc1801.SHF,CFCICA.JZ'}
    # props = {'symbol': '000001.SZ,600001.SH'}
    # props = {'symbol': 'CFCICA.JZ,600001.SH'}
    ins.init_from_config(props, strat)
    ins.run()
    gateway.run()
    
    # ds.subscribe('CFCICA.JZ,000001.SZ')
    # ds.subscribe('rb1710.SHF')
    ds.subscribe(props['symbol'])
    time.sleep(1000)


if __name__ == "__main__":
    remote_data_service_mkt_data_callback()

    
    
    
    
    
    
    
    
    
    
    
    
    
