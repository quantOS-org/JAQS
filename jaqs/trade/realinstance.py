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

    def run(self):
        """
        market_data are from DataService;
        trade&order indications are from TradeApi.

        """
        self.register(EVENT_TYPE.MARKET_DATA, self.on_quote)
        self.register(EVENT_TYPE.ORDER_RSP, self.on_order_rsp)
        self.register(EVENT_TYPE.TRADE_IND, self.on_trade_ind)
        self.register(EVENT_TYPE.ORDER_STATUS_IND, self.on_order_status_ind)
        
        self.start(timer=False)
    
    def on_quote(self, event):
        quote_dic = event.dic['quote']
        quote = Quote.create_from_dict(quote_dic)
        self.strategy.on_quote(quote)
    
    def on_order_rsp(self, event):
        rsp = event.dic['rsp']
        self.strategy.on_order_rsp(rsp)

    def on_trade_ind(self, event):
        ind = event.dic['ind']
        self.strategy.on_trade_ind(ind)

    def on_order_status_ind(self, event):
        ind = event.dic['ind']
        self.strategy.on_order_rsp(ind)


def test_remote_data_service_mkt_data_callback():
    import time
    from jaqs.data.dataservice import RemoteDataService
    from jaqs.trade import model
    from jaqs.example.eventdriven.realtrade import RealStrategy
    from jaqs.trade.gateway import RealGateway
    
    ds = RemoteDataService()
    ins = RealInstance()
    gateway = RealGateway()
    strat = RealStrategy()
    
    gateway.trade_api.use_strategy(3)
    df_univ, msg = gateway.trade_api.query_universe()
    print(df_univ)
    
    context = model.Context(data_api=ds, gateway=gateway, instance=ins)

    # order dependent is not good! we should make it not dependent or hard-code the order
    props = {'symbol': '000001.SZ,rb1801.SHF'}
    ins.init_from_config(props, strat)
    ins.run()
    gateway.run()
    
    # ds.subscribe('CFCICA.JZ')
    ds.subscribe(props['symbol'])
    time.sleep(1000)


if __name__ == "__main__":
    test_remote_data_service_mkt_data_callback()

    
    
    
    
    
    
    
    
    
    
    
    
    
