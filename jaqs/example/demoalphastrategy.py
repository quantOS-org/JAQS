# encoding: utf-8

from jaqs.trade.strategy import AlphaStrategy


class DemoAlphaStrategy(AlphaStrategy):
    def init_from_config(self, props):
        super(DemoAlphaStrategy, self).init_from_config(props)
        print "DemoAlphaStrategy Initialized."
        pass

    def on_after_rebalance(self, total):
        print "\n\n{}, cash all = {:9.4e}".format(self.trade_date, total)  # DEBUG
        
    def pc_new(self):
        pass
