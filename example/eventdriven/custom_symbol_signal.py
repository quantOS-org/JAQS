# encoding: utf-8

from __future__ import print_function
from __future__ import absolute_import

from jaqs.trade import EventDrivenStrategy
from jaqs.trade import model

from jaqs.data import RemoteDataService, EventDataView
from jaqs.trade import EventBacktestInstance, BacktestTradeApi, PortfolioManager
import jaqs.trade.analyze as ana
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

result_dir_path = '../../output/calendar_spread'
dataview_dir_path = '../../output/calendar_spread/dataview'

backtest_props = {
    "symbol"                : "001.JZ",
    "start_date"            : 20170601,
    "end_date"              : 20171231,
    "bar_type"              : "1d",
    "init_balance"          : 1e3,
    "commission_rate": 0.0
}
backtest_props.update(data_config)
backtest_props.update(trade_config)


def prepare_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    symbols = ['600036.SH', '000001.SZ']
    dv_props = {'symbol': ','.join(symbols),
                'start_date': backtest_props['start_date'],
                'end_date': backtest_props['end_date'],
                'benchmark': '000300.SH'}
    dv = EventDataView()
    
    dv.init_from_config(dv_props, ds)
    dv.prepare_data()

    import pandas as pd
    # target security
    diff_cols = ['open', 'high', 'low', 'close']
    df0 = dv.get_symbol(symbols[0], fields=','.join(diff_cols))
    df1 = dv.get_symbol(symbols[1], fields=','.join(diff_cols))
    df_diff = df0 - df1
    
    # calculate signal
    df_signal = pd.concat([df0[['close']], df1[['close']], df_diff[['close']]], axis=1)
    df_signal.columns = symbols + ['diff']
    roll = df_signal['diff'].rolling(window=10)
    df_signal.loc[:, 'signal'] = (df_signal['diff'] - roll.mean()) / roll.std()
    
    dv.append_df_symbol(df_diff, '001.JZ')
    dv.data_custom = df_signal
    dv.save_dataview(dataview_dir_path)


class DoubleMaStrategy(EventDrivenStrategy):
    """"""
    def __init__(self):
        super(DoubleMaStrategy, self).__init__()
        
        # 标的
        self.symbol = ''
        
        # 快线和慢线周期
        self.fast_ma_len = 0
        self.slow_ma_len = 0
        
        # 记录当前已经过的天数
        self.window_count = 0
        self.window = 0
        
        # 快线和慢线均值
        self.fast_ma = 0
        self.slow_ma = 0
        
        # 固定长度的价格序列
        self.price_arr = None
        
        # 当前仓位
        self.pos = 0
        
        # 下单量乘数
        self.buy_size_unit = 1
        self.output = True
    
    def init_from_config(self, props):
        """
        将props中的用户设置读入
        """
        super(DoubleMaStrategy, self).init_from_config(props)
        # 标的
        self.symbol = props.get('symbol')
        
    def on_bar(self, quote_dic):
        quote = quote_dic.get(self.symbol)
        
        df_signal = self.ctx.dataview.data_custom
        row = df_signal.loc[quote.trade_date]
        signal = row['signal']
        
        THRESHOLD = 1.05
        if signal < -THRESHOLD:
            if self.pos == 0:
                self.buy(quote, 100)
            elif self.pos < 0:
                self.buy(quote, 200)
        elif signal > THRESHOLD:
            if self.pos == 0:
                self.sell(quote, 100)
            elif self.pos > 0:
                self.sell(quote, 200)
    
    def on_trade(self, ind):
        """
        交易完成后通过self.ctx.pm.get_pos得到最新仓位并更新self.pos
        """
        print("\nStrategy on trade: ")
        print(ind)
        self.pos = self.ctx.pm.get_pos(self.symbol)


def run_strategy():
    dv = EventDataView()
    dv.load_dataview(dataview_dir_path)
    
    tapi = BacktestTradeApi()
    ins = EventBacktestInstance()
    
    strat = DoubleMaStrategy()
    pm = PortfolioManager()
    
    context = model.Context(dataview=dv, # data_api=ds,
                            trade_api=tapi, instance=ins,
                            strategy=strat, pm=pm)
    
    ins.init_from_config(backtest_props)
    ins.run()
    ins.save_results(folder_path=result_dir_path)


def analyze():
    ta = ana.EventAnalyzer()
    
    dv = EventDataView()
    dv.load_dataview(dataview_dir_path)
    ta.initialize(dataview=dv,
                  file_folder=result_dir_path)
    
    ta.do_analyze(result_dir=result_dir_path, selected_sec=backtest_props['symbol'].split(','))

if __name__ == "__main__":
    #prepare_dataview()
    run_strategy()
    analyze()
