# -*- encoding: utf-8 -*-

"""
pe_ttm between 10 and 20
net profit growth between 0.2 and 4
equal weight
universe : hs300
init_balance = 1e8
start_date 20170101
end_date   20171001
"""
import time

from jaqs.data.dataservice import RemoteDataService
from jaqs.trade.backtest import AlphaBacktestInstance

from jaqs.util import fileio
import jaqs.trade.analyze.analyze as ana
from jaqs.trade.strategy import AlphaStrategy
from jaqs.trade.gateway import DailyStockSimGateway
from jaqs.trade import model
from jaqs.data.dataview import DataView


def read_props(fp):
    props = fileio.read_json(fp)
    
    enum_props = {}
    for k, v in enum_props.iteritems():
        props[k] = v.to_enum(props[k])
        
    return props


def save_dataview(sub_folder='test_dataview'):
    ds = RemoteDataService()
    dv = DataView()
    
    props = {'start_date': 20170101, 'end_date': 20171001, 'universe': '000300.SH',
             'fields': ('pe_ttm,net_profit_incl_min_int_inc'),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'Return(net_profit_incl_min_int_inc, 4)'
    factor_name = 'net_profit_growth'
    dv.add_formula(factor_name, factor_formula, is_quarterly=True)
    
    dv.save_dataview(folder_path=fileio.join_relative_path('../output/prepared'), sub_folder=sub_folder)

def my_selector(context, user_options=None):
    dv = context.dataview
    growth_rate = dv.get_snapshot(context.trade_date, fields='net_profit_growth')
    return (growth_rate >= 0.2) & (growth_rate <= 4)


def my_selector2(context, user_options=None):
    dv = context.dataview
    pe_ttm = dv.get_snapshot(context.trade_date, fields='pe_ttm')
    return (pe_ttm >= 10) & (pe_ttm <= 20)


def my_factor(context=None, user_options=None):
    dv = context.dataview
    res = dv.get_snapshot(context.trade_date, fields='close')
    return res

dv_subfolder_name = 'test_dataview'

def test_save_dataview():
    save_dataview(sub_folder=dv_subfolder_name)

def test_alpha_strategy_dataview():
    dv = DataView()

    fullpath = fileio.join_relative_path('../output/prepared', dv_subfolder_name)
    dv.load_dataview(folder=fullpath)
    
    props = {
        "benchmark": "000300.SH",
        "universe": ','.join(dv.symbol),
    
        "start_date": dv.start_date,
        "end_date": dv.end_date,
    
        "period": "month",
        "days_delay": 0,
    
        "init_balance": 1e8,
        "position_ratio": 1.0,
        }

    gateway = DailyStockSimGateway()
    gateway.init_from_config(props)

    context = model.Context(dataview=dv, gateway=gateway)

    stock_selector = model.StockSelector(context)
    stock_selector.add_filter(name='net_profit_growth', func=my_selector)
    stock_selector.add_filter(name='pe', func=my_selector2)

    strategy = AlphaStrategy(stock_selector=stock_selector, pc_method='equal_weight')

    bt = AlphaBacktestInstance()
    bt.init_from_config(props, strategy, context=context)
    
    bt.run_alpha()
    
    bt.save_results(fileio.join_relative_path('../output/'))


def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    data_service = RemoteDataService()
    
    out_folder = fileio.join_relative_path("../output")
    
    ta.initialize(data_service, out_folder)
    
    print "process trades..."
    ta.process_trades()
    print "get daily stats..."
    ta.get_daily()
    print "calc strategy return..."
    ta.get_returns()
    # position change info is huge!
    # print "get position change..."
    # ta.get_pos_change_info()
    
    selected_sec = list(ta.universe)[:3]
    if len(selected_sec) > 0:
        print "Plot single securities PnL"
        for symbol in selected_sec:
            df_daily = ta.daily.get(symbol, None)
            if df_daily is not None:
                ana.plot_trades(df_daily, symbol=symbol, save_folder=out_folder)
    
    print "Plot strategy PnL..."
    ta.plot_pnl(out_folder)
    
    print "generate report..."
    static_folder = fileio.join_relative_path("trade/analyze/static")
    ta.gen_report(source_dir=static_folder, template_fn='report_template.html',
                  out_folder=out_folder,
                  selected=selected_sec)

if __name__ == "__main__":
    t_start = time.time()

    test_save_dataview()
    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)

