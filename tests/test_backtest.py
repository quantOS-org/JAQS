# -*- encoding: utf-8 -*-

"""
1. filter universe: separate helper functions
2. calc weights
3. generate trades

------------------------

- modify models: register function (with context parameter)
- modify AlphaStrategy: inheritate

------------------------

suspensions and limit reachers:
1. deal with them in re_balance function, not in filter_universe
2. do not care about them when construct portfolio
3. subtract market value and re-normalize weights (positions) after (daily) market open, before sending orders
"""
import time

from jaqs.data.dataservice import RemoteDataService
from jaqs.trade.strategy import AlphaStrategy

from jaqs.util import fileio
import jaqs.trade.analyze.analyze as ana
from jaqs.trade.backtest import AlphaBacktestInstance
from jaqs.trade.gateway import DailyStockSimGateway
from jaqs.trade import model
from jaqs.data.dataview import DataView
sub_folder = 'jli'


def read_props(fp):
    props = fileio.read_json(fp)
    
    enum_props = {}
    for k, v in enum_props.iteritems():
        props[k] = v.to_enum(props[k])
        
    return props


def save_dataview():
    ds = RemoteDataService()
    dv = DataView()
    
    props = {'start_date': 20151114, 'end_date': 20170327, 'universe': '000300.SH',
             'fields': ('open,high,low,close,vwap,volume,turnover,'
                        # + 'pb,net_assets,'
                        + 'eps_basic,oper_exp,tot_profit,int_income'
                        ),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'close >= Delay(Ts_Max(close, 20), 1)'  # 20 days new high
    factor_name = 'new_high'
    dv.add_formula(factor_name, factor_formula, is_quarterly=False)
    
    dv.add_formula('total_profit_growth', formula='Return(tot_profit, 4)', is_quarterly=True)
    
    dv.save_dataview(folder_path=fileio.join_relative_path('../output/prepared'), sub_folder=sub_folder)


def my_selector(context, user_options=None):
    growth_rate = context.snapshot['total_profit_growth']
    return growth_rate > 0.05


def my_selector_no_new_stocks(context, user_options=None):
    import pandas as pd
    td = context.trade_date
    
    df_inst = context.dataview.data_inst
    ser_list_date = pd.to_datetime(df_inst['list_date'], format="%Y%m%d")
    td_dt = pd.to_datetime(td, format="%Y%m%d")
    diff = (td_dt - ser_list_date).dt.days
    mask = diff > 50
    return mask


def my_factor(context, user_options=None):
    res = context.snapshot_sub.loc['new_high']
    return res


def my_commission(symbol, turnover, context=None, user_options=None):
    return turnover * user_options['myrate']


def test_alpha_strategy_dataview():
    # save_dataview()
    
    dv = DataView()
    fullpath = fileio.join_relative_path('../output/prepared', sub_folder)
    dv.load_dataview(folder=fullpath)
    
    props = {
        "start_date": dv.start_date,
        "end_date": dv.end_date,
    
        "period": "month",
        "days_delay": 0,
    
        "init_balance": 1e9,
        "position_ratio": 0.7,
        }

    gateway = DailyStockSimGateway()

    context = model.AlphaContext(dataview=dv, gateway=gateway)
    
    risk_model = model.FactorRiskModel(context)
    signal_model = model.FactorRevenueModel(context)
    cost_model = model.SimpleCostModel(context)
    stock_selector = model.StockSelector(context)
    
    signal_model.add_signal(name='my_factor', func=my_factor)
    cost_model.consider_cost(name='my_commission', func=my_commission, options={'myrate': 1e-2})
    stock_selector.add_filter(name='total_profit_growth', func=my_selector)
    stock_selector.add_filter(name='total_profit_growth2', func=my_selector_no_new_stocks)
    
    # strategy = AlphaStrategy(revenue_model=signal_model, stock_selector=stock_selector,
    #                          cost_model=cost_model, risk_model=risk_model,
    #                          pc_method='factor_value_weight')
    # strategy = AlphaStrategy(revenue_model=signal_model, pc_method='factor_value_weight')
    strategy = AlphaStrategy(stock_selector=stock_selector, pc_method='equal_weight')
    # strategy = AlphaStrategy()
    
    bt = AlphaBacktestInstance()
    bt.init_from_config(props, strategy, context=context)
    
    bt.run_alpha()
    
    bt.save_results(fileio.join_relative_path('../output', sub_folder))


def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    
    fullpath = fileio.join_relative_path('../output/prepared', sub_folder)
    dv = DataView()
    dv.load_dataview(fullpath)

    input_folder = fileio.join_relative_path("../output", sub_folder)
    output_folder = fileio.join_relative_path("../output", sub_folder)
    
    ta.initialize(dataview=dv, file_folder=input_folder)
    
    print "process trades..."
    ta.process_trades()
    print "get daily stats..."
    ta.get_daily()
    print "calc strategy return..."
    ta.get_returns()
    # position change info is huge!
    # print "get position change..."
    ta.get_pos_change_info()
    
    selected_sec = list(ta.universe)[:3]
    if len(selected_sec) > 0:
        print "Plot single securities PnL"
        for symbol in selected_sec:
            df_daily = ta.daily.get(symbol, None)
            if df_daily is not None:
                ana.plot_trades(df_daily, symbol=symbol, save_folder=output_folder)
    
    print "Plot strategy PnL..."
    ta.plot_pnl(output_folder)
    
    print "generate report..."
    static_folder = fileio.join_relative_path("trade/analyze/static")
    ta.gen_report(source_dir=static_folder, template_fn='report_template.html',
                  out_folder=output_folder,
                  selected=selected_sec)


if __name__ == "__main__":
    t_start = time.time()
    
    # test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)

