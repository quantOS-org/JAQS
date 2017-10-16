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
from jaqs.example.demoalphastrategy import DemoAlphaStrategy

from jaqs.util import fileio
import jaqs.trade.analyze.analyze as ana
from jaqs.trade.backtest import AlphaBacktestInstance_dv
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
    
    props = {'start_date': 20141114, 'end_date': 20160327, 'universe': '000300.SH',
             'fields': ('open,high,low,close,vwap,volume,turnover,'
                        # + 'pb,net_assets,'
                        + 's_fa_eps_basic,oper_exp,tot_profit,int_income'
                        ),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'close > Ts_Max(close, 20)'  # 20 days new high
    factor_name = 'new_high'
    dv.add_formula(factor_name, factor_formula, is_quarterly=False)
    
    dv.save_dataview(folder_path=fileio.join_relative_path('../output/prepared'), sub_folder=sub_folder)


def my_factor(context=None, user_options=None):
    dv = context.dataview
    res = dv.get_snapshot(context.trade_date, fields='close')
    return res


def my_commission(symbol, turnover, context=None, user_options=None):
    return turnover * user_options['myrate']


def test_alpha_strategy_dataview():
    dv_subfolder_name = 'test_dataview'
    save_dataview(sub_folder=dv_subfolder_name)
    
    dv = DataView()

    fullpath = fileio.join_relative_path('../output/prepared', dv_subfolder_name)
    dv.load_dataview(folder=fullpath)
    
    props = {
        "benchmark": "000300.SH",
        # "symbol": ','.join(dv.symbol),
        "universe": ','.join(dv.symbol),
    
        "start_date": dv.start_date,
        "end_date": dv.end_date,
    
        "period": "month",
        "days_delay": 0,
    
        "init_balance": 1e9,
        "position_ratio": 0.7,
        }

    gateway = DailyStockSimGateway()
    gateway.init_from_config(props)

    context = model.Context()
    context.register_gateway(gateway)
    context.register_trade_api(gateway)
    context.register_dataview(dv)
    
    risk_model = model.FactorRiskModel()
    signal_model = model.FactorRevenueModel_dv()
    cost_model = model.SimpleCostModel()
    
    risk_model.register_context(context)
    signal_model.register_context(context)
    cost_model.register_context(context)
    
    signal_model.register_func('my_factor', my_factor)
    signal_model.activate_func({'my_factor': {}})
    cost_model.register_func('my_commission', my_commission)
    cost_model.activate_func({'my_commission': {'myrate': 1e-2}})
    
    strategy = DemoAlphaStrategy(risk_model, signal_model, cost_model)
    # strategy.active_pc_method = 'equal_weight'
    # strategy.active_pc_method = 'mc'
    strategy.active_pc_method = 'factor_value_weight'
    
    bt = AlphaBacktestInstance_dv()
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

    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)

