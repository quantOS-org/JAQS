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

import numpy as np
from jaqs.data.dataservice import RemoteDataService
from jaqs.example.demoalphastrategy import DemoAlphaStrategy
from jaqs.util import fileio

from jaqs.util import fileio
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


def my_selector(symbol, trade_date, dataview):
    df = dataview.get_snapshot(trade_date, symbol, 'close,pb')
    close = df.loc[symbol, 'close']
    pb = df.loc[symbol, 'pb']
    
    return close * pb > 123
    
    
def pb_factor(symbol, context=None, user_options=None):
    coef = user_options['coef']
    data_api = context.data_api
    # pb = data_api.get(symbol, field='pb', start_date=20170303, end_date=20170305)
    pb = 1.
    res = np.power(1. / pb, coef)
    return res


def pb_factor_dataview(context=None, user_options=None):
    dv = context.dataview
    return dv.get_snapshot(context.trade_date, fields="open")


def gtja_factor_dataview(context=None, user_options=None):
    dv = context.dataview
    res = dv.get_snapshot(context.trade_date, fields='ret20')
    # res.loc[:, :] = np.random.rand
    # res[res < 1e-2] = 0.0
    res.iloc[:, 0] = np.random.rand(res.shape[0])
    return res


def my_commission(symbol, turnover, context=None, user_options=None):
    return turnover * user_options['myrate']


'''
def test_alpha_strategy():
    gateway = DailyStockSimGateway()
    remote_data_service = RemoteDataService()

    prop_file_path = fileio.join_relative_path('etc', 'alpha.json')
    props = read_props(prop_file_path)
    """
    props = {
        "benchmark": "000300.SH",
        "universe": "600026.SH,600027.SH,600028.SH,600029.SH,600030.SH,600031.SH",
    
        "period": "week",
        "days_delay": 2,
    
        "init_balance": 1e7,
        "position_ratio": 0.7,
    
        "start_date": 20120101,
        "end_date": 20170601,
        }

    """

    remote_data_service.init_from_config(props)
    remote_data_service.initialize()
    gateway.init_from_config(props)

    context = model.Context()
    context.register_data_api(remote_data_service)
    context.register_gateway(gateway)
    context.register_trade_api(gateway)
    
    risk_model = model.FactorRiskModel()
    signal_model = model.FactorRevenueModel()
    cost_model = model.SimpleCostModel()
    
    risk_model.register_context(context)
    signal_model.register_context(context)
    cost_model.register_context(context)
    
    signal_model.register_func('pb_factor', pb_factor)
    signal_model.activate_func({'pb_factor': {'coef': 3.27}})
    cost_model.register_func('my_commission', my_commission)
    cost_model.activate_func({'my_commission': {'myrate': 1e-2}})
    
    strategy = DemoAlphaStrategy(risk_model, signal_model, cost_model)
    # strategy.register_context(context)
    # strategy.active_pc_method = 'equal_weight'
    strategy.active_pc_method = 'mc'
    
    trade = AlphaBacktestInstance_OLD_dataservice()
    trade.init_from_config(props, strategy, context=context)
    
    trade.run_alpha()
    
    trade.save_results('../output/')
    
    
'''


def save_dataview():
    # total 130 seconds
    
    ds = RemoteDataService()
    dv = DataView()
    
    props = {'start_date': 20141114, 'end_date': 20170327, 'universe': '000300.SH',
             # 'symbol': 'rb1710.SHF,rb1801.SHF',
             'fields': ('open,high,low,close,vwap,volume,turnover,'
                        # + 'pb,net_assets,'
                        + 's_fa_eps_basic,oper_exp,tot_profit,int_income'
                        ),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    dv.add_formula('eps_ret', 'Return(s_fa_eps_basic, 3)', is_quarterly=True)
    
    dv.add_formula('ret20', 'Delay(Return(close_adj, 20), -20)', is_quarterly=False)
    
    dv.save_dataview(folder_path=fileio.join_relative_path('../output/prepared'))


def test_alpha_strategy_dataview():
    dv = DataView()

    fullpath = fileio.join_relative_path('../output/prepared/20141114_20170327_freq=1D')
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
    
    signal_model.register_func('gtja', gtja_factor_dataview)
    signal_model.activate_func({'gtja': {}})
    cost_model.register_func('my_commission', my_commission)
    cost_model.activate_func({'my_commission': {'myrate': 1e-2}})
    
    strategy = DemoAlphaStrategy(risk_model, signal_model, cost_model)
    # strategy.active_pc_method = 'equal_weight'
    # strategy.active_pc_method = 'mc'
    strategy.active_pc_method = 'factor_value_weight'
    
    bt = AlphaBacktestInstance_dv()
    bt.init_from_config(props, strategy, context=context)
    
    bt.run_alpha()
    
    bt.save_results('../output/')

if __name__ == "__main__":
    t_start = time.time()

    save_dataview()
    # test_alpha_strategy()
    test_alpha_strategy_dataview()
    # test_prepare()
    # test_read()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)

