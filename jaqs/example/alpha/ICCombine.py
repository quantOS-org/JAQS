# -*- encoding: utf-8 -*-

"""
Weekly rebalance

https://uqer.io/community/share/57b540ef228e5b79a4759398

universe : hs300
init_balance = 1e8
start_date 20140101
end_date   20170301
"""
import time

import numpy as np
import numpy.linalg as nlg
import pandas as pd
import scipy.stats as stats

import jaqs.trade.analyze.analyze as ana
from jaqs.data.dataservice import RemoteDataService
from jaqs.data.dataview import DataView
from jaqs.trade import model
from jaqs.trade.backtest import AlphaBacktestInstance
from jaqs.trade.gateway import DailyStockSimGateway
from jaqs.trade.strategy import AlphaStrategy
from jaqs.util import fileio

dataview_dir_path = fileio.join_relative_path('../output/prepared/ICCombine/dataview')
backtest_result_dir_path = fileio.join_relative_path('../output/ICCombine')

ic_weight_hd5_path = fileio.join_relative_path('../output/ICCombine', 'ic_weight.hd5')
custom_data_path = fileio.join_relative_path('../output/ICCombine', 'custom_date.json')


def test_save_dataview():
    ds = RemoteDataService()
    dv = DataView()
    
    props = {'start_date': 20150101, 'end_date': 20170930, 'universe': '000905.SH',
             'fields': ('turnover,float_mv,close_adj,pe,pb'),
             'freq': 1}
    
    dv.init_from_config(props, ds)
    dv.prepare_data()
    
    factor_formula = 'Cutoff(Standardize(turnover / 10000 / float_mv), 2)'
    dv.add_formula('TO', factor_formula, is_quarterly=False)
    
    factor_formula = 'Cutoff(Standardize(1/pb), 2)'
    dv.add_formula('BP', factor_formula, is_quarterly=False)
    
    factor_formula = 'Cutoff(Standardize(Return(close_adj, 20)), 2)'
    dv.add_formula('REVS20', factor_formula, is_quarterly=False)
    
    factor_formula = 'Cutoff(Standardize(Log(float_mv)), 2)'
    dv.add_formula('float_mv_factor', factor_formula, is_quarterly=False)
    
    factor_formula = 'Delay(Return(close_adj, 1), -1)'
    dv.add_formula('NextRet', factor_formula, is_quarterly=False)

    dv.save_dataview(folder_path=dataview_dir_path)


def ic_calculation(snapshot, factorList):
    """
    Calculate factor IC on single date
    :param snapshot:
    :return: factor IC on single date
    """
    ICresult = []
    for factor in factorList:
        # drop na
        factorPanel = snapshot[[factor, 'NextRet']]
        factorPanel = factorPanel.dropna()
        ic, _ = stats.spearmanr(factorPanel[factor], factorPanel['NextRet'])
        ICresult.append(ic)
    return ICresult


def get_ic(dv):
    """
    Calculate factor IC on all dates and save it in a DataFrame
    :param dv:
    :return: DataFrame recording factor IC on all dates
    """
    factorList = fileio.read_json(custom_data_path)
    ICPanel = {}
    for singleDate in dv.dates:
        singleSnapshot = dv.get_snapshot(singleDate)
        ICPanel[singleDate] = ic_calculation(singleSnapshot, factorList)
    
    ICPanel = pd.DataFrame(ICPanel).T
    return ICPanel


def ic_weight_calculation(icpanel):
    """
    Calculate factor IC weight on single date
    :param icpanel:
    :return:
    """
    mat = np.mat(icpanel.cov())
    mat = nlg.inv(mat)
    weight = mat * np.mat(icpanel.mean()).reshape(len(mat), 1)
    weight = np.array(weight.reshape(len(weight), ))[0]
    return weight


def get_ic_weight(dv):
    """
    Calculate factor IC weight on all dates and save it in a DataFrame
    :param dv:
    :return:
    """
    ICPanel = get_ic(dv)
    ICPanel = ICPanel.dropna()
    N = 10
    IC_weight_Panel = {}
    for i in range(N, len(ICPanel)):
        ICPanel_sub = ICPanel.iloc[i - N:i, :]
        ic_weight = ic_weight_calculation(ICPanel_sub)
        IC_weight_Panel[ICPanel.index[i]] = ic_weight
    IC_weight_Panel = pd.DataFrame(IC_weight_Panel).T
    return IC_weight_Panel


def Schmidt(data):
    """
    :param data: DataFrame containing all factors, with stock code as index and factor value as column
    :return: DataFrame containing orthogonalized factors
    """
    output = pd.DataFrame()
    mat = np.mat(data)
    output[0] = np.array(mat[:, 0].reshape(len(data), ))[0]
    for i in range(1, data.shape[1]):
        tmp = np.zeros(len(data))
        for j in range(i):
            up = np.array((mat[:, i].reshape(1, len(data))) * (np.mat(output[j]).reshape(len(data), 1)))[0][0]
            down = np.array((np.mat(output[j]).reshape(1, len(data))) * (np.mat(output[j]).reshape(len(data), 1)))[0][0]
            tmp = tmp + up * 1.0 / down * (np.array(output[j]))
        output[i] = np.array(mat[:, i].reshape(len(data), ))[0] - np.array(tmp)
    output.index = data.index
    output.columns = data.columns
    return output


def my_selector(context, user_options=None):
    # Preparation
    ic_weight = context.ic_weight
    t_date = context.trade_date
    current_ic_weight = np.mat(ic_weight.loc[t_date,]).reshape(-1, 1)
    factorList = context.factorList
    
    factorPanel = {}
    for factor in factorList:
        factorPanel[factor] = context.snapshot[factor]
    
    factorPanel = pd.DataFrame(factorPanel)
    factorResult = pd.DataFrame(np.mat(factorPanel) * np.mat(current_ic_weight), index=factorPanel.index)
    
    factorResult = factorResult.fillna(-9999)
    s = factorResult.sort_values(0)[::-1]
    
    critical = s.values[30]
    mask = factorResult > critical
    factorResult[mask] = 1.0
    factorResult[~mask] = 0.0
    
    return factorResult


def test_alpha_strategy_dataview():
    dv = DataView()
    dv.load_dataview(folder_path=dataview_dir_path)
    
    props = {
        "start_date": dv.start_date,
        "end_date": dv.end_date,
        
        "period": "week",
        "days_delay": 0,
        
        "init_balance": 1e8,
        "position_ratio": 1.0,
    }
    
    gateway = DailyStockSimGateway()
    gateway.init_from_config(props)
    
    context = model.Context(dataview=dv, gateway=gateway)
    
    store = pd.HDFStore(ic_weight_hd5_path)
    factorList = fileio.read_json(custom_data_path)
    context.ic_weight = store['ic_weight']
    context.factorList = factorList
    store.close()
    
    stock_selector = model.StockSelector(context)
    stock_selector.add_filter(name='myselector', func=my_selector)
    
    strategy = AlphaStrategy(stock_selector=stock_selector, pc_method='equal_weight')
    
    bt = AlphaBacktestInstance()
    bt.init_from_config(props, strategy, context=context)
    
    bt.run_alpha()
    
    bt.save_results(folder_path=backtest_result_dir_path)


def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    dv = DataView()
    dv.load_dataview(folder_path=dataview_dir_path)

    ta.initialize(dataview=dv, file_folder=backtest_result_dir_path)
    
    print "process trades..."
    ta.process_trades()
    print "get daily stats..."
    ta.get_daily()
    
    print "calc strategy return..."
    ta.get_returns()
    # position change info is huge!
    # print "get position change..."
    # ta.get_pos_change_info()
    
    selected_sec = []  # list(ta.universe)[:5]
    if len(selected_sec) > 0:
        print "Plot single securities PnL"
        for symbol in selected_sec:
            df_daily = ta.daily.get(symbol, None)
            if df_daily is not None:
                ana.plot_trades(df_daily, symbol=symbol, save_folder=backtest_result_dir_path)
    
    print "Plot strategy PnL..."
    ta.plot_pnl(backtest_result_dir_path)
    
    print "generate report..."
    static_folder = fileio.join_relative_path("trade/analyze/static")
    ta.gen_report(source_dir=static_folder, template_fn='report_template.html',
                  out_folder=backtest_result_dir_path,
                  selected=selected_sec)


def store_ic_weight():
    """
    Calculate IC weight and save it to file
    """
    dv = DataView()
    
    dv.load_dataview(folder_path=dataview_dir_path)
    
    factorList = ['TO', 'BP', 'REVS20', 'float_mv_factor']
    
    orthFactor_dic = {}
    
    for factor in factorList:
        orthFactor_dic[factor] = {}
    
    # add the orthogonalized factor to dataview
    for trade_date in dv.dates:
        snapshot = dv.get_snapshot(trade_date)
        factorPanel = snapshot[factorList]
        factorPanel = factorPanel.dropna()
        
        if len(factorPanel) != 0:
            orthfactorPanel = Schmidt(factorPanel)
            orthfactorPanel.columns = [x + '_adj' for x in factorList]
            
            snapshot = pd.merge(left=snapshot, right=orthfactorPanel,
                                left_index=True, right_index=True, how='left')
            
            for factor in factorList:
                orthFactor_dic[factor][trade_date] = snapshot[factor]
    
    for factor in factorList:
        dv.append_df(pd.DataFrame(orthFactor_dic[factor]).T, field_name=factor + '_adj', is_quarterly=False)
    dv.save_dataview(dataview_dir_path)
    
    factorList_adj = [x + '_adj' for x in factorList]
    
    fileio.save_json(factorList_adj, custom_data_path)
    
    w = get_ic_weight(dv)
    
    store = pd.HDFStore(ic_weight_hd5_path)
    store['ic_weight'] = w
    store.close()


if __name__ == "__main__":
    t_start = time.time()
    
    test_save_dataview()
    store_ic_weight()
    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
