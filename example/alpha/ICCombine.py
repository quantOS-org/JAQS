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
from __future__ import print_function
from __future__ import absolute_import
import time
import numpy as np
import numpy.linalg as nlg
import pandas as pd
import scipy.stats as stats
import jaqs.trade.analyze as ana

from jaqs.trade import PortfolioManager
from jaqs.data import RemoteDataService
from jaqs.data import DataView
from jaqs.trade import model
from jaqs.trade import AlphaBacktestInstance
from jaqs.trade import AlphaTradeApi
from jaqs.trade import AlphaStrategy
import jaqs.util as jutil

from config_path import DATA_CONFIG_PATH, TRADE_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)
trade_config = jutil.read_json(TRADE_CONFIG_PATH)

dataview_dir_path = '../../output/prepared/ICCombine/dataview'
backtest_result_dir_path = '../../output/ICCombine'

ic_weight_hd5_path = '../../output/ICCombine', 'ic_weight.hd5'
custom_data_path = '../../output/ICCombine', 'custom_date.json'


def save_dataview():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
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
    factorList = jutil.read_json(custom_data_path)
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

    jutil.save_json(factorList_adj, custom_data_path)

    w = get_ic_weight(dv)

    store = pd.HDFStore(ic_weight_hd5_path)
    store['ic_weight'] = w
    store.close()


def test_alpha_strategy_dataview():
    
    dv = DataView()
    dv.load_dataview(folder_path=dataview_dir_path)
    
    props = {
        "start_date": dv.start_date,
        "end_date": dv.end_date,
    
        "period": "week",
        "days_delay": 0,
    
        "init_balance": 1e8,
        'commission_rate': 0.0
        }
    props.update(data_config)
    props.update(trade_config)

    trade_api = AlphaTradeApi()
    bt = AlphaBacktestInstance()

    stock_selector = model.StockSelector()
    stock_selector.add_filter(name='myselector', func=my_selector)

    strategy = AlphaStrategy(stock_selector=stock_selector,
                             pc_method='equal_weight')
    pm = PortfolioManager()

    context = model.AlphaContext(dataview=dv, trade_api=trade_api,
                                 instance=bt, strategy=strategy, pm=pm)

    store = pd.HDFStore(ic_weight_hd5_path)
    factorList = jutil.read_json(custom_data_path)
    context.ic_weight = store['ic_weight']
    context.factorList = factorList
    store.close()

    for mdl in [stock_selector]:
        mdl.register_context(context)

    bt.init_from_config(props)

    bt.run_alpha()
    
    bt.save_results(folder_path=backtest_result_dir_path)


def test_backtest_analyze():
    ta = ana.AlphaAnalyzer()
    
    dv = DataView()
    dv.load_dataview(folder_path=dataview_dir_path)

    ta.initialize(dataview=dv, file_folder=backtest_result_dir_path)

    ta.do_analyze(result_dir=backtest_result_dir_path, selected_sec=list(ta.universe)[:3])
    

if __name__ == "__main__":
    t_start = time.time()
    save_dataview()
    store_ic_weight()
    test_alpha_strategy_dataview()
    test_backtest_analyze()
    
    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))

