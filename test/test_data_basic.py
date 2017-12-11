# encoding: utf-8
import numpy as np
import pandas as pd

from jaqs.data.basic import Bar, InstManager, Instrument
from jaqs.data import RemoteDataService
from jaqs.trade import common
from jaqs.data.py_expression_eval import Parser
import jaqs.util as jutil
from jaqs.data.align import align

from config_path import DATA_CONFIG_PATH
data_config = jutil.read_json(DATA_CONFIG_PATH)


def test_instrument():
    inst = Instrument()
    
    inst.inst_type = 1
    assert inst.is_stock
    
    for i in [101, 102, 103]:
        inst.inst_type = i
        assert inst.is_future


def test_inst_manager():
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    mgr = InstManager(ds)
    mgr.load_instruments()
    sym = '000001.SZ'
    inst_obj = mgr.get_instrument(sym)
    assert inst_obj.market == 'SZ'
    assert inst_obj.symbol == sym
    assert inst_obj.multiplier == 1
    assert inst_obj.inst_type == 1


def test_bar():
    from jaqs.data import RemoteDataService
    from jaqs.trade.common import QUOTE_TYPE
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    df_quotes, msg = ds.bar(symbol='rb1710.SHF,hc1710.SHF', start_time=200000, end_time=160000,
                            trade_date=20170704, freq=QUOTE_TYPE.MIN)
    bar_list = Bar.create_from_df(df_quotes)
    bar = bar_list[0]
    print(str(bar))


def test_order_trade_task():
    from jaqs.data.basic import TradeStat, Trade, Task, TaskInd, Order, OrderStatusInd
    
    order = Order()
    order.symbol = 'SPY'
    order.task_id = 10000001
    order.entrust_no = '123'
    
    order.entrust_size = 10
    order.entrust_action = 'Short'
    order.entrust_price = 1.2
    order.entrust_time = 95055
    order.entrust_date = 20171201
    
    order.fill_price = 1.19
    order.fill_size = 3
    order.commission = 0.001
    
    str(order)
    
    o2 = Order(order)
    o2.entrust_no = '124'
    
    o3 = Order.new_order('SPY', 'Buy', 10, 10, 20111111, 143029, 'Limit')
    
    oind = OrderStatusInd(order)
    OrderStatusInd.create_from_dict({'symbol': 'SPY'})
    str(oind)
    
    task = Task(order.task_id, 'vwap', {'a': 'b'}, order, 'place_order', order.entrust_date)
    assert (not task.is_finished)
    task.task_status = common.TASK_STATUS.DONE
    assert task.is_finished
    
    tind = TaskInd(task.task_id, task.task_status, task.algo, 'success')
    str(tind)
    
    tind2 = TaskInd.create_from_dict({'task_id': 2011223})
    
    trade = Trade(order)
    trade.set_fill_info(15, 20, 20171202, 112311, 12345)
    str(trade)
    
    t2 = Trade.create_from_dict({'symbol': 'SPY'})
    
    tstat = TradeStat()
    str(tstat)


def test_quote():
    from jaqs.data.basic import Quote
    quote = Quote()
    
    quote = Quote.create_from_dict({'bidprice1': 11.1,
                                    'askprice1': 11.2,
                                    'symbol': 'SPY',
                                    'limit_up': 22.0})
    print(str(quote))


def test_align():
    # -------------------------------------------------------------------------------------
    # input and pre-process demo data
    ds = RemoteDataService()
    ds.init_from_config(data_config)
    
    raw, msg = ds.query_lb_fin_stat('income', '000001.SZ,600000.SH,601328.SH,601988.SH',
                         20160505, 20170505, fields='oper_rev,oper_cost')
    #fp = '../output/test_align.csv'
    #raw = pd.read_csv(fp)
    
    idx_list = ['report_date', 'symbol']
    raw_idx = raw.set_index(idx_list)
    raw_idx.sort_index(axis=0, level=idx_list, inplace=True)
    
    # -------------------------------------------------------------------------------------
    # get DataFrames
    df_ann = raw_idx.loc[pd.IndexSlice[:, :], 'ann_date']
    df_ann = df_ann.unstack(level=1)
    
    df_value = raw_idx.loc[pd.IndexSlice[:, :], 'oper_rev']
    df_value = df_value.unstack(level=1)
    
    # -------------------------------------------------------------------------------------
    # get data array and align
    # date_arr = ds.get_trade_date(20160325, 20170625)
    date_arr = np.array([20160325, 20160328, 20160329, 20160330, 20160331, 20160401, 20160405, 20160406,
                         20160407, 20160408, 20160411, 20160412, 20160413, 20160414, 20160415, 20160418,
                         20160419, 20160420, 20160421, 20160422, 20160425, 20160426, 20160427, 20160428,
                         20160429, 20160503, 20160504, 20160505, 20160506, 20160509, 20160510, 20160511,
                         20160512, 20160513, 20160516, 20160517, 20160518, 20160519, 20160520, 20160523,
                         20160524, 20160525, 20160526, 20160527, 20160530, 20160531, 20160601, 20160602,
                         20160603, 20160606, 20160607, 20160608, 20160613, 20160614, 20160615, 20160616,
                         20160617, 20160620, 20160621, 20160622, 20160623, 20160624, 20160627, 20160628,
                         20160629, 20160630, 20160701, 20160704, 20160705, 20160706, 20160707, 20160708,
                         20160711, 20160712, 20160713, 20160714, 20160715, 20160718, 20160719, 20160720,
                         20160721, 20160722, 20160725, 20160726, 20160727, 20160728, 20160729, 20160801,
                         20160802, 20160803, 20160804, 20160805, 20160808, 20160809, 20160810, 20160811,
                         20160812, 20160815, 20160816, 20160817, 20160818, 20160819, 20160822, 20160823,
                         20160824, 20160825, 20160826, 20160829, 20160830, 20160831, 20160901, 20160902,
                         20160905, 20160906, 20160907, 20160908, 20160909, 20160912, 20160913, 20160914,
                         20160919, 20160920, 20160921, 20160922, 20160923, 20160926, 20160927, 20160928,
                         20160929, 20160930, 20161010, 20161011, 20161012, 20161013, 20161014, 20161017,
                         20161018, 20161019, 20161020, 20161021, 20161024, 20161025, 20161026, 20161027,
                         20161028, 20161031, 20161101, 20161102, 20161103, 20161104, 20161107, 20161108,
                         20161109, 20161110, 20161111, 20161114, 20161115, 20161116, 20161117, 20161118,
                         20161121, 20161122, 20161123, 20161124, 20161125, 20161128, 20161129, 20161130,
                         20161201, 20161202, 20161205, 20161206, 20161207, 20161208, 20161209, 20161212,
                         20161213, 20161214, 20161215, 20161216, 20161219, 20161220, 20161221, 20161222,
                         20161223, 20161226, 20161227, 20161228, 20161229, 20161230, 20170103, 20170104,
                         20170105, 20170106, 20170109, 20170110, 20170111, 20170112, 20170113, 20170116,
                         20170117, 20170118, 20170119, 20170120, 20170123, 20170124, 20170125, 20170126,
                         20170203, 20170206, 20170207, 20170208, 20170209, 20170210, 20170213, 20170214,
                         20170215, 20170216, 20170217, 20170220, 20170221, 20170222, 20170223, 20170224,
                         20170227, 20170228, 20170301, 20170302, 20170303, 20170306, 20170307, 20170308,
                         20170309, 20170310, 20170313, 20170314, 20170315, 20170316, 20170317, 20170320,
                         20170321, 20170322, 20170323, 20170324, 20170327, 20170328, 20170329, 20170330,
                         20170331, 20170405, 20170406, 20170407, 20170410, 20170411, 20170412, 20170413,
                         20170414, 20170417, 20170418, 20170419, 20170420, 20170421, 20170424, 20170425,
                         20170426, 20170427, 20170428, 20170502, 20170503, 20170504, 20170505, 20170508,
                         20170509, 20170510, 20170511, 20170512, 20170515, 20170516, 20170517, 20170518,
                         20170519, 20170522, 20170523, 20170524, 20170525, 20170526, 20170531, 20170601,
                         20170602, 20170605, 20170606, 20170607, 20170608, 20170609, 20170612, 20170613,
                         20170614, 20170615, 20170616, 20170619, 20170620, 20170621, 20170622, 20170623])
    # df_res = align(df_ann, df_evaluate, date_arr)
    
    res_align = align(df_value, df_ann, date_arr)
    
    for symbol, ser_value in df_value.iteritems():
        ser_ann = df_ann[symbol]
        ann_date_last = 0
        
        assert res_align.loc[: ser_ann.iat[0]-1, symbol].isnull().all()
        for i in range(len(ser_value)):
            value = ser_value.iat[i]
            ann_date = ser_ann.iat[i]
            if i+1 >= len(ser_value):
                ann_date_next = 99999999
            else:
                ann_date_next = ser_ann.iat[i+1]
            assert (res_align.loc[ann_date: ann_date_next-1, symbol] == value).all()

        # assert (res_align.loc[ser_ann.iat[-1]: , symbol] == ser_value.iat[-1]).all()


def test_position():
    from jaqs.data.basic import Position, GoalPosition
    pos = Position()
    df = pd.DataFrame({'symbol': ['SPY', 'WTI'],
                       'current_size': [12, 15]})
    l = Position.create_from_df(df)
    for p in l:
        str(p)
    
    gp = GoalPosition()
    str(gp)


def test_common():
    from jaqs.trade import common
    
    assert common.QUOTE_TYPE.TICK == '0'
    assert common.RUN_MODE.BACKTEST == 1
    assert common.ORDER_TYPE.MARKET == 'market'
    assert common.ORDER_STATUS.FILLED == 'Filled'
    
    assert common.TASK_STATUS.DONE.full_name == 'TASK_STATUS_DONE'
    buy = common.ORDER_ACTION.BUY
    assert buy == 'Buy'
    assert common.ORDER_ACTION.to_enum('Buy') is buy
    
    assert common.ORDER_ACTION.is_positive(common.ORDER_ACTION.BUY)
    assert common.ORDER_ACTION.is_positive(common.ORDER_ACTION.COVER)
    assert common.ORDER_ACTION.is_negative(common.ORDER_ACTION.SELL)
    assert common.ORDER_ACTION.is_negative(common.ORDER_ACTION.SHORT)


if __name__ == "__main__":
    import time
    t_start = time.time()

    g = globals()
    g = {k: v for k, v in g.items() if k.startswith('test_') and callable(v)}

    for test_name, test_func in g.items():
        print("\n==========\nTesting {:s}...".format(test_name))
        test_func()
    print("Test Complete.")

    t3 = time.time() - t_start
    print("\n\n\nTime lapsed in total: {:.1f}".format(t3))
