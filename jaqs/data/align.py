# encoding: utf-8
import numpy as np
import pandas as pd


def get_neareast(df_ann, df_value, date):
    """
    Get the value whose ann_date is earlier and nearest to date.
    
    Parameters
    ----------
    df_ann : np.ndarray
        announcement dates. shape = (n_quarters, n_securities)
    df_value : np.ndarray
        announcement values. shape = (n_quarters, n_securities)
    date : np.ndarray
        shape = (1,)

    Returns
    -------
    res : np.array
        The value whose ann_date is earlier and nearest to date. shape (n_securities)

    """
    """
    df_ann.fillna(99999999, inplace=True)  # IMPORTANT: At cells where no quarterly data is available,
                                           # we know nothing, thus it will be filled nan in the next step
    """
    mask = date[0] >= df_ann
    # res = np.where(mask, df_value, np.nan)
    n = df_value.shape[1]
    res = np.empty(n, dtype=df_value.dtype)
    
    # for each column, get the last True value
    for i in xrange(n):
        v = df_value[:, i]
        m = mask[:, i]
        r = v[m]
        res[i] = r[-1] if len(r) else np.nan
    
    return res
    

def align(df_value, df_ann, date_arr):
    """
    Expand low frequency DataFrame df_value to frequency of data_arr using announcement date from df_ann.
    
    Parameters
    ----------
    df_ann : pd.DataFrame
        DataFrame of announcement dates. shape = (n_quarters, n_securities)
    df_value : pd.DataFrame
        DataFrame of announcement values. shape = (n_quarters, n_securities)
    date_arr : list or np.array
        Target date array. dtype = int

    Returns
    -------
    df_res : pd.DataFrame
        Expanded DataFrame. shape = (n_days, n_securities)

    """
    df_ann = df_ann.fillna(99999999).astype(int)
    
    date_arr = np.asarray(date_arr, dtype=int)
    
    res = np.apply_along_axis(lambda date: get_neareast(df_ann.values, df_value.values, date), 1, date_arr.reshape(-1, 1))

    df_res = pd.DataFrame(index=date_arr, columns=df_value.columns, data=res)
    return df_res


def demo_usage():
    # -------------------------------------------------------------------------------------
    # input and pre-process demo data
    fp = '../output/test_align.csv'
    raw = pd.read_csv(fp)
    raw.columns = [u'symbol', u'ann_date', u'report_period', u'oper_rev', u'oper_cost']
    raw.drop(['oper_cost'], axis=1, inplace=True)
    
    idx_list = ['report_period', 'symbol']
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

    # -------------------------------------------------------------------------------------
    # demo usage of parser
    parser = Parser()
    # expr_formula = 'Delta(revenue, 1) / Delay(revenue,1)'
    expr_formula = 'Delay(revenue,0)'
    expression = parser.parse(expr_formula)
    df_res = parser.evaluate({'revenue': df_value}, df_ann, date_arr)
    
    # -------------------------------------------------------------------------------------
    # print to validate results
    sec = '600000.SH'
    # print "\nValue:"
    # print df_value.loc[:, sec]
    print "\n======Expression Formula:\n{:s}".format(expr_formula)
    
    print "\n======Report date, ann_date and evaluation value:"
    tmp = pd.concat([df_ann.loc[:, sec], df_evaluate.loc[:, sec]], axis=1)
    tmp.columns = ['ann_date', 'eval_value']
    print tmp
    
    print "\n======Selection of result of expansion:"
    print "20161028  {:.4f}".format(df_res.loc[20161028, sec])
    print "20161031  {:.4f}".format(df_res.loc[20161031, sec])
    print "20170427  {:.4f}".format(df_res.loc[20170427, sec])
    
    print

if __name__ == "__main__":
    import time
    t_start = time.time()
    
    demo_usage()
    
    t3 = time.time() - t_start
    print "\n\n\nTime lapsed in total: {:.1f}".format(t3)
