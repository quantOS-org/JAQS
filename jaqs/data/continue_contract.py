import numpy as np
import pandas as pd
import datetime
from jaqs.data import RemoteDataService
import re

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
dataview_dir_path = '../../output/continue_contract'

ds = RemoteDataService()
ds.init_from_config(data_config)


def get_symbol_date_map(data_service, df_inst_, start_date, end_date, days_to_delist):
    """
    Get a map {trade_date -> front month contract symbol}

    """
    # get the trade date list between start_date and end_date
    dates = data_service.query_trade_dates(start_date, end_date)
    symbols_list = []
    
    mask_within_range = (df_inst_['delist_date'] > dates[0]) & (df_inst_['list_date'] < dates[-1])
    df_inst_ = df_inst_.loc[mask_within_range]
    
    j = 0
    for i, td in enumerate(dates):
        delist_date = df_inst_['delist_date'].iat[j]
        idx = np.nonzero(dates == delist_date)
        
        if (delist_date <= dates[-1]) and (idx[0][0] - i <= days_to_delist):
            j += 1
            delist_date = df_inst_['delist_date'].iat[j]
        symbol = df_inst_['symbol'].iat[j]
        symbols_list.append(symbol)
    
    res = pd.DataFrame(data={'trade_date': dates, 'symbol': symbols_list})
    res.loc[:, 'symbol_next'] = res['symbol'].shift(-1).fillna(method='ffill')
    return res


def get_continuous_contract(symbol, start_date, end_date, change_date, fields):
    # start_date = ds.get_last_trade_date(start_date)
    
    # get information of all instruments and filter only relevant contracts
    df_inst = ds.query_inst_info(symbol="", fields=','.join(['symbol', 'inst_type',
                                                             'market', 'status',
                                                             'multiplier', 'list_date',
                                                             'delist_date']))
    df_inst = df_inst.reset_index()
    def is_target_symbol(s):
        return len(re.findall(symbol + r'\d+', s)) > 0
    mask_stock_index_future = df_inst['symbol'].apply(is_target_symbol)
    df_inst = df_inst.loc[mask_stock_index_future].sort_values('delist_date')
    # df_inst.index = range(len(df_inst))

    # get the front month contract for each trading date
    # prevent the case that no contracts exist near start_date
    first_list_date = df_inst['list_date'].min()
    if start_date < first_list_date:
        start_date = first_list_date

    df_inst_map = get_symbol_date_map(ds, df_inst, start_date, end_date, change_date)
    # df_inst_map['pre_trade_date'] = df_inst_map['trade_date'].shift(1)
    # df_inst_map = df_inst_map.dropna()
    # df_inst_map['pre_trade_date'] = df_inst_map['pre_trade_date'].astype(np.integer)

    # get the daily info
    df_future_daily = query_daily_data(ds, df_inst_map, fields)

    df_future_daily = df_future_daily.drop(['freq'], axis=1)
    '''
    df_future_daily['move'] = np.where(df_future_daily['symbol'] != df_future_daily['symbol'].shift(-1).fillna(method='ffill'), 1, 0)
    # df_future_daily['ret_shift'] = 0.0

    #df_future_daily.index = range(len(df_future_daily))

    for i in range(len(df_future_daily) - 1):
        if df_future_daily.ix[i, 'move'] == 1.0:
            df_future_daily.ix[i, 'ret_shift'] = np.log(df_future_daily.ix[i+1, 'close']) - np.log(df_future_daily.ix[i, 'close'])

    df_future_daily.index = df_future_daily['trade_date']
    df_future_daily['trade_date'] = df_future_daily['trade_date'].apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d').date())
    '''
    return df_future_daily


def query_daily_data(data_service, df_inst_map_, fields_):
    l = []
    
    for symbol in df_inst_map_['symbol'].unique():
        df_symbol = df_inst_map_.loc[df_inst_map_['symbol'] == symbol]
        start_date, end_date = df_symbol['trade_date'].iat[0], df_symbol['trade_date'].iat[-1]
        symbol_next = df_symbol['symbol_next'].iat[-1]
        
        df, _ = data_service.daily(symbol, start_date=start_date, end_date=end_date, fields=fields_)
        df_next, _ = data_service.daily(symbol_next, start_date=end_date, end_date=end_date, fields=fields_)
        
        close_diff = df_next['close'].iat[-1] - df['close'].iat[-1]
        for col in ['open', 'high', 'low', 'close']:
            df.loc[:, col] = df[col] + close_diff
            
        df.loc[:, 'symbol_next'] = df['symbol']
        df.loc[:, 'symbol_next'].iat[-1] = symbol_next
        df.loc[:, 'close_diff'] = 0.0
        df.loc[:, 'close_diff'].iat[-1] = close_diff
        # df_future['pre_close'] = df_future['close'].shift(1)
        # df_future['log_ret'] = np.log(df_future['close']).diff()
        # df_future = df_future.dropna()
        l.append(df)
    
    res = pd.concat(l, axis=0)
    return res


if __name__ == '__main__':
    future_df = get_continuous_contract('rb', 20170101, 20171225, 3, 'open,high,low,close')