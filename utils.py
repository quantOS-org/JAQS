import pandas as pd
from collections import namedtuple

def _to_date(row):
    date = int(row['DATE'])
    return pd.datetime( year=date/10000, month = date/ 100 % 100, day = date%100)

def _to_datetime(row):
    date = int(row['DATE'])
    time = int(row['TIME']) / 1000
    return pd.datetime( year=date/10000, month=date/ 100 % 100, day = date%100,
                        hour=time/10000, minute=time/100%100, second =time%100)

def _to_dataframe(cloumset, index_func = None, index_column = None):
    df = pd.DataFrame(cloumset)
    if index_func:
        df.index = df.apply(index_func, axis = 1)
    elif index_column:
        df.index = df[index_column]
        del df.index.name

    return df

def _error_to_str(error):
    if error:
        if error.has_key('message'):
            return str(error['error']) + "," + error['message']
        else:
            return str(error['error']) + ","
    else:
        return ","


def to_obj(class_name, data):
    try:
        if isinstance(data, [list, tuple]):
            result = []
            for d in data:
                result.append( namedtuple(class_name, d.keys())(*d.values()) )
            return result

        elif type(data) == dict :
            result = namedtuple(class_name, data.keys())(*data.values())
            return result
        else:
            return data
    except Exception, e:
        print class_name, data, e
        return data


def extract_result(cr, format="", index_column=None, class_name=""):
    """
        format supports pandas, obj.
    """
    
    err = _error_to_str(cr['error']) if cr.has_key('error') else None
    if cr.has_key('result'):
        if format == "pandas":
            if index_column :
                return (_to_dataframe(cr['result'], None, index_column), err)
            if 'TIME' in cr['result']:
                return (_to_dataframe(cr['result'], _to_datetime), err)
            elif 'DATE' in cr['result']:
                return (_to_dataframe(cr['result'], _to_date), err)
            else:
                return (_to_dataframe(cr['result']), err)

        elif format == "obj" and cr['result'] and class_name:
            r = cr['result']
            if isinstance(r, [list, tuple]):
                result = []
                for d in r:
                    result.append( namedtuple(class_name, d.keys())(*d.values()) )
            elif isinstance(data, [dict]):
                result = namedtuple(class_name, r.keys())(*r.values())
            else:
                result = r

            return (result, err)
        else:            
            return (cr['result'], err)
    else:
        return (None, err)
