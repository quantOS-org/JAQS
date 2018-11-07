# encoding: utf-8
"""
Classes defined in py_expression_eval moduel are used to parse string expressions
and do corresponding calculations. They are used in DataView. Since expression parsing
is error-prone, we do not recommend directly modifying this module .

"""

# Author: AxiaCore S.A.S. http://axiacore.com
#
# Based on js-expression-eval, by Matthew Crumley (email@matthewcrumley.com, http://silentmatt.com/)
# https://github.com/silentmatt/js-expression-eval
#
# Ported to Python and modified by Vera Mazhuga (ctrl-alt-delete@live.com, http://vero4ka.info/)
#
# You are free to use and modify this code in anyway you find useful. Please leave this comment in the code
# to acknowledge its original source. If you feel like it, I enjoy hearing about projects that use my code,
# but don't feel like you have to let me know or ask permission.

# modified by symbol from quantOS.org

from __future__ import division
from __future__ import print_function

import math

import numpy as np
import pandas as pd

from jaqs.data.align import align
import jaqs.util.numeric as numeric
from jaqs.util import rank_with_mask

TNUMBER = 0
TOP1 = 1
TOP2 = 2
TVAR = 3
TFUNCALL = 4


'''
single quarter / TTM + year on year / month on month
'''


def cum_to_single_quarter(df, report_date):
    df = df.copy()
    is_nan = df.isnull()
    df = df.fillna(method='ffill').fillna(0.0)
    year = report_date // 10000
    
    def cum_to_single_within_year(df_):
        first_row = df_.iloc[0, :].copy()
        df_ = df_.diff(1, axis=0)
        df_.iloc[0, :] = first_row
        return df_
    single_quarter = df.groupby(by=year).apply(cum_to_single_within_year)
    single_quarter[is_nan] = np.nan
    return single_quarter


def calc_ttm(df):
    return df.rolling(window=4, axis=0).sum()


def calc_year_on_year_return(df):
    return df.pct_change(4, axis=0)


def calc_quarter_on_quarter_return(df):
    return df.pct_change(1, axis=0)


class Expression(object):
    
    def __init__(self, tokens, ops1, ops2, functions):
        self.tokens = tokens
        self.ops1 = ops1
        self.ops2 = ops2
        self.functions = functions
        self.ann_dts = None
        self.trade_dts = None
        self.index_member = None
    
    def simplify(self, values):
        values = values or {}
        nstack = []
        newexpression = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TNUMBER:
                nstack.append(item)
            elif type_ == TVAR and item.index_ in values:
                item = Token(TNUMBER, 0, 0, values[item.index_])
                nstack.append(item)
            elif type_ == TOP2 and len(nstack) > 1:
                n2 = nstack.pop()
                n1 = nstack.pop()
                f = self.ops2[item.index_]
                item = Token(TNUMBER, 0, 0, f(n1.number_, n2.number_))
                nstack.append(item)
            elif type_ == TOP1 and nstack:
                n1 = nstack.pop()
                f = self.ops1[item.index_]
                item = Token(TNUMBER, 0, 0, f(n1.number_))
                nstack.append(item)
            else:
                while len(nstack) > 0:
                    newexpression.append(nstack.pop(0))
                newexpression.append(item)
        while nstack:
            newexpression.add(nstack.pop(0))
        
        return Expression(newexpression, self.ops1, self.ops2, self.functions)
    
    def substitute(self, variable, expr):
        if not isinstance(expr, Expression):
            pass  # expr = Parser().parse(str(expr))
        newexpression = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TVAR and item.index_ == variable:
                for j in range(0, len(expr.tokens)):
                    expritem = expr.tokens[j]
                    replitem = Token(
                            expritem.type_,
                            expritem.index_,
                            expritem.prio_,
                            expritem.number_,
                    )
                    newexpression.append(replitem)
            else:
                newexpression.append(item)
        
        ret = Expression(newexpression, self.ops1, self.ops2, self.functions)
        return ret
    
    def evaluate(self, values, ann_dts=None, trade_dts=None):
        self.ann_dts = ann_dts
        self.trade_dts = trade_dts
        values = values or {}
        nstack = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TNUMBER:
                nstack.append(item.number_)
            elif type_ == TOP2:
                n2 = nstack.pop()
                n1 = nstack.pop()
                f = self.ops2[item.index_]
                nstack.append(f(n1, n2))
            elif type_ == TVAR:
                if item.index_ in values:
                    nstack.append(values[item.index_])
                elif item.index_ in self.functions:
                    nstack.append(self.functions[item.index_])
                else:
                    raise Exception('undefined variable: ' + item.index_)
            elif type_ == TOP1:
                n1 = nstack.pop()
                f = self.ops1[item.index_]
                nstack.append(f(n1))
            elif type_ == TFUNCALL:
                n1 = nstack.pop()
                f = nstack.pop()
                if f.apply and f.call:
                    if type(n1) is list:
                        nstack.append(f.apply(None, n1))
                    else:
                        nstack.append(f.call(None, n1))
                else:
                    raise Exception(f + ' is not a function')
            else:
                raise Exception('invalid Expression')
        if len(nstack) > 1:
            raise Exception('invalid Expression (parity)')
        return nstack[0]
    
    def toString(self, toJS=False):
        nstack = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TNUMBER:
                nstack.append(item.number_)
            elif type_ == TOP2:
                n2 = nstack.pop()
                n1 = nstack.pop()
                f = item.index_
                if toJS and f == '^':
                    nstack.append('math.pow(' + n1 + ',' + n2 + ')')
                else:
                    nstack.append('(' + n1 + f + n2 + ')')
            elif type_ == TVAR:
                nstack.append(item.index_)
            elif type_ == TOP1:
                n1 = nstack.pop()
                f = item.index_
                if f == '-':
                    nstack.append('({0}{1})'.format(f, n1))
                else:
                    nstack.append('{0}({1})'.format(f, n1))
            elif type_ == TFUNCALL:
                n1 = nstack.pop()
                f = nstack.pop()
                nstack.append(f + '(' + n1 + ')')
            else:
                raise Exception('invalid Expression')
        if len(nstack) > 1:
            raise Exception('invalid Expression (parity)')
        return nstack[0]
    
    def variables(self):
        vars = []
        for i in range(0, len(self.tokens)):
            item = self.tokens[i]
            if item.type_ == TVAR and \
                    not item.index_ in vars and \
                    True : #item.index_ not in self.functions:
                vars.append(item.index_)
        return vars


class Token(object):
    def __init__(self, type_, index_, prio_, number_):
        self.type_ = type_
        self.index_ = index_ or 0
        self.prio_ = prio_ or 0
        self.number_ = number_ if number_ != None else 0
    
    def to_str(self):
        if self.type_ == TNUMBER:
            return self.number_
        if self.type_ == TOP1 or self.type_ == TOP2 or self.type_ == TVAR:
            return self.index_
        elif self.type_ == TFUNCALL:
            return 'CALL'
        else:
            return 'Invalid Token'


class Parser(object):
    def __init__(self, allow_future_data =False):
        self.success = False
        self.errormsg = ''
        self.expression = ''
        
        self.pos = 0
        
        self.tokens = None
        self.tokennumber = 0
        self.tokenprio = 0
        self.tokenindex = 0
        self.tmpprio = 0
        
        self.PRIMARY = 1
        self.OPERATOR = 2
        self.FUNCTION = 4
        self.LPAREN = 8
        self.RPAREN = 16
        self.COMMA = 32
        self.SIGN = 64
        self.CALL = 128
        self.NULLARY_CALL = 256
        
        # do not need parenthesis
        self.ops1 = {
            'Sin': np.sin,
            'Cos': np.cos,
            'Tan': np.tan,
            #             'asin': np.asin,
            #             'acos': np.acos,
            #             'atan': np.atan,
            #            'Mean':         np.mean,
            'Sqrt': np.sqrt,
            'Log': np.log,
            'Abs': np.abs,
            'Ceil': np.ceil,
            'Floor': np.floor,
            'Round': np.round,
            '-': self.neg,
            '!': self.logicalNot,
            'Sign': np.sign,
            #            'Rank':         self.rank,
            'exp': np.exp
        }
        
        self.ops2 = {
            '+': self.add,
            '-': self.sub,
            '*': self.mul,
            '/': self.div,
            '%': self.mod,
            '^': np.power,
            ',': self.append,
            # '||': self.concat,
            "==": self.equal,
            "!=": self.notEqual,
            ">": self.greaterThan,
            "<": self.lessThan,
            ">=": self.greaterThanEqual,
            "<=": self.lessThanEqual,
            "&&": self.andOperator,
            "||": self.orOperator
        }

        # need parenthesis
        self.functions = {
            # cross section
            'Min': np.minimum,
            'Max': np.maximum,
            'Percentile': self.percentile,
            'GroupPercentile': self.group_percentile,
            'Quantile': self.to_quantile,
            'GroupQuantile': self.group_quantile,
            'Rank': self.rank,
            'GroupRank': self.group_rank,
            'Mask': self.mask,
            'ConditionRank': self.cond_rank,
            'ConditionPercentile': self.cond_percentile,
            'ConditionQuantile': self.cond_quantile,
            'Standardize': self.standardize,
            'Cutoff': self.cutoff,
            # 'GroupApply': self.group_apply,
            # time series
            'CumToSingle': self.cum_to_single,
            'TTM': self.calc_ttm,
            'TTM_jl': self.calc_ttm_jli,
            'YOY': calc_year_on_year_return,
            'QOQ': calc_quarter_on_quarter_return,
            'Ts_Rank': self.ts_rank,
            'Ts_Percentile': self.ts_percentile,
            'Ts_Quantile': self.ts_quantile,
            'Ewma': self.ewma,
            'Sma':self.sma,
            'Ts_Sum': self.ts_sum,
            'Ts_Product': self.ts_product,  # rolling product
            'CountNans': self.count_nans,  # rolling count Nans
            'StdDev': self.std_dev,
            'Covariance': self.cov,
            'Correlation': self.corr,
            'Corr': self.corr,
            'Delay': self.delay,
            'Delta': self.delta,
            'Return': self.calc_return,
            'Ts_Mean': self.ts_mean,
            'Ts_Min': self.ts_min,
            'Ts_Max': self.ts_max,
            'Ts_Skewness': self.ts_skew,
            'Ts_Kurtosis': self.ts_kurt,
            'Tail': self.tail,
            'Step': self.step,
            'Decay_linear': self.decay_linear,
            'Decay_exp': self.decay_exp,
            # inplace
            'Pow': np.power,
            'SignedPower': self.signed_power,
            'IsNan': self.is_nan,
            # others
            'If': self.ifFunction,
            'FillNan': self.fill_nan,
            'Return_Abs': self.calc_return_abs,
            'Return_Fwd': self.calc_return_fwd,
            # test
        }
        
        self.consts = {
            'E': math.e,
            'PI': math.pi,
        }
        
        # no use
        self.values = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sqrt': math.sqrt,
            'log': math.log,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'round': round,
            'random': self.random,
            'fac': self.fac,
            'exp': math.exp,
            'min': min,
            'max': max,
            'pyt': self.pyt,
            'pow': math.pow,
            'atan2': math.atan2,
            'E': math.e,
            'PI': math.pi
        }
        
        self.ann_dts = None
        self.trade_dts = None

        self.allow_future_data = allow_future_data
    
    # -----------------------------------------------------
    # functions
    def add(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        return a + b
    
    def sub(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        return a - b
    
    def mul(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        return a * b
    
    def div(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        res = a / b
        if isinstance(res, pd.DataFrame):
            res = res.replace([np.inf, -np.inf], np.nan)
        return res
    
    def mod(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        return a % b
    
    def pow(self, a, b):
        return np.power(a, b)

    def signed_power(self, x, e):
        signs = np.sign(x)
        return signs * np.power(np.abs(x), e)
    
    def concat(self, a, b, *args):
        result = u'{0}{1}'.format(a, b)
        for arg in args:
            result = u'{0}{1}'.format(result, arg)
        return result
    
    @staticmethod
    def _to_array(x):
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.values
        elif isinstance(x, np.ndarray):
            return x
        elif isinstance(x, (int, float, bool, np.integer, np.float, np.bool)):
            return np.asarray(x)
        else:
            print(x)
            raise ValueError("Cannot convert type {} to numpy array".format(repr(type(x))))
    
    def equal(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = arr == brr
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def notEqual(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = arr != brr
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def greaterThan(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = arr > brr
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def lessThan(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = arr < brr
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def greaterThanEqual(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = arr >= brr
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def lessThanEqual(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = arr <= brr
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def andOperator(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = np.logical_and(arr, brr)
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def orOperator(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        arr, brr = self._to_array(a), self._to_array(b)
        mask = np.logical_or(np.isnan(arr), np.isnan(brr))
        res = np.logical_or(arr, brr)
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def neg(self, a):
        return -a
    
    def logicalNot(self, a):
        arr = self._to_array(a)
        mask = np.isnan(arr)
        res = np.logical_not(arr)
        res = res.astype(float)
        res[mask] = np.nan
        return pd.DataFrame(index=a.index, columns=a.columns, data=res)
    
    def random(self, a):
        return np.random.rand() * (a or 1)
    
    def fac(self, a):  # a!
        return np.math.factorial(a)
    
    def pyt(self, a, b):
        (a, b) = self._align_bivariate(a, b)
        return np.sqrt(a * a + b * b)
    
    def ifFunction(self, cond, b, c):
        mask = np.isnan(cond)
        data = np.where(cond, b, c)
        data = data.astype(float)
        data[mask] = np.nan
        df = pd.DataFrame(data, columns=cond.columns, index=cond.index)
        return df
    
    def tail(self, x, lower, upper, neweval):
        data = np.where((x >= lower) & (x <= upper), neweval, x)
        df = pd.DataFrame(data, columns=x.columns, index=x.index)
        return df

    def append(self, a, b):
        if type(a) != list:
            return [a, b]
        a.append(b)
        return a

    # -----------------------------------------------------
    # Time Series functions. For two parameter, must align
    @staticmethod
    def ewma(df, halflife):
        r = df.ewm(halflife=halflife, axis=0)
        return r.mean()
    
    @staticmethod
    def sma(df, n, m):
        a = n * 1.0 / m - 1
        r = df.ewm(com=a, axis=0)
        return r.mean()
    
    def std_dev(self, x, n):
        return x.rolling(n).std()
    
    def ts_sum(self, x, n):
        return x.rolling(n).sum()
    
    def count_nans(self, x, n):
        return n - x.rolling(n).count()
    
    def delay(self, x, n):
        if not self.allow_future_data and n < 0:
            raise RuntimeError("Can't use future data")

        return x.shift(n)
    
    def delta(self, x, n):
        if not self.allow_future_data and n < 0:
            raise RuntimeError("Can't use future data")

        return x.diff(n)
    
    def calc_return(self, df, forward=1, log=False):
        if not self.allow_future_data and forward < 0:
            raise ValueError("Can't use future data")

        if log:
            res = np.log(df).diff(forward)
        else:
            shift = df.shift(forward)
            res = (df - shift) / shift
        return res
    
    def ts_mean(self, x, n):
        return x.rolling(n).mean()
    
    def ts_min(self, x, n):
        return x.rolling(n).min()
    
    def ts_max(self, x, n):
        return x.rolling(n).max()
    
    def ts_kurt(self, x, n):
        return x.rolling(n).kurt()
    
    def ts_skew(self, x, n):
        return x.rolling(n).skew()
    
    def ts_product(self, x, n):
        return x.rolling(n).apply(np.product)
    
    @staticmethod
    def ts_rank(df, window):
        """Return a DataFrame with values ranging from 0.0 to 1.0"""
        roll = df.rolling(window=window)
    
        def _rank_arr(arr, norm=1.0):
            norm = norm * 1.0
            ranks = np.argsort(np.argsort(arr))[-1] + 1
            return ranks / norm
    
        res = roll.apply(_rank_arr)
        return res

    @staticmethod
    def ts_percentile(df, window):
        """Return a DataFrame with values ranging from 0.0 to 1.0"""
        roll = df.rolling(window=window)
    
        def _rank_arr(arr, norm=1.0):
            norm = norm * 1.0
            ranks = np.argsort(np.argsort(arr))[-1] + 1
            return ranks / norm
    
        res = roll.apply(_rank_arr, kwargs={'norm': window})
        return res

    # Time Series Two Parameters
    def corr(self, x, y, n):
        (x, y) = self._align_bivariate(x, y)
        return x.rolling(n).corr(y)

    def cov(self, x, y, n):
        (x, y) = self._align_bivariate(x, y)
        return x.rolling(n).cov(y)

    # financial statement data
    @staticmethod
    def calc_ttm(df):
        return calc_ttm(cum_to_single_quarter(df, df.index))
    
    @staticmethod
    def calc_ttm_jli(df):
        return calc_ttm(df)

    @staticmethod
    def cum_to_single(df):
        return cum_to_single_quarter(df, df.index)

    # no use
    def step(self, x, n):
        st = x.copy()
        n = n + 1
        begin = n - len(x.index)
        for col in st.columns:
            st.loc[:, col] = range(begin, n, 1)
        return st
    
    def decay_exp_array(self, x, f):
        n = len(x)
        step = range(0, n)
        step = step[::-1]
        fs = np.power(f, step)
        return np.dot(x, fs) / np.sum(fs)
    
    def decay_linear_array(self, x):
        n = len(x) + 1
        step = range(1, n)
        return np.dot(x, step) / np.sum(step)
    
    def decay_linear(self, x, n):
        return x.rolling(n).apply(self.decay_linear_array)
    
    def decay_exp(self, x, f, n):
        return x.rolling(n).apply(self.decay_exp_array, args=[f])
    
    @staticmethod
    def is_nan(df):
        return df.isnull()

    def fill_nan(self, df, value=None, fmethod = 'ffill'):
        if value != None:
            df = df.fillna(value=value)
            return df
        else:
            df = df.fillna(method = fmethod)
            return df

    def calc_return_abs(self, df, forward=1):
        if not self.allow_future_data and forward < 0:
            raise RuntimeError("Can't use future data")


        shift = df.shift(forward)
        res = (df - shift) / abs(shift)
        return res

    def calc_return_fwd(self, df, forward=1):
        if not self.allow_future_data and forward > 0:
            raise RuntimeError("Can't use future data")

        shift = df.shift(-forward)
        res = (shift - df) / df
        return res

    # -----------------------------------------------------
    # Cross Section functions
    
    @staticmethod
    def mask(df, mask):
        df[mask] = np.nan
        return df
        
    def cond_rank(self, df, cond):
        cond = cond.fillna(0.0).astype(bool)
        df, cond = self._align_bivariate(df, cond)
        df = self._mask_non_index_member(df)

        rank = rank_with_mask(df, mask=cond, axis=1, normalize=False)
        return rank

    def cond_percentile(self, df, cond):
        cond = cond.fillna(0.0).astype(bool)
        df, cond = self._align_bivariate(df, cond)
        df = self._mask_non_index_member(df)
    
        rank = rank_with_mask(df, mask=cond, axis=1, normalize=True)
        return rank

    def cond_quantile(self, df, cond, n_quantiles):
        cond = cond.fillna(0.0).astype(bool)
        df, cond = self._align_bivariate(df, cond)
        df = self._mask_non_index_member(df)
        
        res_arr = numeric.quantilize_without_nan(df[cond].values, n_quantiles=n_quantiles, axis=1)
        res = pd.DataFrame(index=df.index, columns=df.columns, data=res_arr)
        return res
    
    # -----------------------------------------------------
    # cross section functions
    def _mask_non_index_member(self, df):
        if self.index_member is not None:
            self.index_member = self.index_member.astype(bool)
            df[~self.index_member] = np.nan
        return df
    
    @staticmethod
    def _mask_df(df, mask):
        if mask is not None:
            mask = mask.astype(bool)
            df[~mask] = np.nan
        return df

    def rank(self, df, mask=None):
        """Return a DataFrame with values ranging from 0.0 to 1.0"""
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)
        df = self._mask_df(df, mask)
        
        rank = rank_with_mask(df, axis=1, normalize=False)
        return rank

    def percentile(self, df, mask=None):
        """Return a DataFrame with values ranging from 0.0 to 1.0"""
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)
        df = self._mask_df(df, mask)
        
        rank = rank_with_mask(df, axis=1, normalize=True)
        return rank
    
    # TODO: all cross-section operations support in-group modification: neutral, extreme values, standardize.
    def group_rank(self, df, group, mask=None):
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)
        df = self._mask_df(df, mask)
        
        vals = np.unique(pd.Series(group.values.flatten()).dropna())
        res = None
        for val in vals:
            mask = (group == val)
            rank = rank_with_mask(df, mask=mask, axis=1, normalize=False)
            if res is None:
                res = rank
            else:
                res = res.fillna(rank)
        return res
    
    def group_percentile(self, df, group, mask=None):
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)
        df = self._mask_df(df, mask)
        
        vals = np.unique(pd.Series(group.values.flatten()).dropna())
        res = None
        for val in vals:
            mask = (group == val)
            rank = rank_with_mask(df, mask=mask, axis=1, normalize=True)
            if res is None:
                res = rank
            else:
                res = res.fillna(rank)
        return res
    
    def ts_quantile(self, df, window=3, n_quantiles=5):
        roll = df.rolling(window=window)
    
        func = lambda arr: numeric.quantilize_without_nan(arr, n_quantiles=n_quantiles, axis=0)[-1]
        res = roll.apply(func)
        return res
    
    def to_quantile(self, df, n_quantiles=5, axis=1, mask=None):
        """
        Convert cross-section values to the quantile number they belong.
        Small values get small quantile numbers.
        
        Parameters
        ----------
        df : DataFrame
            index date, column symbols
        n_quantiles : int
            The number of quantile to be divided to.
        axis : int
            Axis to apply quantilize.

        Returns
        -------
        res : DataFrame
            index date, column symbols

        """
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)
        df = self._mask_df(df, mask)

        # TODO: unnecesssary warnings
        # import warnings
        # warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='py_exp')
        res_arr = numeric.quantilize_without_nan(df.values, n_quantiles=n_quantiles, axis=axis)
        res = pd.DataFrame(index=df.index, columns=df.columns, data=res_arr)
        return res

    def group_quantile(self, df, group, n_quantiles=5, mask=None):
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)
        df = self._mask_df(df, mask)


        res = None
        groups = np.unique(pd.Series(group.values.flatten()).dropna())
        for val in groups:
            val_res = self.to_quantile(df[group == val], n_quantiles=n_quantiles)
            if res is None:
                res = val_res
            else:
                res = res.fillna(val_res)
        return res

    '''
        def group_apply(self, func, df_arg, *args, **kwargs):
        """
        Group on cross section (axis=1). Rank, Mean, Std, Max, Min, Standardize, cutoff.
        Single parameter.
        
        Parameters
        ----------
        func : callable
            Single parameter
        df_arg : pd.DataFrame
            The single argument of func.
            index is date, column is symbol.

        Returns
        -------
        res : pd.DataFrame

        """
        df_group = self.df_group
        
        def gp_apply(df_value, df_group_):
            """df has date index and symbol columns."""
            gp = df_value.groupby(by=df_group_, axis=1)
            res_apply = gp.apply(func, *args, **kwargs)
            return res_apply

        # align for quarterly data
        df_arg = self._align_univariate(df_arg)
        
        # validity check
        if isinstance(df_group, pd.DataFrame):
            if df_group.shape[0] == 1 or df_group.shape[1] == 1:
                df_group = df_group.squeeze()
                return gp_apply(df_arg, df_group)
            else:
                pass
        elif isinstance(df_group, pd.Series):
            return gp_apply(df_arg, df_group)
        else:
            raise NotImplementedError("type of df_group{}".format(type(df_group)))
    
        # for time-variant industry classification, we have to loop
        res_list = []
        for idx in df_arg.index:
            row = df_arg.loc[[idx], :]  # must be DataFrame, because func has certain operation axis
            row_group = df_group.loc[idx, :]  # must be Series, because groupby only support series
            tmp = gp_apply(row, row_group)
            res_list.append(tmp)
            
        res = pd.concat(res_list, axis=0)
        return res

    '''
    def standardize(self, df):
        """Cross section."""
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)
        
        axis = 1
        mean = df.mean(axis=axis)
        std = df.std(axis=axis)
        return df.sub(mean, axis=0).div(std, axis=0)
    
    def cutoff(self, df, z_score=3.0):
        """
        Cut off extreme values using Median Absolute Deviation
        
        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame

        """
        df = self._align_univariate(df)
        df = self._mask_non_index_member(df)

        axis = 1
        x = df.values
        
        median = np.nanmedian(x, axis=axis).reshape(-1, 1)
        diff = x - median
        diff_abs = np.abs(diff)
        mad = np.nanmedian(diff_abs, axis=axis).reshape(-1, 1)
        
        mask = diff_abs > z_score * mad
        x[mask] = 0
        x = x + z_score * mad * np.sign(diff * mask) + mask * median
        
        return pd.DataFrame(index=df.index, columns=df.columns, data=x)
    
    def industry_netural(self, x, group):
        pass
    
    # -----------------------------------------------------
    # align functions
    def _align_bivariate(self, df1, df2, force_align=False):
        if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
            len1 = len(df1.index)
            len2 = len(df2.index)
            if (self.ann_dts is not None) and (self.trade_dts is not None):
                if len1 > len2:
                    df2 = align(df2, self.ann_dts, self.trade_dts)
                elif len1 < len2:
                    df1 = align(df1, self.ann_dts, self.trade_dts)
                elif force_align:
                    df1 = align(df1, self.ann_dts, self.trade_dts)
                    df2 = align(df2, self.ann_dts, self.trade_dts)
        return (df1, df2)

    def _align_univariate(self, df1):
        if isinstance(df1, pd.DataFrame):
            if (self.ann_dts is not None) and (self.trade_dts is not None):
                len1 = len(df1.index)
                len2 = len(self.trade_dts)
                if len1 != len2:
                    return align(df1, self.ann_dts, self.trade_dts)
        return df1

    # -----------------------------------------------------
    # helper methods
    def set_capital(self, style='camel'):
        """
        Set capital style of function names.
        
        Parameters
        ----------
        style : {'upper', 'lower'}
            upper for 'Rank', lower for 'rank'
        
        """
        
        def set_dic_key_capital(dic, style='camel'):
            """
            
            Parameters
            ----------
            dic : dict
            style : {'upper', 'lower'}
                upper for 'Rank', lower for 'rank'

            Returns
            -------
            dict

            """
            if style == 'camel':
                # TODO: not implement
                # deli = '_'
                # res = {deli.join(s.title() for s in k.split(deli)): v for k, v in dic.items()}
                res = dic
            elif style == 'lower':
                res = {k.lower(): v for k, v in dic.items()}
            else:
                raise NotImplementedError("style = {}".format(style))
            return res
        
        self.functions = set_dic_key_capital(self.functions, style=style)
        self.ops1 = set_dic_key_capital(self.ops1, style=style)
    
    def register_function(self, name, func):
        """Register a new function to function map.
        
        Parameters
        ----------
        name : str
        func : callable
        
        """
        if name in self.functions:
            print("Register function failed: name [{:s}] already exist. Try another name.".format(name))
            return
        
        self.functions[name] = func

    # -----------------------------------------------------
    # parse and evaluate
    def parse(self, expr):
        """
        Parse a string expression.
        
        Parameters
        ----------
        expr : str
            Format of expr should follow our document.

        Returns
        -------
        Expression

        """
        self.errormsg = ''
        self.success = True
        operstack = []
        tokenstack = []
        self.tmpprio = 0
        expected = self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
        noperators = 0
        self.expression = expr
        self.pos = 0
        
        while self.pos < len(self.expression):
            if self.is_operator():
                if self.isSign() and expected & self.SIGN:
                    if self.isNegativeSign():
                        self.tokenprio = 5
                        self.tokenindex = '-'
                        noperators += 1
                        self.addfunc(tokenstack, operstack, TOP1)                    
                    elif self.isLogicNot():
                        self.tokenprio = 5
                        self.tokenindex = '!'
                        noperators += 1
                        self.addfunc(tokenstack, operstack, TOP1)
                    expected = \
                        self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
                elif self.isComment():
                    pass
                else:
                    if expected and self.OPERATOR == 0:
                        self.error_parsing(self.pos, 'unexpected operator')
                    noperators += 2
                    self.addfunc(tokenstack, operstack, TOP2)
                    expected = \
                        self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
            elif self.is_number():
                if expected and self.PRIMARY == 0:
                    self.error_parsing(self.pos, 'unexpected number')
                token = Token(TNUMBER, 0, 0, self.tokennumber)
                tokenstack.append(token)
                expected = self.OPERATOR | self.RPAREN | self.COMMA
            elif self.is_str():
                if (expected & self.PRIMARY) == 0:
                    self.error_parsing(self.pos, 'unexpected string')
                token = Token(TNUMBER, 0, 0, self.tokennumber)
                tokenstack.append(token)
                expected = self.OPERATOR | self.RPAREN | self.COMMA
            elif self.isLeftParenth():
                if (expected & self.LPAREN) == 0:
                    self.error_parsing(self.pos, 'unexpected \"(\"')
                if expected & self.CALL:
                    noperators += 2
                    self.tokenprio = -2
                    self.tokenindex = -1
                    self.addfunc(tokenstack, operstack, TFUNCALL)
                expected = \
                    self.PRIMARY | self.LPAREN | self.FUNCTION | \
                    self.SIGN | self.NULLARY_CALL
            elif self.isRightParenth():
                if expected & self.NULLARY_CALL:
                    token = Token(TNUMBER, 0, 0, [])
                    tokenstack.append(token)
                elif (expected & self.RPAREN) == 0:
                    self.error_parsing(self.pos, 'unexpected \")\"')
                expected = \
                    self.OPERATOR | self.RPAREN | self.COMMA | \
                    self.LPAREN | self.CALL
            elif self.isComma():
                if (expected & self.COMMA) == 0:
                    self.error_parsing(self.pos, 'unexpected \",\"')
                self.addfunc(tokenstack, operstack, TOP2)
                noperators += 2
                expected = \
                    self.PRIMARY | self.LPAREN | self.FUNCTION | self.SIGN
            elif self.is_const():
                if (expected & self.PRIMARY) == 0:
                    self.error_parsing(self.pos, 'unexpected constant')
                consttoken = Token(TNUMBER, 0, 0, self.tokennumber)
                tokenstack.append(consttoken)
                expected = self.OPERATOR | self.RPAREN | self.COMMA
            elif self.isOp2():
                if (expected & self.FUNCTION) == 0:
                    self.error_parsing(self.pos, 'unexpected function')
                self.addfunc(tokenstack, operstack, TOP2)
                noperators += 2
                expected = self.LPAREN
            elif self.isOp1():
                if (expected & self.FUNCTION) == 0:
                    self.error_parsing(self.pos, 'unexpected function')
                self.addfunc(tokenstack, operstack, TOP1)
                noperators += 1
                expected = self.LPAREN
            elif self.isVar():
                if (expected & self.PRIMARY) == 0:
                    self.error_parsing(self.pos, 'unexpected variable')
                
                vartoken = Token(TVAR, self.tokenindex, 0, 0)
                tokenstack.append(vartoken)
                expected = \
                    self.OPERATOR | self.RPAREN | \
                    self.COMMA | self.LPAREN | self.CALL
            elif self.isWhite():
                pass
            else:
                if self.errormsg == '':
                    self.error_parsing(self.pos, "You have an unknown character in your formula;"
                                                 "at '{:s}'".format(self.expression[self.pos: 10]))
                else:
                    self.error_parsing(self.pos, self.errormsg)
        if self.tmpprio < 0 or self.tmpprio >= 10:
            self.error_parsing(self.pos, 'unmatched \"()\"')
        while len(operstack) > 0:
            tmp = operstack.pop()
            tokenstack.append(tmp)
        if (noperators + 1) != len(tokenstack):
            self.error_parsing(self.pos, 'parity')
        self.tokens = tokenstack
        return Expression(tokenstack, self.ops1, self.ops2, self.functions)
    
    def evaluate(self, values, ann_dts=None, trade_dts=None, index_member=None):
        """
        Evaluate the value of expression using. Data of different frequency will be automatically expanded.
        
        Parameters
        ----------
        values : dict
            Key is variable name, value is pd.DataFrame (index is date, column is symbol)
        ann_dts : pd.DataFrame
            Announcement dates of financial statements of securities.
        trade_dts : np.ndarray
            The date index of result.
        index_member : pd.DataFrame

        Returns
        -------
        pd.DataFrame

        """
        self.ann_dts = ann_dts
        self.trade_dts = trade_dts
        self.index_member = index_member
        
        values = values or {}
        nstack = []
        L = len(self.tokens)
        for i in range(0, L):
            item = self.tokens[i]
            type_ = item.type_
            if type_ == TNUMBER:
                nstack.append(item.number_)
            elif type_ == TOP2:
                n2 = nstack.pop()
                n1 = nstack.pop()
                f = self.ops2[item.index_]
                nstack.append(f(n1, n2))
            elif type_ == TVAR:
                if item.index_ in values:
                    nstack.append(values[item.index_])
                elif item.index_ in self.functions:
                    nstack.append(self.functions[item.index_])
                else:
                    raise Exception('undefined variable: ' + item.index_)
            elif type_ == TOP1:
                n1 = nstack.pop()
                f = self.ops1[item.index_]
                nstack.append(f(n1))
            elif type_ == TFUNCALL:
                n1 = nstack.pop()
                f = nstack.pop()
                if callable(f):
                    # FIXME: Should set value for factor if it is in list.
                    if type(n1) is list:
                        nstack.append(f(*n1))
                    else:
                        nstack.append(f(n1))  # call(f, n1)
                else:
                    raise Exception(f + ' is not a function')
            else:
                raise Exception('invalid Expression')
        if len(nstack) > 1:
            raise Exception('invalid Expression (parity)')
        return nstack[0]

    # -----------------------------------------------------
    # Other
    def error_parsing(self, column, msg):
        self.success = False
        self.errormsg = 'parse error [column ' + str(column) + ']: ' + msg
        raise Exception(self.errormsg)
    
    def addfunc(self, tokenstack, operstack, type_):
        operator = Token(
                type_,
                self.tokenindex,
                self.tokenprio + self.tmpprio,
                0,
        )
        while len(operstack) > 0:
            if operator.prio_ <= operstack[len(operstack) - 1].prio_:
                tokenstack.append(operstack.pop())
            else:
                break
        operstack.append(operator)
    
    def is_number(self):
        r = False
        str = ''
        while self.pos < len(self.expression):
            code = self.expression[self.pos]
            if (code >= '0' and code <= '9') or code == '.':
                str += self.expression[self.pos]
                self.pos += 1
                if '.' in str:
                    self.tokennumber = float(str)
                else:
                    self.tokennumber = int(str)
                r = True
            else:
                break
        return r
    
    def unescape(self, v, pos):
        buffer = []
        escaping = False
        
        for i in range(0, len(v)):
            c = v[i]
            
            if escaping:
                if c == "'":
                    buffer.append("'")
                    break
                elif c == '\\':
                    buffer.append('\\')
                    break
                elif c == '/':
                    buffer.append('/')
                    break
                elif c == 'b':
                    buffer.append('\b')
                    break
                elif c == 'f':
                    buffer.append('\f')
                    break
                elif c == 'n':
                    buffer.append('\n')
                    break
                elif c == 'r':
                    buffer.append('\r')
                    break
                elif c == 't':
                    buffer.append('\t')
                    break
                elif c == 'u':
                    # interpret the following 4 characters
                    # as the hex of the unicode code point
                    codePoint = int(v[i + 1, i + 5], 16)
                    buffer.append(unichr(codePoint))
                    i += 4
                    break
                else:
                    raise self.error_parsing(
                            pos + i,
                            'Illegal escape sequence: \'\\' + c + '\'',
                    )
                escaping = False
            else:
                if c == '\\':
                    escaping = True
                else:
                    buffer.append(c)
        
        return ''.join(buffer)
    
    def is_str(self):
        r = False
        str = ''
        startpos = self.pos
        if self.pos < len(self.expression) and self.expression[self.pos] == "'":
            self.pos += 1
            while self.pos < len(self.expression):
                code = self.expression[self.pos]
                if code != '\'' or (str != '' and str[-1] == '\\'):
                    str += self.expression[self.pos]
                    self.pos += 1
                else:
                    self.pos += 1
                    self.tokennumber = self.unescape(str, startpos)
                    r = True
                    break
        return r
    
    def is_const(self):
        for i in self.consts:
            L = len(i)
            str = self.expression[self.pos:self.pos + L]
            if i == str:
                if len(self.expression) <= self.pos + L:
                    self.tokennumber = self.consts[i]
                    self.pos += L
                    return True
                if not self.expression[self.pos + L].isalnum() and self.expression[self.pos + L] != "_":
                    self.tokennumber = self.consts[i]
                    self.pos += L
                    return True
        return False
    
    def is_operator(self):
        ops = (
            ('+', 2, '+'),
            ('-', 2, '-'),
            ('*', 3, '*'),
            (u'\u2219', 3, '*'),  # bullet operator
            (u'\u2022', 3, '*'),  # black small circle
            ('/', 4, '/'),
            ('%', 4, '%'),
            ('^', 6, '^'),
            ('&&', 1, '&&'),
            ('||', 1, '||'),
            ('==', 1, '=='),
            ('!=', 1, '!='),
            ('<=', 1, '<='),
            ('>=', 1, '>='),
            ('<', 1, '<'),
            ('>', 1, '>'),
            ('and', 0, 'and'),
            ('or', 0, 'or'),
            ('!', 5, '!'),
        )
        for token, priority, index in ops:
            if self.expression.startswith(token, self.pos):
                self.tokenprio = priority
                self.tokenindex = index
                self.pos += len(token)
                return True
        return False
    
    def isSign(self):
        code = self.expression[self.pos - 1]
        return (code == '+') or (code == '-') or (code == '!')
    
    def isPositiveSign(self):
        code = self.expression[self.pos - 1]
        return code == '+'
    
    def isNegativeSign(self):
        code = self.expression[self.pos - 1]
        return code == '-' 
    
    def isLogicNot(self):
        code = self.expression[self.pos - 1]
        return code == '!' 
    
    def isLeftParenth(self):
        code = self.expression[self.pos]
        if code == '(':
            self.pos += 1
            self.tmpprio += 10
            return True
        return False
    
    def isRightParenth(self):
        code = self.expression[self.pos]
        if code == ')':
            self.pos += 1
            self.tmpprio -= 10
            return True
        return False
    
    def isComma(self):
        code = self.expression[self.pos]
        if code == ',':
            self.pos += 1
            self.tokenprio = -1
            self.tokenindex = ","
            return True
        return False
    
    def isWhite(self):
        code = self.expression[self.pos]
        if code.isspace():
            self.pos += 1
            return True
        return False
    
    def isOp1(self):
        str = ''
        for i in range(self.pos, len(self.expression)):
            c = self.expression[i]
            if c.upper() == c.lower():
                if i == self.pos or (c != '_' and (c < '0' or c > '9')):
                    break
            str += c
        if len(str) > 0 and str in self.ops1:
            self.tokenindex = str
            self.tokenprio = 7
            self.pos += len(str)
            return True
        return False
    
    def isOp2(self):
        str = ''
        for i in range(self.pos, len(self.expression)):
            c = self.expression[i]
            if c.upper() == c.lower():
                if i == self.pos or (c != '_' and (c < '0' or c > '9')):
                    break
            str += c
        if len(str) > 0 and (str in self.ops2):
            self.tokenindex = str
            self.tokenprio = 7
            self.pos += len(str)
            return True
        return False
    
    def isVar(self):
        str = ''
        inQuotes = False
        for i in range(self.pos, len(self.expression)):
            c = self.expression[i]
            if c.lower() == c.upper():
                if ((i == self.pos and c != '"') or (not (c in '_."') and (c < '0' or c > '9'))) and not inQuotes:
                    break
            if c == '"':
                inQuotes = not inQuotes
            str += c
        if str:
            self.tokenindex = str
            self.tokenprio = 4
            self.pos += len(str)
            return True
        return False
    
    def isComment(self):
        code = self.expression[self.pos - 1]
        if code == '/' and self.expression[self.pos] == '*':
            self.pos = self.expression.index('*/', self.pos) + 2
            if self.pos == 1:
                self.pos = len(self.expression)
            return True
        return False
