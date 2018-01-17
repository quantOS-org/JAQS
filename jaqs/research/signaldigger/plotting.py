# encoding: utf-8

from __future__ import print_function
from functools import wraps

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import seaborn as sns

from . import performance as pfm
import jaqs.util as jutil


DECIMAL_TO_BPS = 10000
DECIMAL_TO_PCT = 100
COLOR_MAP = cm.get_cmap('rainbow') # cm.get_cmap('RdBu')
MPL_RCPARAMS = {'figure.facecolor': '#F6F6F6',
                'axes.facecolor': '#F6F6F6',
                'axes.edgecolor': '#D3D3D3',
                'text.color': '#555555',
                'grid.color': '#B1B1B1',
                'grid.alpha': 0.3,
                # scale
                'axes.linewidth': 2.0,
                'axes.titlepad': 12,
                'grid.linewidth': 1.0,
                'grid.linestyle': '-',
                # font size
                'font.size': 13,
                'axes.titlesize': 18,
                'axes.labelsize': 14,
                'legend.fontsize': 'small',
                'lines.linewidth': 2.5,
                }
mpl.rcParams.update(MPL_RCPARAMS)

# -----------------------------------------------------------------------------------
# plotting settings


def customize(func):
    """
    Decorator to set plotting context and axes style during function call.
    """
    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            with plotting_context(), axes_style():
                sns.despine(left=True)
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return call_w_context


def plotting_context(context='notebook', font_scale=1.5, rc=None):
    """
    Create signaldigger default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by signal font_scale.
    rc : dict, optional
        Config flags.
        By default, {'lines.linewidth': 1.5}
        is being used and will be added to any
        rc passed in, unless explicitly overriden.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with signaldigger.plotting.plotting_context(font_scale=2):
        signaldigger.create_full_report(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().
    """
    if rc is None:
        rc = {}

    rc_default = {'lines.linewidth': 1.5}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.plotting_context(context=context, font_scale=font_scale, rc=rc)


def axes_style(style='darkgrid', rc=None):
    """Create signaldigger default axes style context.

    Under the hood, calls and returns seaborn.axes_style() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    style : str, optional
        Name of seaborn style.
    rc : dict, optional
        Config flags.

    Returns
    -------
    seaborn plotting context

    Example
    -------
    with signaldigger.plotting.axes_style(style='whitegrid'):
        signaldigger.create_full_report(..., set_context=False)

    See also
    --------
    For more information, see seaborn.plotting_context().

    """
    if rc is None:
        rc = {}

    rc_default = {}

    # Add defaults if they do not exist
    for name, val in rc_default.items():
        rc.setdefault(name, val)

    return sns.axes_style(style=style, rc=rc)


class GridFigure(object):
    def __init__(self, rows, cols, height_ratio=1.0):
        self.rows = rows * 2
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7 * height_ratio))
        self.gs = gridspec.GridSpec(self.rows, self.cols, wspace=0.1, hspace=0.5)
        self.curr_row = 0
        self.curr_col = 0
        
        self._in_row = False
    
    def next_row(self):
        if self._in_row:
            self.curr_row += 2
            self.curr_col = 0
            self._in_row = False
        
        subplt = plt.subplot(self.gs[self.curr_row: self.curr_row + 2, :])
        self.curr_row += 2
        return subplt
    
    def next_subrow(self):
        if self._in_row:
            self.curr_row += 2
            self.curr_col = 0
            self._in_row = False
        
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt
    
    def next_cell(self):
        subplt = plt.subplot(self.gs[self.curr_row: self.curr_row + 2, self.curr_col])
        self.curr_col += 1
        self._in_row = True
        if self.curr_col >= self.cols:
            self.curr_row += 2
            self.curr_col = 0
            self._in_row = False
        return subplt


# -----------------------------------------------------------------------------------
# Functions to Plot Tables


def plot_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)
    
    if isinstance(table, pd.DataFrame):
        table.columns.name = name
    
    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))
    
    print(table)
    
    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def plot_information_table(ic_summary_table):
    print("Information Analysis")
    plot_table(ic_summary_table.apply(lambda x: x.round(3)).T)


def plot_quantile_statistics_table(tb):
    print("\n\nValue of signals of Different Quantiles Statistics")
    plot_table(tb)


# -----------------------------------------------------------------------------------
# Functions to Plot Returns


'''


def plot_quantile_returns_bar(mean_ret_by_q,
                              # ylim_percentiles=None,
                              ax=None):
    """
    Plots mean period wise returns for signal quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
        
    """
    mean_ret_by_q = mean_ret_by_q.copy().loc[:, ['mean']]
    
    ymin = None
    ymax = None
    
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    
    mean_ret_by_q.multiply(DECIMAL_TO_BPS) \
        .plot(kind='bar',
              title="Mean Return (on symbol, time) By signal Quantile", ax=ax)
    ax.set(xlabel='Quantile', ylabel='Mean Return (bps)',
           ylim=(ymin, ymax))
    
    return ax


'''


def plot_quantile_returns_ts(mean_ret_by_q, ax=None):
    """
    Plots mean period wise returns for signal quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
        
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    
    ret_wide = pd.concat({k: v['mean'] for k, v in mean_ret_by_q.items()}, axis=1)
    ret_wide.index = pd.to_datetime(ret_wide.index, format="%Y%m%d")
    ret_wide = ret_wide.mul(DECIMAL_TO_PCT)
    # ret_wide = ret_wide.rolling(window=22).mean()
    
    ret_wide.plot(lw=1.2, ax=ax, cmap=COLOR_MAP)
    df = pd.DataFrame()
    ax.legend(loc='upper left')
    ymin, ymax = ret_wide.min().min(), ret_wide.max().max()
    ax.set(ylabel='Return (%)',
           title="Daily Quantile Return (equal weight within quantile)",
           xlabel='Date',
           # yscale='symlog',
           # yticks=np.linspace(ymin, ymax, 5),
           ylim=(ymin, ymax))
    
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)
    
    return ax


def plot_mean_quantile_returns_spread_time_series(mean_returns_spread, period,
                                                  std_err=None,
                                                  bandwidth=1,
                                                  ax=None):
    """
    Plots mean period wise returns for signal quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series
        Series with difference between quantile mean returns by period.
    std_err : pd.Series
        Series with standard error of difference between quantile
        mean returns each period.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    
    if False:  # isinstance(mean_returns_spread, pd.DataFrame):
        if ax is None:
            ax = [None for a in mean_returns_spread.columns]
        
        ymin, ymax = (None, None)
        for (i, a), (name, fr_column) in zip(enumerate(ax),
                                             mean_returns_spread.items()):
            stdn = None if std_err is None else std_err[name]
            stdn = mean_returns_spread.loc
            a = plot_mean_quantile_returns_spread_time_series(fr_column,
                                                              std_err=stdn,
                                                              ax=a)
            ax[i] = a
            curr_ymin, curr_ymax = a.get_ylim()
            ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
            ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)
        
        for a in ax:
            a.set_ylim([ymin, ymax])
        
        return ax
    
    periods = period
    title = ('Top Minus Bottom Quantile Return'
             .format(periods if periods is not None else ""))
    
    if ax is None:
        f, ax = plt.subplots(figsize=(18, 6))
    
    mean_returns_spread.index = pd.to_datetime(mean_returns_spread.index, format="%Y%m%d")
    mean_returns_spread_bps = mean_returns_spread['mean_diff'] * DECIMAL_TO_PCT

    std_err_bps = mean_returns_spread['std'] * DECIMAL_TO_PCT
    upper = mean_returns_spread_bps.values + (std_err_bps * bandwidth)
    lower = mean_returns_spread_bps.values - (std_err_bps * bandwidth)
    
    mean_returns_spread_bps.plot(alpha=0.4, ax=ax, lw=0.7, color='navy')
    mean_returns_spread_bps.rolling(22).mean().plot(color='green',
                                                    alpha=0.7,
                                                    ax=ax)
    # ax.fill_between(mean_returns_spread.index, lower, upper,
    #                 alpha=0.3, color='indianred')
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)

    ax.legend(['mean returns spread', '1 month moving avg'], loc='upper right')
    ylim = np.nanpercentile(abs(mean_returns_spread_bps.values), 95)
    ax.set(ylabel='Difference In Quantile Mean Return (%)',
           xlabel='',
           title=title,
           ylim=(-ylim, ylim))
    
    return ax


def plot_cumulative_return(ret, ax=None, title=None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    ret : pd.Series
        Period wise returns of dollar neutral portfolio weighted by signal
        value.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    
    ret = ret.copy()
    
    cum = ret  # pfm.daily_ret_to_cum(ret)
    cum.index = pd.to_datetime(cum.index, format="%Y%m%d")
    cum = cum.mul(DECIMAL_TO_PCT)
    
    cum.plot(ax=ax, lw=3, color='indianred', alpha=1.0)
    ax.axhline(0.0, linestyle='-', color='black', lw=1)
    
    metrics = pfm.calc_performance_metrics(cum, cum_return=True, compound=False)
    ax.text(.85, .30,
            "Ann.Ret. = {:.1f}%\nAnn.Vol. = {:.1f}%\nSharpe = {:.2f}".format(metrics['ann_ret'],
                                                                           metrics['ann_vol'],
                                                                           metrics['sharpe']),
            fontsize=12,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')
    if title is None:
        title = "Cumulative Return"
    ax.set(ylabel='Cumulative Return (%)',
           title=title,
           xlabel='Date')
    
    return ax


def plot_cumulative_returns_by_quantile(quantile_ret, ax=None):
    """
    Plots the cumulative returns of various signal quantiles.

    Parameters
    ----------
    quantile_ret : int: pd.DataFrame
        Cumulative returns by signal quantile.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """
    
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    
    cum_ret = quantile_ret
    cum_ret.index = pd.to_datetime(cum_ret.index, format="%Y%m%d")
    cum_ret = cum_ret.mul(DECIMAL_TO_PCT)
    
    cum_ret.plot(lw=2, ax=ax, cmap=COLOR_MAP)
    ax.axhline(0.0, linestyle='-', color='black', lw=1)
    
    ax.legend(loc='upper left')
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    ax.set(ylabel='Cumulative Returns (%)',
           title='Cumulative Return of Each Quantile (equal weight within quantile)',
           xlabel='Date',
           # yscale='symlog',
           # yticks=np.linspace(ymin, ymax, 5),
           ylim=(ymin, ymax))
    
    sharpes = ["sharpe_{:d} = {:.2f}".format(col, pfm.calc_performance_metrics(ser, cum_return=True,
                                                                               compound=False)['sharpe'])
               for col, ser in cum_ret.iteritems()]
    ax.text(.02, .30,
            '\n'.join(sharpes),
            fontsize=12,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')

    ax.yaxis.set_major_formatter(ScalarFormatter())
    
    return ax


# -----------------------------------------------------------------------------------
# Functions to Plot IC


def plot_ic_ts(ic, period, ax=None):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given signal.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()
    if isinstance(ic, pd.DataFrame):
        ic = ic.iloc[:, 0]
    mean, std = ic.mean(), ic.std()

    if ax is None:
        num_plots = 1
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ic.plot(ax=ax, lw=0.6, color='navy', label='daily IC', alpha=0.8)
    ic.rolling(22).mean().plot(ax=ax, color='royalblue', lw=2, alpha=0.6, label='1 month MA')
    ax.axhline(0.0, linestyle='-', color='black', linewidth=1, alpha=0.8)

    ax.text(.05, .95,
            "Mean {:.3f} \n Std. {:.3f}".format(mean, std),
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top',
            )
    
    ymin, ymax = (None, None)
    curr_ymin, curr_ymax = ax.get_ylim()
    ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
    ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    ax.legend(loc='upper right')
    ax.set(ylabel='IC', xlabel="", ylim=[ymin, ymax],
           title="Daily IC and Moving Average".format(period))
    
    return ax


def plot_ic_hist(ic, period, ax=None):
    """
    Plots Spearman Rank Information Coefficient histogram for a given signal.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()
    if isinstance(ic, pd.DataFrame):
        ic = ic.iloc[:, 0]
    mean, std = ic.mean(), ic.std()
    
    if ax is None:
        v_spaces = 1
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    sns.distplot(ic.replace(np.nan, 0.), ax=ax,
                 hist_kws={'color': 'royalblue'},
                 kde_kws={'color': 'navy', 'alpha': 0.5},
                 # hist_kws={'weights':},
                 )
    ax.axvline(mean, color='indianred', linestyle='dashed', linewidth=1.0, label='Mean')
    ax.text(.05, .95,
            "Mean {:.3f} \n Std. {:.3f}".format(mean, std),
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')
    
    ax.set(title="Distribution of Daily IC",
           xlabel='IC',
           xlim=[-1, 1])
    ax.legend(loc='upper right')

    return ax


def plot_monthly_ic_heatmap(mean_monthly_ic, period, ax=None):
    """
    Plots a heatmap of the information coefficient or returns by month.

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    MONTH_MAP = {1: 'Jan',
                 2: 'Feb',
                 3: 'Mar',
                 4: 'Apr',
                 5: 'May',
                 6: 'Jun',
                 7: 'Jul',
                 8: 'Aug',
                 9: 'Sep',
                 10: 'Oct',
                 11: 'Nov',
                 12: 'Dec'}
    
    mean_monthly_ic = mean_monthly_ic.copy()
    
    num_plots = 1.0
    
    v_spaces = ((num_plots - 1) // 3) + 1
    
    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()
    
    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(MONTH_MAP[date.month])
    
    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
            [new_index_year, new_index_month],
            names=["year", "month"])
    
    ic_year_month = mean_monthly_ic['ic'].unstack()
    sns.heatmap(
            ic_year_month,
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor='white',
            cmap=cm.get_cmap('RdBu'),
            cbar=False,
            ax=ax)
    ax.set(ylabel='', xlabel='')
    
    ax.set_title("IC Monthly Mean".format(period))
    
    return ax


# -----------------------------------------------------------------------------------
# Functions to Plot Others
def plot_event_bar_OLD(mean, std, ax):
    idx = mean.index
    
    DECIMAL_TO_PERCENT = 100.0
    ax.errorbar(idx, mean * DECIMAL_TO_PERCENT, yerr=std * DECIMAL_TO_PERCENT,
                marker='o',
                ecolor='lightblue', elinewidth=5)
    
    ax.set(xlabel='Period Length (trade days)', ylabel='Return (%)',
           title="Annual Return Mean and StdDev")
    ax.set(xticks=idx)
    return ax


def plot_event_bar(df, x, y, hue, ax):
    DECIMAL_TO_PERCENT = 100.0
    
    n = len(np.unique(df[hue]))
    palette_gen = (c for c in sns.color_palette("Reds_r", n))
    
    gp = df.groupby(hue)
    
    for p, dfgp in gp:
        idx = dfgp[x]
        mean = dfgp[y]
        # std = dfgp['Annu. Vol.']
        c = next(palette_gen)
        
        ax.errorbar(idx, mean * DECIMAL_TO_PERCENT,
                    marker='o', color=c,
                    # yerr=std * DECIMAL_TO_PERCENT, ecolor='lightblue', elinewidth=5,
                    label="{}".format(p))
    ax.axhline(0.0, color='k', ls='--', lw=1, alpha=.5)
    ax.set(xlabel='Period Length (trade days)', ylabel='Return (%)',
           title="Average Annual Return")
    ax.legend(loc='upper right')
    ax.set(xticks=idx)
    return ax


def plot_event_dist(df_events, date, axs):
    i = 0
    for period, ser in df_events.iteritems():
        ax = axs[i]
        sns.distplot(ser, ax=ax)
        ax.axvline(ser.mean(), lw=1, ls='--', label='Average', color='red')
        ax.legend(loc='upper left')
        ax.set(xlabel='Return (%)', ylabel='',
               title="{} Distribution of return after {:d} trade dats".format(date, period))
        # self.show_fig(fig, 'event_return_{:d}days.png'.format(my_period))
        i += 1
    
    # print(mean)


'''
def plot_event_dist_NEW(df_events, axs, grouper=None):
    i = 0
    def _plot(ser):
        ax = axs[i]
        sns.distplot(ser, ax=ax)
        ax.axvline(ser.mean(), lw=1, ls='--', label='Average', color='red')
        ax.legend(loc='upper left')
        ax.set(xlabel='Return (%)', ylabel='',
               title="Distribution of return after {:d} trade dats".format(period))
    if grouper is None:
    
    for (date, period), row in df_events.iterrows():
        ax = axs[i]
        sns.distplot(ser, ax=ax)
        ax.axvline(ser.mean(), lw=1, ls='--', label='Average', color='red')
        ax.legend(loc='upper left')
        ax.set(xlabel='Return (%)', ylabel='',
               title="Distribution of return after {:d} trade dats".format(period))
        # self.show_fig(fig, 'event_return_{:d}days.png'.format(my_period))
        i += 1
        
        # print(mean)

'''
def plot_calendar_distribution(signal, monthly_signal, yearly_signal, ax1, ax2):
    idx = signal.index.values
    start = jutil.convert_int_to_datetime(idx[0]).date()
    end = jutil.convert_int_to_datetime(idx[-1]).date()
    count = np.sum(yearly_signal.values.flatten())

    print("\n       " + "Calendar Distribution    ({} occurance from {} to {}):".format(count, start, end))

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), dpi=72)

    # sns.barplot(data=monthly_signal.reset_index(), x='Month', y='Time', ax=ax1£©
    # sns.barplot(x=monthly_signal.index.values, y=monthly_signal.values, ax=ax1)
    ax1.bar(monthly_signal.index, monthly_signal['Time'].values)
    ax1.axhline(monthly_signal.values.mean(), lw=1, ls='--', color='red', label='Average')
    ax1.legend(loc='upper right')
    months_str = ['Jan', 'Feb', 'March', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax1.set(xticks=range(len(months_str)), xticklabels=months_str,
            title="Monthly Distribution",
            xlabel='Month', ylabel='Time')

    # sns.barplot(data=yearly_signal.reset_index(), x='Year', y='Times', ax=ax2, color='forestgreen')
    ax2.bar(yearly_signal.index, yearly_signal['Time'].values)
    ax2.axhline(yearly_signal.values.mean(), lw=1, ls='--', color='red', label='Average')
    ax2.legend(loc='upper right')
    ax2.set(xticks=yearly_signal.index,
            title="Yearly Distribution",
            xlabel='Month', ylabel='Time')


def plot_event_pvalue(pv, ax):
    idx = pv.index
    v = pv.values
    ax.plot(idx, v, marker='D')
    
    ax.set(xlabel='Period Length (trade days)', ylabel='p-value',
           title="P Value of Test: Mean(return) == 0")
    ax.set(xticks=idx)
    return ax


def plot_ic_decay(df_ic, ax):
    df_ic.mul(DECIMAL_TO_PCT).plot(marker='x', lw=1.2, ax=ax, cmap=COLOR_MAP)
    ax.axhline(0.0, color='k', ls='--', lw=0.7, alpha=.5)
    ax.set(xlabel="Period Length (trade days)", ylabel="IC (%)",
           title="IC Decay",
           xticks=df_ic.index,
           xlim=(0, df_ic.index[-1] + 1))


def plot_quantile_return_mean_std(dic, ax):
    n_quantiles = len(dic)
    palette_gen = (COLOR_MAP(x) for x in np.linspace(0, 1, n_quantiles))
    #palette_gen_light = (COLOR_MAP(x) for x in np.linspace(0, 1, n_quantiles))
    # palette_gen = (c for c in sns.color_palette("RdBu", n_quantiles, desat=0.5))
    # palette_gen =\
       # (c for c in sns.cubehelix_palette(n_quantiles,
       #                                             start=0, rot=0.5,
       #                                             dark=0.1, light=0.8, reverse=True,
       #                                             gamma=.9))
    # palette_gen_light = (c for c in sns.color_palette("RdBu", n_quantiles, desat=0.5))
    # palette_gen_light = (c for c in sns.cubehelix_palette(n_quantiles,
    #                                                start=0, rot=0.5,
    #                                                dark=0.1, light=0.8, reverse=True,
    #                                                gamma=.3))
    df_tmp = list(dic.values())[0]
    idx = df_tmp.columns
    offsets = np.linspace(-0.3, 0.3, n_quantiles)
    
    for i, (quantile, df) in enumerate(dic.items()):
        mean = df.loc['mean', :]
        std = df.loc['std', :]
        c = next(palette_gen)
        c_light = list(c)
        c_light[3] = c_light[3] * .2
        # c_light = next(palette_gen_light)
    
        ax.errorbar(idx + offsets[i], mean * DECIMAL_TO_PCT,
                    marker='x', color=c, lw=1.2,
                    yerr=std * DECIMAL_TO_PCT, ecolor=c_light, elinewidth=1,
                    label="Quantile {}".format(int(quantile)))
    ax.axhline(0.0, color='k', ls='--', lw=0.7, alpha=.5)
    ax.set(xlabel='Period Length (trade days)', ylabel='Return (%)',
           title="Mean & Std of Return",
           xticks=idx)
    ax.legend(loc='upper left')
    #ax.set(xticks=idx)


def plot_batch_backtest(df, ax):
    """
    
    Parameters
    ----------
    df : pd.DataFrame
    ax : axes

    """
    df = df.copy()
    df.index = jutil.convert_int_to_datetime(df.index)
    df.mul(DECIMAL_TO_PCT).plot(# marker='x',
                                lw=1.2, ax=ax, cmap=COLOR_MAP)
    ax.axhline(0.0, color='k', ls='--', lw=0.7, alpha=.5)
    ax.set(xlabel="Date", ylabel="Cumulative Return (%)",
           title="Cumulative Return for Different Buy Condition", )
    

