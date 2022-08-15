#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/16 18:45:28
# @Author  : Michael_Liu @ QTG
# @File    : plotting
# @Software: PyCharm

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
import empyrical as ep

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from functools import wraps

from . import utils
from . import performance as perf

DECIMAL_TO_BPS = 10000


# %% setting up the format and style, creating decorator
def customize(func):
    """
    [ORIGINAL]
        Decorator to set plotting context and axes style during function call.
    """

    @wraps(func)
    def call_w_context(*args, **kwargs):
        set_context = kwargs.pop('set_context', True)
        if set_context:
            color_palette = sns.color_palette('colorblind')
            with plotting_context(), axes_style(), color_palette:
                sns.despine(left=True)
                mpl.rcParams['font.sans-serif'] = ['SimHei']  # for Chinese display
                mpl.rcParams['axes.unicode_minus'] = False  # for negative display after Chinese display
                return func(*args, **kwargs)
        else:
            mpl.rcParams['font.sans-serif'] = ['SimHei']  # for Chinese display
            mpl.rcParams['axes.unicode_minus'] = False  # for negative display after Chinese display
            return func(*args, **kwargs)

    return call_w_context


def plotting_context(context='notebook', font_scale=1.5, rc=None):
    """
    [ORIGINAL]
    Create alphalens default plotting style context.

    Under the hood, calls and returns seaborn.plotting_context() with
    some custom settings. Usually you would use in a with-context.

    Parameters
    ----------
    context : str, optional
        Name of seaborn context.
    font_scale : float, optional
        Scale font by factor font_scale.
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
    with alphalens.plotting.plotting_context(font_scale=2):
        alphalens.create_full_tear_sheet(..., set_context=False)

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
    """
    [ORIGINAL]
    Create alphalens default axes style context.

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
    with alphalens.plotting.axes_style(style='whitegrid'):
        alphalens.create_full_tear_sheet(..., set_context=False)

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


# %% specific plottings for each tasks
def plot_returns_table(alpha_beta,
                       alpha_beta_eq,
                       mean_ret_quantile,
                       mean_ret_spread_quantile,
                       notes=''):
    """
    [ORIGINAL]
    Parameters
    ----------
    alpha_beta
    alpha_beta_eq
    mean_ret_quantile
    mean_ret_spread_quantile

    Returns
    -------

    """
    returns_table = pd.DataFrame()
    returns_table = returns_table.append(alpha_beta)
    returns_table = returns_table.append(alpha_beta_eq)
    returns_table.loc["Mean Period Wise Return Top Quantile (bps) - RateRet"] = \
        mean_ret_quantile.iloc[-1] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Return Bottom Quantile (bps) - RateRet"] = \
        mean_ret_quantile.iloc[0] * DECIMAL_TO_BPS
    returns_table.loc["Mean Period Wise Spread Top-Bot (bps) - RateRet"] = \
        mean_ret_spread_quantile.mean() * DECIMAL_TO_BPS
    returns_table.loc["Std. of Period Wise Spread Top-Bot (bps) - RateRet"] = \
        mean_ret_spread_quantile.std() * DECIMAL_TO_BPS

    print("Returns Analysis " + notes)
    utils.print_table(returns_table.apply(lambda x: x.round(3)))


# %% returns related part -------------------------------------------------------------
def plot_quantile_returns_bar(mean_ret_by_q,
                              by_group=False,
                              ylim_percentiles=None,
                              ax=None):
    """
    [ORIGINAL]
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.

    by_group : bool
        Disaggregated figures by group.

    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.

    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    mean_ret_by_q = mean_ret_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if by_group:  # if by_group, suppose ax will be same shape for group Disaggregation
        num_group = len(
            mean_ret_by_q.index.get_level_values('group').unique())

        if ax is None:
            v_spaces = ((num_group - 1) // 2) + 1
            f, ax = plt.subplots(nrows=v_spaces, ncols=2, sharex=False,
                                 sharey=True, figsize=(18, 6 * v_spaces))
            ax = ax.flatten()

        for a, (sc, cor) in zip(ax, mean_ret_by_q.groupby(level='group')):
            (cor.xs(sc, level='group')
             .multiply(DECIMAL_TO_BPS)
             .plot(kind='bar', title=sc, ax=a))

            a.set(xlabel='', ylabel='Mean Return (bps)',  # xlabel='quantiles'
                  ylim=(ymin, ymax))

        if num_group < len(ax):
            ax[-1].set_visible(False)

        return ax

    else:
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(18, 6))

        (mean_ret_by_q.multiply(DECIMAL_TO_BPS)
         .plot(kind='bar',
               title="Mean Period Wise Return By Factor Quantile", ax=ax))
        ax.set(xlabel='', ylabel='Mean Return (bps)',  # xlabel='quantiles'
               ylim=(ymin, ymax))

        return ax


def plot_quantile_returns_violin(return_by_q,
                                 ylim_percentiles=None,
                                 ax=None):
    """
    [ORIGINAL]

    Plots a violin box plot of period wise returns for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex - (date, quantile)
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.

    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.

    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    # long-form data modified to wide-form data to suit for sns.plot mechanism
    unstacked_dr = (return_by_q
                    .multiply(DECIMAL_TO_BPS))
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    sns.violinplot(data=unstacked_dr,
                   x='factor_quantile',
                   hue='forward_periods',
                   y='return',
                   orient='v',
                   cut=0,
                   inner='quartile',
                   ax=ax)
    ax.set(xlabel='', ylabel='Return (bps)',
           title="Period Wise Return By Factor Quantile",
           ylim=(ymin, ymax))

    ax.axhline(0.0, linestyle='-', color='black', lw=0.7, alpha=0.6)

    return ax


def plot_mean_quantile_returns_spread_time_series(mean_returns_spread,
                                                  std_err=None,
                                                  bandwidth=1,
                                                  ax=None):
    """
    [COMMENTED]:
    Plots mean period wise time-series-returns for factor quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series / DataFrame
        [IMPORTANT]: rateret / return depends on application
        [IMPORTANT]: if DataFrame, nested application for each column as Series
        Series with difference between quantile mean returns by period.

    std_err : pd.Series / DataFrame
        [IMPORTANT]: corresponding to format of "mean_returns_spread"
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

    if isinstance(mean_returns_spread, pd.DataFrame):
        if ax is None:
            ax = [None for a in mean_returns_spread.columns]

        ymin, ymax = (None, None)
        for (i, a), (name, fr_column) in zip(enumerate(ax),
                                             mean_returns_spread.iteritems()):  # df.iteritems()  column-wise
            stdn = None if std_err is None else std_err[name]
            a = plot_mean_quantile_returns_spread_time_series(fr_column,
                                                              std_err=stdn,
                                                              ax=a)  # nested-application, DataFrame -> Series
            ax[i] = a
            curr_ymin, curr_ymax = a.get_ylim()
            ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
            ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        for a in ax:
            a.set_ylim([ymin, ymax])

        return ax  # DataFrame case return

    if mean_returns_spread.isnull().all():  # empty return
        warnings.warn("Data is empty for mean_returns_spread plotting of period {}".format(mean_returns_spread.name),
                      UserWarning)
        return ax

    period = mean_returns_spread.name
    title = ('Top Minus Bottom Quantile Mean Return ({} Period Forward Return)'
             .format(period if period is not None else ""))

    if ax is None:
        f, ax = plt.subplots(figsize=(18, 6))

    mean_returns_spread_bps = mean_returns_spread * DECIMAL_TO_BPS

    mean_returns_spread_bps.plot(alpha=0.4, ax=ax, lw=0.7, color='forestgreen')
    mean_returns_spread_bps.rolling(window=22).mean().plot(
        color='orangered',
        alpha=0.7,
        ax=ax
    )
    ax.legend(['mean returns spread', '1 month moving avg'], loc='upper right')

    if std_err is not None:
        std_err_bps = std_err * DECIMAL_TO_BPS
        upper = mean_returns_spread_bps.values + (std_err_bps * bandwidth)
        lower = mean_returns_spread_bps.values - (std_err_bps * bandwidth)
        ax.fill_between(mean_returns_spread.index,
                        lower,
                        upper,
                        alpha=0.3,
                        color='steelblue')

    ylim = np.nanpercentile(abs(mean_returns_spread_bps.values), 95)
    ax.set(ylabel='Difference In Quantile Mean Return (bps)',
           xlabel='',
           title=title,
           ylim=(-ylim, ylim))
    ax.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)

    return ax


# %% multi-period generalized operations --------------------------
def plot_cumulative_returns(factor_returns,
                            period,
                            freq=None,
                            title=None,
                            ax=None):
    """
    [MODIFIED][IMPORTANT]: multi-period generalization
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series - single period for multi-periods
        usually, setting series.name as the freq, i.e. '1D'
        Period wise returns of dollar neutral portfolio weighted by factor
        value.

    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)

    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns

    title: string, optional
        Custom title

    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    # modified to suitable for multi-period rebalancing case
    # factor_returns = perf.cumulative_returns(factor_returns)
    factor_returns_cumlatively = perf.cumulative_returns(
        # rename factor_returns to factor_returns_cumlatively for clarification
        returns=factor_returns,
        period=period,
        freq=freq)

    factor_returns_cumlatively.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    ax.set(ylabel='Cumulative Returns',
           title=("Portfolio Cumulative Return ({} Fwd Period)".format(period)
                  if title is None else title),
           xlabel='')
    ax.axhline(1.0, linestyle='-', color='black', lw=1)
    ax.text(.05, .95,
            " Ann. ret: {:.2f}% \n Ann. vol: {:.2f}% \n Sharpe: {:.2f} \n MaxDD: {:.2f}%"
            .format(ep.annual_return(factor_returns_cumlatively.pct_change()) * 100,
                    ep.annual_volatility(factor_returns_cumlatively.pct_change()) * 100,
                    ep.sharpe_ratio(factor_returns_cumlatively.pct_change()),
                    ep.max_drawdown(factor_returns_cumlatively.pct_change()) * 100),
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')

    return ax


def plot_cumulative_returns_by_quantile(quantile_returns,
                                        period,
                                        freq=None,
                                        title=None,
                                        ax=None):
    """
    [MODIFIED][IMPORTANT]: multi-period generalization
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    quantile_returns : pd.DataFrame for single period,
        multi-indexed by ['datetime', 'factor_quantile']
        Daily returns by factor quantile

    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)

    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns

    title: string, optional
        Custom title

    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """

    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    # transform quantile_returns from long-form to wide form
    ret_wide = quantile_returns.unstack('factor_quantile')

    # modified to suitable for multi-period rebalancing case
    # cum_ret = ret_wide.apply(perf.cumulative_returns)
    cum_ret = ret_wide.apply(perf.cumulative_returns, period=period, freq=freq)

    cum_ret = cum_ret.loc[:, ::-1]  # we want negative quantiles as 'red'

    cum_ret.plot(lw=2, ax=ax, cmap=cm.coolwarm)
    ax.legend()
    ymin, ymax = cum_ret.min().min(), cum_ret.max().max()
    #ymin=0
    #ymax = 3.5
    ax.set(ylabel='Cumulative Returns',
           title=('''Cumulative Return by Quantile
                    ({} Period Forward Return)'''.format(period) if title is None else title),
           xlabel='',
           yscale = 'symlog',
           #yticks=[0,0.5,1,1.5,2,2.5,3,3.5],
           yticks=np.linspace(ymin, ymax, 5),
           ylim=(ymin, ymax))

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.axhline(1.0, linestyle='-', color='black', lw=1)

    return ax


# %% information coefficient part -------------------------------------------------------------
def plot_information_table(ic_data):
    """
    [ORIGINAL]: plot_information_table of stats.

    Parameters
    ----------
    ic_data: single-indexed as date
        columns like   [1D	5D	10D]

    Returns
    -------
        None
    """
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC(ICIR)"] = \
        ic_data.mean() / ic_data.std()  # 即 ICIR ！！！！
    t_stat, p_value = stats.ttest_1samp(ic_data,
                                        0)  # scipy 的包 scipy.stats.    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html
    ic_summary_table["t-stat(IC)"] = t_stat  # 这个t 值本身没什么意义， 需要参考 自由度 和 相对应 置信区间， 越大越拒绝
    ic_summary_table["p-value(IC)"] = p_value  # 主要是看p 值， 跟 confidence level 比，小于，就拒绝H0， 大于就无法拒绝H0
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    print("Information Analysis")
    utils.print_table(ic_summary_table.apply(lambda x: x.round(3)).T)


def plot_ic_ts(ic, ax=None):
    """
    [ORIGINAL]

    Plots 【Spearman Rank Information Coefficient】 and IC moving
    average for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame - single-indexed
        DataFrame indexed by date, with IC for each forward return.

    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    ic = ic.copy()

    num_plots = len(ic.columns)
    if ax is None:
        f, ax = plt.subplots(num_plots, 1, figsize=(18, num_plots * 7))
        ax = np.asarray([ax]).flatten()

    ymin, ymax = (None, None)
    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        ic.plot(alpha=0.7, ax=a, lw=0.7, color='steelblue')
        ic.rolling(window=22).mean().plot(  # pandas 对于含 nan 的点 怎么画图
            # 粗浅测试是 pandas plot 会跳过nan 对应的点
            # rolling 需配合上 min_periods 来去除 nan
            ax=a,
            color='forestgreen',
            lw=2,
            alpha=0.8
        )

        a.set(ylabel='IC', xlabel="")
        a.set_title(
            "{} Period Forward Return Information Coefficient (IC)"
                .format(period_num))
        a.axhline(0.0, linestyle='-', color='black', lw=1, alpha=0.8)
        a.legend(['IC', '1 month moving avg'], loc='upper right')
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')

        curr_ymin, curr_ymax = a.get_ylim()
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

    for a in ax:
        a.set_ylim([ymin, ymax])  # set to common

    return ax


def plot_ic_hist(ic, ax=None):
    """
    [ORIGINAL]
    Plots 【Spearman Rank Information Coefficient】 histogram for a given factor.

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

    num_plots = len(ic.columns)

    v_spaces = ((num_plots - 1) // 3) + 1

    if ax is None:
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))  # TODO: 3 ??? general ???
        ax = ax.flatten()

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        # Depreciation warning
        # sns.distplot(ic.replace(np.nan, 0.), norm_hist=True, ax=a)
        sns.histplot(ic.replace(np.nan, 0.), stat='density', ax=a, kde=True)
        a.set(title="%s Period IC" % period_num, xlabel='IC')
        a.set_xlim([-1, 1])
        a.text(.05, .95, "Mean %.3f \n Std. %.3f" % (ic.mean(), ic.std()),
               fontsize=16,
               bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
               transform=a.transAxes,
               verticalalignment='top')
        a.axvline(ic.mean(), color='w', linestyle='dashed', linewidth=2)

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


def plot_ic_qq(ic, theoretical_dist=stats.norm, ax=None):
    """
    [ORIGINAL]
    [IMPORTANT]: 【Q-Q plot】 of the quantiles of x versus the quantiles/ppf of a distribution
    Plots 【Spearman Rank Information Coefficient】 "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    ic = ic.copy()

    num_plots = len(ic.columns)

    if ax is None:
        v_spaces = ((num_plots - 1) // 3) + 1
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = 'Normal'
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = 'T'
    else:
        dist_name = 'Theoretical'

    for a, (period_num, ic) in zip(ax, ic.iteritems()):
        sm.qqplot(ic.replace(np.nan, 0.).values, theoretical_dist, fit=True,  # fillna
                  line='45', ax=a)  # qqplot 使用 statsmodel 即 sm 来画的   sm 的输入是基于 array-like
        # statsmodel 和 scipy as stats 是相通的？？ https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html#statsmodels.graphics.gofplots.qqplot
        # dist： Comparison distribution. The default is scipy.stats.distributions.norm (a standard normal)
        a.set(title="{} Period IC {} Dist. Q-Q".format(
            period_num, dist_name),
            ylabel='Observed Quantile',
            xlabel='{} Distribution Quantile'.format(dist_name))

    return ax


def plot_ic_by_group(ic_group, ax=None):
    """
    [ORIGINAL]
    Plots 【Spearman Rank Information Coefficient】 for a given factor over
    provided forward returns. Separates by group.

    Parameters
    ----------
    ic_group : pd.DataFrame
        group-wise mean period wise returns.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))
    ic_group.plot(kind='bar', ax=ax)

    ax.set(title="Information Coefficient By Group", xlabel="")
    ax.set_xticklabels(ic_group.index, rotation=45)  # generally for group and freq-grouping only,
    # otherwise the graph will not be friendly, test it

    return ax


def plot_monthly_ic_heatmap(mean_monthly_ic, ax=None):
    """
    [ORIGINAL]
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

    mean_monthly_ic = mean_monthly_ic.copy()

    num_plots = len(mean_monthly_ic.columns)

    if ax is None:
        v_spaces = ((num_plots - 1) // 3) + 1
        f, ax = plt.subplots(v_spaces, 3, figsize=(18, v_spaces * 6))
        ax = ax.flatten()

    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    # 将原本的日期index转换为年、月的multiindex
    # 目的是为了符合后面一部绘制热力图时对输入数据的格式要求
    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month],
        names=["year", "month"])

    for a, (periods_num, ic) in zip(ax, mean_monthly_ic.iteritems()):
        sns.heatmap(
            ic.unstack(),
            annot=True,
            alpha=1.0,
            center=0.0,
            annot_kws={"size": 7},
            linewidths=0.01,
            linecolor='white',
            cmap=cm.coolwarm_r,
            cbar=False,
            ax=a)
        a.set(ylabel='', xlabel='')

        a.set_title("Monthly Mean {} Period IC".format(periods_num))

    if num_plots < len(ax):
        ax[-1].set_visible(False)

    return ax


# %% turnover part  -------------------------------------------------------------
def plot_turnover_table(autocorrelation_data, quantile_turnover):
    """[ORIGINAL]"""
    turnover_table = pd.DataFrame()
    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].iteritems():
            turnover_table.loc["Quantile {} Mean Turnover ".format(quantile),
                               "{}D".format(period)] = p_data.mean()

    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation_data.iteritems():
        auto_corr.loc["Mean Factor Rank Autocorrelation",
                      "{}D".format(period)] = p_data.mean()

    print("Turnover Analysis")
    utils.print_table(turnover_table.apply(lambda x: x.round(3)))
    utils.print_table(auto_corr.apply(lambda x: x.round(3)))


def plot_top_bottom_quantile_turnover(quantile_turnover, period=1, ax=None):
    """
    [ORIGINAL]
    Plots period wise top and bottom quantile factor turnover.

    Parameters
    ----------
    quantile_turnover: pd.Dataframe
        Quantile turnover (each DataFrame's column == a quantile).

    period: int, optional
        Period over which to calculate the turnover.

    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    turnover = pd.DataFrame()
    turnover['top quantile turnover'] = quantile_turnover[max_quantile]
    turnover['bottom quantile turnover'] = quantile_turnover[min_quantile]
    turnover.plot(title='{}D Period Top and Bottom Quantile Turnover'
                  .format(period), ax=ax, alpha=0.6, lw=0.8)
    ax.set(ylabel='Proportion Of Names New To Quantile', xlabel="")

    return ax


def plot_factor_rank_auto_correlation(factor_autocorrelation,
                                      period=1,
                                      ax=None):
    """
    [ORIGINAL]
    Plots factor rank autocorrelation over time.
    See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    factor_autocorrelation : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation
        of factor values.
    period: int, optional
        Period over which the autocorrelation is calculated
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    factor_autocorrelation.plot(title='{}D Period Factor Rank Autocorrelation - 1 Period spacing'  # pd.Series
                                .format(period), ax=ax)
    ax.set(ylabel='Autocorrelation Coefficient', xlabel='')
    ax.axhline(0.0, linestyle='-', color='black', lw=1)
    ax.text(.05, .95, "Mean %.3f" % factor_autocorrelation.mean(),
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')

    return ax


# %% whole tear_sheet part
def plot_quantile_statistics_table(factor_data):
    """[ORIGINAL]"""
    quantile_stats = factor_data.groupby('factor_quantile') \
        .agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] \
                                / quantile_stats['count'].sum() * 100.

    print("Quantiles Statistics")
    utils.print_table(quantile_stats)


# %% event studies part
def plot_quantile_average_cumulative_return(avg_cumulative_returns,
                                            by_quantile=False,
                                            std_bar=False,
                                            title=None,
                                            ax=None):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    ...

    """
    pass


def plot_events_distribution(events, num_bars=50, ax=None):
    """
    Plots the distribution of events in time.

    ...

    """
    pass


# %% QTG added
def plot_factor_value_ts(factor_data: pd.DataFrame, symbol, start=None, end=None):
    """
        plot time-series of factor-value of symbol for checking

    Parameters
    ----------
    factor_data: pd.DataFrame, multi-indexed

    symbol: str or List[str]

    start: str or pd.Timestamp

    end: str or pd.Timestamp

    Returns
    -------
        None

    """
    if start is None:
        start = factor_data.index.levels[0][0]
    if end is None:
        end = factor_data.index.levels[0][-1]
    selected = factor_data.loc[start:end]
    if isinstance(symbol, str):
        symbol = [symbol]
    selected = selected[['factor']].reset_index()
    selected = selected[selected['asset'].isin(symbol)].set_index(['date', 'asset'])
    selected = selected.unstack().droplevel(0, axis=1)
    selected.plot(kind='line', figsize=(16, 9), ylabel='factor value')
