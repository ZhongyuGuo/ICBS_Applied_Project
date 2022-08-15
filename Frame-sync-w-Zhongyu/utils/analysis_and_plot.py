#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/19 18:34:21
# @Author  : Michael_Liu @ QTG
# @File    : analysis_and_plot
# @Software: PyCharm

import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import qtgQuants.alphalens as al
import empyrical as ep


def plot_factor_lag_auto_corr(factor_data, horizon: int = 60):
    """

    Parameters
    ----------
    factor_data: multi-indexed DataFrame
        factor_value saved as 'factor' column

    horizon: period horizon


    Returns
    -------
        axes
    """
    # freq = factor_data.index.levels[0].freq
    full_factor = factor_data.factor.unstack()
    # full_factor = full_factor.fillna(method='ffill')
    autocorr_df = pd.DataFrame()
    for i in range(0, horizon, 3):
        p = i + 1
        corr = full_factor.apply(pd.Series.autocorr, lag=p)
        corr.name = p
        autocorr_df = pd.concat([autocorr_df, corr], axis=1)
    # autocorr_df.fillna(axis=0, method='ffill')
    # seaborn will automatically drop nan data and linking nan points

    fig, axes = plt.subplots(4, 1, figsize=(20, 40))
    sns.lineplot(ax=axes[0], data=autocorr_df.T)
    axes[0].set_title('All Symbol Autocorrelation')
    sns.lineplot(ax=axes[1], data=autocorr_df.mean().T)
    axes[1].set_title('Average Autocorrelation')
    high_ac = autocorr_df[autocorr_df[22] >= 0.7]
    low_ac = autocorr_df[autocorr_df[22] < 0.7]
    sns.lineplot(ax=axes[2], data=high_ac.T)
    sns.lineplot(ax=axes[3], data=low_ac.T)
    axes[2].set_title('Lag20>0.7 Symbol Autocorrelation')
    axes[3].set_title('Lag20<0.7  AC Symbol Autocorrelation')
    return axes


def plot_factor_rank_auto_corr(factor_data, horizon: int = 60):
    """

    Parameters
    ----------
    factor_data: DataFrame

    horizon: period horizon

    Returns
    -------

    """
    autocorr_df = pd.DataFrame()
    for i in range(0, horizon, 3):
        period = i + 1
        corr = al.performance.factor_rank_autocorrelation(factor_data=factor_data, period=period)
        corr.name = period
        autocorr_df = pd.concat([autocorr_df, corr], axis=1)

    fig, axes = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(ax=axes[0], data=autocorr_df.T)
    axes[0].set_title('Periodic Rank Autocorrelation')


def plot_return_series(cum_rets, ax=None):
    """

    Parameters
    ----------
    cum_rets: Series or DataFrame
        with multiple return series in each columns

    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(18, 6))

    cum_rets.plot(ax=ax, lw=3, color='forestgreen', alpha=0.6)
    ax.set(ylabel=cum_rets.name,
           title=(''),
           xlabel='')
    ax.axhline(1.0, linestyle='-', color='black', lw=1)
    ax.text(.05, .95, " Ann. ret: {:.2f}% \n Ann. vol: {:.2f}% \n Sharpe: {:.2f} \n MaxDD: {:.2f}%"
            .format(ep.annual_return(cum_rets.pct_change()) * 100,
                    ep.annual_volatility(cum_rets.pct_change()) * 100,
                    ep.sharpe_ratio(cum_rets.pct_change()),
                    ep.max_drawdown(cum_rets.pct_change()) * 100),
            fontsize=16,
            bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
            transform=ax.transAxes,
            verticalalignment='top')

    return ax


def show_simple_stats(ret):
    """
        使用 empyrical 包，显示最常用的 metrics 信息：
            ann_ret
            ann_vol
            cum_ret
            sharpe
            calmar
            maxdd

    Parameters
    ----------
    ret

    Returns
    -------

    """
    ann_ret = ep.annual_return(ret) * 100
    ann_volatility = ep.annual_volatility(ret) * 100
    cum_ret = ep.cum_returns_final(ret) * 100
    sharpe = ep.sharpe_ratio(ret)
    calmar = ep.calmar_ratio(ret)
    maxdd = ep.max_drawdown(ret) * 100
    df = pd.DataFrame([ann_ret, ann_volatility, cum_ret, sharpe, calmar, maxdd],
                      index=['Annual return(%)', 'Annual volatility(%)', 'Cumulative returns(%)', 'Sharpe Ratio',
                             'Calmar Ratio', 'Max drawdown(%)'], columns=['backtest'])
    df = np.round(df, 3)
    return df


def weekly_shape_ratio(cum_ret):
    """

    Parameters
    ----------
    cum_ret: Series, indexed by 'date'
        daily_return

    Returns
    -------

    """
    weekly_cum_ret = cum_ret.resample('W').last()
    weekly_cum_ret.plot()
    cum_ret.plot()
    weekly_ret = weekly_cum_ret.pct_change()
    return ep.sharpe_ratio(weekly_ret, period='weekly')
