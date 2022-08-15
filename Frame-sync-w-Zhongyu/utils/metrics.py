#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/17 20:57:29
# @Author  : Michael_Liu @ QTG
# @File    : metrics
# @Software: PyCharm

import pandas as pd
import numpy as np
from pandas import (DataFrame, Series)
import qtgQuants.alphalens as al
import qtgQuants.pyfolio as pf


# %% metrics analysis based on averaging N independent sub-portfolio analysis
def get_period_metrics_analysis(factor_data: DataFrame,
                                weights: Series,
                                period: str,
                                level: str = 'all',
                                detail: bool = True):
    """

    Parameters
    ----------
    factor_data: DataFrame, multi-indexed,
        classical alphalen's factor_data with forward-returns

    weights: Series, multi-indexed
        prepared ahead, statisfing leverage==1

    period: str
        example: '5D'

    level: str, level of statisical analysis
        example: 'all', 'group', 'asset'

    detail: bool
        if show detailed N sub-period results

    Returns
    -------
        result: Dict[key, DataFrame]

    """
    factor_info = get_factor_info(factor_data=factor_data,
                                  weights=weights,
                                  period=period)

    result = get_period_metrics(factor_info=factor_info,
                                period=period,
                                level=level,
                                detail=detail)

    for key in result:
        al.utils.print_table(result[key], name=key)

    return result


def get_period_metrics(factor_info: DataFrame,
                       period: str,
                       level: str = 'all',
                       detail: bool = True):
    """

    Parameters
    ----------
    factor_info: multi-indexed, same as input of factor_data
        created by get_factor_info() from factor_data
        columns:
            added:
                'weights'
                'weighted_return'
    period: str
        example: '5D'

    level: str, level of statisical analysis
        example: 'all', 'group', 'asset'

    detail: bool
        if show detailed N sub-period results

    Returns
    -------

    """

    factor_info = factor_info.copy()
    period_int = pd.Timedelta(period).days
    freq = factor_info.index.levels[0].freq
    full_date_idx = factor_info.index.get_level_values(0).unique()
    factor_info['all'] = 'all'
    factor_info['abs_weights'] = factor_info['weights'].abs()
    cols = ['weights', 'weighted_return', 'abs_weights']

    grouper = ['date', level]
    return_data = factor_info.groupby(grouper)[cols].sum()

    long_rate_list = []
    short_rate_list = []
    participate_rate_list = []
    win_rate_list = []
    long_win_rate_list = []
    short_win_rate_list = []
    gain_loss_raio_list = []
    long_gain_loss_ratio_list = []
    short_gain_loss_ratio_list = []
    turnover_ratio_list = []

    result = {}

    for idx in range(period_int):
        # suffix用来标记是哪条线的result
        line_suffix = '_' + str(idx + 1)
        # 按照freq取换仓日，这样即使某一天的数据有缺失也不会导致换仓日取错
        rebalanced_days = pd.date_range(start=full_date_idx[i], end=full_date_idx[-1], freq=period_int * freq)
        sub_return_data = return_data.loc[rebalanced_days]
        # for turnover calculation
        sub_factor_info = factor_info.loc[rebalanced_days]

        long_sub_return_data = sub_return_data[factor_info['weights'] > 0]
        short_sub_return_data = sub_return_data[factor_info['weights'] < 0]

        # long, short, participate rate
        long_rate = sub_return_data.groupby(level)['weights'].apply(get_long_rate)
        long_rate.name = 'long_rate' + line_suffix
        long_rate_list.append(long_rate)

        short_rate = 1 - long_rate
        short_rate.name = 'short_rate' + line_suffix
        short_rate_list.append(short_rate)

        participate_rate = sub_return_data.groupby(level)['abs_weights'].apply(get_participate_rate, rebalanced_days)
        participate_rate.name = 'participate_rate' + line_suffix
        participate_rate_list.append(participate_rate)

        def get_full_win_rate(df, level, suffix):
            win_rate = df.groupby(level)['weighted_return'].apply(get_win_rate)
            win_rate.name = 'win_rate' + suffix
            return win_rate

        # win_rate
        win_rate_list.append(get_full_win_rate(sub_return_data, level, line_suffix))
        long_win_rate_list.append(get_full_win_rate(long_sub_return_data, level, line_suffix))
        short_win_rate_list.append(get_full_win_rate(short_sub_return_data, level, line_suffix))

        def get_full_gain_loss_ratio(df, level, suffix):
            gain_loss_ratio = df.groupby(level)['weighted_return'].apply(get_gain_loss_ratio)
            gain_loss_ratio.name = 'gain_loss_ratio' + suffix
            return gain_loss_ratio

        gain_loss_raio_list.append(get_full_gain_loss_ratio(sub_return_data, level, line_suffix))
        long_gain_loss_ratio_list.append(get_full_gain_loss_ratio(long_sub_return_data, level, line_suffix))
        short_gain_loss_ratio_list.append(get_full_gain_loss_ratio(short_sub_return_data, level, line_suffix))

        def get_turnover_ratio(factor_info, period, level, suffix):
            grouper = ['date', level]
            turnover_info = create_turnover_info(factor_info, period)
            freq_adjust = pd.Timedelta('252Days') / pd.Timedelta(period)
            turnover_ratio = turnover_info.groupby(grouper)['turnover'].sum()
            turnover_ratio = turnover_ratio.groupby(level).mean() * freq_adjust

            return turnover_ratio

        # turnover
        turnover_ratio = get_turnover_ratio(factor_info, period, level, line_suffix)
        turnover_ratio.name = 'turnover_ratio' + line_suffix
        turnover_ratio_list.append(turnover_ratio)

    # 将对应的 dataframe 放入 dict 中
    long_rate_df = pd.concat(long_rate_list, axis=1)
    short_rate_df = pd.concat(short_rate_list, axis=1)
    participate_rate_df = pd.concat(participate_rate_list, axis=1)
    win_rate_df = pd.concat(win_rate_list, axis=1)
    long_win_rate_df = pd.concat(long_win_rate_list, axis=1)
    short_win_rate_df = pd.concat(short_win_rate_list, axis=1)
    gain_loss_ratio_df = pd.concat(gain_loss_raio_list, axis=1)
    long_gain_loss_ratio_df = pd.concat(long_gain_loss_ratio_list, axis=1)
    short_gain_loss_ratio_df = pd.concat(short_gain_loss_ratio_list, axis=1)
    turnover_ratio_df = pd.concat(turnover_ratio_list, axis=1)

    result['long_rate_df'] = long_rate_df
    result['short_rate_df'] = short_rate_df
    result['participate_rate_df'] = participate_rate_df
    result['win_rate_df'] = win_rate_df
    result['long_win_rate_df'] = long_win_rate_df
    result['short_win_rate_df'] = short_win_rate_df
    result['gain_loss_ratio_df'] = gain_loss_ratio_df
    result['long_gain_loss_ratio_df'] = long_gain_loss_ratio_df
    result['short_gain_loss_ratio_df'] = short_gain_loss_ratio_df
    result['turnover_ratio_df'] = turnover_ratio_df

    if not detail:
        for key in result:
            result[key] = result[key].mean(axis=1)

    return result


def get_factor_info(factor_data: DataFrame,
                    weights: Series,
                    period: str):
    """

    Parameters
    ----------
    factor_data: DataFrame, multi-indexed, classical alphalen's factor_data with forward-returns

    weights: Series, multi-indexed
        prepared ahead, statisfing leverage==1

    period: str

    Returns
    -------
        factor_info: multi-indexed, same as input of factor_data
        columns:
            added:
                'weights'
                'weighted_return'

    """
    weights.name = 'weights'
    factor_info = factor_data.merge(weights, left_index=True, right_index=True, how='left')
    factor_info['weighted_return'] = factor_info[period] * factor_info['weights']
    return factor_info


# %%
def get_cum_ret_metrics_analysis(cumulative_return: Series,
                                 benchmark_rets: Series = None):
    """
        make analysis inheritate from pyfolio from cumulative_return

    Parameters
    ----------
    cumulative_return: Series
        daily cumulative return

    benchmark_rets: Series

    Returns
    -------

    """
    pf.tears.create_returns_tear_sheet(returns=cumulative_return,
                                       benchmark_rets=benchmark_rets)


# %% support function for apply(groupby)
def get_long_rate(s: Series):
    """

    Parameters
    ----------
    s: Series, weight/signal, only sign matters

    Returns
    -------
    rate: scalar
        基于 s 的 多、空比率，用来反映品种的多空对称性
        long_rate + short_rate = 1
    """
    # 去掉奇异值日期
    s = s.dropna()

    long_count = (s > 0).sum()
    participate_num = (s != 0).sum()
    if participate_num > 0:
        rate = long_count / participate_num
    else:
        rate = 0
    return rate


def get_participate_rate(s: Series, full_idx: pd.DatetimeIndex):
    """

    Parameters
    ----------
    s: rebalancing weight each round

    full_idx: full datetime_index for calculation

    Returns
    -------

    """
    # 构建品种、行业覆盖日期长度
    s = s.unstack()
    s = s.reindex(full_idx)
    first_valid_idx = s.first_valid_index()
    s = s.loc[first_valid_idx:]
    s = s.fillna(0)
    s = s.stack()

    total_num = len(s)
    participate_num = (s != 0).sum()

    if total_num > 0:
        rate = participate_num / total_num
    else:
        rate = 0
    return rate


def get_win_rate(s: Series):
    """

    Parameters
    ----------
    s: Series
        return series, single indexed by datetime

    Returns
    -------
        基于s 的胜率计算
    """
    win_count = (s > 0).sum()
    participate_num = (s != 0).sum()
    if participate_num > 0:
        rate = win_count / participate_num
    else:
        rate = 0
    return rate


def get_gain_loss_ratio(s: Series):
    """

    Parameters
    ----------
    s: Series
        (weighted) return (contribution)

    Returns
    -------
        平均盈利 vs 平均亏损 比率
    """
    gain_avg = s[s > 0].mean()
    loss_avg = s[s < 0].mean().abs()
    if loss_avg == 0 and gain_avg > 0:
        rate = np.inf
    elif loss_avg == 0 and gain_avg == 0:
        rate = 0
    else:
        rate = gain_avg / loss_avg
    return rate


def create_turnover_info(factor_info: DataFrame, period: str) -> DataFrame:
    """
        基于 factor_info (单一 multi-period 序列)，生成品种级的逐期 turnover (rate)
            Note: turnover (rate) defined as:
                abs(weight-diff before and after the rebalancing)
                thus, the value will be influenced by leverage(usually set to 1 by
                letting weight.abs().sum() == 1 on rebalancing)

    Parameters
    ----------
    factor_info: DataFrame, multi-indexed by ('date', 'asset')
        columns:
            weights(rebalancing)
            period(forward_return period),
            weighted_return(weighted forward_return)
            (others)
    period: str
        example: '5D'

    Returns
    -------
        DataFrame, multi-indexed by ('date', 'asset')
        columns:
            added:
                'prev_weight',
                'portfolio_return'
                'turnover'

    """
    data = factor_info.copy()
    prev_weights = data['weights'].unstack().fillna(0).shift(1).stack()
    prev_weights.name = 'prev_weights'

    prev_weighted_return = data['weighted_return'].unstack().fillna(0).shift(1).stack()
    prev_weighted_return.name = 'prev_weighted_return'

    data = data.merge(prev_weights, how='outer', left_index=True, right_index=True)
    data = data.merge(prev_weighted_return, how='outer', left_index=True, right_index=True)
    data['portfolio_return'] = data['prev_weighted_return'].groupby('date').transform('sum')
    data['turnover'] = ((data['weights']) - (data['prev_weights'] + data['prev_weighted_return']) / (
            1 + data['portfolio_return'])).abs()
    # drop NAN
    data = data.loc[~data.turnover.isnull()]

    return data
