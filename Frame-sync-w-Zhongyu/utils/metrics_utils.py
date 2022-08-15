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

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# %% metrics analysis based on averaging N independent sub-portfolio analysis
def get_period_metrics_analysis(factor_data: DataFrame,
                                weights: Series,
                                period: str,
                                level: str = 'all',
                                long_short_detail: bool = True,
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

    long_short_detail: bool
        说明：long_short_detail if True 主要是计算多空头单独的胜率和盈亏比。 之前尝试分行业去分析的时候，发现如果binnin_by_group=False,
         1 vs 5 quantiile内可能会发生没有属于这个行业的品种，然后会导致无法分出 long 和 short return data。
         那个时候为了防止这个报错写了这个参数。这个情况只会在 binning_by_group = False 然后单独看一个行业内品种的时候发生，
         需要用到的场合比较少。

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
                                long_short_detail=long_short_detail,
                                detail=detail)

    for key in result:
        al.utils.print_table(result[key], name=key)

    return result


def get_period_metrics(factor_info: DataFrame,
                       period: str,
                       level: str = 'all',
                       long_short_detail=True,
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

    if level != 'asset':
        grouper = ['date', level]
        return_data = factor_info.groupby(grouper)[cols].sum()
        if long_short_detail:
            long_return_data = factor_info[factor_info['weights'] > 0].groupby(grouper)[cols].sum()
            short_return_data = factor_info[factor_info['weights'] < 0].groupby(grouper)[cols].sum()
    else:
        return_data = factor_info[cols]
        if long_short_detail:
            long_return_data = factor_info.loc[factor_info['weights'] > 0, cols]
            short_return_data = factor_info.loc[factor_info['weights'] < 0, cols]

    long_rate_list = []
    short_rate_list = []
    participate_rate_list = []
    win_rate_list = []
    long_win_rate_list = []
    short_win_rate_list = []
    gain_loss_ratio_list = []
    long_gain_loss_ratio_list = []
    short_gain_loss_ratio_list = []
    turnover_ratio_list = []

    result = {}

    for idx in range(period_int):
        # suffix用来标记是哪条线的result
        line_suffix = '_' + str(idx + 1)
        # 按照freq取换仓日，这样即使某一天的数据有缺失也不会导致换仓日取错
        rebalanced_days = pd.date_range(start=full_date_idx[idx], end=full_date_idx[-1],
                                        freq=period_int * freq)
        sub_return_data = return_data.loc[rebalanced_days]
        # for turnover calculation
        sub_factor_info = factor_info.loc[rebalanced_days]
        if long_short_detail:
            sub_long_return_data = long_return_data.loc[rebalanced_days]
            sub_short_return_data = short_return_data.loc[rebalanced_days]

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
        if long_short_detail:
            long_win_rate_list.append(get_full_win_rate(sub_long_return_data, level, line_suffix))
            short_win_rate_list.append(get_full_win_rate(sub_short_return_data, level, line_suffix))

        def get_full_gain_loss_ratio(df, level, suffix):
            gain_loss_ratio = df.groupby(level)['weighted_return'].apply(get_gain_loss_ratio)
            gain_loss_ratio.name = 'gain_loss_ratio' + suffix
            return gain_loss_ratio

        gain_loss_ratio_list.append(get_full_gain_loss_ratio(sub_return_data, level, line_suffix))
        if long_short_detail:
            long_gain_loss_ratio_list.append(get_full_gain_loss_ratio(sub_long_return_data, level, line_suffix))
            short_gain_loss_ratio_list.append(get_full_gain_loss_ratio(sub_short_return_data, level, line_suffix))

        def get_turnover_ratio(sub_factor_info, period, level, suffix):
            turnover_info = create_turnover_info(sub_factor_info)
            freq_adjust = pd.Timedelta('252Days') / pd.Timedelta(period)
            if level != 'asset':
                grouper = ['date', level]
                turnover_ratio = turnover_info.groupby(grouper)['turnover'].sum()
            else:
                turnover_ratio = turnover_info['turnover']

            turnover_ratio = turnover_ratio.groupby(level).mean() * freq_adjust

            return turnover_ratio

        # turnover
        turnover_ratio = get_turnover_ratio(sub_factor_info, period, level, line_suffix)
        turnover_ratio.name = 'turnover_ratio' + line_suffix
        turnover_ratio_list.append(turnover_ratio)

    # 将对应的 dataframe 放入 dict 中
    long_rate_df = pd.concat(long_rate_list, axis=1)
    short_rate_df = pd.concat(short_rate_list, axis=1)
    participate_rate_df = pd.concat(participate_rate_list, axis=1)
    win_rate_df = pd.concat(win_rate_list, axis=1)
    gain_loss_ratio_df = pd.concat(gain_loss_ratio_list, axis=1)
    turnover_ratio_df = pd.concat(turnover_ratio_list, axis=1)
    if long_short_detail:
        long_win_rate_df = pd.concat(long_win_rate_list, axis=1)
        short_win_rate_df = pd.concat(short_win_rate_list, axis=1)
        long_gain_loss_ratio_df = pd.concat(long_gain_loss_ratio_list, axis=1)
        short_gain_loss_ratio_df = pd.concat(short_gain_loss_ratio_list, axis=1)

    result['long_rate'] = long_rate_df
    result['short_rate'] = short_rate_df
    result['participate_rate'] = participate_rate_df
    result['win_rate'] = win_rate_df
    result['gain_loss_ratio'] = gain_loss_ratio_df
    result['turnover_ratio'] = turnover_ratio_df
    if long_short_detail:
        result['long_win_rate'] = long_win_rate_df
        result['short_win_rate'] = short_win_rate_df
        result['long_gain_loss_ratio'] = long_gain_loss_ratio_df
        result['short_gain_loss_ratio'] = short_gain_loss_ratio_df

    if not detail:
        for key in result:
            result[key] = result[key].mean(axis=1)
            result[key] = result[key].to_frame(key).T
        metrics = pd.concat(result.values(), axis=0)
        result = {}
        result['metrics'] = metrics

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
    # TODO: 通过引入额外的 weights_wo_na 来保证后续问题的合理解决

    weights.name = 'weights'
    factor_info = factor_data.merge(weights, left_index=True, right_index=True, how='left')
    factor_info['weighted_return'] = factor_info[period] * factor_info['weights']
    # 列出所有品种每日的weights,如果原本weights中没有，是因为该品种当天没有weight，用0填补
    weights_wo_na = weights.unstack().fillna(0).stack()
    weights_wo_na.name = 'weights_wo_na'
    # 将weights_wo_na merge进入factor_info中，weights_wo_na的index会更多，使用how=right，保留这些多出来的日期、品种
    factor_info = factor_info.merge(weights_wo_na, left_index=True, right_index=True, how='right')
    factor_info['weighted_return_wo_na'] = factor_info[period] * factor_info['weights_wo_na']
    # weights被fillna的日期对应的return依然是nan，所以weighted_return_wo_na里会有na,实际上就是weighted return为0，用0fill
    factor_info['weighted_return_wo_na'] = factor_info['weighted_return_wo_na'].fillna(0)
    # 上一步merge how='right’会导致freq日期的freq变成None,重新加上去。
    # factor_data本身可能会缺少日期，但是factor_data的freq中日期是全的
    # 这样的话factor_info在unstack再stack回来时无法直接设置freq，需要先reindex
    # reindex不会对数据本身产生影响
    factor_info = factor_info.reindex(factor_data.index.levels[0], level=0)
    factor_info.index.levels[0].freq = factor_data.index.levels[0].freq
    return factor_info


# %% decomposition of N independent sub-portfolio and plot
def detail_and_all_level_decomp(ret: DataFrame, weights: DataFrame, period: str, group: dict,
                                initial_total_value: float = 100000.0,
                                commission_rate: float = 0.0002):
    """
        create detail result dict and plot decomposition of the portfolio at 'all' level

    Parameters
    ----------
    ret
    weights
    period
    group
    initial_total_value
    commission_rate

    Returns
    -------

    """
    result = create_period_portfolio_series(ret=ret, weights=weights, period=period,
                                            initial_total_value=initial_total_value, commission_rate=commission_rate)
    cum_ret_by_sub_level(result=result, group=group, period=period, level='all',
                         initial_total_value=initial_total_value)
    cum_ret_by_ls(result=result, group=group, period=period, level='all', initial_total_value=initial_total_value)
    return result


# %% create and record N independent sub-portfolio asset
def create_period_portfolio_series(ret: DataFrame,
                                   weights: DataFrame,
                                   period: str,
                                   subs: int = None,
                                   spacing: int = 1,
                                   start_day: int = 0,
                                   initial_total_value: float = 100000.0,
                                   commission_rate: float = 0.0002):
    """
        create dict of detailed daily asset info for N independent period-rebalancing portfolio

        See Also: backtester_old.get_hold_profit_df()

        See Also: 海通 FICC 系列研报 二、 五、 十三， 关于 等分五资金，间隔投资来达到多起始日平滑的目标

    Parameters
    ----------
    ret: Series, multi-indexed
        daily forward return, thus, '1D' in factor_data

    weights: Series, multi-indexed
        daily weight-value according to factor-value and weighting mechanism

    period: str
        example: '5D'

    subs: int
        number of subportfolios to be created, 比如说  总共就分成 5 份资金

    spacing: int
        Number of days to wait before creating the next portfolio. E.g. spacing = 4 means to create on the 1st,
        5th, 9th day etc.

    start_day: int
        Start the first sub portfolio on the Nth day. 0 means to build the first sub portfolio on the first day

    initial_total_value: float
        initial base asset

    commission_rate: float
        commission rate of the turnover notional

    Returns
    -------
        result: dict[str, DataFrame]
            key:
                profit: DataFrame, multi-indexed
                    每个品种每日在每个 subporfolio 上的收益金额
                long_profit: DataFrame, multi-indexed
                    每个品种每日在每个 subporfolio 的多头收益金额
                short_profit: DataFrame, multi-indexed
                    每个品种每日在每个 subporfolio 的空头收益金额
                value: DataFrame, multi-indexed
                    每个品种每日在每个 subporfolio 的资金量。正负号代表多空，数值代表资金量
                abs_value: DataFrame, multi-indexed
                    每个品种每日在每个 subporfolio 的资金量的绝对值。
                    在 leverage==1 的情况下， 每个品种的资金量的绝对值相加等于组合的总资金量
                portfolio_value: DataFrame, multi-indexed
                    每个sub portfolio每日的nav。计算方法为上一期portfolio_value加当期total_profit
                turnover_rate_nav: DataFrame, multi-indexed
                    品种当日的换手率（换手金额的绝对值/当日sub portfolio的nav）
                turnover_rate_notional: DataFrame, multi-indexed
                    品种当日的换手率（换手金额的绝对值/当日sub portfolio的notional value）
                comission: DataFrame, multi-indexed
                    手续费，换手金额的绝对值乘以手续费率

    """
    full_date_index = weights.index.get_level_values(0).unique()
    period_int = pd.Timedelta(period).days
    # 如果没有指定sub portfolio数量，则默认每日建仓
    if subs is None:
        subs = period_int
    # 获得建仓日期的index
    starting_day_idx_list = [int(start_day + (spacing * x)) for x in range(subs)]
    # 将当日没有权重和return的品种在对应位置填上0，为了计算每个品种每日的收益
    weights = weights.unstack().fillna(0).stack()
    freq = ret.index.levels[0].freq
    ret = ret.unstack().fillna(0).stack()
    profit_df_list = []
    value_df_list = []
    portfolio_value_df_list = []
    turnover_nav_df_list = []
    turnover_notional_df_list = []
    long_profit_df_list = []
    short_profit_df_list = []
    commission_df_list = []
    # result为储存输出df的dict
    result = {}
    # 每个subportfolio循环
    for j in range(subs):
        # 按照freq获取这条线的换仓日
        starting_day_idx = starting_day_idx_list[j]
        rebalanced_days = pd.date_range(start=full_date_index[starting_day_idx], end=full_date_index[-1],
                                        freq=period_int * freq)
        # 每个sub-portfolio开始的日期不一样，保留开始以后的全部日期
        dates = full_date_index[starting_day_idx:]
        profit_dict = {}
        long_profit_dict = {}
        short_profit_dict = {}
        turnover_nav_dict = {}
        turnover_notional_dict = {}
        commission_dict = {}
        value_dict = {}
        portfolio_value_dict = {}
        # 每天循环
        for i in range(len(dates)):
            # 建仓日
            if i == 0:
                # 获取当日的品种权重
                curr_weight = weights.loc[dates[i]]
                # 按权重分配资金
                # 每条subportfolio使用全部资金，之后使用get_field_level_data转换成多起始日平滑情况
                # profolio value即nav
                portfolio_value = initial_total_value
                portfolio_value_dict[dates[i]] = portfolio_value
                # weight包含杠杆信息，value的abs sum受杠杆影响
                value = portfolio_value * curr_weight
                value_dict[dates[i]] = value

                # 建仓日profit为0
                profit = pd.Series(0, index=value.index)
                profit_dict[dates[i]] = profit
                long_profit = profit.copy()
                short_profit = profit.copy()
                long_profit[curr_weight < 0.0] = 0.0
                short_profit[curr_weight > 0.0] = 0.0
                long_profit_dict[dates[i]] = long_profit
                short_profit_dict[dates[i]] = short_profit

                # 建仓日 turnover value即为第一天每个品种上的value
                turnover_value = value.abs()
                turnover_nav = turnover_value / portfolio_value
                turnover_nav_dict[dates[i]] = turnover_nav
                turnover_notional = turnover_value / (value.abs().sum())
                turnover_notional_dict[dates[i]] = turnover_notional
                commission = commission_rate * turnover_value
                commission_dict[dates[i]] = commission


            else:
                # 换仓日
                if dates[i] in rebalanced_days:
                    # 品种收益为上一日品种资金乘以上一日的forward return
                    profit = value * ret.loc[dates[i - 1]]
                    # sub portfolio当日总收益为各品种当日收益之和
                    total_profit = profit.sum()
                    # 按照收益更新品种的资金
                    value = value + np.sign(value) * profit
                    # sub portfolio当日nav
                    portfolio_value += total_profit
                    portfolio_value_dict[dates[i]] = portfolio_value
                    # 按照上一日的权重获得当日的多头、空头收益
                    long_profit = profit.copy()
                    short_profit = profit.copy()
                    long_profit[curr_weight < 0.0] = 0.0
                    short_profit[curr_weight > 0.0] = 0.0
                    long_profit_dict[dates[i]] = long_profit
                    short_profit_dict[dates[i]] = short_profit

                    # 记录当日数据
                    profit_dict[dates[i]] = profit

                    # 将权重、品种资金更新为换仓日的对应数据
                    curr_weight = weights.loc[dates[i]]
                    new_value = portfolio_value * curr_weight
                    # 品种换手金额
                    turnover_value = (new_value - value).abs()

                    # 计算换手所产生的手续费
                    commission = turnover_value * commission_rate
                    # 品种换手率为换手金额除以当日nav或notional
                    turnover_nav = (turnover_value / portfolio_value)
                    turnover_notional = turnover_value / (value.abs().sum())
                    # 用新的品种资金覆盖旧的品种资金
                    value = new_value
                    # 数据记录
                    value_dict[dates[i]] = value
                    turnover_nav_dict[dates[i]] = turnover_nav
                    turnover_notional_dict[dates[i]] = turnover_notional
                    commission_dict[dates[i]] = commission
                # 非换仓日（品种权重保持不变）
                else:
                    # 计算当日品种收益、自己以及总收益、总资金，并记录
                    profit = value * ret.loc[dates[i - 1]]
                    total_profit = profit.sum()
                    value = value + np.sign(value) * profit
                    portfolio_value = portfolio_value + total_profit
                    portfolio_value_dict[dates[i]] = portfolio_value

                    long_profit = profit.copy()
                    short_profit = profit.copy()
                    long_profit[curr_weight < 0.0] = 0.0
                    short_profit[curr_weight > 0.0] = 0.0
                    long_profit_dict[dates[i]] = long_profit
                    short_profit_dict[dates[i]] = short_profit

                    profit_dict[dates[i]] = profit
                    value_dict[dates[i]] = value

                    turnover_nav = pd.Series(0,index=value.index)
                    turnover_notional = pd.Series(0, index=value.index)
                    commission = pd.Series(0, index=value.index)
                    turnover_nav_dict[dates[i]] = turnover_nav
                    turnover_notional_dict[dates[i]] = turnover_notional
                    commission_dict[dates[i]] = commission



        # 用来将dict转换为df，同时加上index的列名
        def _dict_to_df(d, j):
            # 这样输出的df会有两列index，对于分品种的数据分别是 date, asset
            df = pd.DataFrame.from_dict(d, orient='index')
            df.index.name = 'date'
            df.columns.name = 'asset'
            df = df.stack().to_frame(j)
            return df

        # 将这条线的每日数据变成一个df，加入对应list中
        profit_df_list.append(_dict_to_df(profit_dict, j))
        long_profit_df_list.append(_dict_to_df(long_profit_dict, j))
        short_profit_df_list.append(_dict_to_df(short_profit_dict, j))
        value_df_list.append(_dict_to_df(value_dict, j))
        portfolio_value_df_list.append(_dict_to_df(portfolio_value_dict, j))
        turnover_nav_df_list.append(_dict_to_df(turnover_nav_dict, j))
        turnover_notional_df_list.append(_dict_to_df(turnover_notional_dict, j))
        commission_df_list.append(_dict_to_df(commission_dict, j))

    # 将所有线的df合并，并存入result中
    result['profit'] = pd.concat(profit_df_list, axis=1)
    result['long_profit'] = pd.concat(long_profit_df_list, axis=1)
    result['short_profit'] = pd.concat(short_profit_df_list, axis=1)
    result['value'] = pd.concat(value_df_list, axis=1)
    result['abs_value'] = result['value'].abs()
    result['portfolio_value'] = pd.concat(portfolio_value_df_list, axis=1)
    result['turnover_rate_nav'] = pd.concat(turnover_nav_df_list, axis=1)
    result['turnover_rate_notional'] = pd.concat(turnover_notional_df_list, axis=1)
    result['commission'] = pd.concat(commission_df_list, axis=1)
    return result


# %% summarize field data from the details of N independent sub-portfolio info
def get_field_level_data(result: dict, group: dict, period: str,
                         field: str, level: str, subs: int = None,
                         initial_total_value=100000.0, by_date: bool = True, detail: bool = True):
    """

    Parameters
    ----------
    result: dict
        result of the above 'create_period_portfolio_series' function

    group: dict,
        asset -> group mapping

    period: str

    field: str
        keys of result for summarize task

    level: str
        'asset', 'group', 'all'

    intial_total_value: float
        initial base asset

    by_date: bool
        if need by_date info

    detail: bool
        if need info for each sub-portfolio or get mean of info of N sub-portfolio's

    Note: here, by_data will be dealt first and detail will be dealt later
        if, need other ordering, manually operate by full detail_info

    Returns
    -------
    data: DataFrame

        If field is not turnover(i.e. a money value),the values will be divided by the previous portfolio value
        to transform the money value to a ratio
        for profits:
            the ratio is the Net-profit +/-  relative to previous portfolio value
        for value:
            the ratio is net(long-short)-exposure of the specific group
        for abs_value:
            the ratio is the abs(long+short)-exposure of the specific group
        for commission:
            the ratio is the commission relative to previous portfolio value

    """

    data = result[field]
    period_int = pd.Timedelta(period).days
    portfolio_value = result['portfolio_value'].reset_index(1, drop=True)
    # Fillna (1)是因为我拿到在建仓前profolio value为nan，导致建仓第一天的money value为nan
    prev_portfolio_value = portfolio_value.shift(1).fillna(initial_total_value)
    if subs is None:
        subs = period_int
    # 处理profit、commission等量纲为金额的数据时，先除以最初的总资金
    if field not in ['turnover_rate_nav', 'turnover_rate_notional']:
        data = data / prev_portfolio_value
    # 在生成result时，每个subportfolio是使用全部资金的，但是多起始日平滑下实际上只用1/N资金
    # 这一差别会使得换仓手续费放大N倍，此处除以N以还原多起始日平滑的真实情况
    if field == 'commission':
        data = data / period_int
    # 将品种所属行业加入data中
    group_df = pd.DataFrame.from_dict(group, orient='index')
    group_df = group_df.rename(columns={0: 'group'})
    group_df.index.name = 'asset'
    data = data.join(group_df, how='left', on='asset')
    grouper = []
    # 如果需要逐日数据，则在grouper中加入'date'
    # 无论逐日与否 turnover 都需要先要逐日sum
    if by_date or field in ['turnover_rate_nav', 'turnover_rate_notional']:
        grouper = ['date']
    if level == 'all':
        # 如果需要整体级别的数据，将所有品种的行业都改为all，即都在一组中
        data['group'] = 'all'
        # all 现在就可以也被视为一个行业（全行业），所以将他的level改成group，方便后面groupby处理
        level = 'group'
    # 在grouper中加入需要的level
    grouper.append(level)
    # 按照要求将数据sum起来
    # 现在grouper中可能会有'date',一定会有'asset','group'中的一个
    # 对于turnover来说,grouper是['date','asset']或者['date',group]，即每日品种换手率或每日行业的换手率（当日行业内品种换手率相加）
    # 在groper为['date','asset']时，sum会导致原本nan的地方被填入0，需要避免
    if grouper != ['date', 'asset']:
        data_list = []
        for i in range(subs):
            # 取出每一个subporfolio中的数据
            sub_data = data[[data.columns[i], 'group']]
            # 不在sub portfolio中日期也会被取到，全为np.nan，但是下一步sum会导致这些日期的值为0，所以先drop掉
            sub_data = sub_data.dropna()
            data_list.append(sub_data.groupby(grouper).sum())
        # 将每个sub portfolio的数据重新拼在一起
        data = pd.concat(data_list, axis=1)

    # 获取年化平均换手率
    if field in ['turnover_rate_nav', 'turnover_rate_notional'] and by_date == False:
        # 用以年化的系数
        freq_adjust = pd.Timedelta('252Days') / pd.Timedelta(period)
        data = data.groupby(level).mean() * freq_adjust

    # 是否需要每条线的数据
    if not detail:
        # 对于换手率，将每日每个品种五条线上的换手率平均
        # 实际上每日5条线上的只有1条会有换手率，其他四条为nan，所以平均就是取这条线上的换手率
        data = data.mean(axis=1).to_frame(field)

    # 在前面mean的操作中，如果grouper!=['date','asset']，之前加入的行业分组会丢失，这里重新加入
    if 'group' not in data.reset_index().columns:
        data = data.join(group_df, how='left', on='asset')

    return data


# %% 将累计走势图，分解成 level 级别的子贡献，获得拆解
def cum_ret_by_sub_level(result: dict, group: dict, period: str, level: str,
                         group_name: str = None, direction: str = None, initial_total_value: float = 100000.0):
    """
        将某个level的累计收益率拆分至下一个level，并绘制
        例：将整体的多头累计收益率拆分为每个行业的多头累计收益率
            将某个行业的累计收益率拆分为行业内每个品种的累计收益率
        level只能是某一个行业（比如‘有色_贵金属’）或者all， asset level不能再继续拆分成别的level

    Parameters
    ----------
    result: dict
        result of the above 'create_period_portfolio_series' function

    group: dict,
        asset -> group mapping

    period: str

    level: str
        'asset', 'group', 'all'

    group_name: str
        specific group name for group analysis

    direction: str
        'long', 'short' , None for 'long-short'

    intial_total_value: float
        initial base asset

    Returns
    -------

    """

    # diretion的意思是要求多头/空头数据，不输入即为整体的
    # 按照direction的要求制作field（取哪个dataframe的数据）
    if level != 'all' and direction is not None:
        field = direction + '_profit'
    else:
        field = 'profit'

    # 如果是整体层面拆分，那取整体层面和行业层面每个行业的profit即可，需要驻日数据，不需要每条数据
    if level == 'all':
        df_big = get_field_level_data(result, group, period=period, field=field, level='all',
                                      initial_total_value=initial_total_value, by_date=True, detail=False)
        df_small = get_field_level_data(result, group, period=period, field=field, level='group',
                                        initial_total_value=initial_total_value, by_date=True, detail=False)
    # 行业层面拆分
    else:
        # 取出的行业层面数据是各个行业的profit，所以需要再提取出需要的行业
        df_big = get_field_level_data(result, group, period=period, field=field, level='group',
                                      initial_total_value=initial_total_value, by_date=True, detail=False)

        df_big = df_big.xs(group_name, level=1, drop_level=False)
        # 同样地，asset层面数据包含所有品种的profit，需要取出在行业内的品种
        df_small = get_field_level_data(result, group, period=period, field=field, level='asset',
                                        initial_total_value=initial_total_value, by_date=True, detail=False)
        df_small = df_small[df_small.group == group_name]
        # 取出品种之后group列就没有用了，drop掉
        df_small = df_small.drop(columns='group')

    # 将两个层面的数据合并 合并是一个multiindex dataframe
    #                          short_profit
    #    date       group
    #    2010-01-05 HC         0.000000
    #               I          0.000000
    #               J          0.000000
    #               JM         0.000000
    #               RB        -0.000055
    #    ...                        ...
    #    2021-05-17 SF         0.001298
    #               SM         0.000000
    #               SS         0.000000
    #               ZC         0.000000
    #               黑色       -0.001578

    df = pd.concat([df_big, df_small], axis=0).sort_index()
    # 将收益率转换为累计收益率
    # TODO：实际上每条 subportfolio 是复利的操作和思维， 只是在期初采用了平均分配资金的方式
    #  因此，每天 profit / initial_total_value 实际上是，考虑的按一个常数的折算和约化，
    #  不改变每条 subportfolio 复利的本质， 因此，用 cumsum + 1，而非 +1再 cumprod 的做法
    #  获得的是 NAV 层面的结果
    df = df.unstack().cumsum() + 1
    # columns有两层index,将第一层drop掉
    df.columns = df.columns.droplevel()

    plt.rcParams["figure.figsize"] = (20, 12)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    ax = df.plot()
    ax.axhline(y=1, color='black', linestyle='--', lw=2, alpha=0.5)
    ax.set_title(level + ' ' + (direction if direction else '') + ' Cumulative return decomposition ')

    return ax


# %% 将累计走势图，分解成 level 级别的多空贡献，获得拆解
def cum_ret_by_ls(result: dict, group: dict, period: str, level: str,
                  group_name: str = None, initial_total_value: float = 100000.0):
    """
        将某个层面的累计收益率拆分为同层面的多头和空头

    result: dict
        result of the above 'create_period_portfolio_series' function

    group: dict,
        asset -> group mapping

    period: str

    level: str
        'asset', 'group', 'all'

    group_name: str
        specific group name for group analysis

    intial_total_value: float
        initial base asset

    Returns
    -------

    """
    if group_name is None:
        group_name = level
    df_big = get_field_level_data(result, group, period=period, field='profit', level=level, by_date=True, detail=False)
    df_long = get_field_level_data(result, group, period=period, field='long_profit', level=level, by_date=True,
                                   detail=False)
    df_short = get_field_level_data(result, group, period=period, field='short_profit', level=level, by_date=True,
                                    detail=False)
    # 对于行业，品种的拆分，需要从上面获取的数据中再提取特定的行业，品种
    df_big = df_big.xs(group_name, level=1, drop_level=False)
    df_long = df_long.xs(group_name, level=1, drop_level=False)
    df_short = df_short.xs(group_name, level=1, drop_level=False)
    # 整合数据，整合后是multiindex: data, group（或asset），三列数据，对应total,long,short
    df = pd.concat([df_big, df_long, df_short], axis=1)
    # asset情况下df_big,df_long,df_short 都会有一列group，在concat后将这三列group一起drop掉
    if level == 'asset':
        df = df.drop(columns='group')
    df = df.cumsum() + 1
    # 将第二层index舍弃
    df = df.reset_index(1, drop=True)
    ax = df.plot()
    ax.axhline(y=1, color='black', linestyle='--', lw=2, alpha=0.5)
    ax.set_title(group_name + ' Cumulative return long short decomposition')
    return ax


# %% metrics analysis based on cumulative_return
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
    # 按照factor_info中的操作，多出来的index的weighted_return会是nan，会被记入pariticipate num
    # 实际上没有参与，所以需要先dropna
    s = s.dropna()
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
    loss_avg = np.abs(s[s < 0].mean())
    if loss_avg == 0 and gain_avg > 0:
        rate = np.inf
    elif loss_avg == 0 and gain_avg == 0:
        rate = 0
    else:
        rate = gain_avg / loss_avg
    return rate


def create_turnover_info(factor_info: DataFrame) -> DataFrame:
    """
        基于 factor_info (单一 multi-period 序列) 往往是
            sub_factor_info = factor_info.loc[rebalanced_days，
            生成品种级的逐期 turnover (rate)
        【IMPORTANT】: 此处计算出的 turnover 是逐【期】逐品种的权重变动， 如果需要计算年化值，
        还需要综合考虑factor_info 对应的调仓period， 此处不显性含有 period 信息

            Note: turnover (rate) defined as:
                abs(weight-diff before and after the rebalancing)
                thus, the value will be influenced by leverage(usually set to 1 by
                letting weight.abs().sum() == 1 on rebalancing)

    Parameters
    ----------
    factor_info: DataFrame, multi-indexed by ('date', 'asset')，隐形含有 period 信息，此处统一假设输入的factor_info是【逐期排列】的
        columns:
            weights(rebalancing)
            period(forward_return period),
            weighted_return(weighted forward_return)
            (others)

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

    prev_weights = data['weights_wo_na'].unstack().fillna(0).shift(1).stack()

    prev_weights.name = 'prev_weights'

    prev_weighted_return = data['weighted_return_wo_na'].unstack().fillna(0).shift(1).stack()
    prev_weighted_return.name = 'prev_weighted_return'
    data = data.merge(prev_weights, how='outer', left_index=True, right_index=True)
    data = data.merge(prev_weighted_return, how='outer', left_index=True, right_index=True)
    data['portfolio_return'] = data['prev_weighted_return'].groupby('date').transform('sum')
    data['turnover'] = ((data['weights_wo_na']) - (data['prev_weights'] + data['prev_weighted_return']) / (
            1 + data['portfolio_return'])).abs()

    # drop NAN
    # data = data.loc[~data.turnover.isnull()]
    # data['turnover'] = data['turnover'].fillna(0)

    return data


def get_turnover_with_positions(factor_data, by_date: bool = False, positions=None, weights=None, period=None,
                                leverage_up=True):
    """
    基于 positions，使用与 get_period_metrics中获得 sub portfolio turnover 的类似逻辑计算整体 portfolio 的年化 turnover
    此处的 turnover 是以双边计算的，如果需要计算手续费，乘以的是 单边手续费率

    factor_data: DataFrame, multi-indexed, classical alphalen's factor_data with forward-returns

    by_date: bool, default as False
        whether to return daily_turnover, dafault will return annualized turnover ratio

    positions: default as None

    following 3 params are used for calculating positions only:

        weights: Series, multi-indexed
            prepared ahead, statisfing leverage==1 (usually)

        period: str
            rebalane days

        leverage_up: bool
        If true, leverage up to 1 when calculating positions

    """
    # 按照period计算positions，并转换成long form
    if positions is None:
        freq = factor_data.index.levels[0].freq
        positions = al.performance.positions(weights=weights, period=period, freq=freq, leverage_up=leverage_up)[0]
    positions = positions.stack()  # 后续展开过程中往往伴随fillna(0)
    # 使用与get_period_metrics中相同的办法计算turnover
    # 使用get_period_info得到各品种每日的weighted return，设定period='1D', 因为 positions 输出的就是每日rebalance的组合的权重
    factor_info = get_factor_info(factor_data=factor_data, weights=positions, period='1D')
    # 使用create_turnover_info 基于上述 factor_info 即逐日换仓的 portfolio
    # 得到每个品种上一天的weighted_return以及portfolio上一天的的return，并计算各品种每日换手率
    turnover_info = create_turnover_info(factor_info)
    # 每日各品种换手率相加，得到portfolio每日换手率
    turnover_ratio = turnover_info.groupby(['date'])['turnover'].sum()
    if by_date:
        return turnover_ratio
    else:
        # 获得年化换手率
        # 此处年化的方案，有 factor_info 的意义上的频率决定，比如此处 factor_info 是逐日换仓的组合，所以，直接mean 年化
        ann_turnover = turnover_ratio.mean() * 252
        return ann_turnover


def dependent_cum_ret_wo_fee(factor_data, weights, period, freq, commission_rate: float,leverage_up:True):
    """
    基于cumulative_returns_precisely_w_positions,get_turnover_with_positions，得到去除每日手续费后的daily return series
    以及cumulative returns series

    factor_data: DataFrame, multi-indexed, classical alphalen's factor_data with forward-returns

    weights: Series, multi-indexed
        prepared ahead, statisfing leverage==1 (usually)

    period: str
        rebalance days

    freq:
        freq of factor_data datetime index

    commission_rate: float
        单边手续费率

    """
    cumulative_ret, positions = al.performance.cumulative_returns_precisely_w_positions(returns=factor_data['1D'],
                                                                                        weights=weights,
                                                                                        period=period,
                                                                                        freq=freq,
                                                                                        leverage_up=leverage_up,
                                                                                        ret_pos=True)
    rets = cumulative_ret.pct_change().fillna(0)
    turnover = get_turnover_with_positions(factor_data=factor_data, by_date=True, positions=positions)
    rets = rets - turnover * commission_rate
    cum_ret = (rets + 1).cumprod()
    return cum_ret, rets


def independent_cum_ret_wo_fee(result, group, period, subs):
    """
    以create_period_portfolio_series的输出dict作为输入，获取independent方法下的每日return series和手续费。每日数值相减得到
    除去手续费后的daily return series
    result: dict
        output from create_period_portfolio_series

    group: dict
        asset -> group mapping

    period: str
        example: '5D'

    subs: int
        number of subportfolios to be created

    Returns
    ------
    ret_after_comm: Series
        daily return of the portfolio taken away commission

    """
    # 获得每日return和commission
    all_return = get_field_level_data(result, group, period=period, field='profit', level='all', subs=subs,
                                      by_date=True, detail=False)
    all_fee = get_field_level_data(result, group, period=period, field='commission', level='all', subs=subs,
                                   by_date=True, detail=False)
    # return - commission
    # 每个sub protfolio第一次建仓的手续费之前没有计算，此处用0填上
    merged = all_return.merge(all_fee, on=['date', 'group'], how='left').fillna(0)
    ret_wo_fee = merged['profit'] - merged['commission']
    ret_wo_fee = ret_wo_fee.reset_index('group', drop=True)
    cum_ret = (ret_wo_fee + 1).cumprod()

    return cum_ret, ret_wo_fee


# %% data prepare for pyfolio
def prepare_pyfolio_input(factor_data: DataFrame,
                          cumulative_ret: Series,
                          benchmark_period: str):
    """

        self made preparation function for pyfolio output

    """
    freq = factor_data.index.levels[0].freq
    returns = cumulative_ret.pct_change().fillna(0)
    benchmark_data = factor_data.copy()
    benchmark_data['factor'] = benchmark_data['factor'].abs()
    benchmark_rets = al.performance.factor_cumulative_returns(factor_data=benchmark_data,
                                                              period=benchmark_period,
                                                              long_short=False,
                                                              group_neutral=False,
                                                              equal_weight=True,
                                                              freq=freq,
                                                              precisely=True)
    benchmark_rets = benchmark_rets.pct_change().fillna(0)
    benchmark_rets.name = 'benchmark'
    return (returns, benchmark_rets)
