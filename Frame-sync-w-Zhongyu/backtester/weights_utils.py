#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/18 21:32:48
# @Author  : Michael_Liu @ QTG
# @File    : weights
# @Software: PyCharm

"""

    original ideas and basic design are based on 海通 FICC 研报三，
    其中， 单品种目标波动率，和 相应的组合目标波动率 方案，都引入了一些额外的
    推导和理解

    equal_weights_from_quantiles, equal_weights_from_signals:
        可选择从factor data, or signals data 得到对应的等权权重
        default demeaned=True，也就是多头内等权、空头内等权，同时杠杆=1，多空对称、各用资金50%
        此处 demeaned 的设计，最初是考虑到 时间序列因子 的等权设置

    volatility_inverse_normed_weights:
        【波动率倒数加权归一化】，着重在可以选择 demeaned = True or False
        If True，则会对多空分别处理，在多头内部进行 1/sigma 归一化，同时在空头内部 1/sigma 归一化，最后整体归一化使得杠杆=1（因为分别
        作了多空头的归一化，整体的归一化变得很天然），如此，多空完全对称，没有【净暴露】
        If False, 那么对【多空统一处理】，首先对于全部要交易的多空品种进行1/sigma归一化，保证组合杠杆=1 之后，再添加信号方向（符号），
        因为多空组合中各品种波动率分布的不对称性，因此头寸大概率【会出现净暴露】
        注： 理论上，demeaned=False 是完全 refer to 研报的方式

        实际中，无法确定 demeaned=False 是否一定比 demeaned=True，带来更高的年化收益和夏普，需要分别测试

    volatility_normed_weights:
        All the others same as volatility_inverse_normed_weights
        只是将波动率倒数加权归一化，改成了【波动率加权归一化】（没有取倒数）

    ATR_inverse_normed_weights"
        【ATR倒数加权归一化】，着重在可以选择 demeaned = True or False， 类似上述波动率倒数加权归一化的思维，只是
        用 close/ATR 代替了 sigma inverse
        If True，则会对多空分别处理，在多头内部进行 close/ATR 的归一化，同时在空头内部 close/ATR 归一化
        最后整体归一化使得杠杆=1，如此，多空完全对称，没有净暴露
        If False, 那么对多空统一处理，首先对于全部要交易的多空品种进行 close/ATR 归一化，保证组合杠杆=1
        之后，再添加信号方向（符号），因为多空两端的 close/ATR 不对称，因此对净头寸无约束，允许净暴露
        理论上，demeaned=False是完全 refer to 研报的方式

        实际中，无法确定demeaned=False是否一定比demeaned=True，带来更高的年化收益和夏普，需要分别测试

    ATR_normed_weights:
        All the others same as ATR_inverse_normed_weights
        只是将 close/ATR 归一化，改成了 【ATR/close 归一化】

    single_target_volatility_inverse_weights:
        【单品种目标波动率】，以对多空统一处理的方式，来强制净暴露=0
        此功能的实现是： 在一个约束多空对称的基础（波动率倒数加权【多空独立】归一化）权重的基础上，进行杠杆调节

        这里，我们暂时使用的基础权重的是波动率倒数加权【多空独立】
        adj_weights = vol_inverse_weights * daily adj factor
        daily adj factor = sigma target * (vol inverse mean within the whole long-short portfolio)

        此函数与single_target_volatility_inverse_w_exposure_weights的区别在于，前者无净暴露，后者有净暴露
        但是两者的实现方式都是：基于一个约束多空对称的基础权重（如波动率倒数加权），再加上一个动态adj factor
        这个adj factor，是波动率倒数的平均 * tgt vol，区别在于这个平均是在什么范围内做平均
        如果要约束净头寸，必须在whole ls portfolio内求平均
            保持原有多空组合的暴露对称性，两个组合同乘以 同一个数 （对应杠杆调节）
        如果是with exposure，那么就是对多头和空头分别求平均
            不保持原有组合的暴露对称性， 两个组合分别乘以 不同的数 （对杠杆调节），因而使得原有对称的多空组合，出现
            了扭曲，造成了净敞口

    single_target_volatility_inverse_w_exposure_weights:
        【单品种目标波动率】，允许净暴露，但是对多空分别处理
        同时，是在一个约束多空对称的基础权重的基础上，进行杠杆调节

        这里，我们暂时使用的基础权重的是波动率倒数加权
        adj_weights = vol_inverse_weights * daily adj factor
        For long assets, daily adj factor = sigma target * (vol inverse mean within long side)
        For short assets, daily adj factor = sigma target * (vol inverse mean withn short side)
        上述这个公式，实际上等价于一个：对于多头空头，分别处理 的 不约束净暴露 的单品种目标波动率策略

        【Note】：这个w_exposure_weights 和 simple_weights 两者之间【没有必然可拆解的关系】
        后者是多空统一处理，而此函数是在一个基础权重的基础上，多空分别处理。虽然都有净暴露和动态杠杆，但两者之间无推导关系。
        因此，两者孰优孰劣，没有general结论，需要case by case对每个因子分别测试

    single_target_volatility_simple_weights:
        【单品种目标波动率】，完全复制研报的实现，对多空统一处理
        首先，对于每一个多空组合中的品种，计算 sigma_target / N / sigma_i
        Here, N = N_long + N_short, number of traded assets
        之后，对于每一个品种添加信号方向（符号）
        由于多空两端的 1/sigma 分布不对称，所以对于净头寸无约束，允许净暴露浮动
        同时，有一个动态的杠杆调节

        【Note】：这个simple_weights 和 w_exposure_weights 两者之间没有必然可拆解的关系
        前者是多空统一处理，后者是在一个基础权重的基础上，多空分别处理。虽然都有净暴露和动态杠杆，但两者之间无推导关系。
        因此，两者孰优孰劣，没有general结论，需要case by case对每个因子分别测试

    target_volatility_inverse_weights：
        【策略目标波动率】，在基础权重上，以对多空统一求adj factor的方式，约束净暴露=0，动态杠杆
        = single_target_volatility_inverse_weights * correlation factor
        CF = np.sqrt( 1/ correlation mean of daily traded assets, including one and oneself)

    target_volatility_inverse_w_exposure_weights:
        【策略目标波动率】，在基础权重上，不约束净暴露，对于多空分别求daily adj factor
        此处 _w_exposure_weights 和 _weights 版本的区别，类似 单品种目标波动率中的情况
        For long, = single_target_volatility_inverse_w_exposure_weights * correlation factor long
        For short, = single_target_volatility_inverse_w_exposure_weights * correlation factor short
        CF long = np.sqrt( 1/ correlation mean of daily long assets)
        CF short = np.sqrt( 1/ correlation mean of daily short assets)

    target_volatility_simple_weights:
        【策略目标波动率】，完全复制研报的实现方式，对于多空统一处理，不约束净暴露，动态杠杆
        也不需要再做基础权重 * 调节系数的拆解，no need to calculate daily adj factor

        具体的，首先，对每个品种求： sigma target / N / sigma i * corr factor
        N = num of traded assets
        corr factor = np.sqrt( 1/ correlation mean of daily traded assets, including one and oneself)
        其次，在每个品种的weights上添加信号方向（符号）即可

        这三个函数之间的关系，和其他细节的实现，实际上all the same as single_target_volatility
        策略目标波动率仅仅是在单品种的基础上，多叠加一个调节系数correlation factor

        for 约束净暴露=0的target_volatility_inverse_weights, CF = np.sqrt( 1/ correlation mean of daily traded assets)
        for target_volatility_simple_weights, CF also = np.sqrt( 1/ correlation mean of daily traded assets)
        for target_volatility_inverse_w_exposure_weights, 多头和空头对应不同的CF，这是因为对多空分别处理
        CF long = np.sqrt( 1/ correlation mean of daily long assets)
        CF short = np.sqrt( 1/ correlation mean of daily short assets)

"""

import pandas as pd
import numpy as np
from pandas import (DataFrame, Series)
from typing import List
import qtgQuants.alphalens as al
import qtgQuants.pyfolio as pf
from . import signals_utils
from IPython.display import display

import seaborn as sns
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import talib as ta


def _to_weights(group, demeaned: bool = True):
    """
        默认 demeaned as True, 对 多头 空头 分别归一的（保证对空对称）的
        然后，再返回 根据 group 值归一的(abs().sum()==1)

        general to_weights, suitable for arbitrary signal, not only 1,0,-1 mask

    Parameters
    ----------
    group

    demeaned

    Returns
    -------

    """
    # 因为下述赋值要对 group 做 大小判断， 所以，此处需要 fillna 操作
    # group = group.fillna(0)
    # nan !=0是True, nan<0, nan>0, nan==0是False
    # 但需要保证group里面没有inf, -inf

    negative_mask = group < 0
    positive_mask = group > 0

    # 多空对称，各用资金50%
    if demeaned:
        if negative_mask.any():
            group[negative_mask] /= group[negative_mask].abs().sum()
        if positive_mask.any():
            group[positive_mask] /= group[positive_mask].abs().sum()
    # 约束杠杆=1
    return group / group.abs().sum()


def equal_weights_from_signals(signals_data: Series,
                               demeaned: bool = True):
    """
        set equal weights by sign of signal, default as long=short (demeaned)
        which means long and short normalized independently ahead of leveraging to 1

    Parameters
    ----------
    signals_data

    demeaned: long_short or not, default as True

    Returns
    -------

    """

    signals_data = np.sign(signals_data)
    weights_data = signals_data.groupby('date').apply(_to_weights, demeaned)
    return weights_data


def volatility_normed_weights(signals_data: Series,
                              prices,
                              window: int = 20,
                              demeaned: bool = True):
    """
        滚动（历史）波动率加权归一， 会对波动率为 0 的品种强行 剔除 （归 0）

    Parameters
    ----------
    signals_data

    prices: DataFrame, wide-form, indexed by 'date', columns by assets-symbols

    window: int, 滚动历史波动率窗口

    demeaned: bool, default True
        默认是多空对称的，如果不是，那么仅约束杠杆=1而保留净暴露

    Returns
    -------

    """
    # long form
    signals_data = np.sign(signals_data)

    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    # wide form
    volatility_series = prices.pct_change().rolling(window, min_periods=1).std()
    # 下述处理等价于强行剔除了波动率为 0 的品种
    volatility_series.replace(0, np.nan, inplace=True)

    # 强制净暴露=0，多空对称
    if demeaned == True:
        # 先标记每个品种的波动率倒数 with sign，再group by date，保证每天的多头空头对称，资金各50%，杠杆=1
        # stack to long form so also to group by date
        volatility_signed = (signals_data.unstack().multiply(volatility_series) ).stack()
        weights_data = volatility_signed.groupby('date').apply(_to_weights, demeaned)

    # 允许有净暴露，不约束多空对称
    else:
        # here, demeaned=False, 代表着只约束杠杆=1，不再约束净暴露
        volatility_wo_sign = (signals_data.abs().unstack().multiply(volatility_series)).stack()
        weights_wo_sign = volatility_wo_sign.groupby('date').apply(_to_weights, demeaned)
        # 因为是wo sign，所以需要加上之前的信号方向，此时生成的weights不再是多空对称的
        weights_data = weights_wo_sign.multiply(signals_data)

    weights_data.dropna(inplace=True)
    return weights_data


def volatility_inverse_normed_weights(signals_data: Series,
                                      prices: DataFrame,
                                      window: int = 20,
                                      demeaned: bool = True):
    """
        滚动（历史）波动率倒数加权归一， 会对波动率为 0 的品种强行 剔除 （归 0）

    Parameters
    ----------
    signals_data

    prices: DataFrame, wide-form, indexed by 'date', columns by assets-symbols

    window: int, 滚动历史波动率窗口

    demeaned: bool

    Returns
    -------

    """
    # long form
    signals_data = np.sign(signals_data)

    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    # wide form
    volatility_inverse_series = prices.pct_change().rolling(window, min_periods=1).std().rdiv(1)
    # NaN 的 rdiv 是 NaN, 0 的 rdiv 是 np.inf 符号由 rdiv决定， 都可以用 np.isinf 来判断
    # 下述处理等价于强行剔除了波动率为 0 的品种
    volatility_inverse_series.replace(np.inf, np.nan, inplace=True)

    # 强制净暴露=0，多空对称
    if demeaned == True:
        # 先标记每个品种的波动率倒数 with sign，再group by date，保证当日多头空头对称，资金各50%，杠杆=1
        volatility_inverse_signed = (signals_data.unstack().multiply(volatility_inverse_series) ).stack()
        weights_data = volatility_inverse_signed.groupby('date').apply(_to_weights, demeaned)

    # 允许有净暴露，不约束多空对称
    else:
        # here, demeaned=False, 代表着只约束杠杆=1，不再约束净暴露
        volatility_inverse_wo_sign = (signals_data.abs().unstack().multiply(volatility_inverse_series) ).stack()
        weights_wo_sign = volatility_inverse_wo_sign.groupby('date').apply(_to_weights, demeaned)
        # 因为是wo sign，所以需要加上之前的信号方向，此时生成的weights不再是多空对称的
        weights_data = weights_wo_sign.multiply(signals_data)

    weights_data.dropna(inplace=True)
    return weights_data


# 单品种目标波动率（强制净暴露==0）
def single_target_volatility_inverse_weights(signals_data: Series,
                                             prices: DataFrame,
                                             window: int = 20,
                                             target_volatility: float = 0.1,
                                             demeaned: bool = True):
    """
        单品种目标波动率，并强制多空对称，净头寸 !=1，会对波动率为 0 的品种强行 剔除 （归 0）
        具体实现算法为：对于波动率倒数加权归一化的权重，乘以一个调节系数

        调节系数 = 目标波动率（日） * 当日多空组合中的品种的波动率倒数的平均

    Parameters
    ----------
    signals_data

    prices: DataFrame, wide-form, indexed by 'date', columns by assets-symbols

    window: int, 滚动历史波动率窗口

    target_volatility: float，设定的单品种 年化波动率
        如果使用中需要转化到日波动率, 需要做调整：比如年化20% 对应 0.2*np.sqrt(1/252)

    demeaned: bool

    Returns
    -------

    """
    target_daily_vol = target_volatility * np.sqrt(1 / 252)
    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    # same as vol-inverse part
    signals_data = np.sign(signals_data)
    volatility_inverse_series = prices.pct_change().rolling(window, min_periods=1).std().rdiv(1)
    # NaN 的 rdiv 是 NaN, 0 的 rdiv 是 np.inf 符号由 rdiv决定， 都可以用 np.isinf 来判断
    # 下述处理等价于强行剔除了波动率为 0 的品种
    volatility_inverse_series.replace(np.inf, np.nan, inplace=True)

    # 波动率倒数加权（未归一）
    # long-form Series
    volatility_inverse_weighted = signals_data.unstack().multiply(volatility_inverse_series).stack()
    # 波动率倒数加权归一化的权重
    weights_data = volatility_inverse_weighted.groupby('date').apply(_to_weights, demeaned)

    # following for target-vol adjust
    # 0 means invalid here, stack() 默认是 dropna 的
    # 筛选条件决定了， universe 是多空组合中品种，且 波动率倒数有意义的
    universe_mask = volatility_inverse_weighted != 0
    # long-form mask 应该是稳定 work 的
    daily_univ_mean_vol_inverse = volatility_inverse_weighted[universe_mask].abs().groupby('date').mean()

    adj_weights = weights_data * daily_univ_mean_vol_inverse * target_daily_vol

    return adj_weights


def single_target_volatility_inverse_w_exposure_weights(signals_data: Series,
                                                        prices,
                                                        window: int = 20,
                                                        target_volatility: float = 0.1,
                                                        demeaned: bool = True):
    """
        单品种目标波动率，不约束多空暴露， 杠杆！=1，净头寸！=0，会对波动率为 0 的品种强行 剔除 （归 0）
        具体实现算法为：对于波动率倒数加权归一化的权重的多空部分，分别乘以一个调节系数
        调节系数 = 目标波动率（日） * 当日多头或该空头组合中的品种的波动率倒数的平均

    Parameters
    ----------
    signals_data

    prices: wide-form

    window: int, 滚动历史波动率窗口

    target_volatility: float，设定的单品种 年化波动率
        如果使用中需要转化到日波动率, 需要做调整：比如年化20% 对应 0.2*np.sqrt(1/252)

    demeaned: bool

    Returns
    -------

    """
    target_daily_vol = target_volatility * np.sqrt(1 / 252)
    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    signals_data = np.sign(signals_data)
    volatility_inverse_series = prices.pct_change().rolling(window, min_periods=1).std().rdiv(1)
    # NaN 的 rdiv 是 NaN, 0 的 rdiv 是 np.inf 符号由 rdiv决定， 都可以用 np.isinf 来判断
    # 下述处理等价于强行剔除了波动率为 0 的品种
    volatility_inverse_series.replace(np.inf, np.nan, inplace=True)

    # 波动率倒数加权（未归一）
    # long-form Series
    volatility_inverse_weighted = signals_data.unstack().multiply(volatility_inverse_series).stack()
    # 波动率倒数加权归一化的权重
    weights_data = volatility_inverse_weighted.groupby('date').apply(_to_weights, demeaned)

    # following for target-vol adjust
    # 0 means invalid here, stack() 默认是 dropna 的
    # 筛选条件决定了， long 是多头组合中品种，且 波动率倒数有意义的
    long_mask = volatility_inverse_weighted > 0
    short_mask = volatility_inverse_weighted < 0

    # long-form 的 mask机制 应该是稳定 work 的
    daily_univ_mean_vol_inverse_long = volatility_inverse_weighted[long_mask].abs().groupby('date').mean()
    daily_univ_mean_vol_inverse_short = volatility_inverse_weighted[short_mask].abs().groupby('date').mean()

    weights_data[long_mask] = weights_data[long_mask] * daily_univ_mean_vol_inverse_long * target_daily_vol
    weights_data[short_mask] = weights_data[short_mask] * daily_univ_mean_vol_inverse_short * target_daily_vol

    return weights_data


# 完全复制研报的简易版实现
# 单品种目标波动率简易版
def single_target_volatility_simple_weights(signals_data: Series,
                                            prices,
                                            window: int = 20,
                                            target_volatility: float = 0.1):
    """
        海通 FICC 三
        单品种目标波动率，不约束多空暴露， 杠杆！=1，净头寸！=0，会对波动率为 0 的品种强行 剔除 （归 0）
        w i = sigma(tgt) / (N1+N2) / sigma i

    Parameters
    ----------
    signals_data

    prices: wide-form

    window: int, 滚动历史波动率窗口

    target_volatility: float，设定的单品种 年化波动率
        如果使用中需要转化到日波动率, 需要做调整：比如年化20% 对应 0.2*np.sqrt(1/252)
    -------
    """
    target_daily_vol = target_volatility * np.sqrt(1 / 252)
    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    # long form
    signals_data = np.sign(signals_data)
    # wide form
    volatility_inverse_series = prices.pct_change().rolling(window, min_periods=1).std().rdiv(1)
    # NaN 的 rdiv 是 NaN, 0 的 rdiv 是 np.inf 符号由 rdiv决定， 都可以用 np.isinf 来判断
    # 下述处理等价于强行剔除了波动率为 0 的品种
    volatility_inverse_series.replace(np.inf, np.nan, inplace=True)

    # long-form Series, volatility with signs
    volatility_inverse_signed = signals_data.unstack().multiply(volatility_inverse_series).stack()

    # 计算当日要交易的品种数量, count 会跳过nan
    daily_portfolio_num = volatility_inverse_signed[volatility_inverse_signed != 0].groupby('date').count()
    daily_adj = target_daily_vol / daily_portfolio_num

    weights_data = volatility_inverse_signed.multiply(daily_adj)

    return weights_data

# 完全复制研报的简易版实现
# 策略目标波动率简易版
def target_volatility_simple_weights(signals_data: Series,
                                     prices,
                                     window: int = 20,
                                     target_volatility: float = 0.1):
    """
        策略目标波动率，不约束多空暴露， 杠杆！=1，净头寸！=0，会对波动率为 0 的品种强行 剔除 （归 0）
        w i = sigma(tgt) / (N1+N2) / sigma i * CF

    Parameters
    ----------
    signals_data

    prices: wide-form

    window: int, 滚动历史波动率窗口

    target_volatility: float，设定的单品种 年化波动率
        如果使用中需要转化到日波动率, 需要做调整：比如年化20% 对应 0.2*np.sqrt(1/252)
    -------
    """
    target_daily_vol = target_volatility * np.sqrt(1 / 252)

    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    # pairwise correlation, using past 60 days close prices
    all_corr = prices.pct_change().rolling(60, min_periods=1).corr()  # 这个60天，暂时固定不动

    # long form
    signals_data = np.sign(signals_data)
    # wide form
    volatility_inverse_series = prices.pct_change().rolling(window, min_periods=1).std().rdiv(1)
    # 下述处理等价于强行剔除了波动率为 0 的品种
    volatility_inverse_series.replace(np.inf, np.nan, inplace=True)
    # long-form multi-index Series, volatility with signs
    volatility_inverse_signed = signals_data.unstack().multiply(volatility_inverse_series).stack()

    def cal_daily_adj(group, all_corr, target_daily_vol):
        # group = daily vol inverse with signs
        num = group[group != 0].count() # will skip na
        date = group.index[0][0]
        # traded asset
        asset = group[group != 0].index.get_level_values(1)

        # Multiindex slicing, .loc[(level0, level1), level0]
        # daily_related_asset = all_corr.loc[(date, asset), asset].copy()
        correlation_factor = np.sqrt(1 / all_corr.loc[(date, asset), asset].mean().mean())

        # daily adj = daily tgt vol / num * corr factor = scalar
        adj_weights = group * target_daily_vol * correlation_factor / num
        return adj_weights

    # for each date, cal daily adjusted weights
    adj_weights = volatility_inverse_signed.groupby('date').apply(cal_daily_adj, all_corr, target_daily_vol)
    return adj_weights


def target_volatility_inverse_weights(signals_data,
                                      prices,
                                      window=20,
                                      target_volatility=0.2,
                                      demeaned=True):
    """
        策略目标波动率，并强制多空对称，净头寸 ==0，同时杠杆不再等于1，会对波动率为 0 的品种强行 剔除 （归 0）
        具体实现算法为：对于波动率倒数加权归一化的权重，乘以一个调节系数

        调节系数 = 目标波动率（日） * 当日多空组合中的品种的波动率倒数的平均 * 相关性因子
        相关性因子的计算公式为：np.sqrt( N / (1+ (N-1)* corr_mean) )

    Parameters
    ----------
    signals_data

    prices: DataFrame, wide-form, indexed by 'date', columns by assets-symbols

    window: int, 滚动历史波动率窗口

    target_volatility: float，设定的单品种 年化波动率
        如果使用中需要转化到日波动率, 需要做调整：比如年化20% 对应 0.2*np.sqrt(1/252)

    demeaned: bool

    Returns
    -------

    """

    target_daily_vol = target_volatility * np.sqrt(1 / 252)
    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    signals_data = np.sign(signals_data)

    # vol inverse, wide-form
    volatility_inverse_series = prices.pct_change().rolling(window, min_periods=1).std().rdiv(1)
    volatility_inverse_series.replace(np.inf, np.nan, inplace=True)
    # long-form Series, with signals
    volatility_inverse_weighted = signals_data.unstack().multiply(volatility_inverse_series).stack()
    # 波动率倒数加权归一化的权重
    weights_data = volatility_inverse_weighted.groupby('date').apply(_to_weights, demeaned)

    # pairwise correlation, using past 60 days close prices
    all_corr = prices.pct_change().rolling(60, min_periods=1).corr()  # 这个60天，暂时固定不动

    def cal_daily_adjusted_weights(group, volatility_inverse_weighted, all_corr, target_daily_vol):
        # calculate portfolio's commodity number, corresponding asset and date
        # num = group[group != 0].shape[0]
        date = group.index[0][0]
        asset = group[group != 0].index.get_level_values(1)

        # volatility inverse mean within portfolio
        daily_vol_inverse_signed = volatility_inverse_weighted.loc[date]
        daily_univ_mean_vol_inverse = daily_vol_inverse_signed[
            daily_vol_inverse_signed != 0].abs().mean()  # weights data has sign

        # Multiindex slicing, .loc[(level0, level1), level0]
        # daily_related_asset = all_corr.loc[(date, asset), asset].copy()
        correlation_factor = np.sqrt(1 / all_corr.loc[(date, asset), asset].mean().mean())

        # 非对角元素求mean
        # rho_mean = daily_related_asset.mean().mean()    # 这个方法会引入对角元素， 即使令对角元素为0
        # 修正为去掉对角元素的求 mean
        # rho_mean = (daily_related_asset.sum().sum() - np.diag(daily_related_asset).sum()) / num / (num - 1)
        # correlation_factor = np.sqrt(num / (1 + (num - 1) * rho_mean))

        adj_weights = group * daily_univ_mean_vol_inverse * target_daily_vol * correlation_factor
        return adj_weights

    # for each date, cal daily adjusted weights
    adj_weights = weights_data.groupby('date').apply(cal_daily_adjusted_weights, volatility_inverse_weighted, all_corr,
                                                     target_daily_vol)

    return adj_weights


def target_volatility_inverse_w_exposure_weights(signals_data,
                                                 prices,
                                                 window=20,
                                                 target_volatility=0.2,
                                                 demeaned=True):
    """
            策略目标波动率，不约束多空暴露， 杠杆！=1，净头寸！=0，会对波动率为 0 的品种强行 剔除 （归 0）
            具体实现算法为：对于波动率倒数加权归一化的权重的多空部分，分别乘以一个调节系数
            调节系数 = 目标波动率（日） * 当日多头或该空头组合中的品种的波动率倒数的平均 * 多头或者空头组合的相关性因子
            相关性因子中的N = 多头或者空头组合的商品数量，rho_mean = 多头或者空头组合的两两相关性的平均

        Parameters
        ----------
        signals_data

        prices: wide-form

        window: int, 滚动历史波动率窗口

        target_volatility: float，设定的单品种 年化波动率
            如果使用中需要转化到日波动率, 需要做调整：比如年化20% 对应 0.2*np.sqrt(1/252)

        demeaned: bool

        Returns
        -------

        """
    target_daily_vol = target_volatility * np.sqrt(1 / 252)
    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    # same as vol-inverse part
    signals_data = np.sign(signals_data)
    # vol inverse, wide-form
    volatility_inverse_series = prices.pct_change().rolling(window, min_periods=1).std().rdiv(1)
    volatility_inverse_series.replace(np.inf, np.nan, inplace=True)
    # long-form Series, with signals
    volatility_inverse_weighted = signals_data.unstack().multiply(volatility_inverse_series).stack()
    weights_data = volatility_inverse_weighted.groupby('date').apply(_to_weights, demeaned)

    # pairwise correlation, past 60 days close prices
    all_corr = prices.pct_change().rolling(60, min_periods=1).corr()

    # TODO: 考虑在三处使用这个机制，能否将三处用法合并，获得简便的写法
    def cal_daily_adjusted_weights(group, volatility_inverse_weighted, all_corr, target_daily_vol):
        date = group.index[0][0]
        daily_vol_inverse_signed = volatility_inverse_weighted.loc[date]

        # Long side
        asset_posi = group[group > 0].index.get_level_values(1)
        # If time series momentum, this could be nan since no assets in posi group
        correlation_factor_posi = np.sqrt(1 / all_corr.loc[(date, asset_posi), asset_posi].mean().mean())
        daily_mean_vol_inverse_posi = daily_vol_inverse_signed[daily_vol_inverse_signed > 0].abs().mean()

        group[group > 0] *= daily_mean_vol_inverse_posi * target_daily_vol * correlation_factor_posi

        # Short side
        asset_nega = group[group < 0].index.get_level_values(1)
        # also this could be nan if time series momentum
        correlation_factor_nega = np.sqrt(1 / all_corr.loc[(date, asset_nega), asset_nega].mean().mean())
        daily_mean_vol_inverse_nega = daily_vol_inverse_signed[
            daily_vol_inverse_signed < 0].abs().mean()  # weights data has sign

        group[group < 0] *= daily_mean_vol_inverse_nega * target_daily_vol * correlation_factor_nega
        return group

    adj_weights = weights_data.groupby('date').apply(cal_daily_adjusted_weights, volatility_inverse_weighted, all_corr,
                                                     target_daily_vol)

    return adj_weights


def _get_atr(group, window):
    atr_series = ta.ATR(group.high.values, group.low.values, group.close.values, timeperiod=window)
    # 因为 ta 返回的是一个 numpy array， 所以，要重新索引
    atr_series = pd.Series(atr_series, index=group.index, name='ATR')
    return atr_series


def ATR_normed_weights(signals_data: Series,
                       prices,
                       bar_data: DataFrame,
                       window: int = 20,
                       demeaned: bool = True):
    """

    Parameters
    ----------
    signals_data

    prices

    bar_data: columns = ['date', 'asset', 'preclose', 'high', 'low', 'close'], # 我们需要使用的是high low and preclose
    thus, bar_data created from ContDominantDataManager.get_cont_dominant_adj_bar
    will need to rename the index (from ['datetime', 'underlying_symbol',....])

    window

    demeaned

    Returns
    -------
    """
    signals_data = np.sign(signals_data)
    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    bar_data = bar_data.copy()
    bar_data.set_index(['datetime', 'underlying_symbol'], inplace=True)
    bar_data.index.names = ['date', 'asset']
    # 生成的Multiindex是asset, date, asset, 因此需要剔除多余的第一个index: asset
    bar_data['ATR'] = bar_data.groupby('asset').apply(_get_atr, window).droplevel(0)

    # 信号 * 当天的ATR / 当天的close
    # changed to wide form to divide with prices
    ATR_adj = bar_data['ATR'].unstack().multiply(prices.rdiv(1)).stack()
    # 去除 prices 为 0 数据导致的问题
    ATR_adj.replace(np.inf, np.nan, inplace=True)

    # 强制净暴露=0，多空对称
    if demeaned == True:
        # 先标记每个品种的ATR倒数 with sign，再group by date，保证多头空头对称，资金各50%，杠杆=1
        ATR_signed = (signals_data.unstack().multiply(ATR_adj.unstack()) ).stack()
        weights_data = ATR_signed.groupby('date').apply(_to_weights, demeaned)

    # 允许有净暴露，不约束多空对称
    else:
        # here, demeaned=False, 代表着只约束杠杆=1，不再约束净暴露
        ATR_wo_sign = (signals_data.abs().unstack().multiply(ATR_adj.unstack()) ).stack()
        weights_wo_sign = ATR_wo_sign.groupby('date').apply(_to_weights, demeaned)
        # 因为是wo sign，所以需要加上之前的信号方向，此时生成的weights不再是多空对称的
        weights_data = weights_wo_sign.multiply(signals_data)

    weights_data.dropna(inplace=True)
    return weights_data


def ATR_inverse_normed_weights(signals_data: Series,
                               prices,
                               bar_data: DataFrame,
                               window: int = 20,
                               demeaned: bool = True):
    """

    Parameters
    ----------
    signals_data

    prices

    bar_data: columns = ['date', 'asset', 'open', 'high', 'low', 'close'],
    thus, bar_data created from ContDominantDataManager.get_cont_dominant_adj_bar
    will need to rename the index (from ['datetime', 'underlying_symbol',....])

    window

    demeaned

    Returns
    -------
    """
    # long form
    signals_data = np.sign(signals_data)
    # change names to date and asset
    prices = prices.copy()
    prices.index.name = 'date'
    prices.columns.name = 'asset'

    bar_data = bar_data.copy()
    bar_data.set_index(['datetime', 'underlying_symbol'], inplace=True)
    bar_data.index.names = ['date', 'asset']
    # multi-indexed
    # 生成的Multiindex是asset, date, asset, 因此需要剔除多余的第一个index: asset
    bar_data['ATR'] = bar_data.groupby('asset').apply(_get_atr, window).droplevel(0)

    # 下述处理等价于强行剔除了 ATR为 0 的品种
    bar_data.ATR.replace(0, np.nan, inplace=True)
    bar_data['ATR_inverse'] = bar_data['ATR'].rdiv(1)

    # 强制净暴露=0，多空对称
    if demeaned == True:
        # 先标记每个品种的ATR倒数 with sign，再group by date，保证多头空头对称，资金各50%，杠杆=1
        # 信号 * 当天close * 当天的ATR的倒数
        ATR_inverse_signed = signals_data.multiply(bar_data['ATR_inverse']).unstack().multiply(prices).stack()
        weights_data = ATR_inverse_signed.groupby('date').apply(_to_weights, demeaned)

    # 允许有净暴露，不约束多空对称
    else:
        # here, demeaned=False, 代表着只约束杠杆=1，不再约束净暴露
        # convert to long form
        ATR_inverse_wo_sign = (signals_data.abs().multiply(bar_data['ATR_inverse'])).unstack().multiply(prices).stack()
        weights_wo_sign = ATR_inverse_wo_sign.groupby('date').apply(_to_weights, demeaned)
        # 因为是wo sign，所以需要加上之前的信号方向，此时生成的weights不再是多空对称的
        weights_data = weights_wo_sign.multiply(signals_data)

    weights_data.dropna(inplace=True)
    return weights_data


def equal_weights_from_quantiles(factor_data: DataFrame,
                                 long_quantiles: List[int] = [1],
                                 short_quantiles: List[int] = [5],
                                 demeaned: bool = True,
                                 style: str = 'equal'):
    """
        by default: equal weights for long and short quantiles-list

    Parameters
    ----------
    factor_data
    long_quantiles
    short_quantiles

    demeaned: bool, default as True
        long/short or not

    Returns
    -------

    """
    signals_data = signals_utils.set_signal_by_quantiles(factor_data=factor_data,
                                                         long_quantiles=long_quantiles,
                                                         short_quantiles=short_quantiles)
    return equal_weights_from_signals(signals_data=signals_data, demeaned=demeaned)


def get_yearly_weights(weights: Series, group: dict, level: str):
    """

    Parameters
    ----------
    weights: Series, multi-indexed

    group: dict,
        asset -> group mapping

    level: str
        'asset', 'group'
        no 'all' here, because it is constant by strategy,
        for example: long-short strategy
            abs_weights sum up to 1 as leverage
            weights sum up to 0 as long-short

    Returns
    -------

        DataFrame with yearly and all-year statistics
    """
    weights = weights.unstack().fillna(0).stack().to_frame('weights')

    # group_df = pd.DataFrame.from_dict(group, orient='index')
    # group_df = group_df.rename(columns={0: 'group'})
    # group_df.index.name = 'asset'
    # weights = weights.join(group_df, how='left', on='asset')

    weights['group'] = weights.reset_index()['asset'].map(group).values

    weights['abs_weights'] = weights['weights'].abs()
    # level = group的话将每日的每个行业中的品种weight、abs_weight相加。level= asset时等于没有操作
    group_weights = weights.groupby(['date', level])[['weights', 'abs_weights']].sum()
    # 获取年份，按年份取平均值
    group_weights['year'] = group_weights.index.get_level_values(0).year
    group_weights_mean = group_weights.groupby(['year', level])[['weights', 'abs_weights']].mean()
    # 计算每年abs_weight的std
    group_weights_mean['abs_weights_std'] = group_weights.groupby(['year', level])['abs_weights'].std()
    # 计算全时间的weight、abs_weight均值、abs_weight的std
    all_weights_mean = group_weights.groupby(level)[['weights', 'abs_weights']].mean()
    all_weights_mean['abs_weights_std'] = group_weights.groupby(level)['abs_weights'].std()
    all_weights_mean['year'] = 'all_year'
    all_weights_mean = all_weights_mean.reset_index().set_index(['year', level])
    result = pd.concat([group_weights_mean, all_weights_mean])
    result.rename(columns={'weights': 'weights_mean', 'abs_weigths': 'abs_weights_mean'})
    display(result)
    return result


def weights_plot(weights: Series, group: dict, group_name: str):
    """
        weights/exposure plot
        abs_weights(range from 0 to 1) vs weights(net of long-short, maximum 0.5 for long-short)

    Parameters
    ----------
    weights: Series, multi-indexed

    group: dict,
        asset -> group mapping

    group_name: str

    Returns
    -------

    """
    weights = weights.unstack().fillna(0).stack().to_frame('weights')
    abs_weights = weights.abs()

    # group_df = pd.DataFrame.from_dict(group, orient='index')
    # group_df = group_df.rename(columns={0: 'group'})
    # group_df.index.name = 'asset'
    # weights = weights.join(group_df, how='left', on='asset')
    # abs_weights = abs_weights.join(group_df, how='left', on='asset')

    weights['group'] = weights.index.get_level_values(1).map(group).values
    abs_weights['group'] = weights['group']

    group_weights = weights.groupby(['date', 'group']).sum().unstack()
    group_weights.columns = group_weights.columns.droplevel()
    abs_group_weights = abs_weights.groupby(['date', 'group']).sum().unstack()
    abs_group_weights.columns = abs_group_weights.columns.droplevel()
    plt.rcParams["figure.figsize"] = (20, 12)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    group_weights.plot(title='Industry weights plot')
    # bs_group_weights.plot(title='Industry abs weights plot')
    abs_group_weights.plot.area(title='Industry abs weights area plot')
    return abs_group_weights

def equal_weight_from_stock_index(factor_data,index_const,industry_df):
    """
    To generate weight that simulates stock index
    Equal weight for each assset in a industry

    index_const: pd.DataFrame
        columns: date, code, name, weight, industry
    industry_df: pd.DataFrame
        columns: code, industry
    Return:
        pd.Series, MultiIndex[date,asset]
            series of weight
            same format as al.performance.factor_weights()
    """
    top_quantile = factor_data.factor_quantile.unique().max()
    factor_data = factor_data.reset_index()
    a = index_const.merge(industry_df,on=['code'],how='left')
    a=a[['date','code','weight','industry_y']].dropna().reset_index(drop=True)
    a = a.rename(columns={'industry_y':'industry'})
    total_w = a.groupby('date')['weight'].sum().to_frame('tot_weight').reset_index()
    a = a.merge(total_w,on='date',how='left')
    a['weight'] = a['weight']/a.tot_weight
    index_indus_weight = a.groupby(['date','industry'])['weight'].sum().to_frame()
    index_indus_weight = index_indus_weight.reset_index()
    index_indus_weight = index_indus_weight.rename(columns = {'industry':'group'})
    top10 = factor_data[factor_data.factor_quantile==top_quantile].reset_index(drop=True)
    top10_count = top10.groupby(['date','group']).asset.count().to_frame('number').reset_index()
    top10_count = top10_count.merge(index_indus_weight,how='left',on=['date','group'])
    top10_count['asset_weight'] = top10_count.weight/top10_count.number
    top10_count = top10_count[['date','group','asset_weight']]
    top10_count['factor_quantile']=top_quantile
    weight = factor_data.merge(top10_count,on=['date','group','factor_quantile'],how='left').set_index(['date','asset'])
    weights_value = weight.asset_weight.fillna(0)
    weights_value.name=None
    return weights_value

