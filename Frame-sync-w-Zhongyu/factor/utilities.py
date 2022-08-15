#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 14:38:15
# @Author  : Michael_Liu @ QTG
# @File    : tools
# @Software: PyCharm

import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Union
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import logging
import warnings

warnings.filterwarnings('ignore')


def get_other_contract_except_main(daily_data: DataFrame, dominant: DataFrame = None, allow_delist_month=False,
                                   filter_col: str = None, others_type='near') -> DataFrame:
    """
        以volume为例, 获取单品种除主力合约以外的有交易的近月合约(非进入交割月) 和 主力合约

    Parameters
    ----------
    daily_data: DataFrame,
        单品种日线行情数据

    dominant: DataFrame: indexed by 'datetime', columns=['dominant_contract']
              set dafault to be None for general purpose (近月合约)

        if None, thus, the dominant filtering step will be skip
        在 generalization 的过程中， dominant 被代替为任意一合约序列（DataFrame）
        See Also: 'get_near_far_contract()'

    allow_delist_month: bool
        允不允许交割月列入

    filter_col: str
        volume/open_interest
        如果 others_type == 'near' or 'second', filter_col 用来做过滤
        如果 others_type == 'second', filter_col 不仅用来做过滤，还用来定义次主力（按volume， or， open_interest）

    others_type: str
        除主力合约外的另一合约的生成方式
        "near": 最近月
        "second": 次主力
        "far": 最远月

    Returns
    -------
        单品种每日 X 合约，注意是shift前的 X 合约，
            X is determined by others_type
            # DataFrame，index by datetime，columns=['contract']

    """

    fields = ['datetime', 'contract']

    if others_type == 'second' and filter_col is None:
        raise ValueError(filter_col)

    if filter_col is not None:
        fields.append(filter_col)

    daily_data_df = daily_data[fields].set_index('datetime')

    if dominant is not None:
        df = pd.merge(left=dominant, right=daily_data_df, on='datetime')
        # 寻找另一个合约，按照需求，可以是除主力合约外最早交割、且交易量不为0的合约，也可以是主力合约以外持仓量最大的合约
        df = df[df['contract'] != df['dominant_contract']]
        tradingday_index = dominant.index
    else:
        df = daily_data_df
        tradingday_index = df.index.unique()

    df['delist_date'] = pd.to_datetime('20' + df['contract'].str[-4:] + '01')

    # 如果不考虑进入交割月的合约，则剔除日期大于交割月月初日期的合约数据
    if allow_delist_month == False:
        df = df[df['delist_date'] > df.index]

    # 根据 filter_col 去掉 filter_col 不满足的：
    if filter_col is not None:
        df = df[df[filter_col] > 0]

    if others_type == 'near':
        other_df = df.groupby('datetime').first()
    elif others_type == 'far':
        other_df = df.groupby('datetime').last()
    else:
        # 次主力
        other_df = df.sort_values(filter_col).groupby('datetime').last()

    other_df = other_df[['contract']]

    other_df = other_df.reindex(tradingday_index)

    return other_df


def get_near_far_contract(daily_data: DataFrame, symbol: str, dominant_df: Union[str, DataFrame],
                          allow_delist_month=False,
                          filter_col: str = None,
                          others_type='near'):
    """
        选取 dominant 和 others 方法生成的合约， 构成 近、远 月合约，返回 dataFrame
        在 generalization 的过程中， dominant 被代替为任意一合约序列（DataFrame），甚至可以是 str， 用来主动生成序列

    Parameters
    ----------
    daily_data:
        单品种日线行情数据

    dominant_df: str or DataFrame(index by datetime，columns=['contract'] 主力合约序列， 后扩展为
        在 pair 中，首先确定的序列 for specific symbol)
        if is str, thus, the str defined the ‘dominant’ DataFrame creating method
        See Also: others_type in get_other_contract_except_main()
        类似： cddm.get_cont_dominant_data(contract='main', price='close', rebalance_days=1, style='k') 获得主力合约序列

    allow_delist_month: bool
        允不允许交割月列入

    filter_col: str
        volume/open_interest

    others_type: str
        除主力合约外的另一合约的生成方式
        "near": 最近月
        "second": 次主力
        "far": 最远月

    Returns
    -------
        Tuple(DataFrame, DataFrame, boolean Series, boolean Series)
            as near, far, dominant_invalid_mask, other_invalid_mask
            and all indexed by 'datetime':
            near/far_contract: DataFrame,  index by datetime，columns=['contract']
            dominant_invalid_mask/other_invalid_mask: boolean Series indexed by datetime
            where the index is full-index inheriting from dominant_df DataFrame

    """

    if isinstance(dominant_df, str):
        dominant = get_other_contract_except_main(daily_data=daily_data,
                                                  dominant=None,
                                                  allow_delist_month=allow_delist_month,
                                                  filter_col=filter_col,
                                                  others_type=dominant_df)
    else:
        dominant = dominant_df[['contract']]
    dominant = dominant.rename(columns={'contract': 'dominant_contract'})

    # check 1:
    # 主力合约构建导致的 NA 错误
    dominant_invalid_mask = dominant['dominant_contract'].isnull()
    if dominant_invalid_mask.any():
        logging.warning(u'\n Check 1 - {} of main_contract cannot be decided by symbol- : {}'.format(
            dominant_invalid_mask.sum(), symbol))
        # dominant.fillna(method='ffill', inplace=True)             # let the item to be np.nan explicitly

    other_df = get_other_contract_except_main(daily_data=daily_data,
                                              dominant=dominant,
                                              allow_delist_month=allow_delist_month,
                                              filter_col=filter_col,
                                              others_type=others_type)

    # check 2:
    # 筛选出的 other 合约无法生成导致的 NA
    other_invalid_mask = other_df['contract'].isnull()
    if other_invalid_mask.any():
        logging.warning(u'\n Check 2 - {} of other_contract cannot be decided by symbol- : {}'.format(
            other_invalid_mask.sum(), symbol))
        # other_df.fillna(method='ffill', inplace=True)             # let the item to be np.nan explicitly

    # 根据合约的名称重拍名称, reordering
    near_contract_mask = dominant['dominant_contract'] < other_df['contract']
    near_contract = other_df[['contract']].copy()
    near_contract.loc[near_contract_mask, 'contract'] = dominant.loc[near_contract_mask, 'dominant_contract'].values
    far_contract = other_df[['contract']].copy()
    far_contract.loc[~near_contract_mask, 'contract'] = dominant.loc[~near_contract_mask, 'dominant_contract'].values

    return (near_contract, far_contract, dominant_invalid_mask, other_invalid_mask)


def get_absolue_month(contract: str) -> int:
    """
        get absolute month since 2000
    Parameters
    ----------
    contract: str
        standard commodity contract name

    Returns
    -------
        abs_month: int
    """
    if isinstance(contract, str):
        dt = pd.to_datetime('20' + contract[-4:] + '01')
        abs_month = (dt.year - 2000) * 12 + dt.month
    else:
        # contract 信息缺失
        abs_month = np.nan

    return abs_month


def get_carry_factor_and_tvalue(daily_data, maturity_date_df, symbol, price, allow_delist_month, filterby):
    """
        辅助完成斜率形式的期限结构因子的计算， 对给定 symbol，构建 因子值（斜率）序列 和 t-value 序列

    maturity_date_df: DataFrame, naive-indexed, columns =['contract', 'maturity_date']

    daily_data： DataFrame, naive-indexed

    symbol: str, 品种名称

    price: str, 用来计算 斜率的基准、参考价格

    allow_delist_month: bool， 是否允许交割进入交割月的合约进入评估/计算

    filterby: str, 'volume'/'open_interest' , 去除volume或open_interest为0的合约。之后再按照filterby排序

    Returns
    -------
        Tuple of series ('factor', 't_value'), indexed by 'datetime'

    """
    # 按filterby剔除不活跃的合约
    # 假设 daily_data[filterby] 没有 NaN， 缺失已经填0.0
    # 去掉 price invalid 的 data
    df = daily_data[(daily_data[filterby] != 0) & (daily_data[price] != 0)]
    df = df.merge(maturity_date_df, on='contract', how='left')
    # 按要求剔除进入交割月的合约
    if ~allow_delist_month:
        df['delist_date'] = pd.to_datetime('20' + df['contract'].str[-4:] + '01')
        df = df[df['delist_date'] > df['datetime']]
    # 计算合约距离到期日的日期，作为自变量
    df['to_delist'] = (df.maturity_date - df['datetime']).dt.days
    # 取出每日交易量/交易金额最高的4个合约
    df = df.sort_values(by=filterby, ascending=False).reset_index(drop=True)
    top4 = df.groupby('datetime').head(4)
    # 计算每日取出的合约数量
    contract_count = top4.groupby('datetime')['contract'].count()
    if (contract_count <= 1).any():
        logging.warning(u'\n Check 1 - {} days have 1 or less active contract for symbol- : {}'.format(
            (contract_count <= 1).sum(), symbol))
    if (contract_count == 2).any():
        logging.warning(u'\n Check 2 - {} days have only 2 active contract for symbol- : {}'.format(
            (contract_count == 2).sum(), symbol))
    # 取出回归需要的数据
    # 只排序了 datetime, 同一 datetime 内， 合约并没有排序
    top4 = top4[['datetime', price, 'to_delist']].set_index('datetime').sort_index()
    # 踢出只有1个合约的日期
    drop_dates = contract_count[contract_count <= 1].index
    ols_df = top4.drop(index=drop_dates)
    # 取收盘价的ln
    ols_df[price] = np.log(ols_df[price])

    # 每日OLS回归，记录当天的tvalue, 和斜率
    def ols_reg(df):
        x = df['to_delist']
        y = df[price]
        x = add_constant(x)
        reg_fit = OLS(y, x).fit()
        # 斜率相当于(ln(P_far)-ln(P_near)/(T_far-T_near),与之前计算的因子差一个负号
        return pd.Series(data=[reg_fit.tvalues[1], reg_fit.params[1] * (-1)], index=['t_value', 'factor'])

    result = ols_df.groupby('datetime').apply(ols_reg)
    return result.factor, result.t_value


def get_interval_ret(contract_df, price, window, daily_data):
    """
        get interval return of the contract for each datetime
        where interval return is defined by 'window' trading days return of 'price' ahead
        i.e. price_datetime / price_datetime-window

    Parameters
    ----------
    contract_df: DataFrame, indexed by 'datetime', columns by 'contract'

    price: str

    window: days ahead for interval-ret calculation

    daily_data: dafault price-source, for original-pair-interval calculation

    Returns
    -------
        DataFrame: indexed by 'datetime', columns=['contract','interval_ret']

    """
    contract_df['end_date'] = contract_df.index
    contract_df['start_date'] = contract_df['end_date'].shift(window)

    # for default pair-interval scheme
    contract_df = contract_df.merge(daily_data[['datetime', 'contract', price]], on=['datetime', 'contract'],
                                    how='left').rename(columns={price: 'end_price'})
    contract_df = contract_df.merge(
        daily_data[['datetime', 'contract', price]].rename(columns={'datetime': 'start_date'}),
        on=['start_date', 'contract'], how='left').rename(columns={price: 'start_price'})

    contract_df['interval_ret'] = contract_df['end_price'] / contract_df['start_price'] - 1
    return contract_df[['datetime', 'contract', 'interval_ret']].set_index('datetime')


def get_cont_contract_series_adjusted(daily_data, contract_df, price):
    """
        Construct continuous price from Dataframe of contracts of an asset.
        Use k method and 1 day rebalance

        [Notice]: 因为输入的 contract_df 会有合约的缺失，因此，
            1. 引入 contract_invalid_mask， 方便后续应用层处理
            2. 为了使得连续合约的拼接不引入额外的价格跳变
                1. 参见 下面 step-1 的 ffill
                2. 参加 下面 step-2 的 ffill
            3. 这种设计，在最后的价格序列中不再有 Nan， 因此， pct_change 不再会有 nan
            4. 由于 contract_invalid_mask 标记的日期的价格填入的随意性，导致，相应的日期的 daily_ret 需要剔除

    Parameters
    ----------
        contract_df： DataFrame
            output of get_near_far_contract. Can use either of near_contract or far_contract
            datetime indexed with a column of contract

        price: str
            type of price to be used. e.g. 'close'

        daily_data, DataFrame(naive-indexed)
            output of daily_data_manager.get_symbol(symbol=symbol)

    Returns
    -------
        Tuple of (DataFrame, boolean Series)
            contract_df, DataFrame: indexed by 'datetime', columns=['contract', price]
            contract_invalid_mask, boolen Series: indexed by 'datetime'
    """
    contract_df = contract_df.copy()
    # Record dates where contract is nan, for overwriting daily return with nan later
    contract_invalid_mask = contract_df['contract'].isnull()

    price_next = price + '_next'
    # Get the contract for next day
    contract_df['next_contract'] = contract_df.contract.shift(-1)
    # Get the prices for contract and contract_next
    contract_df = contract_df.merge(daily_data[['contract', 'datetime', price]], how='left',
                                    on=['datetime', 'contract'])
    # Ffill nans in price (close) of current contract -- step 1
    # 这个价格的用途主要是两个：
    # 1. 直接： 连续合约中，当日的连续合约 price 的计算
    # 2. 直接： 下一日，单日复权调整系数 - adj（ 以致 累计复权调整系数 - cum_adj 的计算 ）
    # 3. 间接（后续）： 当日合约 daily_ret 的计算， 我们会利用 contract_invalid_mask 把这个 daily_ret 做清理

    contract_df[price].fillna(method='ffill', inplace=True)
    contract_df = contract_df.merge(
        daily_data[['contract', 'datetime', price]].rename(columns={'contract': 'next_contract'}),
        how='left', on=['datetime', 'next_contract'], suffixes=('', '_next'))

    # Ffill nans in price (close) of next contract -- step 2
    # 这个价格的用途主要就是一个：
    # 1. 直接： 下一日（合约缺失日 contract_invalid_mask）的单日复权系数的计算 - adj
    contract_df[price_next].fillna(method='ffill', inplace=True)

    # Calculate adjsument factor
    contract_df['adj'] = (contract_df[price_next] / contract_df[price]).shift(1)
    # contract_df['cum_adj'] = contract_df['adj'].cumprod().fillna(method='ffill')
    # 此处不会 fillna， 保持了 cum_adj 的 Nan，就是在保持 pair 序列以及 pair 对应 price 序列中的 Nan 信息
    # 把这个信息彰显出来，留给具体的使用因子去决定如何处理这样的 price 断点
    contract_df['cum_adj'] = contract_df['adj'].cumprod()
    # Overwrite price as adjusted price
    contract_df[price] = contract_df[price] / contract_df.cum_adj
    contract_df = contract_df[['datetime', 'contract', price]].set_index('datetime')

    return contract_df, contract_invalid_mask
