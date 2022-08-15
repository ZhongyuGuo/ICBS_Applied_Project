#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 15:04:44
# @Author  : Michael_Liu @ QTG
# @File    : DynamicPool7
# @Software: PyCharm

import numpy as np
from pandas import DataFrame
from commodity_pool.DynamicPool.base import BaseDynamicPool
from typing import Dict, List, Union


class DynamicPool7(BaseDynamicPool):
    """
    动态商品池7: 返回一个对应商品池的 mask (boolean DataFrame)
    Note: 商品池mask 的起始日期，需要覆盖回测期，因为 mask 的一开始部分，会因为rolling(min_periods)的设置，而自动NaN

    描述：
        1. 考虑 warm_up_days 过滤， rolling(min_periods)   NaN
        2. 考略 window 期的平滑，取平均
        3. 考虑平均成交额的分位数过滤

    比如：
        首先剔除上市不满warm_up_days的品种, warm_up_days = 0代表不进行预热
        其次，计算每个品种主力合约的滚动过去window天的成交额（amount）的平均，大于q分位的品种才能纳入商品池
        Ex: DynamicPool7(q=0.4, window=60, warm_up_days=0, exclusionList='ExclusionList4')

    参考：
        华泰CTA系列 期限结构因子

    Attributes
    __________
    q: float, default 0.4
        分位数

    window: int, default 60
        滚动窗口

    warm_up_days: int, default 63
        新品种上市热身期， 热身期之后才参与计算调入

    exclusionList: List[str]
        需要预先剔除的品种，比如金融期货

    """

    def __init__(self,
                 dm=None,
                 q: float = 0.4,
                 window: int = 60,
                 warm_up_days: int = 0,
                 exclusionList: Union[str, List[str]] = 'ExclusionList4') -> None:

        super().__init__(dm=dm,
                         q=q,
                         window=window,
                         warm_up_days=warm_up_days,
                         exclusionList=exclusionList)

    def compute_commodity_pool_value(self) -> DataFrame:
        """

        Returns
        -------
            DataFrame:
                boolean mask for commodity pool
        """

        group = self.industry_data_manager.get_symbol_industry_map(group='actual_industry', name='actual_five_industry')
        org_dominant = self.cont_dominant_data_manager.get_cont_dominant_data(contract='main', price='close',
                                                                              rebalance_days=1, style='k')
        basics = self.all_instruments

        # params
        window = self.window
        warm_up_days = self.warm_up_days

        symbol_keep = set(group.keys())
        if self.exclusion_symbol_list is not None:
            symbol_keep = symbol_keep - set(self.exclusion_symbol_list)
        dominant = org_dominant[org_dominant.underlying_symbol.isin(symbol_keep)]
        dominant.contract = dominant.contract.fillna(dominant.contract_open)

        # 计算主力合约的日均成交额
        multiplier = basics[['contract', 'contract_multiplier']]
        dominant = dominant.merge(multiplier, on='contract', how='left')
        # 使用交易量*收盘价*multiplier得到正确的 amount 成交额
        dominant['amount'] = dominant.volume * dominant.close * dominant.contract_multiplier

        dominant_contract_df = dominant.pivot(index='datetime', columns='underlying_symbol', values='contract')
        list_date_mask = dominant_contract_df.shift(warm_up_days).isnull()

        dominant = dominant.set_index('datetime')
        # 计算 window 滚动 filterby value, min_periods set to window as default
        rolling_value = dominant.groupby('underlying_symbol')['amount'].rolling(window).mean()
        rolling_value_wide = rolling_value.unstack(level=0)

        rolling_value_wide.mask(list_date_mask, np.nan, inplace=True)
        rolling_value = rolling_value_wide.stack()

        # 求出每天对应的q分位，将大于q分位的品种纳入商品池
        q_amount = rolling_value.groupby('datetime').quantile(self.q, interpolation='higher').reindex(rolling_value_wide.index)
        rolling_value_wide = rolling_value_wide.apply(lambda x: x >= q_amount)

        self.commodity_pool_value = rolling_value_wide
        return self.commodity_pool_value


# %%
if __name__ == "__main__":
    self = DynamicPool7(q=0.4, window=60, warm_up_days=0, exclusionList='ExclusionList4')
    com_pool = self.compute_commodity_pool_value()
    print(com_pool)