#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/21 15:04:44
# @Author  : Michael_Liu @ QTG
# @File    : DynamicPool2
# @Software: PyCharm

import numpy as np
from pandas import DataFrame
from commodity_pool.DynamicPool.base import BaseDynamicPool
from typing import Dict, List, Union


class DynamicPool2(BaseDynamicPool):
    """
    动态商品池2: 返回一个对应商品池的 mask (boolean DataFrame，wide-form)

    描述：
        1. 考虑 warm_up_days 过滤， rolling(min_periods)   NaN
        2. 考略 window 期的平滑，取平均
        3. 考虑基于平均成交量的分位数过滤

    比如：
        首先，剔除上市不满warm_up_days天的品种
        其次，计算每个品种每日volume(当日品种各合约的成交量之和）的滚动window日平均，将大于q分位数的品种纳入商品池
        注：仅支持filter by volume
        Ex: DynamicPool2(q=0.25, window=20, warm_up_days=63)

    Attributes
    __________
    q: float, default 0.25
        分位数

    window: int, default 126
            滚动窗口

    warm_up_days: int, default 63
        新品种上市热身期， 热身期之后才参与计算调入

    """

    def __init__(self,
                 dm=None,
                 q: float = 0.25,
                 window: int = 126,
                 warm_up_days: int = 63,
                 exclusionList: Union[str, List[str]] = 'ExclusionList1') -> None:

        super().__init__(dm=dm,
                         q=q,
                         window=window,
                         warm_up_days=warm_up_days,
                         exclusionList=exclusionList)
        self.daily_TotVol = self.get_volume_per_symbol()

    def compute_commodity_pool_value(self) -> DataFrame:
        """

        Returns
        -------
            DataFrame:
                boolean mask for commodity pool
        """

        daily_rolling_volume = self.daily_TotVol.unstack().rolling(window=self.window,
                                        min_periods=min(self.window, self.warm_up_days)).mean()
        # TODO: speical commodity exclusion
        # daily_rolling_volume.loc[:, ['IF', 'IH', 'IC', 'T', 'TF', 'TS', 'SC', 'NR', 'LU', 'BC']] = np.nan
        if self.exclusion_symbol_list:
            daily_rolling_volume.loc[:, self.exclusion_symbol_list] = np.nan
        daily_quantile = daily_rolling_volume.quantile(q=self.q, axis=1, interpolation='higher')
        commodity_pool_value = daily_rolling_volume.apply(lambda x: x >= daily_quantile, axis=0)
        # commodity_pool_value = (daily_rolling_volume.T > daily_quantile).T

        self.commodity_pool_value = commodity_pool_value
        return commodity_pool_value


# %%
if __name__ == "__main__":
    self = DynamicPool2(q=0.25, window=126, warm_up_days=63)
    com_pool = self.compute_commodity_pool_value()