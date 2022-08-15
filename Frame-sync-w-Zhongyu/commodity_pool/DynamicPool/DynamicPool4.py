#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 15:43:27
# @Author  : Michael_Liu @ QTG
# @File    : DynamicPool4
# @Software: PyCharm

import warnings
import numpy as np
from pandas import DataFrame
from commodity_pool.DynamicPool.base import BaseDynamicPool
from typing import Dict, List, Union

warnings.filterwarnings('ignore')


class DynamicPool4(BaseDynamicPool):
    """
    动态商品池4: New, 返回一个对应商品池的 mask:  boolean DataFrame, wide-form
        index = 'datetime'
        columns = 'underlying_symbol'
        value = boolean, True or False (NaN means False)

    描述：
        1. 考虑 warm_up_days 过滤， rolling(min_periods)   NaN
        2. 考略 window 期的平滑，取平均
        3. 考虑平均成交额的分位数过滤
        4. 考虑2018年前后的可能的不同分位数操作
        5. 考虑是否强制只需调入 entry_only = True or False
        6. 考虑是否引入 min_value 的强制过滤， min_value = 0 等于不引入过滤,

    比如：
        首先剔除上市不满warm_up_days的品种
        之后，对某个filterby指标（可以是volume, amount等）：既做分位数q过滤，也做min value的强制过滤

        比如，在每一期，将过去window天的指标filterby均值在q分位以下的品种剔除
        考虑到活跃度的变动，在2018年1月前后，q可以选取不同的值
        同时，强制要求只有过去window天filterby均值 大于min_value手的品种才能纳入商品池
        Ex: DynamicPool4(q1=0.2, q2=0.3, window=126, warm_up_days=63, entry_only=1, min_value=10000, filterby='volume',
                        exclusionList='ExclusionList1')

    Attributes
    __________
    q1: float, default 0.3
        分位数, filtering the low-liquidity symbols after 2018-01

    q2: float, default 0.4
        分位数, filtering the low-liquidity symbols before 2018-01

    window: int, default 126
        滚动窗口

    warm_up_days: int, default 63
        新品种上市热身期， 热身期之后才参与计算调入

    entry_only： int, default 1 as True
        允不允许品种入池之后调出 —— 因为会引入额外的turnover，可能也会对因子收益产生影响

    min_value: int, default None
        if None, no filter
        else, symbols below this threshold will be filter out at the end of the pool operation

    exclusionList: str or List[str], default to be 'ExclusionList1'
        str: read specific common exclusionlist's from json
        List[str]: specific exclusion symbol list

    """

    def __init__(self,
                 dm=None,
                 q1: float = 0.3,
                 q2: float = 0.4,
                 window: int = 126,
                 warm_up_days: int = 63,
                 entry_only: int = 1, # 只进不出
                 min_value: int = 0,
                 filterby: str = 'volume',
                 exclusionList: Union[str, List[str]] = 'ExclusionList1') -> None:

        super().__init__(dm=dm,
                         q1=q1,
                         q2=q2,
                         window=window,
                         warm_up_days=warm_up_days,
                         entry_only=entry_only,
                         min_value=min_value,
                         filterby=filterby,
                         exclusionList=exclusionList)

    def compute_commodity_pool_value(self) -> DataFrame:
        """

        Returns
        -------
            DataFrame:
                boolean mask for commodity pool, wide-form indexed by 'datetime', columned by 'underlying_symbol'

        """
        # fixed for below group category
        group = self.industry_data_manager.get_symbol_industry_map(group='actual_industry', name='actual_five_industry')
        org_dominant = self.cont_dominant_data_manager.get_cont_dominant_data(contract='main', price='close',
                                                                              rebalance_days=1, style='k')
        basics = self.all_instruments

        # params
        filterby = self.filterby
        window = self.window
        warm_up_days = self.warm_up_days
        min_value = self.min_value
        entry_only=self.entry_only

        # 剔除排除的或没有行业分组的品种
        symbol_keep = set(group.keys())
        if self.exclusion_symbol_list is not None:
            symbol_keep = symbol_keep - set(self.exclusion_symbol_list)
        # TODO: chain indexing warning, copy instead of view
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        dominant = org_dominant[org_dominant.underlying_symbol.isin(symbol_keep)]
        dominant.contract = dominant.contract.fillna(dominant.contract_open)

        if filterby == 'amount':
            multiplier = basics[['contract', 'contract_multiplier']]
            # 主力合约数据中contract列仅有LR2020年7月21至23日为nan，填补为contract_open, 为了保证后续LR可以动态引入后续操作
            # NaN * multiplier = NaN， thus, 在rolling操作中会被剔除
            dominant = dominant.merge(multiplier, on='contract', how='left')
            # 使用交易量*收盘价*multiplier得到正确的 amount 成交额
            dominant['amount'] = dominant.volume * dominant.close * dominant.contract_multiplier

        dominant_contract_df = dominant.pivot(index='datetime', columns='underlying_symbol', values='contract')
        list_date_mask = dominant_contract_df.shift(warm_up_days).isnull()

        # first_list_date = dominant.sort_values('datetime').groupby('underlying_symbol').first()['datetime']

        dominant = dominant.set_index('datetime')
        # 计算 window 滚动 filterby value, min_periods set to window as default
        rolling_value = dominant.groupby('underlying_symbol')[filterby].rolling(window).mean()
        rolling_value_wide = rolling_value.unstack(level=0)

        pre2018 = rolling_value_wide.loc[:'2018-01-01']
        post2018 = rolling_value_wide.loc['2018-01-01':]

        rolling_value_wide.mask(list_date_mask, np.nan, inplace=True)
        rolling_value = rolling_value_wide.stack()

        # TODO：计算 30、40 分位值，选取这两个分位数的原因参见 "分位数选择" 文档
        # 从 "分位数选择" 文档中看到，2018年及之前，品种成交额区分梯度断档分位点有变动
        q1_amount = rolling_value.groupby('datetime').quantile(self.q1, interpolation='higher').reindex(pre2018.index)
        q2_amount = rolling_value.groupby('datetime').quantile(self.q2, interpolation='higher').reindex(post2018.index)
        # 2018年前后处理分别
        pre2018 = pre2018.apply(lambda x: x >= q1_amount)
        post2018 = post2018.apply(lambda x: x >= q2_amount)
        pool = pre2018.append(post2018)

        # 为了后面 first_valid_index 功能的使用
        pool = pool.replace(False, np.nan)

        if entry_only == 1:
            # 获取每个品种第一个进入商品池的日期
            first_enter = pool.apply(lambda x: x.first_valid_index())
            # 第一个日期后所有都强制变为True
            for col in pool.columns:
                pool.loc[first_enter[col]:, col] = True

        # fillna to 0.0 to filter out low_value data
        rolling_value_wide.fillna(0.0, inplace=True)
        pool.mask(rolling_value_wide <= min_value, np.nan, inplace=True)

        pool.fillna(False, inplace=True)

        self.commodity_pool_value = pool

        return self.commodity_pool_value


# %%
if __name__ == "__main__":
    self = DynamicPool4(q1=0.2, q2=0.3, window=126, warm_up_days=63, entry_only=1, min_value=10000, filterby='volume',
                        exclusionList='ExclusionList1')
    com_pool = self.compute_commodity_pool_value()
    print(com_pool)