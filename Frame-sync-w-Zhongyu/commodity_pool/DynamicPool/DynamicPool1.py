#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/12 10:11:55
# @Author  : Michael_Liu @ QTG
# @File    : DynamicPool1
# @Software: PyCharm

import warnings
import numpy as np
from pandas import DataFrame
from commodity_pool.DynamicPool.base import BaseDynamicPool
from typing import Dict, List, Union

warnings.filterwarnings('ignore')

# mask是个df形式，index=时间，cols=品种，value=bool
class DynamicPool1(BaseDynamicPool):
    """
    动态商品池1: New, 返回一个对应商品池的 mask:  boolean DataFrame, wide-form
        index = 'datetime'
        columns = 'underlying_symbol'
        value = boolean, True or False (NaN means False)

    描述：
        1. 考虑 warm_up_days 过滤， rolling(min_periods)   NaN
        2. 考略 window 期的平滑，取平均
        3. 考虑平均成交额的分位数过滤
        4. 考虑2018年前后的可能的不同分位数操作
        5. 考虑是否强制只需调入 entry_only = True or False，表示只进不出
        6. 考虑是否引入 min_volume 的强制过滤， min_volume = 0 等于不引入过滤

    比如：
        首先剔除上市不满warm_up_days天的品种
        其次，在每一期，将过去window天的平均成交额（amount）在q分位以下的品种剔除
        考虑到活跃度的变动，在2018年1月之前，q=0.4，在2018年1月之后，q=0.3
        如果是entry_only，代表品种一旦被选入那么不会被剔除
        最后，只有过去window天平均成交量大于min_volume手的品种才能纳入商品池
        Ex: DynamicPool1(q1=0.2, q2=0.3, window=126, warm_up_days=63, entry_only=1,
        min_volume=10000, exclusionList='ExclusionList1')


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

    min_volume: int, default None
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
                 entry_only: int = 1,
                 min_volume: int = 0,
                 exclusionList: Union[str, List[str]] = 'ExclusionList1') -> None:

        super().__init__(dm=dm,
                         q1=q1,
                         q2=q2,
                         window=window,
                         warm_up_days=warm_up_days,
                         entry_only=entry_only,
                         min_volume=min_volume,
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
        org_dominant = self.cont_dominant_data_manager.get_cont_dominant_data(contract='main', price='close', rebalance_days=1, style='k')
        basics = self.all_instruments

        # 剔除排除的或没有行业分组的品种
        symbol_keep = set(group.keys())
        if self.exclusion_symbol_list is not None:
            symbol_keep = symbol_keep - set(self.exclusion_symbol_list)
        # TODO: chain indexing warning, copy instead of view
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        dominant = org_dominant[org_dominant.underlying_symbol.isin(symbol_keep)]

        multiplier = basics[['contract', 'contract_multiplier']]
        # 主力合约数据中contract列仅有LR2020年7月21至23日为nan，填补为contract_open, 为了保证后续LR可以动态引入后续操作
        # NaN * multiplier = NaN， thus, 在rolling操作中会被剔除
        dominant.contract = dominant.contract.fillna(dominant.contract_open)

        dominant = dominant.merge(multiplier, on='contract', how='left')
        dominant = dominant.set_index('datetime')
        # 使用交易量*收盘价*multiplier得到正确的 amount 成交额
        dominant['amount'] = dominant.volume * dominant.close * dominant.contract_multiplier

        # 计算120日滚动成交额平均值
        # 新品种推出天数大于min(预热期，回滚期)后即开始计算
        rolling_amount = dominant.groupby('underlying_symbol')['amount'].\
            rolling(self.window, min_periods=min(self.window, self.warm_up_days)).mean()
        # 计算120日滚动成交量平均值，后面要求这个大于10000
        rolling_volume = dominant.groupby('underlying_symbol')['volume'].\
            rolling(self.window, min_periods=min(self.window, self.warm_up_days)).mean()
        # TODO：计算 30、40 分位值，选取这两个分位数的原因参见 "分位数选择" 文档
        # 从 "分位数选择" 文档中看到，2018年及之前，品种成交额区分梯度断档分位点有变动
        q1_amount = rolling_amount.groupby('datetime').quantile(self.q1, interpolation='higher').loc[:'2018-01-01']
        q2_amount = rolling_amount.groupby('datetime').quantile(self.q2, interpolation='higher').loc['2018-01-01':]

        # 将 rolling_amount 这个series变成wide form。Index 为日期，column为品种，方便后面使用apply比较（需要日期能够完全对应，不能缺少）
        rolling_amount_wide = rolling_amount.unstack().T
        pre2018 = rolling_amount_wide.loc[:'2018-01-01']
        post2018 = rolling_amount_wide.loc['2018-01-01':]

        # 2018年前后处理分别
        pre2018 = pre2018.apply(lambda x: x >= q1_amount)
        post2018 = post2018.apply(lambda x: x >= q2_amount)
        pool = pre2018.append(post2018)

        # 为了后面 first_valid_index 功能的使用
        pool = pool.replace(False, np.nan)

        if self.entry_only==1:
            # 获取每个品种第一个进入商品池的日期
            first_enter = pool.apply(lambda x: x.first_valid_index())
            # 第一个日期后所有都强制变为True
            for i in pool.columns:
                pool.loc[first_enter[i]:, i] = True

        # rolling_volume changed to wide-form
        rolling_volume = rolling_volume.unstack().T.fillna(0.0)
        pool.mask(rolling_volume<=self.min_volume, np.nan, inplace=True)

        pool.fillna(False, inplace=True)

        self.commodity_pool_value = pool

        return self.commodity_pool_value


# %%
if __name__ == "__main__":
    self = DynamicPool1(q1=0.2, q2=0.3, window=126, warm_up_days=63, entry_only=1, min_volume=10000, exclusionList='ExclusionList1')
    # com_pool = self.compute_commodity_pool_value()
    # print(com_pool)