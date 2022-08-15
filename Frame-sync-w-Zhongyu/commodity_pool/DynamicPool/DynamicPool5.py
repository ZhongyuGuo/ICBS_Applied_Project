#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 15:43:27
# @Author  : Michael_Liu @ QTG
# @File    : DynamicPool5
# @Software: PyCharm

import warnings
import numpy as np
from pandas import DataFrame
from commodity_pool.DynamicPool.base import BaseDynamicPool
from typing import Dict, List, Union

warnings.filterwarnings('ignore')


class DynamicPool5(BaseDynamicPool):
    """
    动态商品池5: New, 返回一个对应商品池的 mask:  boolean DataFrame, wide-form
        index = 'datetime'
        columns = 'underlying_symbol'
        value = boolean, True or False (NaN means False)

    描述：
        1. 考虑 warm_up_days 过滤， rolling(min_periods)   NaN
        2. 考略 window 期的平滑，取平均
        3. 考虑是否强制只需调入 entry_only = True or False
        4. 考虑是否引入 min_value 的强制过滤， min_value = 0 等于不引入过滤,

    比如：
        首先剔除上市不满warm_up_days天的品种
        其次，只有过去window天的filterby均值大于min_value的品种才能入选商品池
        如果选择entry_only=1, 那么品种一旦被选入就不做剔除
        Ex: DynamicPool5(window=20, warm_up_days=126, entry_only=1, min_value=10000, filterby='volume',
                        exclusionList='ExclusionList1')

    参考：
        海通FICC系列中关于期限结构因子的商品池的设计逻辑，以及申万《挖掘商品期货风险溢价因子》

    Attributes
    __________
    window: int, default 126
        滚动窗口

    warm_up_days: int, default 63
        新品种上市热身期， 热身期之后才参与计算调入

    entry_only： int, default 1 as True
        允不允许品种入池之后调出 —— 因为会引入额外的turnover，可能也会对因子收益产生影响

    filter_by: str
        基于该指标的min_value过滤，可以是volume, amount or open interest

    min_value: int, default None
        if None, no filter
        else, symbols below this threshold will be filter out at the end of the pool operation

    exclusionList: str or List[str], default to be 'ExclusionList1'
        str: read specific common exclusionlist's from json
        List[str]: specific exclusion symbol list

    """

    def __init__(self,
                 dm=None,
                 window: int = 20,
                 warm_up_days: int = 120,
                 entry_only: int = 1,
                 min_value: int = 10000,
                 filterby: str = 'volume',
                 exclusionList: Union[str, List[str]] = 'ExclusionList1') -> None:

        super().__init__(dm=dm,
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

        # 上市不满120天的date对应的asset的合约名称都需要mask
        list_date_mask = dominant_contract_df.shift(warm_up_days).isnull()

        # first_list_date = dominant.sort_values('datetime').groupby('underlying_symbol').first()['datetime']

        dominant = dominant.set_index('datetime')
        # 计算 window 滚动 filterby value, min_periods set to window as default
        rolling_value = dominant.groupby('underlying_symbol')[filterby].rolling(window).mean() 
        rolling_value_wide = rolling_value.unstack(level=0)

        rolling_value_wide.mask(list_date_mask, np.nan, inplace=True)
        pool = rolling_value_wide > min_value
        # 为了后面 first_valid_index 功能的使用
        pool = pool.replace(False, np.nan)

        if entry_only == 1:
            # 获取每个品种第一个进入商品池的日期
            first_enter = pool.apply(lambda x: x.first_valid_index())
            # 第一个日期后所有都强制变为True
            for col in pool.columns:
                pool.loc[first_enter[col]:, col] = True

        # fillna to 0.0 to filter out low_value data
        #rolling_value_wide.fillna(0.0, inplace=True)
        #pool.mask(rolling_value_wide <= min_value, np.nan, inplace=True)

        pool.fillna(False, inplace=True)

        self.commodity_pool_value = pool

        return self.commodity_pool_value


# %%
if __name__ == "__main__":
    self = DynamicPool5(window=20, warm_up_days=126, entry_only=1, min_value=10000, filterby='volume',
                        exclusionList='ExclusionList1')
    com_pool = self.compute_commodity_pool_value()
    print(com_pool)

