#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 13:07
# @Author  : Zhongyu_Guo @ QTG
# @File    : BasisMomentum4
# @Software: PyCharm

import numpy as np
import pandas as pd
from pandas import DataFrame

from factor.CarryFactor.base import BaseCarryFactor
from factor import utilities
import logging


class BasisMomentum4(BaseCarryFactor):
    """
    利用主力合约和去除主力合约后的如下合约 (不允许进交割月, 且 交易量大于 0) 计算的基差动量因子
        最近合约： others = ‘near’
        最远合约:  others = 'far'
        次主力合约: others = 'second'

    合约: 主力合约和去除主力合约后其他合约，两者中近的合约定义为近月合约，远的合约定义为远月合约. 若合约只有一个，则因子值为缺失值

    计算方法: 获得每日近、远月合约pair后，将所有近月、远月合约用k,rebalance_days=1的方式拼接成两个连续合约
            在获取连续合约价格后，先计算daily_return，再用连续合约得到的contract_invalid_mask将对应日期的return改为nan，再进行R天的rolling mean。
            对近月、远月的rolling mean相减得到因子值


    用于 TS2（近月、主力）、TS4（主力、次主力）因子连续合约化，在连续合约上进行回溯，区别于 BasisMomentum1 在当日合约pair上进行回溯


    Attributes
    __________
    price: str
            用于代表合约价格的字段, close或settlement

    window: int
            因子平滑参数，所有因子都具有

    See Also
    ________
    factor.CarryFactor.base.BaseCarryFactor
    """

    def __init__(self,
                 dm=None,
                 price: str = 'close', R: int = 50, window: int = 1,
                 delist: int = 0, filterby: str = 'volume', others: str = 'near'):
        """
        Constructor

        Parameters
        ----------
        price: str
            用于代表合约价格的字段，close或settlement

        window: int
            因子平滑参数，所有因子都具有

        delist: int as bool, 默认不允许
            允不允许交割月列入, 1 允许， 0 不允许

        filterby: str
            volume/open_interest

        others: str
            除主力合约外的另一合约的生成方式
            "near": 最近月
            "second": 次主力,  此时， fliterby 是定义次主力的方式， volume 或者 openInterest
            "far": 最远月

        """
        super().__init__(dm=dm, price=price, R=R, window=window, delist=delist, filterby=filterby, others=others)

    def compute_single_factor(self, symbol: str) -> DataFrame:
        """
        计算单品种的因子值

        Attributes
        ----------
        near_contract: DataFrame, indexed by 'datetime', single columned by 'contract'

        far_contract: DataFrame, indexed by 'datetime', single columned by 'contract'

        Parameters
        ----------
        symbol: str
                品种代码

        Returns
        -------
        因子值: DataFrame, indexed by 'datetime', columns=[symbol]

        """
        price = self.price
        delist = self.delist
        filterby = self.filterby
        others = self.others
        R=self.R

        daily_data = self.daily_data_manager.get_symbol(symbol=symbol)
        dominant = self.get_continuous_data(symbol=symbol, price=price)
        dominant = dominant.set_index('datetime')
        # basics = self.basics_data_manager.get_all_instruments()

        near_contract, far_contract, dominant_invalid_mask, other_invalid_mask = \
            utilities.get_near_far_contract(daily_data=daily_data,
                                            symbol=symbol,
                                            dominant_df=dominant,               # 【主力合约】
                                            allow_delist_month=(delist == 1),
                                            filter_col=filterby,
                                            others_type=others)

        near_contract, near_invalid_mask = utilities.get_cont_contract_series_adjusted(daily_data=daily_data,contract_df=near_contract,price=price)
        far_contract, far_invalid_mask = utilities.get_cont_contract_series_adjusted(daily_data=daily_data, contract_df=far_contract, price=price)

        near_contract['daily_ret'] = near_contract[price].pct_change()
        near_contract['daily_ret'].loc[near_invalid_mask]=np.nan
        near_contract['period_mean_ret'] = near_contract['daily_ret'].rolling(R,min_periods=1).mean()

        far_contract['daily_ret'] = far_contract[price].pct_change()
        far_contract['daily_ret'].loc[far_invalid_mask] = np.nan
        far_contract['period_mean_ret'] = far_contract['daily_ret'].rolling(R, min_periods=1).mean()

        near_far_df = pd.merge(near_contract, far_contract, on='datetime', suffixes=('_near', '_far'))
        factor = (near_far_df['period_mean_ret_near'] - near_far_df['period_mean_ret_far'])

        # check summary
        if factor.isnull().any():
            logging.warning(u'\n Check 3 - {} of factors cannot be decided by symbol- : {}'.format(
                factor.isnull().sum(), symbol))

        factor = factor.to_frame(symbol)
        return factor


# %%
if __name__ == "__main__":
    self = BasisMomentum4(price='close', R=20)
    # exclusion_list = ['LR', 'PM']
    # self.set_exclusion_list(exclusion_list)
    #factor = self.compute_factor()

    symbol = 'BB'
    # symbol = 'JM'
    # symbol = 'LR'
    # symbol = 'PM'
    # symbol = 'J'
    # symbol = 'HC'
    #
    factor_s = self.compute_single_factor(symbol)

    # daily_data = self.daily_data_manager.get_symbol(symbol)