#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 22:20:19
# @Author  : Michael_Liu @ QTG
# @File    : BasisMomentum2
# @Software: PyCharm

import numpy as np
import pandas as pd
from pandas import DataFrame

from factor.CarryFactor.base import BaseCarryFactor
from factor import utilities
import logging


class BasisMomentum2(BaseCarryFactor):
    """
    利用【近月合约】和去除近月合约后的如下合约 (不允许进交割月, 且 交易量大于 0) 计算的基差动量因子
        最近合约： others = ‘near’
        最远合约:  others = 'far'
        次主力合约: others = 'second'

    合约: 主力合约和去除主力合约后其他合约，两者中近的合约定义为近月合约，远的合约定义为远月合约. 若合约只有一个，则因子值为缺失值

    计算方法: (近月合约区间收益-远月合约区间收益)
    TODO: 此处不同合约的近远月结构的月份差不同，需不需要将上述基差动量除以月份差/到期差，获得更好的横向对比，可以据此构建 BasisMomentum2,3

    类似 BasisMomentum1,只是参与求差的合约 从 主力和近月 变成了， 近月 和 others

    用于 TS1（近月、次近月） pair only， 因为 TS3（近月、远月）的设计不能保证在长回溯期下的能计算因子值
    因为是基于 TS1 only， 因此，不用考虑月化的操作

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
                 delist: int = 0, filterby: str = 'volume', others: str = 'far'):
        """
        Constructor

        Parameters
        ----------
        price: str
            用于代表合约价格的字段，close或settlement

        window: int
            因子平滑参数，所有因子都具有

        delist: int as bool,
            允不允许交割月列入, 1 允许， 2 不允许

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
        window = self.window
        delist = self.delist
        filterby = self.filterby
        others = self.others
        R = self.R

        daily_data = self.daily_data_manager.get_symbol(symbol=symbol)
        # basics = self.basics_data_manager.get_all_instruments()

        near_contract, far_contract, dominant_invalid_mask, other_invalid_mask = \
            utilities.get_near_far_contract(daily_data=daily_data,
                                            symbol=symbol,
                                            dominant_df='near',
                                            allow_delist_month=(delist == 1),
                                            filter_col=filterby,
                                            others_type=others)

        near_contract = utilities.get_interval_ret(contract_df=near_contract,
                                                 price=price,
                                                 window=R,
                                                 daily_data=daily_data)

        far_contract = utilities.get_interval_ret(contract_df=far_contract,
                                                price=price,
                                                window=R,
                                                daily_data=daily_data)

        near_far_df = pd.merge(near_contract, far_contract, on='datetime', suffixes=('_near','_far'))
        factor = near_far_df['interval_ret_near'] - near_far_df['interval_ret_far']

        # check summary
        if factor.isnull().any():
            logging.warning(u'\n Check 3 - {} of factors cannot be decided by symbol- : {}'.format(
                factor.isnull().sum(), symbol))

        # invalid price causing factor to be set np.nan
        factor.loc[dominant_invalid_mask] = np.nan
        factor.loc[other_invalid_mask] = np.nan

        factor = factor.to_frame(symbol)
        return factor


# %%
if __name__ == "__main__":
    self = BasisMomentum2(price='close', R=120)
    # exclusion_list = ['LR', 'PM']
    # self.set_exclusion_list(exclusion_list)
    #factor = self.compute_factor()

    symbol = 'A'
    # symbol = 'JM'
    # symbol = 'LR'
    # symbol = 'PM'
    # symbol = 'J'
    # symbol = 'HC'
    #
    factor_s = self.compute_single_factor(symbol)

    # daily_data = self.daily_data_manager.get_symbol(symbol)
