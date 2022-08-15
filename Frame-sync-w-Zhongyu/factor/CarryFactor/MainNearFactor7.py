#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/6 13:12:22
# @Author  : Michael_Liu @ QTG
# @File    : MainNearFactor7
# @Software: PyCharm

import numpy as np
import pandas as pd
from pandas import DataFrame

from factor.CarryFactor.base import BaseCarryFactor
from factor import utilities
import logging


class MainNearFactor7(BaseCarryFactor):
    """
    展期收益率因子： 斜率因子版

    利用过滤后的【多合约】 (不允许进交割月, 且 交易量大于 0)中的最多4个合约的log(price) 与 到期天数的斜率 计算的期限结构因子

    合约:

    See Also: MainNearFactor3.py, 只是展期收益率的具体计算方式不同

    时间差: 交割日差

    计算方法: log(合约价格) ~ constant + beta * 到期天数 回归得到日化展期收益率，再乘以 365，得到年化值，
            方便与期限结构因子的其他表现形式做对比

    Attributes
    __________
    price: str
        用于代表合约价格的字段, close或settlement

    window: int
        因子平滑参数，所有因子都具有

    delist: int as bool, 默认不允许
        允不允许交割月列入, 1 允许， 0 不允许

    filterby: str
        volume/open_interest

    See Also
    ________
    factor.CarryFactor.base.BaseCarryFactor
    """

    def __init__(self,
                 dm=None,
                 price: str = 'close', window: int = 1,
                 delist: int = 0, filterby: str = 'volume'):
        """
        Constructor

        Parameters
        ----------
        price: str
                用于代表合约价格的字段，close或settlement

        window: int
                因子平滑参数，所有因子都具有
        """
        super().__init__(dm=dm, price=price, window=window, delist=delist, filterby=filterby)

    def compute_single_factor(self, symbol: str) -> DataFrame:
        """
        计算单品种的因子值

        Attributes
        ----------
        main_contract: DataFrame, indexed by 'datetime', single columned by 'contract'

        near_contract: DataFrame, indexed by 'datetime', single columned by 'contract'

        Parameters
        ----------
        symbol: str
                品种代码

        Returns
        -------
        因子值: DataFrame, indexed by 'datetime', columns=[symbol]

        """

        price = self.price
        filterby = self.filterby
        delist = self.delist
        window = self.window

        daily_data = self.daily_data_manager.get_symbol(symbol=symbol)
        maturity_date_df = self.basics_data_manager.get_maturity_date(symbol)

        factor, tvalue = utilities.get_carry_factor_and_tvalue(daily_data=daily_data,
                                                               maturity_date_df=maturity_date_df,
                                                               symbol=symbol,
                                                               price=price,
                                                               allow_delist_month=(delist == 1),
                                                               filterby=filterby)

        factor = factor.to_frame(symbol)
        return factor


# %%
if __name__ == "__main__":
    self = MainNearFactor7(price='close')
    exclusion_list = ['LR', 'PM']
    self.set_exclusion_list(exclusion_list)
    factor = self.compute_factor()

    # symbol = 'TA'
    # symbol = 'JM'
    # symbol = 'LR'
    # symbol = 'PM'
    # symbol = 'J'
    # symbol = 'WR'
    # symbol = 'HC'

    # factor_s = self.compute_single_factor(symbol)

    # daily_data = self.daily_data_manager.get_symbol(symbol)
