#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/9 13:12:22
# @Author  : Michael_Liu @ QTG
# @File    : MainNearFactor9
# @Software: PyCharm

import numpy as np
import pandas as pd
from pandas import DataFrame

from factor.CarryFactor.base import BaseCarryFactor
from factor import utilities
import logging


class MainNearFactor9(BaseCarryFactor):
    """
    利用【近月合约】和去除近月合约后的如下合约 (不允许进交割月, 且 交易量大于 0) 计算的期限结构因子
        次近月合约： others = ‘near’
        最远月合约:  others = 'far'

    合约: 近月合约和去除近月合约后其他合约，两者中近的合约定义为近月合约，远的合约定义为远月合约. 若合约只有一个，则因子值为缺失值

    See Also: MainNearFactor6.py, 只是展期收益率的具体计算方式不同

    时间差: 交割日差

    计算方法: 365 * (Pnear / Pfar - 1) / (Dfar- Dnear)

    非常类似 MainNearFactor6

    Attributes
    __________
    price: str
        用于代表合约价格的字段, close或settlement

    window: int
        因子平滑参数，所有因子都具有

    delist: int as bool,
        允不允许交割月列入, 1 允许， 2 不允许

    filterby: str
        volume/open_interest

    others: str
        除主力合约外的另一合约的生成方式
        "near": 最近月
        "second": 次主力
        "far": 最远月

    See Also
    ________
    factor.CarryFactor.base.BaseCarryFactor
    """

    def __init__(self,
                 dm=None,
                 price: str = 'close', window: int = 1,
                 delist: int = 1, filterby: str = 'volume', others: str = 'near'):
        """
        Constructor

        Parameters
        ----------
        price: str
                用于代表合约价格的字段，close或settlement

        window: int
                因子平滑参数，所有因子都具有
        """
        super().__init__(dm=dm, price=price, window=window, delist=delist, filterby=filterby, others=others)

    def compute_single_factor(self, symbol: str) -> DataFrame:
        """
        计算单品种的因子值

        Attributes
        ----------
        main_contract_df: DataFrame
            reshape to 'datatime' indexed

        near_contract_df: DataFrame
            reshape to 'datatime' indexed

        Parameters
        ----------
        symbol: str
                品种代码

        Returns
        -------
        因子值: DataFrame, naive indexed,
                columns = [datetime, underlying_symbol(均为symbol), factor_value]

        """

        price = self.price
        delist = self.delist
        filterby = self.filterby
        others = self.others

        daily_data = self.daily_data_manager.get_symbol(symbol=symbol)
        dominant = self.get_continuous_data(symbol=symbol, price=price)
        dominant = dominant.set_index('datetime')
        # basics = self.basics_data_manager.get_all_instruments()

        near_contract, far_contract, dominant_invalid_mask, other_invalid_mask = \
            utilities.get_near_far_contract(daily_data=daily_data,
                                            symbol=symbol,
                                            dominant_df='near',
                                            allow_delist_month=(delist == 1),
                                            filter_col='volume',
                                            others_type=others)

        maturity_date_df = self.basics_data_manager.get_maturity_date(symbol)

        near_contract = near_contract.reset_index().merge(maturity_date_df, on='contract')
        near_contract = near_contract.merge(daily_data[['datetime', 'contract', price]], on=['datetime', 'contract'])

        far_contract = far_contract.reset_index().merge(maturity_date_df, on='contract')
        far_contract = far_contract.merge(daily_data[['datetime', 'contract', price]], on=['datetime', 'contract'])

        near_far_df = pd.merge(near_contract, far_contract, on='datetime', suffixes=('_near', '_far')).set_index(
            'datetime')

        near_price = price + '_near'
        far_price = price + '_far'

        # check 3:
        price_invalid_mask = (near_far_df[near_price] == 0) | (near_far_df[far_price] == 0)
        if price_invalid_mask.any():
            logging.warning(u'\n Check 3 - {} of prices are missing by symbol- : {}'.format(
                price_invalid_mask.sum(), symbol))

        # 基差因子 = 365 * (Pnear / Pfar - 1) / (Dfar- Dnear)
        factor = (near_far_df[near_price]/near_far_df[far_price]-1) * 365 / (near_far_df['maturity_date_far'] - near_far_df['maturity_date_near']).dt.days

        # check summary
        if factor.isnull().any():
            logging.warning(u'\n Check 3 - {} of factors cannot be decided by symbol- : {}'.format(
                factor.isnull().sum(), symbol))

        # invalid price causing factor to be set np.nan
        factor.loc[dominant_invalid_mask] = np.nan
        factor.loc[other_invalid_mask] = np.nan
        factor.loc[price_invalid_mask] = np.nan

        factor = factor.to_frame('factor')
        factor['underlying_symbol'] = symbol
        factor.reset_index(inplace=True)

        return factor


# %%
if __name__ == "__main__":
    self = MainNearFactor9(price='close')
    exclusion_list = 'ExclusionList3'
    self.set_exclusion_list('ExclusionList3') #剔除金融期货
    factor = self.compute_factor()
