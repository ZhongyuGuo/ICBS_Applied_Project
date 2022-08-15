#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 14:03:58
# @Author  : Michael_Liu @ QTG
# @File    : MomentumFactor1
# @Software: PyCharm

from pandas import DataFrame
import numpy as np
from factor.MomentumFactor.base import BaseMomentumFactor

class MomentumFactor2(BaseMomentumFactor):

    """
    将过去一段时间的日平均日收益率的符号作为动量因子。

    Attributes
    __________
    R: int, default 5
        回溯期，即过去R天的平均收益率

    contract: str, default main
              连续合约以什么合约为基础，main or active_near

    price: str, default close
            选取连续合约的什么价格, close or settlement

    rebalance_days: int, default 1
                    选择几日换仓, 1 or 3 or 5

    """

    def __init__(self, R: int = 5, contract: str = 'main', price: str = 'close', rebalance_days: int = 1) -> None:
        """
        Constructor

        Parameters
        ----------
        R: int, default 5
            回溯期，即过去R天的平均收益率

        contract: str, default main
                连续合约以什么合约为基础，main or active_near

        price: str, default close
                选取连续合约的什么价格, close or settlement

        rebalance_days: int, default 1
                    选择几日换仓, 1 or 3 or 5
        """
        super().__init__(R=R, contract=contract, price=price, rebalance_days=rebalance_days)

    def compute_factor(self) -> DataFrame:
        """
        计算因子值

        Parameters
        ----------

        Returns
        -------
        momentum: DataFrame, pivoted
                    index = datetime
                    columns = underlying_symbol
                    动量因子
        """
        # 获取收盘价
        # ------- 获取参数 开始 -----------------------
        params = self.get_params()
        R = params['R']
        contract = params['contract']
        price = params['price']
        rebalance_days = params['rebalance_days']
        # ------- 获取参数 结束 -----------------------
        price_df = self.get_continuous_field(contract=contract, price=price, rebalance_days=rebalance_days, field='continuous_price')
        return_df = price_df.pct_change()
        momentum = return_df.rolling(window=R, min_periods=0).mean()
        self.factor_value = np.sign(momentum)
        return momentum


# %%
if __name__ == "__main__":
    self = MomentumFactor2(R=10)
    factor_data = self.compute_factor()
    # factor_data = factor_data.drop(columns=['FB'])