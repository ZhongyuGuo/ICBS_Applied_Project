#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 13:47:58
# @Author  : Michael_Liu @ QTG
# @File    : base
# @Software: PyCharm

import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from abc import abstractmethod
from factor.base import BaseFactor


class BaseCarryFactor(BaseFactor):
    """
    期限结构因子基类

    See Also
    ________
    bases.base.BaseClass
    factor.base.BaseFactor
    """

    def __init__(self, **params) -> None:
        """
        Constructor

        Parameters
        __________
            **params: 关键字可变参数

        See Also
        ________
            factor.base.BaseFactor
        """
        super().__init__(**params)

    @abstractmethod
    def compute_single_factor(self, symbol: str) -> DataFrame:
        """
        计算单品种的期限结构因子, 抽象方法, 需要各期限结构因子重写

        Parameters
        ----------
        symbol: str
                品种代码

        Returns
        -------
        因子值: DataFrame, indexed by 'datetime', columns=[symbol]

        """
        raise NotImplementedError

    def compute_factor(self) -> DataFrame:
        """
        计算因子，通过对compute_single_factor的结果再组装

        Returns
        -------
        因子值: DataFrame, pivoted
                index为datetime, columns为underlying_symbol, values为factor
        """
        # ------- 获取参数 开始 -----------------------
        params = self.get_params()
        window = params['window']
        # ------- 获取参数 结束 -----------------------

        symbol_list = self.get_symbol_list()
        symbol_list = list(set(symbol_list) - set(self.exclusion_list))
        symbol_list.sort()
        factor_list = []
        for symbol in tqdm(symbol_list):
            # create DataFrame from single symbol, indexed by 'datetime', columns = [symbol]
            factor = self.compute_single_factor(symbol)
            factor_list.append(factor)
        # wide-formed dataframe
        factor = pd.concat(factor_list, axis=1)
        factor.columns.name = 'underlying_symbol'
        factor = factor.rolling(window=window, min_periods=1).mean()
        # TODO factor dataframe， NaN
        self.factor_value = factor
        return factor

    def get_factor_value(self) -> DataFrame:
        """
        获取因子值, 如果为None， 计算

        Returns
        -------
        factor_value: DataFrame
                      因子值DataFrame,index为交易时间, columns为品种代码, , values为因子值
        """

        if self.factor_value is None:
            self.compute_factor()

        return self.factor_value