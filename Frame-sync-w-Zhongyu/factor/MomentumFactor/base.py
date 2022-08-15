#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 14:02:22
# @Author  : Michael_Liu @ QTG
# @File    : base
# @Software: PyCharm

from pandas import DataFrame
from factor.base import BaseFactor

class BaseMomentumFactor(BaseFactor):
    """
    动量因子基类
    """

    def __init__(self, **params) -> None:
        """Constructor"""
        super().__init__(**params)

    def compute_factor(self) -> DataFrame:
        pass