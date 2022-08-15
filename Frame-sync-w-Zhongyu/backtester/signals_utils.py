#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/18 21:35:16
# @Author  : Michael_Liu @ QTG
# @File    : signals
# @Software: PyCharm


import pandas as pd
import numpy as np
from pandas import (DataFrame, Series)
from typing import List
import qtgQuants.alphalens as al
import qtgQuants.pyfolio as pf


def set_signal_by_quantiles(factor_data: DataFrame,
                            long_quantiles: List[int] = [1],
                            short_quantiles: List[int] = [5]):
    signals = pd.Series(0, index=factor_data.index)
    long_idx = factor_data['factor_quantile'].isin(long_quantiles)
    signals.loc[long_idx] = 1
    short_idx = factor_data['factor_quantile'].isin(short_quantiles)
    signals.loc[short_idx] = -1
    return signals