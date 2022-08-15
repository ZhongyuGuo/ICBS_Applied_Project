#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/27 13:37:50
# @Author  : Michael_Liu @ QTG
# @File    : tools
# @Software: PyCharm

from typing import Dict, Tuple, Any, List
from pandas import DataFrame


def reverse_dict(input_dict: Dict[str, List[str]]) -> Dict[str, str]:
    """
        change ‘1 - multiple’ dict into  ‘1-1’ dict

    Parameters
    ----------
    input_dict: Dict[str, List[str]]   1 -> multiple

    Returns
    -------
        Dict[str, str]          1 -> 1

    """
    output_dict: Dict[str, str] = {}
    for key in input_dict:
        for value in input_dict[key]:
            output_dict[value] = key

    return output_dict


def df_to_dict(df: DataFrame, key: str, value: str) -> Dict[str, List[str]]:
    """
        create dict with key columns

    Parameters
    ----------
    df
    key: str
    value: str

    Returns
    -------

    """
    output_dict = {}
    for k in set(df[key].values):
        output_dict[k] = list(df.loc[df[key] == k, value].values)
    return output_dict