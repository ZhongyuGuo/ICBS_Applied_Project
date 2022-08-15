#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/13 17:16:40
# @Author  : Michael_Liu @ QTG
# @File    : base
# @Software: PyCharm

from commodity_pool.base import BaseCommodityPool
import json
import logging

class BaseDynamicPool(BaseCommodityPool):
    """

    动态商品池

    """

    def __init__(self, **params) -> None:
        """Constructor"""
        super().__init__(**params)
        if isinstance(self.exclusionList, str):
            self.get_exclusion_list_from_json(self.exclusionList)
        elif isinstance(self.exclusionList, list):
            self.set_exclusion_symbol_list(self.exclusionList)
        else:
            logging.warning(u' - invalid exclusionList input - : {}'.format(self.exclusionList))

    def get_exclusion_list_from_json(self, exclusionList: str) -> None:
        with open(self.commodity_pool_info_file_path, 'r', encoding='utf-8') as file:
             commodity_pool_info = json.load(file)

        exclusion_list = commodity_pool_info['ExclusionDict'][exclusionList]

        self.set_exclusion_symbol_list(exclusion_list)

    def compute_commodity_pool_value(self):
        pass