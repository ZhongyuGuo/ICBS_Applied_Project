#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/21 10:20:38
# @Author  : Michael_Liu @ QTG
# @File    : FixedPool1
# @Software: PyCharm

from pathlib import Path
import pandas as pd
from pandas import DataFrame
from commodity_pool.base import BaseCommodityPool
import json
import inspect


class FixedPool1(BaseCommodityPool):

    def __init__(self, warmUpDays: int = 60, fixedList: str = 'FixedPool1List1') -> None:
        super().__init__(warmUpDays=warmUpDays, fixedList=fixedList)

    def get_fixed_pool_list(self):
        """

        Returns
        -------
            fixed_pool: List[str]
                manually set pool list
        """
        with open(self.commodity_pool_info_file_path, 'r', encoding='utf-8') as file:
             commodity_pool_info = json.load(file)
        group = Path(inspect.getfile(self.__class__)).parent.name
        name = self.__class__.__name__

        fixed_pool = commodity_pool_info[group][name][self.fixedList]

        return fixed_pool

    # previous version, return only partial mask
    # def compute_commodity_pool_value(self) -> DataFrame:
    #     """
    #         mannually select symbols and create commodity pool
    #
    #     Returns
    #     -------
    #         DataFrame:
    #             with only selected symbol columns, not the full DataFrame
    #
    #     """
    #
    #     fixed_pool = self.get_fixed_pool_list()
    #
    #     # fixed_pool = ['A', 'AG', 'AL', 'AU', 'C', 'CF', 'CS', 'CU', 'FG', 'I', 'J', 'JD', 'JM', 'L', 'M', 'MA', 'NI',
    #     #               'OI', 'P', 'PP', 'RB', 'RM', 'RU', 'SR', 'TA', 'V', 'Y', 'ZC', 'ZN']
    #
    #     listed_date_df = self.get_active_date(self.warmUpDays)
    #
    #     daily_volume = self.get_volume_per_symbol().reset_index()       # .reset_index()
    #
    #     listed_date_df = listed_date_df.loc[fixed_pool].reset_index()       # .reset_index()
    #
    #     # return df with selected symbols
    #     listed_date_df = pd.merge(left=daily_volume[['datetime', 'underlying_symbol']],
    #                                     right=listed_date_df, on='underlying_symbol')
    #
    #     # return df with full symbols columns
    #     # listed_date_df = pd.merge(left=daily_volume[['datetime', 'underlying_symbol']],
    #     #                                 right=listed_date_df, on='underlying_symbol', how='left')
    #
    #     listed_date_df['fixed_pool'] = listed_date_df['datetime'] >= listed_date_df['active_date']
    #     commodity_pool_value = listed_date_df.set_index(['datetime', 'underlying_symbol'])['fixed_pool'].\
    #         unstack(level=-1).fillna(False)
    #
    #     self.commodity_pool_value = commodity_pool_value
    #     return commodity_pool_value

    def compute_commodity_pool_value(self):
        """
            mannually select symbols and create commodity pool

        Returns
        -------
            DataFrame:
                with the full DataFrame mask, not only the selected symbol columns

        """
        fixed_pool = self.get_fixed_pool_list()
        active_date_df = self.get_active_date(self.warmUpDays)
        daily_volume = self.get_volume_per_symbol().reset_index()  # .reset_index()
        df = pd.merge(left=daily_volume[['datetime', 'underlying_symbol']],
                                        right=active_date_df, on='underlying_symbol')
        out_of_list_symbols = list(set(active_date_df.index) - set(fixed_pool))
        df['fixed_pool'] = df['datetime'] >= df['active_date']
        commodity_pool_value = df.set_index(['datetime', 'underlying_symbol'])['fixed_pool'].\
            unstack(level=-1).fillna(False)
        commodity_pool_value[out_of_list_symbols] = False

        self.commodity_pool_value = commodity_pool_value
        return commodity_pool_value

if __name__ == "__main__":
    self = FixedPool1(warmUpDays=60, fixedList='FixedPool1List1')
    print(self.get_fixed_pool_list())
    print(self.compute_commodity_pool_value())