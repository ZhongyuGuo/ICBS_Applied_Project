#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/18 14:21:19
# @Author  : Michael_Liu @ QTG
# @File    : BaseCommodityPool
# @Software: PyCharm

import inspect
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from datetime import timedelta
from abc import ABC, abstractmethod
from typing import Dict, List
from utils.constants import PUBLIC_DATA_MANAGER_DICT

from bases.base import BaseClass
from data_manager.DailyDataManager import DailyDataManager
from data_manager.BasicsDataManager import BasicsDataManager
from data_manager.IndustryDataManager import IndustryDataManager
from data_manager.ContDominantDataManager import ContDominantDataManager
from data_manager.DataManager import DataManager


class BaseCommodityPool(BaseClass):
    """

        BaseClass for commodity pool subclasses, which is the factory for specific instances

    """

    def __init__(self,
                 dm=None,
                 **params) -> None:
        """
            common constructor

        Parameters
        ----------
        params:
            params for specific subclasses

        """

        super().__init__(dm=dm,
                         **params)

        self.group: str = Path(inspect.getfile(self.__class__)).parent.name
        self.name: str = self.__class__.__name__

        if dm is None:
            self.daily_data_manager: DailyDataManager = DailyDataManager()
            self.basics_data_manager: BasicsDataManager = BasicsDataManager()
            self.industry_data_manager: IndustryDataManager = IndustryDataManager()
            self.cont_dominant_data_manager: ContDominantDataManager = ContDominantDataManager()
        else:
            self.daily_data_manager = dm.daily_data_manager
            self.basics_data_manager = dm.basics_data_manager
            self.cont_dominant_data_manager = dm.cont_dominant_data_manager
            self.industry_data_manager = dm.industry_data_manager

        self.all_instruments: DataFrame = self.basics_data_manager.get_all_instruments()
        self.daily_data: DataFrame = self.daily_data_manager.get_daily_data()

        self.exclusion_symbol_list = []

        self.commodity_pool_info_file_path: Path = Path(__file__).parents[1].joinpath("data") \
            .joinpath("commodity_pool").joinpath('commodity_pool_info.json')

        self.daily_TotVol: DataFrame = None
        self.daily_TotOI: DataFrame = None
        self.listed_date_df: DataFrame = None

        self.commodity_pool_value: DataFrame = None

    def get_listed_date(self) -> DataFrame:
        """
            从 all_instruments 获得各品种的首个上市合约的挂牌日期

        Returns
        -------
            DataFrame:      index = ,  column =
                各品种上市日期
        """
        if not isinstance(self.listed_date_df, DataFrame):
            if not isinstance(self.all_instruments, DataFrame):
                all_instruments = self.basics_data_manager.get_all_instruments()
            else:
                all_instruments = self.all_instruments

            listed_date_df = all_instruments.sort_values(by='contract'). \
                groupby('underlying_symbol', as_index=True)['listed_date'].nth(0).to_frame('listed_date')

            self.listed_date_df = listed_date_df

        return self.listed_date_df

    def get_active_date(self, warmUpDays: int = 60) -> DataFrame:
        """
            获得各品种上市后的 warmUpDays 后的日期
            注: 平移的日期为 calendar days
            依赖参数，动态生成，不存储备份

        Parameters
        ----------
        warmUpDays: int
            新上市品种的激活时长

        Returns
        -------
            DataFrame:
                获得各品种上市后 limit_days 的日期
        """
        listed_date_df = self.get_listed_date()
        active_date_df = (listed_date_df['listed_date'] + pd.Timedelta(days=warmUpDays)).to_frame('active_date')
        # active_date_df['listed_date'] = listed_date_df['listed_date']
        return active_date_df

    def get_volume_per_symbol(self) -> DataFrame:
        """
            generally original tot-volume DataFrame with no smoothing window and filter parameters

        Returns
        -------
            DataFrame:
                index = ('datetime', 'underlying_symbol')
                column = 'volume'

        """
        if not isinstance(self.daily_TotVol, DataFrame):
            self.daily_TotVol = self.daily_data.groupby(['datetime', 'underlying_symbol'], as_index=True)[
                'volume'].sum()
        return self.daily_TotVol

    def get_OI_per_symbol(self) -> DataFrame:
        """
            generally original tot-OI DataFrame with no smoothing window and filter parameters

        Returns
        -------
            DataFrame:
                index = ('datetime', 'underlying_symbol')
                column = 'open_interest'

        """
        if not isinstance(self.daily_TotOI, DataFrame):
            self.daily_TotOI = self.daily_data.groupby(['datetime', 'underlying_symbol'], as_index=True)[
                'open_interest'].sum()
        return self.daily_TotOI

    def set_exclusion_symbol_list(self, exclusion_symbol_list: List[str]) -> None:
        self.exclusion_symbol_list = exclusion_symbol_list

    def set_commodity_pool_value(self, commodity_pool_value: DataFrame) -> None:
        self.commodity_pool_value = commodity_pool_value

    def get_commodity_pool_value(self) -> DataFrame:
        return self.commodity_pool_value

    @abstractmethod
    def compute_commodity_pool_value(self) -> DataFrame:
        raise NotImplementedError

    def __repr__(self):
        group = self.group
        name = self.name
        # 添加因子参数
        param_list = []
        for key, value in self.get_params().items():
            if key not in PUBLIC_DATA_MANAGER_DICT:
                param_list.append(f"{key}={value}")
        title = "commodity_pool(group={}, name={}, {})".format(group, name, ", ".join(param_list))
        return title

    def get_string(self) -> str:
        return self.__repr__()

    def get_info(self) -> Dict:
        return {'group': self.group, 'name': self.name}
