#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 12:23:27
# @Author  : Michael_Liu @ QTG
# @File    : base
# @Software: PyCharm

# 导入库
import inspect
from pathlib import Path
from pandas import DataFrame
from typing import Dict, List
from abc import abstractmethod
from typing import Dict
from utils.constants import PUBLIC_DATA_MANAGER_DICT

from bases.base import BaseClass
from data_manager.DailyDataManager import DailyDataManager
from data_manager.BasicsDataManager import BasicsDataManager
from data_manager.ContDominantDataManager import ContDominantDataManager
from data_manager.TradeHoldDataManager import TradeHoldDataManager
from data_manager.WarehouseDataManager import WarehouseDataManager


class BaseFactor(BaseClass):
    """
        因子基类
        Note: complete exclusion_list handling mechanism to exclude the boundary error during factor computation
            for some trivial symbols


    Parameters
    __________
    **params: 因子参数，以关键字参数的形式输入，可通过set_params方法设置参数，可通过get_params方法获取参数

    Attributes
    __________
    all_instruments: DataFrame
                    所有期货合约基础信息，
                    Index(['contract', 'underlying_symbol', 'market_tplus', 'symbol',
                       'margin_rate', 'maturity_date', 'mode', 'trading_code', 'exchange',
                       'product', 'contract_multiplier', 'round_lot', 'trading_hours',
                       'listed_date', 'industry_name', 'de_listed_date',
                       'underlying_order_book_id', 'symbol_cn', 'symbol_full'],
                      dtype='object')

    symbol_industry_map: DataFrame
                        所有期货合约对应的行业（采用all_instruments中的行业）。两列DataFrame，
                        第一列为underlying_symbol，第二列为industry_name

    daily_data_manager: DailyDataManager
                        日线数据管理器

    basics_data_manager: BasicsDataManager
                        基础数据管理器

    cont_dominant_data_manager: ContDominantDataManager
                                        主力连续合约数据管理器

    See Also
    ________
    bases.bases.BaseClass
    data_manager.DailyDataManager.DailyDataManager
    data_manager.BasicsDataManager.BasicsDataManager
    data_manager.ContDominantDataManager.ContDominantDataManager

    """

    def __init__(self,
                 dm=None,
                 **params) -> None:
        """Constructor"""

        super().__init__(dm=dm,
                         **params)

        self.group: str = Path(inspect.getfile(self.__class__)).parent.name
        self.name: str = self.__class__.__name__

        self.all_instruments: DataFrame = None
        self.symbol_industry_map: DataFrame = None

        if dm is None:
            self.daily_data_manager: DailyDataManager = DailyDataManager()
            self.basics_data_manager: BasicsDataManager = BasicsDataManager()
            self.cont_dominant_data_manager: ContDominantDataManager = ContDominantDataManager()
            self.trade_hold_data_manager: TradeHoldDataManager = TradeHoldDataManager()
            self.warehouse_data_manager: WarehouseDataManager = WarehouseDataManager()
        else:
            self.daily_data_manager = dm.daily_data_manager
            self.basics_data_manager = dm.basics_data_manager
            self.cont_dominant_data_manager = dm.cont_dominant_data_manager
            self.trade_hold_data_manager = dm.trade_hold_data_manager
            self.warehouse_data_manager = dm.warehouse_data_manager

        self.maturity_date_dict: Dict[str, DataFrame] = {}

        self.factor_value: DataFrame = None
        self.init_basics_data()

        self.symbol_list: List[str] = None

        self.exclusion_list: List[str] = []

    def init_basics_data(self) -> None:
        """
            初始化基础数据,获取所有期货合约基础信息all_instruments和品种行业对应表symbol_industry_map
        """
        if not isinstance(self.all_instruments, DataFrame):
            all_instruments = self.basics_data_manager.get_all_instruments()
            # 此处的industry_name 是RQData 的分类
            symbol_industry_map = all_instruments[['underlying_symbol', 'industry_name']].drop_duplicates()
            self.all_instruments = all_instruments
            self.symbol_industry_map = symbol_industry_map

    def set_factor_value(self, factor_value: DataFrame) -> None:
        """
            设置因子值

        Parameters
        ----------
        factor_value: DataFrame
                      因子值DataFrame,index为交易时间, columns为品种代码, data为因子值

        Returns
        -------
        None
        """
        self.factor_value = factor_value

    def get_factor_value(self) -> DataFrame:
        """
        获取因子值

        Returns
        -------
        factor_value: DataFrame
                      因子值DataFrame,index为交易时间, columns为品种代码, , values为因子值
        """
        return self.factor_value

    def get_symbol_list(self) -> List[str]:
        """
        获取期货品种列表

        Returns
        -------
        symbol_list: 期货品种列表: List[str]
        """

        if not isinstance(self.symbol_list, List):
            self.symbol_list: List[str] = self.basics_data_manager.get_symbol_list()
        return self.symbol_list

    def set_symbol_list(self, symbol_list: List[str] = None) -> None:
        """
            考虑节约多品种的应用场景，直接限制到个别品种计算因子值

        Parameters
        ----------
        symbol_list: 设定的品种列表

        Returns
        -------

        """
        if isinstance(symbol_list, List):
            self.symbol_list = symbol_list

    def set_exclusion_list(self, exclusion_list: List[str]) -> None:
        self.exclusion_list = exclusion_list

    def get_exclusion_list(self) -> List[str]:
        return self.exclusion_list

    def get_continuous_data(self, symbol: str,
                            contract: str = 'main', price: str = 'close', rebalance_days: int = 1) -> DataFrame:
        """
        获取连续合约数据, 按品种代码取, 不做stack

        Parameters
        ----------
        symbol: str,
                品种代码

        contract: str, default main
                选择以何种合约为基础的连续数据, main主力, active_near活跃近月

        price: str, default close
                选择以什么价格为基础的连续数据, close为收盘价, settlement结算价

        rebalance_days: str, default 1
                换仓天数, 1或3或5

        Returns
        -------
        continuous_contract_data: DataFrame, naive indexed
                连续合约数据，columns:
                    ['datetime', 'contract_close', 'contract_open', 'close_contract_open',
                   'close_contract_close', 'adj', 'cum_adj', 'adj_price', 'adj_style',
                   'contract_close_multidays', 'contract_open_multidays', 'switched',
                   'nth_switching_day', 'cur_contract_exe', 'prev_contract_exe',
                   'cur_contract_weight_exe', 'prev_contract_weight_exe',
                   'cur_contract_weight_eff', 'prev_contract_weight_eff',
                   'cur_contract_eff', 'prev_contract_eff', 'cur_ret_eff', 'prev_ret_eff',
                   'ret', 'continuous_price', 'contract', 'open_interest',
                   'prev_settlement', 'open', 'volume', 'turnover', 'close', 'high',
                   'upper_limit', 'lower_limit', 'low', 'settlement', 'underlying_symbol']

        """
        # 因为使用实例化的方式，通过data_manager取数据，这个数据实际上已经生成了，并不是每次都通过文件打开来获取
        cont_data = self.cont_dominant_data_manager. \
            get_cont_dominant_data(contract=contract, price=price, rebalance_days=rebalance_days)
        cont_data = cont_data[cont_data['underlying_symbol'] == symbol]
        return cont_data

    def get_continuous_field(self, contract: str = 'main', price: str = 'close', rebalance_days: int = 1,
                             field: str = 'continuous_price') -> DataFrame:
        """
        获取连续合约指定字段的数据

        Parameters
        ----------
        contract: str
                合约种类，目前可选有main和active_near, main表示主力合约, active_near表示活跃近月

        price: str
                选择以什么价格为基础的连续数据, close为收盘价, settlement结算价

        rebalance_days: int, default = 1
                换仓天数, 可选天数1,3,5
        field: str, default = 'continuous_price'
                字典，continuous_price

        Returns
        -------
        df: DataFrame, pivoted
            连续合约field字段数据, 一般是开盘价或收盘价
            index为交易时间, columns为品种代码, values为field值
        """
        return self.cont_dominant_data_manager.get_cont_dominant_data_by_field(contract=contract,
                                                         price=price, rebalance_days=rebalance_days, field=field)

    def get_field(self, symbol: str, field: str) -> DataFrame:
        """
            获取单品种所有合约在整个交易时间段某数值字段的DataFrame

        Parameters
        ________
        symbol: string
                品种代码

        field:  string
                要获取的字段，如open, high, low, close, settlement, prev_settlement, open_interest, volume, upper_limit, lower_limit

        Returns
        -------
        data: DataFrame, multiindexed
                # TODO: 形式很差， 单品种所有合约在整个交易时间某数值字段的DataFrame，index是交易日期，columns是合约代码，data是字段值
                # changed to:
                index = ['datetime', 'contract']
                columns = filed
        """
        # 预先检查
        #data = self.daily_data_manager.get_field(symbol, field)
        data = self.daily_data_manager.get_field(symbol, field).stack().to_frame(field)
        return data

    def get_maturity_date(self, symbol: str) -> DataFrame:
        """
        获取某个品种所有合约的到期日

        Parameters
        __________
        symbol: string
                品种代码

        Returns
        __________
        品种到期日期，DataFrame，一共两列，第一列为合约(contract)，第二列为到期日(maturity_date)

        Examples
        ________
        >>> print(self.get_maturity_date('A'))
            contract maturity_date
        0      A0303    2003-03-14
        1      A0305    2003-05-23
        2      A0307    2003-07-14
        3      A0309    2003-09-12
        4      A0311    2003-11-14
        ..       ...           ...
        """
        maturity_date = self.basics_data_manager.get_maturity_date(symbol)
        return maturity_date

    @abstractmethod
    def compute_factor(self, *args, **kwargs) -> DataFrame:
        """
        计算因子抽象方法，需要在所有因子中重写

        Parameters
        ----------
        args: 可变位置参数

        kwargs: 可变关键字参数

        Returns
        -------
            因子值，index为交易时间, columns为品种代码, values为因子值
        """
        raise NotImplementedError

    def __repr__(self):
        group = self.group
        name = self.name
        # 添加因子参数
        param_list = []
        for key, value in self.get_params().items():
            if key not in PUBLIC_DATA_MANAGER_DICT:
                param_list.append(f"{key}={value}")
        title = "factor(group={}, name={}, {})".format(group, name, ", ".join(param_list))
        return title

    def get_string(self) -> str:
        return self.__repr__()

    def get_info(self) -> Dict:
        return {'group': self.group, 'name': self.name}