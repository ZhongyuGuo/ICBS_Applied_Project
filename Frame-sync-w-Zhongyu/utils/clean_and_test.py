#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/15 11:02:33
# @Author  : Michael_Liu @ QTG
# @File    : clean_and_test
# @Software: PyCharm

import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


def nan_test(factor_wide: DataFrame = None, commodity_pool_mask: DataFrame = None,file_path: str = None, factor_name: str = None) -> DataFrame:
    """
        check the nan case for each symbols since the first non-nan and summarize the nan-ratio with plots

    Parameters
    ----------
    factor_wide:   DataFrame
        wide-form:  index = datetime, columns = symbols

    file_path
        loading from file if factor-wide is not input directly

    Returns
    -------
        summary: DataFrame
            Nan case summary with decending ordered for nan-ratio
    """

    if factor_wide is None:
        factor_wide = pd.read_pickle(file_path)

    def _find_nan_last_occurence(s):
        df_nan = s[s.isnull()]
        if len(df_nan) == 0:
            return np.nan
        else:
            return df_nan.index[-1]

    summary = factor_wide.apply(lambda x: len(x.loc[x.first_valid_index():])).to_frame('total_count')
    summary['nan_count'] = factor_wide.apply(lambda x: x.loc[x.first_valid_index():].isnull().sum())
    summary['nan_ratio'] = summary.nan_count / summary.total_count
    summary['last_occurence'] = factor_wide.apply(lambda x: _find_nan_last_occurence(x.loc[x.first_valid_index():]))
    summary = summary[summary.nan_ratio != 0]
    summary = summary.sort_values('nan_ratio', ascending=False)
    #summary.nan_ratio.plot(kind='bar', figsize=(20, 12), title='Nan ratio')
    #把商品池mask中的False改成nan，可以直接使用定义的内置函数
    commodity_pool_mask = commodity_pool_mask.replace(False, np.nan)
    #获取每个品种进入商品池后的nan(即原本的false)
    pool_nan_count = commodity_pool_mask.apply(lambda x: x.loc[x.first_valid_index():].isnull().sum())
    pool_nan_count.name = 'pool_nan_count'
    # Merge到summary中，如果summuary的总nan_ratio为0，那mask部分的nan数量一定为0，how=left不会丢失信息
    summary = summary.merge(pool_nan_count, on='underlying_symbol', how='left')
    #计算品种商品池nan占比
    summary['pool_nan_ratio'] = summary.pool_nan_count / summary.total_count
    #计算因子值nan占比，这里如果一个商品某日的的因子值和商品池mask都为nan,那会被算作商品池nan，此处的是在商品池不是nan，但factor_value是nan
    summary['value_nan_ratio'] = (summary.nan_count - summary.pool_nan_count) / summary.total_count
    if factor_name is None:
        summary.to_csv('nan_summary.csv')
    else:
        summary.to_csv(f'{factor_name}_nan_summary.csv')
    fig, ax = plt.subplots(figsize=(20,12))
    ax.bar(summary.index, summary.value_nan_ratio, label='value_nan')
    ax.bar(summary.index, summary.pool_nan_ratio, bottom=summary.value_nan_ratio, label='mask_nan')
    ax.set_title('nan_ratio')
    ax.legend()
    plt.show()

    return summary


def get_trade_cal_wo_ns(trade_cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
        return the special trade_cal on which the night session is hault for holiday or other reasons

    Parameters
    ----------
    trade_cal: DatetimeIndex

    Returns
    -------
    trade_cal_wo_sc: DatetimeIndex

    """
    trade_cal = pd.Series(trade_cal)
    days_diff = trade_cal.diff().dt.days
    tradingday_af_holiday = pd.DatetimeIndex(trade_cal.loc[(days_diff > 3) | (days_diff == 2)])
    # specific mask for 2020 covid event
    covid_range = pd.date_range(start='2020-02-04', end='2020-05-06', freq='D')
    covid_tradingday = covid_range.intersection(trade_cal)
    trade_cal_wo_ns = tradingday_af_holiday.union(covid_tradingday)
    # special_date mask, i.e. '2015-09-28'
    trade_cal_wo_ns = trade_cal_wo_ns.union(np.array([pd.to_datetime('2015-09-28')]))
    return trade_cal_wo_ns


def min_data_check_by_contract_by_RQData(filepath: str, all_instruments: DataFrame, trade_cal_wo_ns: pd.DatetimeIndex) -> DataFrame:
    """
        count the data missing cases for each contract in "Start-point" trading schedule
        by 'trading_hours' of RQData: all_instruments

    Parameters
    ----------
    filepath: str
        filepath of 1min's data of specific contract for check

    all_instruments: DataFrame
        basic info of all instruments

    trade_cal_wo_ns: DatetimeIndex

    Returns
    -------
        if there's any missing case on specific date, DataFrame will be returned
        DataFrame format:
            index = 'date'
            columns = specific 'trading_hour' session

        otherwise, return None

    """

    contract = pd.read_pickle(filepath)
    # TODO: adjust the time stamp back to start-point
    contract.index -= pd.Timedelta('1minute')
    contract_name = Path(filepath).stem
    basics = all_instruments
    trading_hour = basics[basics.contract == contract_name].trading_hours.values[0]
    trading_hour = trading_hour.split(',')
    start_time = []
    end_time = []

    # 获取每一段交易时间的开始和结束时间
    for i in range(len(trading_hour)):
        t_start, delim, t_end = trading_hour[i].partition('-')
        t_start = (pd.to_datetime(t_start)-pd.Timedelta('1minute')).time()
        t_end = (pd.to_datetime(t_end)-pd.Timedelta('1minute')).time()
        start_time.append(t_start)
        # 将跨日的交易时间分成两段零点前后两段
        if t_end < t_start:
            t_end1 = pd.to_datetime('23:59').time()
            t_start1 = pd.to_datetime('00:00').time()
            start_time.append(t_start1)
            end_time.append(t_end1)
        end_time.append(t_end)
    trading_minutes = []
    axed_trading_hour = []
    night_time_mask = []

    # 获取每一段交易时间的分钟数
    for i in range(len(start_time)):
        time_diff = datetime.combine(datetime.today(), end_time[i]) - datetime.combine(datetime.today(), start_time[i])
        # 因为min bar 使用的是边界方法，差值需要加1
        trading_minutes.append(time_diff.total_seconds() / 60 + 1)
        # 跨日的交易时间被分成两段，需要重新制作交易时间list
        axed_trading_hour.append('-'.join([str(start_time[i]), str(end_time[i])]))
        # 记录哪些交易时间是夜盘
        is_night = (start_time[i] > pd.to_datetime('18:00').time()) | (start_time[i] < pd.to_datetime('09:00').time())
        night_time_mask.append(is_night)
    night_time = np.array(axed_trading_hour)[night_time_mask]
    trading_date = contract.trading_date.unique()
    column_name = dict(zip(list(range(len(axed_trading_hour))), axed_trading_hour))

    # 将每日数据按所在交易时间段进行分组
    def _group(_df, _axed_trading_hour, _start_time, _end_time):
        _df = _df.reset_index()
        _df['time'] = _df['datetime'].dt.time
        for i in range(len(_axed_trading_hour)):
            _df.loc[(_df['time'] >= _start_time[i]) & (_df['time'] <= _end_time[i]), 'group'] = i
        return _df.group.value_counts()

    minutes_count = pd.DataFrame(columns=list(range(len(axed_trading_hour))))
    for i in range(len(trading_date)):
        df = contract[contract.trading_date == trading_date[i]]
        counts = _group(df, axed_trading_hour, start_time, end_time)
        counts.name = trading_date[i]
        minutes_count = minutes_count.append(counts)
    # 计算每日各时间段缺少多少分钟的交易数据
    minutes_count = minutes_count.fillna(0)
    minutes_count = minutes_count.rename(columns=column_name)
    minutes_diff = trading_minutes - minutes_count
    # 无夜盘的交易日夜盘时间段缺失数据量设置为0
    minutes_diff.loc[minutes_diff.index.isin(trade_cal_wo_ns), night_time] = 0
    # 如果当日没有任何数据缺失，剔除当日
    minutes_diff = minutes_diff.replace(0, np.nan).dropna(how='all')
    if len(minutes_diff) > 0:
        return minutes_diff
    else:
        return None
    return minutes_diff


def min_data_check_by_contract_w_change(filepath: str, change_in_trading_hours: DataFrame,
                                        trade_cal_wo_ns: pd.DatetimeIndex) -> DataFrame:
    """
        Count the data missing cases for each contract
        If there are "extra data" not in trading hours, also return the "contract name"

        Logic:
            倒推出的交易时段是每日的，一个合约中可能存在多套交易时段。按照 change_in_trading_hours 中所记录的每个品种
            交易时段发生变更的日期将合约中的交易数据分为多段，每段计算每个交易时段的总数据量、实际数据量、缺少数据量。
            再将每段的数据缺失信息整合在一起，形成整个合约中的数据缺失信息。

        Parameters
        ----------
        filepath: str
            the filepath of the file for check

        change_in_trading_hours: DataFrame
            全品种交易时段变动的日期，及变动前后交易时段。
            增加了上市首日的信息，因此，所有品种都会在change_in_trading_hours有值
            对于上市首日，trading_hours_before设置为第一天的交易时段，即 “前后一致”

        trade_cal_wo_ns： DatetimeIndex
            因假期和特殊规定没有夜盘的交易日

        Returns
        -------
            Tuple: (minutes_diff_contract, extra_data)
            minutes_diff_contract:
                minutes_diff_contract: DataFrame
                    if there's any missing case on specific date, minutes_diff_contract will be returned
                        format:
                            index = 'date'
                            columns = specific 'trading_hour' session
                otherwise, return 'None'

            extra_data: str
                contract_name if have extra mins data
                otherwise 'None'

            thus, clean_complete_data will return (None,None)

    """
    contract = pd.read_pickle(filepath)
    contract.index = contract.index - pd.Timedelta('1minute')
    contract_name = Path(filepath).stem
    symbol = ''.join([i for i in contract_name if not i.isdigit()])
    extra_data = None
    # 获取该品种的交易时段变更时间，其中包含上市日期，所以每个品种都会有一个日期
    change_symbol = change_in_trading_hours[change_in_trading_hours.underlying_symbol == symbol]
    times_of_change = change_symbol.trading_date.values
    minutes_diff_list = []
    # 将整个合约的交易数据按交易时段变动日期分成几段
    # 获得每一套交易时段的开始、结束时间，对应分钟数
    for j in range(len(times_of_change) + 1):
        if j == 0:
            trading_hour = change_symbol.trading_hours_before.values[j]
        elif (j > 0) & (j < len(times_of_change)):
            trading_hour = change_symbol.trading_hours_before.values[j]
        else:
            trading_hour = change_symbol.trading_hours.values[j - 1]
        trading_hour = trading_hour.split(',')
        start_time = []
        end_time = []
        # 获取每一段交易时间的开始和结束时间
        for i in range(len(trading_hour)):
            t_start, delim, t_end = trading_hour[i].partition('-')
            t_start = (pd.to_datetime(t_start) - pd.Timedelta('1minute')).time()
            t_end = (pd.to_datetime(t_end) - pd.Timedelta('1minute')).time()
            start_time.append(t_start)
            # 将跨日的交易时间分成两段零点前后两段
            if t_end < t_start:
                t_end1 = pd.to_datetime('23:59').time()
                t_start1 = pd.to_datetime('00:00').time()
                start_time.append(t_start1)
                end_time.append(t_end1)
            end_time.append(t_end)
        trading_minutes = []
        axed_trading_hour = []
        night_time_mask = []
        # 获取每一段交易时间的分钟数
        for i in range(len(start_time)):
            time_diff = datetime.combine(datetime.today(), end_time[i]) - datetime.combine(datetime.today(),
                                                                                           start_time[i])
            # 因为min bar 使用的是边界方法，差值需要加1
            trading_minutes.append(time_diff.total_seconds() / 60 + 1)
            # 跨日的交易时间被分成两段，需要重新制作交易时间list
            axed_trading_hour.append('-'.join([str(start_time[i]), str(end_time[i])]))
            # 记录哪些交易时间是夜盘
            is_night = (start_time[i] > pd.to_datetime('18:00').time()) | (
                        start_time[i] < pd.to_datetime('09:00').time())
            night_time_mask.append(is_night)
        night_time = np.array(axed_trading_hour)[night_time_mask]
        # 获得每套交易时段对应的交易数据，即按交易时段改变的日期对整个合约的交易数据进行切割。
        # 变动日期为改变后的第一日，所以每段数据的结束日应该在变动日期前一天，下一段的开始日为变动日期当日
        # 对于包含在变动日期中的上市日期，该日期前不会有合约交易数据。
        if j == 0:
            contract_sliced = contract[contract.trading_date < times_of_change[j]]
        elif (j > 0) & (j < len(times_of_change)):
            contract_sliced = contract[
                (contract.trading_date >= times_of_change[j - 1]) & (contract.trading_date < times_of_change[j])]
        else:
            contract_sliced = contract[contract.trading_date >= times_of_change[j - 1]]
        trading_date = contract_sliced.trading_date.unique()
        column_name = dict(zip(list(range(len(axed_trading_hour))), axed_trading_hour))

        # 计算该段数据中每段交易时间的数据量
        # 将该段中每日数据按所在交易时间段进行分组
        def _group(_df, _axed_trading_hour, _start_time, _end_time):
            _df = _df.reset_index()
            _df['time'] = (_df['datetime']).dt.time
            for i in range(len(_axed_trading_hour)):
                _df.loc[(_df['time'] >= _start_time[i]) & (_df['time'] <= _end_time[i]), 'group'] = i
            return _df.group.value_counts()

        minutes_count = pd.DataFrame(columns=list(range(len(axed_trading_hour))))
        for i in range(len(trading_date)):
            df = contract_sliced[contract_sliced.trading_date == trading_date[i]]
            counts = _group(df, axed_trading_hour, start_time, end_time)
            counts.name = trading_date[i]
            minutes_count = minutes_count.append(counts)
        if minutes_count.sum().sum() < len(contract_sliced):
            print(contract_name + '多出数据')
            extra_data = contract_name

        # 计算每日各时间段缺少多少分钟的交易数据
        minutes_count = minutes_count.fillna(0)
        minutes_count = minutes_count.rename(columns=column_name)
        minutes_diff = trading_minutes - minutes_count
        # 无夜盘的交易日夜盘时间段缺失数据量设置为0
        minutes_diff.loc[minutes_diff.index.isin(trade_cal_wo_ns), night_time] = 0
        # 如果当日没有任何数据缺失，剔除当日
        minutes_diff = minutes_diff.replace(0, np.nan).dropna(how='all')
        minutes_diff_list.append(minutes_diff)
    minutes_diff_contract = pd.concat(minutes_diff_list)
    minutes_diff_contract['contract'] = contract_name
    if len(minutes_diff_contract) == 0:
        return None, extra_data
    else:
        return minutes_diff_contract, extra_data


def commodity_symbol_adjust(df: DataFrame, symbol_col: str) -> DataFrame:
    """
        adjust the underlying_symbol of the dataframe and return a DataFrame with adjusted underlying_symbol

    Parameters
    ----------
    df: DataFrame

    symbol_col: str
        'underlying_symbol'

    Returns
    -------
    a_df: DataFrame

    """
    a_df = df.copy()
    a_df[symbol_col] = a_df[symbol_col].replace('WT','PM').replace('WS', 'WH').replace('ER', 'RI').\
        replace('ME', 'MA').replace('TC', 'ZC').replace('RO', 'OI').replace("S", "A")

    return a_df


def commodity_contractName_adjust(df: DataFrame, symbol_col: str, contract_col: str) -> DataFrame:
    """
        adjust the contractName of the dataframe and return a DataFrame with adjusted contractName
    Parameters
    ----------
    df: DataFrame

    symbol_col: str
        'underlying_symbol'

    contract_col: str
        'contract'

    Returns
    -------
    a_df: DataFrame

    """
    a_df = df.copy()
    a_df[contract_col] = a_df[contract_col].str.replace('WT', 'PM').str.replace('WS', 'WH').str.replace('ER','RI').\
        str.replace('ME', 'MA').str.replace('TC', 'ZC').str.replace('RO', 'OI')
    a_df.loc[a_df[symbol_col]=='A', contract_col] = a_df.loc[a_df[symbol_col]=='A', contract_col].str.replace("S", "A")

    return a_df


def commodity_rename_full_adjust(df: DataFrame, symbol_col: str, contract_col: str) -> DataFrame:
    """
        readjust the data with both symbol(first-step) and contractName(second-step) modification

    Parameters
    ----------
    df: DataFrame

    symbol_col: str
        'underlying_symbol'

    contract_col: str
        'contract'

    Returns
    -------
    a_df: DataFrame

    """
    df1 = commodity_symbol_adjust(df, symbol_col=symbol_col)
    df2 = commodity_contractName_adjust(df1, symbol_col=symbol_col, contract_col=contract_col)

    return df2


def rename_to_al_format(o_data):
    """
        DataFrame or Series same as o_data with modified multi-indexed as ('date', 'asset')

    Parameters
    ----------
    o_data: DataFrame or Series, multi-indexed by ('datetime', 'underlying_symbol')

    Returns
    -------
            DataFrame or Series same as o_data with modified multi-indexed as ('date', 'asset')
    """
    return o_data.rename(index={'datetime': 'date', 'underlying_symbol': 'asset'})


# %%
if __name__ == '__main__':
    # nan_test test
    factor_trial = pd.read_pickle(r'factor_trial.pkl')
    summary = nan_test(factor_trial)
