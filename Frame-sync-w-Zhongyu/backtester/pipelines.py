#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/18 22:28:48
# @Author  : Michael_Liu @ QTG
# @File    : pipelines
# @Software: PyCharm

from data_manager.BasicsDataManager import BasicsDataManager
from data_manager.DailyDataManager import DailyDataManager
from data_manager.IndustryDataManager import IndustryDataManager
from data_manager.CommodityPoolManager import CommodityPoolManager
from data_manager.ContDominantDataManager import ContDominantDataManager
from data_manager.FactorDataManager import FactorDataManager

import backtester.weights_utils as wu
import qtgQuants.alphalens as al
import qtgQuants.pyfolio as pf

import utils.clean_and_test as ct
import utils.metrics_utils as mu
import utils.analysis_and_plot as ap
import empyrical as ep
import importlib

# %% create data_managers
bdm = BasicsDataManager()
ddm = DailyDataManager()
idm = IndustryDataManager()
cpm = CommodityPoolManager()
cddm = ContDominantDataManager()
fdm = FactorDataManager()

# %% load factor
# pivoted, wide-form, indexed by datetime, columns underlying_symbol
factor_instace = fdm.get_factor(group='CarryFactor', name='MainNearFactor3',
                                price='close',
                                window=1,
                                delist=0,  # don't allow to include delist month
                                filterby='volume',
                                others='near')
factor_value = factor_instace.get_factor_value()

# %% load group info
# dict[symbol, group]
group = idm.get_symbol_industry_map(group='actual_industry', name='actual_five_industry')

# %% load commodity_pool_mask
# pivoted, wide-form, indexed by datetime, columns underlying_symbol
commodity_pool_instance = cpm.get_commodity_pool(group='DynamicPool', name='DynamicPool1',
                                                 q1=0.3, q2=0.4,
                                                 window=126,
                                                 warm_up_days=63,
                                                 entry_only=0,
                                                 min_volume=10000,
                                                 exclusionList='ExclusionList1')  # check ExclusionList1 and modify it if necessary
commodity_pool_mask = commodity_pool_instance.get_commodity_pool_value()

# %% create exclusion_list and create final group_list_after_exclusion
exclusion_list = ['BB', 'CY', 'FB', 'JR', 'RI', 'RS', 'WH', 'WR'] + ['IF', 'IH', 'IC', 'T', 'TS', 'TF']
group_list_after_exclusion = set(group.keys()) - set(exclusion_list)
symbol_list = factor_instace.get_symbol_list()  # original full symbol list
# TODO: check to set grouping gradually with new symbols
symbol_list_wo_grouping = set(symbol_list) - set(group.keys())

# %% filter out symbols for commodity_pool_mask, wide-form
commodity_pool_mask = commodity_pool_mask.filter(items=group_list_after_exclusion)

# %% filter factor_value by commodity_pool_mask and change to long-form with dropna
factor_value = factor_value[commodity_pool_mask]
factor_value = factor_value.stack()

# %% factor_value test and
ct.nan_test(factor_wide=factor_value.unstack())

# %% load prices
# normal case, for factor can be trade at the close
# pivoted, wide-form, indexed by datetime, columns underlying_symbol
prices = cddm.get_cont_dominant_data_by_field(contract='main', price='close', rebalance_days=1,
                                              style='k', field='continuous_price')

# # factor can only be created after close
# # get original cont_dominant_data
# cont_dominant_data = cddm.get_cont_dominant_data(contract='main', price='close', rebalance_days=1, style='k')
# # adjust the price from the cont_dominant_data
# prices = cddm.get_cont_dominant_price_series(cont_dominant_data, field='open')
# # [IMPORTANT]: shift -1
# prices = prices.shift(-1)

# %% parameters set for analysis
periods = (1, 5, 10)

# %% create alphalens factor_data
# alphalens will rename the index and columns by itself independently with input 'naming'
factor_data = al.utils.get_clean_factor_and_forward_returns(factor=factor_value,  # has to be long-form
                                                            prices=prices,  # has to be wide-form
                                                            groupby=group,
                                                            binning_by_group=False,  # TODO: within group analysis
                                                            quantiles=5,  # default 5 groups
                                                            bins=None,
                                                            periods=periods,  # default multi-periods
                                                            filter_zscore=20,
                                                            groupby_labels=None,
                                                            max_loss=0.35,
                                                            zero_aware=False,
                                                            cumulative_returns=True)

# saving factor_data for repeat analysis
# factor_data.to_pickle()

# set freq for following explicitly usage
freq = factor_data.index.levels[0].freq

# %% Basic analysis of the factor_data by alphalens, will create both universal
# equally and factor-value weighted analysis
# TODO: create qtg full_tear_sheet
al.tears.create_full_tear_sheet(factor_data=factor_data,
                                long_short=True,
                                group_neutral=False,
                                by_group=True)

# al.tears.create_full_tear_sheet(factor_data=factor_data,
#                                 long_short=True,
#                                 group_neutral=False,
#                                 by_group=True,
#                                 equal_weight=False)
#
# al.tears.create_full_tear_sheet(factor_data=factor_data,
#                                 long_short=True,
#                                 group_neutral=False,
#                                 by_group=True,
#                                 equal_weight=True)  # overlapping with above

# %% Advanced analysis with specific setting
weights_value = wu.equal_weights_from_quantiles(factor_data=factor_data,
                                                long_quantiles=[5],
                                                short_quantiles=[1])

# construct cumulative return with weights by positions
# 1. set freq explicitly
# 2. set leverage_up explicitly
# 3. set ret_pos explicitly
cumulative_ret, positions = al.performance.cumulative_returns_precisely_w_positions(returns=factor_data['1D'],
                                                                                    weights=weights_value,
                                                                                    period='5D',
                                                                                    freq=freq,
                                                                                    leverage_up=True,
                                                                                    ret_pos=True)
# plot cum-ret series
ap.plot_return_series(cum_rets=cumulative_ret)

# get ann turnover by positions
mu.get_ann_turnover_with_positions(factor_data=factor_data, positions=positions)

# show general stats
ap.show_simple_stats(ret=cumulative_ret.pct_change().fillna(0))

# decomposition and attribution
wu.get_yearly_weights(weights=weights_value, group=group, level='asset')
wu.get_yearly_weights(weights=weights_value, group=group, level='group')
# example: weight/exposure plot
wu.weights_plot(weights=weights_value, group=group, group_name='黑色')

# %% pyfolio analysis, temporarily not useful, too much redundancy
# rets, benchmark_rets = mu.prepare_pyfolio_input(factor_data=factor_data,
#                                                 cumulative_ret=cumulative_ret,
#                                                 benchmark_period='5D')
#
# pf.tears.create_returns_tear_sheet(returns=rets,
#                                    benchmark_rets=benchmark_rets)

# %%
# ---------------------------------------------------------------------
# analysis 1: N independent sub-portfolio metrics calculation
period_metrics_analysis = mu.get_period_metrics_analysis(factor_data=factor_data,
                                                         weights=weights_value,
                                                         period='5D',
                                                         group=group,
                                                         level='all',
                                                         detail=True)

# ----------------------------------------------------------------------
# analysis 2: N independent sub-portfolio attribution
result = mu.detail_and_all_level_decomp(ret=factor_data['1D'],
                                        weights=weights_value,
                                        period='5D',
                                        group=group)

# example: group decomposition
mu.cum_ret_by_sub_level(result=result, group=group,
                        period='5D', level='group', group_name='有色_贵金属')

# example: group long-short decomposition
mu.cum_ret_by_ls(result=result, group=group, period='5D', level='group', group_name='黑色')
