#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/16 18:45:15
# @Author  : Michael_Liu @ QTG
# @File    : performance
# @Software: PyCharm

import pandas as pd
import numpy as np
import warnings

from pandas.tseries.offsets import BDay
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from . import utils


# %% returns related part -------------------------------------------------------------
def factor_weights(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False):
    """
    [ORIGINAL]
    [COMMENTS]: equal_weight or not (factor_value weight), the total (gross/absolute) weights for each date
     (groupby 'date') only will be 1 no matter demeaned or group_adjust

    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    demeaned : bool == long_short
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio

    group_adjust : bool == group_neutral
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.

    equal_weight : bool, optional
        if True the assets will be equal-weighted instead of factor-weighted
        [CAUTION]: demeaned on median or mean
        If demeaned is True then the factor universe will be split into two
        equal sized groups (by median), top assets with positive weights and bottom assets
        with negative weights

    Returns
    -------
    returns : pd.Series
        Assets weighted by factor value.
    """

    def to_weights(group, _demeaned, _equal_weight):
        """
            apply function for groupby case
            weights sum up to 1 within group

        Parameters
        ----------
        group: candidate groupby

        _demeaned
        _equal_weight

        Returns
        -------
            pd.Series: with whole weights sum up to 1

        """
        if _equal_weight:
            group = group.copy()

            if _demeaned:
                # top assets positive weights, bottom ones negative
                group = group - group.median()

            negative_mask = group < 0
            group[negative_mask] = -1.0
            positive_mask = group > 0
            group[positive_mask] = 1.0

            if _demeaned:
                # positive weights must equal negative weights
                if negative_mask.any():
                    group[negative_mask] /= negative_mask.sum()
                if positive_mask.any():
                    group[positive_mask] /= positive_mask.sum()

        elif _demeaned:
            group = group - group.mean()

        return group / group.abs().sum()  # ensure weights sum up to 1 absolutely

    # MODIFIED
    # grouper = [factor_data.index.get_level_values('date')]
    grouper = ['date']
    if group_adjust:
        grouper.append('group')

    weights = factor_data.groupby(grouper)['factor'] \
        .apply(to_weights, demeaned, equal_weight)

    if group_adjust:
        # now, each group is sum up to 1 for specific date
        # the weights should be adjust further by group numbers
        weights = weights.groupby(level='date').apply(to_weights, False, False)

    return weights


def factor_returns(factor_data,
                   demeaned=True,
                   group_adjust=False,
                   equal_weight=False,
                   by_asset=False,
                   weights_df=None):
    """
    [MODIFIED]: bring-in weights_df input to saving time
    [COMMENTS]: core part is func. "factor_weights"
    Computes period wise returns for portfolio weighted by
    factor values (or other complex adjusted weights see factor_weights)

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    demeaned : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation

    group_adjust : bool
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation

    equal_weight : bool, optional
        Control how to build factor weights
        -- see performance.factor_weights for a full explanation

    by_asset: bool, optional
        [NOTE]: sometime we need to save assets return independently to analyze factor performance for
        each assets, instead of combine the returns as whole for universal analysis
        If True, returns are reported separately for each asset.

    Returns
    -------
    returns : pd.DataFrame
        "Period" wise factor returns, whose "periods" coming from factor_data info
    """

    if weights_df is None:
        weights = \
            factor_weights(factor_data=factor_data,
                           demeaned=demeaned,
                           group_adjust=group_adjust,
                           equal_weight=equal_weight)
    else:
        weights = weights_df

    weighted_returns = \
        factor_data[utils.get_forward_returns_columns(factor_data.columns)] \
            .multiply(weights, axis=0)  # only weighted but not sum up

    if by_asset:
        returns = weighted_returns
    else:
        returns = weighted_returns.groupby(level='date').sum()

    return returns


def factor_returns_by_weight(factor_data, weights, by_asset=False):
    """
        simple version of factor_returns with weights as input
        See Also: al.perf.factor_returns

    Parameters
    ----------
    factor_data

    weights: Series, multi-indexed

    by_asset: bool, optional
        [NOTE]: sometime we need to save assets return independently to analyze factor performance for
        each assets, instead of combine the returns as whole for universal analysis
        If True, returns are reported separately for each asset.

    Returns
    -------
    returns : pd.DataFrame
        "Period" wise factor returns, whose "periods" coming from factor_data info
    """
    weighted_returns = \
        factor_data[utils.get_forward_returns_columns(factor_data.columns)] \
            .multiply(weights, axis=0)  # only weighted but not sum up

    if by_asset:
        returns = weighted_returns
    else:
        returns = weighted_returns.groupby(level='date').sum()

    return returns


def mean_return_by_quantile(factor_data,
                            by_date=False,
                            by_group=False,
                            demeaned=True,
                            group_adjust=False):
    """
    [MODIFIED]
    Computes mean returns for factor quantiles across
    provided forward returns columns(effective periods).

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    by_date : bool
        [IMPORTANT]: useful for cumulative returns for each quantile

        If True, compute quantile bucket returns separately for each date.

    by_group : bool
        [IMPORTANT]: useful for group analysis of the quantize grouping
                    and be combined with [by_date] to create date-by-date data

        If True, compute quantile bucket returns separately for each group.

    demeaned : bool
        Compute demeaned mean returns (long short portfolio)

    group_adjust : bool
        Returns demeaning will occur on the group level.

    [CAUTIONs]： 此处 group_adjust 和 demeaned 写成了互斥关系， 使用的时候要注意！！！

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile / group / date
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile / group / date

    [IMPORTANT]:
        if by_date: multiindex added 'date' index level
        if by_group: multiindex added 'group' index level
        thus, if both true, it will have 3 levels: quantile, date, group
    """

    if group_adjust:
        # TODO: check [SIMPLIFIED]
        # grouper = [factor_data.index.get_level_values('date')] + ['group']
        grouper = ['date', 'group']
        factor_data = utils.demean_forward_returns(factor_data, grouper)
    elif demeaned:
        factor_data = utils.demean_forward_returns(factor_data)
    else:
        factor_data = factor_data.copy()

    # TODO: check [SIMPLIFIED]
    # grouper = ['factor_quantile', factor_data.index.get_level_values('date')]
    grouper = ['factor_quantile', 'date']

    if by_group:
        grouper.append('group')

    group_stats = factor_data.groupby(grouper)[
        utils.get_forward_returns_columns(factor_data.columns)] \
        .agg(['mean', 'std', 'count'])

    # TODO: check [SIMPLIFIED]
    # mean_ret = group_stats.T.xs('mean', level=1).T
    mean_ret = group_stats.xs('mean', axis=1, level=1)

    if not by_date:
        # TODO: check [SIMPLIFIED]
        # grouper = [mean_ret.index.get_level_values('factor_quantile')]
        grouper = ['factor_quantile']
        if by_group:
            # TODO: check [SIMPLIFIED]
            # grouper.append(mean_ret.index.get_level_values('group'))
            grouper.append('group')
        group_stats = mean_ret.groupby(grouper) \
            .agg(['mean', 'std', 'count'])
        # 此时 ’mean‘ 是 daily_return（本身是当天的groupby mean） 的 mean
        # 此时 ’std‘ 是 daily_return（本身是当天的groupby mean） 的 std
        # 此时，是daily_return 的 case 数（天数）
        # TODO: check [SIMPLIFIED]
        # mean_ret = group_stats.T.xs('mean', level=1).T
        mean_ret = group_stats.xs('mean', axis=1, level=1)

    # TODO: check [SIMPLIFIED]
    # std_error_ret = group_stats.T.xs('std', level=1).T \
    #                 / np.sqrt(group_stats.T.xs('count', level=1).T)
    std_error_ret = group_stats.xs('std', axis=1, level=1) / np.sqrt(group_stats.xs('count', axis=1, level=1))
    # by_date=False 时， daily_return 的 日化标准差

    return mean_ret, std_error_ret


def factor_alpha_beta(factor_data,
                      returns=None,
                      demeaned=True,
                      group_adjust=False,
                      equal_weight=False):
    """
    [CLARIFIED]
    Compute the alpha (excess returns), [alpha t-stat (alpha significance)],    # old version?
    and beta (market exposure) of a factor. A regression is run with
    "the period wise factor universe mean return" (benchmark) as the independent variable
    and mean period wise return from a portfolio weighted by factor values (according to equal_weight)
    as the dependent variable.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    returns : pd.DataFrame, optional
        Period wise factor returns. If this is None then it will be computed
        with 'factor_returns' function and the passed flags: 'demeaned',
        'group_adjust', 'equal_weight'

    [CAUTION]:  demeaned, group_adjust, equal_weight are all for rebuilding returns from factor_data
    demeaned : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    group_adjust : bool
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation
    equal_weight : bool, optional
        Control how to build factor returns used for alpha/beta computation
        -- see performance.factor_return for a full explanation

    Returns
    -------
    alpha_beta : pd.DataFrame, single index:
        index = ['Ann. alpha', 'beta']
        A list containing the alpha, beta, [a t-stat(alpha)]
        for the given factor and forward returns.
        columns as the input returns for different period
    """
    suffix = '(Equal_weighted)' if equal_weight else '(Factor_weighted)'

    if returns is None:
        returns = \
            factor_returns(factor_data=factor_data,
                           demeaned=demeaned,
                           group_adjust=group_adjust,
                           equal_weight=equal_weight,
                           by_asset=False)  # by_asset as default to be False,
        # thus, single indexed as 'date'

    universe_ret = factor_data.groupby(level='date')[
        utils.get_forward_returns_columns(factor_data.columns)] \
        .mean().loc[returns.index]  # single index as 'date'

    if isinstance(returns, pd.Series):  # if returns as series, treat it as [default] the basic period series
        # if returns has series.name, treat it as the period
        if returns.name is None:
            # default
            returns.name = universe_ret.columns.values[0]
        returns = pd.DataFrame(returns)

    alpha_beta = pd.DataFrame()
    for period in returns.columns.values:
        x = universe_ret[period].values
        y = returns[period].values
        x = add_constant(x)

        reg_fit = OLS(y, x).fit()
        try:
            alpha, beta = reg_fit.params
        except ValueError:
            alpha_beta.loc['Ann. alpha ' + suffix, period] = np.nan
            alpha_beta.loc['beta ' + suffix, period] = np.nan
        else:
            freq_adjust = pd.Timedelta('252Days') / pd.Timedelta(period)

            alpha_beta.loc['Ann. alpha ' + suffix, period] = \
                (1 + alpha) ** freq_adjust - 1
            alpha_beta.loc['beta ' + suffix, period] = beta

    return alpha_beta


def compute_mean_returns_spread(mean_returns,
                                upper_quant,
                                lower_quant,
                                std_err=None):
    """
    [COMMENTED]: std calculation may have some challenged assumption

    Computes the difference between the mean returns of
    two quantiles. Optionally, computes the standard error
    of this difference.

    Parameters
    ----------
    mean_returns : pd.DataFrame, multiindexed [date, quantile]
        DataFrame of mean period wise returns by quantile.
        MultiIndex containing date and quantile.
        See mean_return_by_quantile.

    upper_quant : int
        Quantile of mean return from which we
        wish to subtract lower quantile mean return.

    lower_quant : int
        Quantile of mean return we wish to subtract
        from upper quantile mean return.

    std_err : pd.DataFrame, optional

        Period wise standard error in mean return by quantile.
        Takes the same form as mean_returns.

    Returns
    -------
    mean_return_difference : pd.DataFrame, single indexed as 'date'
        Period wise difference in quantile returns.

    joint_std_err : pd.DataFrame, single indexed as 'date'
        [Comments]: in whole alphalens presentation/project, this output has no application
        [Assumption]： with challenge: the std_error in different quantiles of specific day is independent
        Period wise standard error of the difference in quantile returns.
        if std_err is None, this will be None
    """

    mean_return_difference = mean_returns.xs(upper_quant,
                                             level='factor_quantile') \
                             - mean_returns.xs(lower_quant, level='factor_quantile')

    if std_err is None:
        joint_std_err = None
    else:
        std1 = std_err.xs(upper_quant, level='factor_quantile')
        std2 = std_err.xs(lower_quant, level='factor_quantile')
        joint_std_err = np.sqrt(std1 ** 2 + std2 ** 2)  # TODO: challenged

    return mean_return_difference, joint_std_err


# %% multi-period generalized operations --------------------------
def cumulative_returns(returns, period, freq=None):
    """
    # TODO: rewritten to be more accurate
    [OLD VERSION]: general multi-periods rebalancing version
    [COMMENTED]
    Builds cumulative returns from 'period' returns. This function simulates
    the cumulative effect that a series of gains or losses (the 'returns')
    have on an original amount of capital over a period of time.

    if F is the frequency at which returns are computed (e.g. 1 day if
    'returns' contains daily values) and N is the period for which the retuns
    are computed (e.g. returns after 1 day, 5 hours or 3 days) then:
    - if N <= F the cumulative retuns are trivially computed as Compound Return
    - if N > F (e.g. F 1 day, and N is 3 days) then the returns overlap and the
      cumulative returns are computed building and averaging N interleaved sub
      portfolios (started at subsequent periods 1,2,..,N) each one rebalancing
      every N periods. This correspond to an algorithm which trades the factor          # TODO: multi-starting-date smoothing
      every single time it is computed, which is statistically more robust and
      with a lower volatity compared to an algorithm that trades the factor
      every N periods and whose returns depend on the specific starting day of
      trading.

    Also note that when the factor is not computed at a specific frequency, for
    exaple a factor representing a random event, it is not efficient to create
    multiples sub-portfolios as it is not certain when the factor will be
    traded and this would result in an underleveraged portfolio. In this case           # TODO： mutli-starting-date is pre-setted N equal portfolios
    the simulated portfolio is fully invested whenever an event happens and if
    a subsequent event occur while the portfolio is still invested in a
    previous event then the portfolio is rebalanced and split equally among the         # TODO: will be ugly, non-linear composition
    active events.

    Parameters
    ----------
    returns: pd.Series or pd.DataFrame
        pd.Series:
        containing factor 'period' forward returns, the index                 # multi-periods forward returns not rate-returns
        contains timestamps at which the trades are computed and the values
        correspond to returns after 'period' time
        pd.DataFrame:
        containing factor '1D'

    period: pandas.Timedelta or string
        Length of period for which the returns are computed (1 day, 2 mins,
        3 hours etc). It can be a Timedelta or a string in the format accepted
        by Timedelta constructor ('1 days', '1D', '30m', '3h', '1D1h', etc)

    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        returns.index.freq (infer) will be used

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957
    """

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = returns.index.freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar",
                      UserWarning)

    #
    # returns index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # Cumulative returns will use 'full_idx' index,because we want a cumulative
    # returns value for each entry in 'full_idx'
    #
    trades_idx = returns.index.copy()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period,
                                                      freq)  # TODO: should be extended one more period
    full_idx = trades_idx.union(returns_idx)  # same as func - positions

    #
    # Build N sub_returns from the single returns Series. Each sub_retuns
    # stream will contain non-overlapping returns.
    # In the next step we'll compute the portfolio returns averaging the
    # returns happening on those overlapping returns streams
    #
    sub_returns = []
    while len(trades_idx) > 0:

        #
        # select non-overlapping returns starting with first timestamp in index
        #
        sub_index = []
        next = trades_idx.min()
        while next <= trades_idx.max():
            sub_index.append(next)
            next = utils.add_custom_calendar_timedelta(next, period, freq)
            # make sure to fetch the next available entry after 'period'
            try:
                i = trades_idx.get_loc(next, method='bfill')  # create non-overlapping rebalancing dates
                next = trades_idx[i]
            except KeyError:
                break

        sub_index = pd.DatetimeIndex(sub_index, tz=full_idx.tz)  # tz will influence normalize and others
        subret = returns[sub_index]

        # make the index to have all entries in 'full_idx'
        subret = subret.reindex(full_idx)  # will create some NaN's

        #
        # compute intermediate returns values for each index in subret that are
        # in between the timestaps at which the factors are computed and the
        # timestamps at which the 'period' returns actually happen                  # [compoundingly average the period-return to each day] - geometrically
        #
        for pret_idx in reversed(sub_index):  #

            pret = subret[pret_idx]

            # get all timestamps between factor computation and period returns
            pret_end_idx = \
                utils.add_custom_calendar_timedelta(pret_idx, period, freq)
            slice = subret[(subret.index > pret_idx) & (
                    subret.index <= pret_end_idx)].index  # dates inbetween

            if pd.isnull(pret):
                continue

            def rate_of_returns(ret, period):
                return ((np.nansum(ret) + 1) ** (1. / period)) - 1  # geometrically averaging

            # compute intermediate 'period' returns values, note that this also
            # moves the final 'period' returns value from trading timestamp to
            # trading timestamp + 'period'
            for slice_idx in slice:
                sub_period = utils.diff_custom_calendar_timedeltas(
                    pret_idx, slice_idx, freq)
                subret[slice_idx] = rate_of_returns(pret,
                                                    period / sub_period)  # fill geometrically averaging to different days inbetween

            subret[pret_idx] = np.nan

            # transform returns as percentage change from previous value
            subret[slice[1:]] = (subret[slice] + 1).pct_change()[
                slice[1:]]  # precise operation taking into account of the date-missing in trades_idx.date

        sub_returns.append(subret)
        trades_idx = trades_idx.difference(sub_index)

    #
    # Compute portfolio cumulative returns averaging the returns happening on
    # overlapping returns streams.
    #
    sub_portfolios = pd.concat(sub_returns, axis=1)
    portfolio = pd.Series(index=sub_portfolios.index)

    for i, (index, row) in enumerate(sub_portfolios.iterrows()):

        # check the active portfolios, count() returns non-nans elements
        active_subfolios = row.count()

        # fill forward portfolio value
        portfolio.iloc[i] = portfolio.iloc[i - 1] if i > 0 else 1.

        if active_subfolios <= 0:
            continue

        # current portfolio is the average of active sub_portfolios
        portfolio.iloc[i] *= (row + 1).mean(skipna=True)  # TODO: [IMPORTANT] N-equal subset

    return portfolio


def cumulative_returns_precisely(returns, weights, period, freq=None):
    """
    # TODO: rewritten to be more accurate
    [OLD VERSION]: general multi-periods rebalancing version
    [COMMENTED]
    Builds cumulative returns from 'period' returns. This function simulates
    the cumulative effect that a series of gains or losses (the 'returns')
    have on an original amount of capital over a period of time.

    if F is the frequency at which returns are computed (e.g. 1 day if
    'returns' contains daily values) and N is the period for which the retuns
    are computed (e.g. returns after 1 day, 5 hours or 3 days) then:
    - if N <= F the cumulative retuns are trivially computed as Compound Return
    - if N > F (e.g. F 1 day, and N is 3 days) then the returns overlap and the
      cumulative returns are computed building and averaging N interleaved sub
      portfolios (started at subsequent periods 1,2,..,N) each one rebalancing
      every N periods. This correspond to an algorithm which trades the factor          # TODO: multi-starting-date smoothing
      every single time it is computed, which is statistically more robust and
      with a lower volatity compared to an algorithm that trades the factor
      every N periods and whose returns depend on the specific starting day of
      trading.

    Also note that when the factor is not computed at a specific frequency, for
    example a factor representing a random event, it is not efficient to create
    multiples sub-portfolios as it is not certain when the factor will be
    traded and this would result in an underleveraged portfolio. In this case           # TODO： mutli-starting-date is pre-setted N equal portfolios
    the simulated portfolio is fully invested whenever an event happens and if
    a subsequent event occur while the portfolio is still invested in a
    previous event then the portfolio is rebalanced and split equally among the         # TODO: non-linear composition, conflict of independency
    active events.

    Parameters
    ----------
    returns: pd.Series, mutli-indexed: ('datetime', 'asset')
        pd.Series: containing '1D' forward returns, the index
        contains timestamps at which the trades are computed and the values
        correspond to returns after '1D' time for each asset

    weights: pd.Series, mutli-indexed: ('datetime', 'asset')
        pd.Series: the assets daily weights from factor_data operation
        for example: weighted by factor value daily

    period: pandas.Timedelta or string
        Length of period for which the returns are computed (1 day, 2 mins,
        3 hours etc). It can be a Timedelta or a string in the format accepted
        by Timedelta constructor ('1 days', '1D', '30m', '3h', '1D1h', etc)

    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        returns.index.freq (infer) will be used

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957
    """

    # returns reformat to single indexed by 'datetime', columned by 'asset'
    returns = returns.unstack()
    # [IMPORTANT]: forward 1D return to current 1D return
    returns = returns.shift(1)

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = returns.index.freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar",
                      UserWarning)

    #
    # returns index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # Cumulative returns will use 'full_idx' index,because we want a cumulative
    # returns value for each entry in 'full_idx'
    #

    trades_idx = returns.index.copy()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period,
                                                      freq)  # TODO: should be extended one more period
    full_idx = trades_idx.union(returns_idx)

    #
    # Build N sub_returns from the single returns Series. Each sub_retuns
    # stream will contain non-overlapping returns.
    # In the next step we'll compute the portfolio precise returns
    # instead of averaging the returns happening on those overlapping returns streams
    #
    sub_returns = []
    while len(trades_idx) > 0:

        #
        # select non-overlapping returns starting with first timestamp in index
        #
        sub_index = []
        next = trades_idx.min()
        while next <= trades_idx.max():
            sub_index.append(next)
            next = utils.add_custom_calendar_timedelta(next, period, freq)
            # make sure to fetch the next available entry after 'period'
            try:
                i = trades_idx.get_loc(next, method='bfill')  # create non-overlapping rebalancing dates
                next = trades_idx[i]
            except KeyError:
                break

        sub_index = pd.DatetimeIndex(sub_index, tz=full_idx.tz)  # tz will influence normalize and others

        # make the index to have all entries in 'full_idx'
        subret = pd.Series(0, index=full_idx)  # initial a 0 returns

        #
        # compute intermediate returns values for each index in subret that are
        # in between the timestaps at which the factors are computed and the
        # timestamps at which the 'period' returns actually happen                  # [compoundingly average the period-return to each day] - geometrically
        #
        for pret_idx in reversed(sub_index):  #

            # get all timestamps between factor computation and period returns
            pret_end_idx = \
                utils.add_custom_calendar_timedelta(pret_idx, period, freq)
            slice = subret[(subret.index > pret_idx) & (
                    subret.index <= pret_end_idx)].index  # dates inbetween

            # TODO: different from the original geometric averaging
            def slice_cumulative_returns(slice_start, weights, ret, slice_end):
                """
                    calculate the precise instead of geometric-averaging daily cumulative return
                    for the sub-slice-period

                Parameters
                ----------
                slice_start
                weights:
                    constant weight for the rebalancing day
                ret:
                    daily current return for the corresponding asset
                slice_end:
                    sub_slice_end index of the slice for sub_slice_cumulative

                Returns
                -------
                    scalar, portfolio cumulative return for sub-slice-period

                """
                return ((ret.loc[slice_start:slice_end] + 1).prod() - 1).multiply(weights).sum()

            # compute intermediate 'period' returns values, note that this also
            # moves the final 'period' returns value from trading timestamp to
            # trading timestamp + 'period'
            for slice_idx in slice:
                sub_period = utils.diff_custom_calendar_timedeltas(
                    pret_idx, slice_idx, freq)
                # TODO: different from the original geometric averaging
                pweight = weights.loc[pret_idx]
                # current return instead of forward return
                subret.loc[slice_idx] = slice_cumulative_returns(slice_start=slice[0],
                                                                 weights=pweight,
                                                                 ret=returns,
                                                                 slice_end=slice_idx)

            subret.loc[pret_idx] = np.nan

            # transform returns as percentage change from previous value
            subret[slice[1:]] = (subret[slice] + 1).pct_change()[slice[1:]]
            # precise operation taking into account of the date-missing in trades_idx.date
            # subret[slice[0]] keep still

        sub_returns.append(subret)
        trades_idx = trades_idx.difference(sub_index)

    #
    # Compute portfolio cumulative returns averaging the returns happening on
    # overlapping returns streams.
    #
    sub_portfolios = pd.concat(sub_returns, axis=1)
    portfolio = pd.Series(index=sub_portfolios.index)

    for i, (index, row) in enumerate(sub_portfolios.iterrows()):

        # check the active portfolios, count() returns non-nans elements
        active_subfolios = row.count()

        # fill forward portfolio value
        portfolio.iloc[i] = portfolio.iloc[i - 1] if i > 0 else 1.

        if active_subfolios <= 0:
            continue

        # current portfolio is the average of active sub_portfolios
        portfolio.iloc[i] *= (row + 1).mean(skipna=True)  # TODO: [IMPORTANT] N-equal subset

    return portfolio


def cumulative_returns_precisely_w_leverage(returns, weights, period, freq=None, leverage_up=True):
    """
    TODO: rewritten to be more accurate
    instead of using pct_change of cumulative return for subportfolio and then sum over subportfolio returns equally
    here, we use daily ret of each assets and corresponding weights similar to positions
    leverage_up will decide whether the portfolio will be leveraged up altogether along with the
    same leverage up in positions

    usually, the daily-sub-portfoilo return will not be different with the one in cumulative_returns_precisely too much,
    thus, the only difference will be caused by the leverage process if leverage_up==True

    See Also:
        cumulative_returns_precisely_w_leverage  vs  cumulative_returns_precisely
        cumulative returns_precisely  vs  cumulative returns
        cumulative returns:  rough idea of the factor
                                geometric mean + mean of independent(periodically rebalanced) sub-portfolio daily return
        cumulative returns precisely:
                                mean of accurate independent(periodically rebalanced) sub-portfolio daily return
        cumulative returns precisely with leverage:
                                mean of dependent(daily rebalanced) sub-portfolio daily return
        where, mean vs independence is conflict to implement/realize

    [OLD VERSION]: general multi-periods rebalancing version
    [COMMENTED]
    Builds cumulative returns from 'period' returns. This function simulates
    the cumulative effect that a series of gains or losses (the 'returns')
    have on an original amount of capital over a period of time.

    if F is the frequency at which returns are computed (e.g. 1 day if
    'returns' contains daily values) and N is the period for which the retuns
    are computed (e.g. returns after 1 day, 5 hours or 3 days) then:
    - if N <= F the cumulative retuns are trivially computed as Compound Return
    - if N > F (e.g. F 1 day, and N is 3 days) then the returns overlap and the
      cumulative returns are computed building and averaging N interleaved sub
      portfolios (started at subsequent periods 1,2,..,N) each one rebalancing
      every N periods. This correspond to an algorithm which trades the factor          # TODO: multi-starting-date smoothing
      every single time it is computed, which is statistically more robust and
      with a lower volatity compared to an algorithm that trades the factor
      every N periods and whose returns depend on the specific starting day of
      trading.

    Also note that when the factor is not computed at a specific frequency, for
    example a factor representing a random event, it is not efficient to create
    multiples sub-portfolios as it is not certain when the factor will be
    traded and this would result in an underleveraged portfolio. In this case           # TODO： mutli-starting-date is pre-setted N equal portfolios
    the simulated portfolio is fully invested whenever an event happens and if
    a subsequent event occur while the portfolio is still invested in a
    previous event then the portfolio is rebalanced and split equally among the         # TODO: non-linear composition, conflict of independency
    active events.

    Parameters
    ----------
    returns: pd.Series, mutli-indexed: ('datetime', 'asset')
        pd.Series: containing '1D' forward returns, the index
        contains timestamps at which the trades are computed and the values
        correspond to returns after '1D' time for each asset

    weights: pd.Series, mutli-indexed: ('datetime', 'asset')
        pd.Series: the assets daily weights from factor_data operation
        for example: weighted by factor value daily

    period: pandas.Timedelta or string
        Length of period for which the returns are computed (1 day, 2 mins,
        3 hours etc). It can be a Timedelta or a string in the format accepted
        by Timedelta constructor ('1 days', '1D', '30m', '3h', '1D1h', etc)

    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        returns.index.freq (infer) will be used

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957
    """

    if leverage_up:
        porfolio_weights, portfolio_leverage_adj = positions(weights=weights, period=period, freq=freq)
        portfolio_leverage_adj = portfolio_leverage_adj.shift(1, freq=freq)

    # returns reformat to single indexed by 'datetime', columned by 'asset'
    returns = returns.unstack()
    # [IMPORTANT]: forward 1D return to current 1D return
    returns = returns.shift(1)

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = returns.index.freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar",
                      UserWarning)

    #
    # returns index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # Cumulative returns will use 'full_idx' index,because we want a cumulative
    # returns value for each entry in 'full_idx'
    #

    trades_idx = returns.index.copy()
    # max index for daily return usage
    o_returns_idx_max = trades_idx.max()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period,
                                                      freq)  # TODO: should be extended one more period
    full_idx = trades_idx.union(returns_idx)

    #
    # Build N sub_returns from the single returns Series. Each sub_retuns
    # stream will contain non-overlapping returns.
    # In the next step we'll compute the portfolio precise returns
    # instead of averaging the returns happening on those overlapping returns streams
    #
    sub_returns = []
    while len(trades_idx) > 0:

        #
        # select non-overlapping returns starting with first timestamp in index
        #
        sub_index = []
        next = trades_idx.min()
        while next <= trades_idx.max():
            sub_index.append(next)
            next = utils.add_custom_calendar_timedelta(next, period, freq)
            # make sure to fetch the next available entry after 'period'
            try:
                i = trades_idx.get_loc(next, method='bfill')  # create non-overlapping rebalancing dates
                next = trades_idx[i]
            except KeyError:
                break

        sub_index = pd.DatetimeIndex(sub_index, tz=full_idx.tz)  # tz will influence normalize and others

        # make the index to have all entries in 'full_idx'
        subret = pd.Series(np.nan, index=full_idx)  # initial a 0 returns

        #
        # compute intermediate returns values for each index in subret that are
        # in between the timestaps at which the factors are computed and the
        # timestamps at which the 'period' returns actually happen                  # [compoundingly average the period-return to each day] - geometrically
        #
        for pret_idx in reversed(sub_index):  #

            # get all timestamps between factor computation and period returns
            pret_end_idx = \
                utils.add_custom_calendar_timedelta(pret_idx, period, freq)
            slice = subret[(subret.index > pret_idx) & (
                    subret.index <= pret_end_idx)].index  # dates inbetween

            def slice_daily_returns(slice_idx, weights, ret):
                return ret.loc[slice_idx].multiply(weights).sum()

            # compute intermediate 'period' returns values, note that this also
            # moves the final 'period' returns value from trading timestamp to
            # trading timestamp + 'period'
            for slice_idx in slice:
                sub_period = utils.diff_custom_calendar_timedeltas(
                    pret_idx, slice_idx, freq)
                # TODO: different from the original geometric averaging
                pweight = weights.loc[pret_idx]

                # if slice_idx out of bound, keep as 0 and skip
                if slice_idx > o_returns_idx_max:
                    continue

                # current return instead of forward return
                subret.loc[slice_idx] = slice_daily_returns(slice_idx=slice_idx,
                                                            weights=pweight,
                                                            ret=returns)

            subret.loc[pret_idx] = np.nan

            # transform returns as percentage change from previous value
            # subret[slice[1:]] = (subret[slice] + 1).pct_change()[slice[1:]]
            # precise operation taking into account of the date-missing in trades_idx.date
            # subret[slice[0]] keep still

        sub_returns.append(subret)
        trades_idx = trades_idx.difference(sub_index)

    #
    # Compute portfolio cumulative returns averaging the returns happening on
    # overlapping returns streams.
    #
    sub_portfolios = pd.concat(sub_returns, axis=1)

    portfolio = pd.Series(index=sub_portfolios.index)

    for i, (index, row) in enumerate(sub_portfolios.iterrows()):

        # check the active portfolios, count() returns non-nans elements
        active_subfolios = row.count()

        # fill forward portfolio value
        portfolio.iloc[i] = portfolio.iloc[i - 1] if i > 0 else 1.

        if active_subfolios <= 0:
            continue

        # current portfolio is the average of active sub_portfolios
        # TODO: [IMPORTANT] N-equal subset
        # TODO: leverage_up or not
        if leverage_up:
            portfolio.iloc[i] *= (portfolio_leverage_adj.loc[index] * row + 1).mean(skipna=True)
        else:
            portfolio.iloc[i] *= (row + 1).mean(skipna=True)

    return portfolio


def cumulative_returns_precisely_w_positions(returns, weights, period, freq=None, leverage_up=True, ret_pos: bool = False):
    """
    [New]
        directly inheritate from positions, and get corresponding cumulative returns

    Parameters
    ----------
    returns: pd.Series, mutli-indexed: ('datetime', 'asset')
        pd.Series: containing '1D' 【forward returns】, the index
        contains timestamps at which the trades are computed and the values
        correspond to returns after '1D' time for each asset

    weights: pd.Series, mutli-indexed: ('datetime', 'asset')
        pd.Series: the assets daily weights from factor_data operation
        for example: weighted by factor value daily

    period: pandas.Timedelta or string
        Length of period for which the returns are computed (1 day, 2 mins,
        3 hours etc). It can be a Timedelta or a string in the format accepted
        by Timedelta constructor ('1 days', '1D', '30m', '3h', '1D1h', etc)

    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        returns.index.freq (infer) will be used

    leverage_up: bool default as True
        whether adjust the sub/tot_weights to keep constant leverage as 1

    ret_pos: bool, default as False
        when set true, return positions to reduce recalculation

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957

    """
    daily_weights = positions(weights, period, freq=freq, leverage_up=leverage_up)[0]
    daily_returns = daily_weights * (returns.unstack())
    daily_returns = daily_returns.sum(axis=1)
    # 目前得到的是当天 portfolio 的 1D forward return
    # 要变成第二天的return 所以shift 1
    daily_returns = daily_returns.shift(1, freq=freq)
    portfolio = (1 + daily_returns).cumprod()
    if ret_pos:
        return portfolio, daily_weights
    else:
        return portfolio


def positions(weights, period, freq=None, leverage_up=True):
    """
    # TODO: rewritten to be more accurate
    [Modified]
    [COMMENTS]:  multi-periods rebalanced smoothing estimates --- [DIFFERENT] from the authentic weights for each day

    Builds net position values time series, the portfolio percentage invested
    in each asset daily.

    Parameters
    ----------
    weights: pd.Series
        pd.Series containing factor weights, the index contains timestamps at
        which the trades are computed and the values correspond to assets
        weights
        - see factor_weights for more details

    period: pandas.Timedelta or string
        Assets holding period (1 day, 2 mins, 3 hours etc). It can be a
        Timedelta or a string in the format accepted by Timedelta constructor
        ('1 days', '1D', '30m', '3h', '1D1h', etc)

    freq : pandas DateOffset, optional
        [IMPORTANT]
        Used to specify a particular trading calendar. If not present
        weights.index.freq ('infer') will be used

    leverage_up: bool default as True
        whether adjust the sub/tot_weights to keep constant leverage as 1

    [MODIFIED][DUMPED]
    rebalance: bool, optional, default as True
        rebalancing to keep leverage constant equals to 1 as default, which is conflict of the thinking of
        N equal-invested sub-portfolio's, see also "the 1/K portfolio composition method" such as
        "Harvesting Commodity Risk Premia"

    Returns
    -------
    pd.DataFrame
        Assets positions DataFrame, datetime on index, assets on columns
        Example: [old]
            index                 'AAPL'         'MSFT'
            2004-01-09 10:30:00
            2004-01-09 15:30:00
            2004-01-12 10:30:00
            2004-01-12 15:30:00
            2004-01-13 10:30:00

        weights instead of the capital

    """

    # changed to wide form
    # [IMPORTANT] fillna as 0 to get consistent with mean operation afterwards when combine different sub-portfolio's
    weights = weights.unstack().fillna(0)

    if not isinstance(period, pd.Timedelta):
        period = pd.Timedelta(period)

    if freq is None:
        freq = weights.index.freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar, STRONGLY SUGGEST specify freq ！！！ ",
                      UserWarning)

    #
    # weights index contains factor computation timestamps, then add returns
    # timestamps too (factor timestamps + period) and save them to 'full_idx'
    # 'full_idx' index will contain an entry for each point in time the weights
    # change and hence they have to be re-computed
    #
    trades_idx = weights.index.copy()
    returns_idx = utils.add_custom_calendar_timedelta(trades_idx, period,
                                                      freq)  # TODO: should be extended one more period
    weights_idx = trades_idx.union(returns_idx)

    #
    # Compute portfolio weights for each point in time contained in the index
    #
    portfolio_weights = pd.DataFrame(index=weights_idx,
                                     columns=weights.columns)

    portfolio_leverageAdj = pd.Series(0, index=weights_idx)

    active_weights = []

    for curr_time in weights_idx:

        #
        # fetch new weights that become available at curr_time and store them
        # in active weights
        #
        if curr_time in weights.index:
            assets_weights = weights.loc[curr_time]
            expire_ts = utils.add_custom_calendar_timedelta(curr_time,
                                                            period, freq)
            active_weights.append((expire_ts, assets_weights))  # start and end

        #
        # remove expired entry in active_weights (older than 'period')
        #
        if active_weights:  # 这种方法计算的weight 是不准确的，存在着被动的rebalance，因为会涨跌，逐日权重是不恒定的 ???
            expire_ts, assets_weights = active_weights[0]
            if expire_ts <= curr_time:
                active_weights.pop(0)

        if not active_weights:
            continue
        #
        # Compute total weights for curr_time and store them
        #
        tot_weights = [w for (ts, w) in active_weights]
        tot_weights = pd.concat(tot_weights, axis=1)

        # TODO: corresponding to cumulative-returns "mean" for each date between different sub-portfolios
        #  different subportfolio entangled, usually deleveraged before adjustment
        #  thus, the positions corresponding to the cumulative_returns mechanism
        #  N-equal(l/N instead, not 1/N) subset each day, instead of only at the very beginning
        #  sum (equally treated) different sub-portfolio's means equally weighting sub-portfolio's daily
        #  and then leverage up to 1 to ensure fully invested daily
        #  thus, portfolio_weights here will be consistent with the ones of cumulative_returns_precisely
        tot_weights = tot_weights.mean(axis=1)
        portfolio_leverage_b_adj = tot_weights.abs().sum()

        if leverage_up:
            tot_weights /= portfolio_leverage_b_adj

        # TODO: below will not to be independent either, because weights will fluctuate since rebalancing
        #  and compounding process since the very beginning, sum() equally conflict
        #  [DUMPED], the below weighting is only constantly proportional to above, thus deleveraged
        # tot_weights = (tot_weights.sum(axis=1)) / (tot_weights.abs().sum().sum())

        portfolio_weights.loc[curr_time] = tot_weights
        portfolio_leverageAdj.loc[curr_time] = 1 / portfolio_leverage_b_adj

    return (portfolio_weights.fillna(0), portfolio_leverageAdj)


# %% information coefficient part -------------------------------------------------------------
def factor_information_coefficient(factor_data,
                                   group_adjust=False,
                                   by_group=False):
    """
    [ORIGINAL]
    Computes the 【Spearman Rank Correlation】 based Information Coefficient (IC)
    between factor values and N period forward returns for each period in
    the factor index bygroup

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    group_adjust : bool
        Demean forward returns by group before computing IC.

    by_group : bool
        If True, compute period wise IC separately for each group.

    Returns
    -------
    ic : pd.DataFrame
        Spearman Rank correlation between factor and
        provided forward returns.
    """

    def src_ic(group):  # 【Spearman Rank Correlation】
        f = group['factor']
        _ic = group[utils.get_forward_returns_columns(factor_data.columns)] \
            .apply(lambda x: stats.spearmanr(x, f)[0])  # calculate IC for each 【period】
        return _ic

    factor_data = factor_data.copy()

    # TODO: SIMPLIFIED
    # grouper = [factor_data.index.get_level_values('date')]
    grouper = ['date']

    if group_adjust:
        factor_data = utils.demean_forward_returns(factor_data,
                                                   grouper + ['group'])  # demeaned or each [date, group]

        # TODO: also demean the factor value accordingly
        #  based on the design of the factor, it will cause absolute return rank or excess-return rank
        #  if the factor is designed to be excess-value of the group factor,
        #  group_adjust = True to check, usually group_neutral
        #  see also: factor_returns calculation and assignment
        #  factor_data['factor'] = factor_data.groupby(grouper + ['group'])['factor'].transform(lambda x: x - x.mean())
        #  group_neutral == True, usually along with by_group==True, while reverse not true

    if by_group:
        grouper.append('group')

    ic = factor_data.groupby(grouper).apply(src_ic)

    return ic


def mean_information_coefficient(factor_data,
                                 group_adjust=False,
                                 by_group=False,
                                 by_time=None):
    """
    [ORIGINAL]
    Get the mean information coefficient of specified groups.
    Answers questions like:
    What is the mean IC for each month?
    What is the mean IC for each group for our whole time-range?
    What is the mean IC for for each group, each week?

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    group_adjust : bool
        Demean forward returns by group before computing IC.

    by_group : bool
        If True, take the mean IC for each group.

    by_time : str (pd time_rule), optional
        Time window to use when taking mean IC.
        See http://pandas.pydata.org/pandas-docs/stable/timeseries.html
        for available options.

    Returns
    -------
    ic : pd.DataFrame
        Mean Spearman Rank correlation between factor and provided
        forward price movement windows.
    """

    ic = factor_information_coefficient(factor_data=factor_data,
                                        group_adjust=group_adjust,
                                        by_group=by_group)

    grouper = []
    if by_time is not None:
        grouper.append(pd.Grouper(freq=by_time))  # 进一步丰富了对 grouper 的理解， 可以是一个list of grouper，
        # 包含 array-like， labels， grouper
    if by_group:
        grouper.append('group')

    if len(grouper) == 0:
        ic = ic.mean()

    else:
        ic = (
            ic.reset_index().set_index('date').groupby(grouper).mean())  # np.nanmean() if there's nan inside the group

    return ic


# %% turnover part  -------------------------------------------------------------
def quantile_turnover(quantile_factor, quantile, period=1):
    """
    [COMMENTED]
    [IMPORTANT]: turnover by quantile instead by weights
    Computes the proportion of names in a factor quantile that were
    not in that quantile in the previous period.                                                # quantile_turnover

    if cross-sectional number of assets keep constant, quantile-turnover is a good approximate of real-turnover-level

    Parameters
    ----------
    quantile_factor : pd.Series, multi-indexed                                                    # series
        DataFrame with (date, asset) and factor quantile.                                         # 不是用 weight 来算，而是用 quantile 来算的

    quantile : int
        Quantile on which to perform turnover analysis.

    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    quant_turnover : pd.Series
        Period by period turnover for that quantile.
    """

    quant_names = quantile_factor[quantile_factor == quantile]  # multiindexed as factor_data["factor_quantile"]
    quant_name_sets = quant_names.groupby(level=['date']).apply(
        lambda x: set(x.index.get_level_values('asset')))  # 因为用的是 set， 所以，天然是 unique 的

    name_shifted = quant_name_sets.shift(period)  # 此处的操作是 shift， 这个是否跟 index 的 freq 有关？？？ 还是简单的numpy shift？？
    # 答：基于获得的办法，此处DatetimeIndex没有frequency，numpy shift
    # TODO: [IMPORTANT] 如果这里的shift 不做 freq specify， 就是 简单 shift， 除非 freq = ‘infer’

    new_names = (quant_name_sets - name_shifted).dropna()  # TODO: 集合的差集运算 ！！！  跟 nan 的 差集 nan？？？
    # 答：只要原本的两个集合中有任意一个是nan，差集就是nan
    # 所以需要做dropna() 处理
    quant_turnover = new_names.apply(
        lambda x: len(x)) / quant_name_sets.apply(lambda x: len(x))
    # 每个quant分组的 个数的 turnover， 跟我们说的turnover不一样？？？
    # 答：不一样，这里的是指新品种占该层所有品种的比率
    # 如果每一期的所有品种个数是固定的N， 那么每个 quantile 内的品种的个数也是固定的，在这种情况下，此 turnover 等于彼 rough weights turnover
    quant_turnover.name = quantile
    # 答：在测试中发现每个quantile内品种的个数是不同的
    return quant_turnover  # 一般计算turnover 都是基于weight 来计算的


def factor_rank_autocorrelation(factor_data, period=1):
    """
    [ORIGINAL]
    Computes autocorrelation of mean factor ranks in specified time spans.
    We must compare period to period factor ranks rather than factor values
    to account for systematic shifts in the factor values of all names or names
    within a group. This metric is useful for measuring the turnover of a                       # factor rank
    factor. If the value of a factor for each name changes randomly from period
    to period, we'd expect an autocorrelation of 0.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns
    period: int, optional
        Number of days over which to calculate the turnover.

    Returns
    -------
    autocorr : pd.Series - single indexed by date
        Rolling 1 period (defined by time_rule) autocorrelation of
        factor values.
    """
    grouper = [factor_data.index.get_level_values('date')]

    ranks = factor_data.groupby(grouper)['factor'].rank()  # factor_value -> rank

    asset_factor_rank = ranks.reset_index().pivot(index='date',
                                                  columns='asset',
                                                  values='factor')

    asset_shifted = asset_factor_rank.shift(period)  # naive np.shift

    autocorr = asset_factor_rank.corrwith(asset_shifted, axis=1)
    autocorr.name = period
    return autocorr


# %% preparation for pyfolio
def factor_positions(factor_data,
                     period,
                     long_short=True,
                     group_neutral=False,
                     equal_weight=False,
                     quantiles=None,
                     groups=None,
                     freq=None):  # add "freq"
    """
    [ORIGINAL] + add "freq"
    [COMMENTS] the whole operation is similar to "factor_cumulative_returns -- cumulative_returns"
    Simulate a portfolio using the factor in input and returns the assets
    positions as percentage of the total portfolio.
    [IMPORTANT]: the leverage will usually "NOT" be 1 due to the multi-period mechanism, N-independent originally equal
    sub-portfolio rebalancing every N-periods

    Parameters                  # same as Parameters of "factor_cumulative_returns"
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns

    long_short : bool, optional
        if True then simulates a dollar neutral long-short portfolio
        - see performance.create_pyfolio_input for more details

    group_neutral : bool, optional
        If True then simulates a group neutral portfolio
        - see performance.create_pyfolio_input for more details

    equal_weight : bool, optional
        Control the assets weights:
        - see performance.create_pyfolio_input for more details.

    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used

    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used

    freq : pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        returns.index.freq (infer) will be used

    Returns
    -------
    assets positions : pd.DataFrame
        Assets positions series, datetime on index, assets on columns.
        Example: [old]
            index                 'AAPL'         'MSFT'
            2004-01-09 10:30:00
            2004-01-09 15:30:00
            2004-01-12 10:30:00
            2004-01-12 15:30:00
            2004-01-13 10:30:00

        weights instead of the capital

    """
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)

    if period not in fwd_ret_cols:
        raise ValueError("Period '%s' not found" % period)

    # ADDED, similiar but not same as 'positions'
    if freq is None:
        freq = factor_data.index.levels[0].freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar",
                      UserWarning)

    todrop = list(fwd_ret_cols)
    todrop.remove(period)
    portfolio_data = factor_data.drop(todrop, axis=1)

    if quantiles is not None:
        portfolio_data = portfolio_data[portfolio_data['factor_quantile'].isin(
            quantiles)]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data['group'].isin(groups)]

    weights = \
        factor_weights(factor_data=portfolio_data,
                       demeaned=long_short,
                       group_adjust=group_neutral,
                       equal_weight=equal_weight)

    return positions(weights=weights, period=period, freq=freq)


def factor_cumulative_returns(factor_data,
                              period,
                              long_short=True,
                              group_neutral=False,
                              equal_weight=False,
                              quantiles=None,
                              groups=None,
                              freq=None,
                              precisely=False):  # add "freq"
    """
    [ORIGINAL] + add "freq"
    [COMMENTS] the whole operation is similar to "factor_positions -- positions"
    Simulate a portfolio using the factor in input and returns the cumulative
    returns of the simulated portfolio

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns                                 # mutli-periods problem

    long_short : bool, optional
        if True then simulates a dollar neutral long-short portfolio
        - see performance.create_pyfolio_input for more details

    group_neutral : bool, optional
        If True then simulates a group neutral portfolio
        - see performance.create_pyfolio_input for more details

    equal_weight : bool, optional
        Control the assets weights:
        - see performance.create_pyfolio_input for more details

    quantiles: sequence[int], optional
        Use only specific quantiles in the computation. By default all
        quantiles are used

    groups: sequence[string], optional
        Use only specific groups in the computation. By default all groups
        are used

    freq: pandas DateOffset, optional
        Used to specify a particular trading calendar. If not present
        returns.index.freq (infer) will be used

    precisely: bool, optional
        whether to construct the multi-period cumulative return precisely instead of geometrically averaging

    Returns
    -------
    Cumulative returns series : pd.Series
        Example:
            2015-07-16 09:30:00  -0.012143
            2015-07-16 12:30:00   0.012546
            2015-07-17 09:30:00   0.045350
            2015-07-17 12:30:00   0.065897
            2015-07-20 09:30:00   0.030957
    """
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)

    if period not in fwd_ret_cols:
        raise ValueError("Period '%s' not found" % period)

    # ADDED, similiar but not same as 'cumulative_returns'
    if freq is None:
        freq = factor_data.index.levels[0].freq

    if freq is None:
        freq = BDay()
        warnings.warn("'freq' not set, using business day calendar",
                      UserWarning)

    todrop = list(fwd_ret_cols)
    todrop.remove(period)
    portfolio_data = factor_data.drop(todrop, axis=1)  # 为什么不使用选取，而是反向使用丢弃， 是对格式通用性的ensure？？
    # 答：是的，factor_data中的列数会根据条件发生变化。比如group在groupby=None的时候不存在
    # 只剩下给定period下的return， factor_data 里的 return 是forward return 没有年化和平滑

    if quantiles is not None:
        portfolio_data = portfolio_data[portfolio_data['factor_quantile'].isin(
            quantiles)]

    if groups is not None:
        portfolio_data = portfolio_data[portfolio_data['group'].isin(groups)]

    # old version, will take multi-periods rebalancing into account
    if ~precisely:
        returns = \
            factor_returns(factor_data=portfolio_data,  # subset of factor_data will be used
                           demeaned=long_short,  # thus, weighting will be calculated by this subsets
                           group_adjust=group_neutral,
                           equal_weight=equal_weight)
        return cumulative_returns(returns=returns[period], period=period, freq=freq)
    else:
        weights = \
            factor_weights(factor_data=factor_data,
                           demeaned=long_short,
                           group_adjust=group_neutral,
                           equal_weight=equal_weight)

        return cumulative_returns_precisely(returns=factor_data['1D'], weights=weights, period=period, freq=freq)


def create_pyfolio_input(factor_data,
                         period,
                         capital=None,
                         long_short=True,
                         group_neutral=False,
                         equal_weight=False,
                         quantiles=None,
                         groups=None,
                         benchmark_period='1D'):
    """
    Simulate a portfolio using the input factor and returns the portfolio
    performance data properly formatted for Pyfolio analysis.

    For more details on how this portfolio is built see:
    - performance.cumulative_returns (how the portfolio returns are computed)
    - performance.factor_weights (how assets weights are computed)

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    period : string
        'factor_data' column name corresponding to the 'period' returns to be
        used in the computation of porfolio returns

    capital : float, optional                                                       # dollar amount or percentage
        If set, then compute 'positions' in dollar amount instead of percentage

    long_short : bool, optional                                                     # for specific returns
        if True enforce a dollar neutral long-short portfolio: asset weights
        will be computed by demeaning factor values and dividing by the sum of
        their absolute value (achieving gross leverage of 1) which will cause
        the portfolio to hold both long and short positions and the total
        weights of both long and short positions will be equal.
        If False the portfolio weights will be computed dividing the factor
        values and  by the sum of their absolute value (achieving gross
        leverage of 1). Positive factor values will generate long positions and
        negative factor values will produce short positions so that a factor
        with only posive values will result in a long only portfolio.

    group_neutral : bool, optional                                                  # for specific returns
        If True simulates a group neutral portfolio: the portfolio weights
        will be computed so that each group will weigh the same.
        if 'long_short' is enabled the factor values demeaning will occur on
        the group level resulting in a dollar neutral, group neutral,
        long-short portfolio.
        If False group information will not be used in weights computation.

    equal_weight : bool, optional                                                  # for specific returns
        if True the assets will be equal-weighted. If long_short is True then
        the factor universe will be split in two equal sized groups with the
        top assets in long positions and bottom assets in short positions.
        if False the assets will be factor-weighed, see 'long_short' argument

    quantiles: sequence[int], optional                                             # for specific returns
        Use only specific quantiles in the computation. By default all
        quantiles are used

    groups: sequence[string], optional                                             # for specific returns
        Use only specific groups in the computation. By default all groups
        are used

    benchmark_period : string, optional                                            # for pyfolio input
        By default benchmark returns are computed as the factor universe mean
        daily returns but 'benchmark_period' allows to choose a 'factor_data'
        column corresponding to the returns to be used in the computation of
        benchmark returns. More generally benchmark returns are computed as the
        factor universe returns traded at 'benchmark_period' frequency, equal
        weighting and long only


    Returns
    -------
     returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - Time series with decimal returns.
         - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

     positions : pd.DataFrame
        Time series of dollar amount (or percentage when 'capital' is not
        provided) invested in each position and cash.
         - Days where stocks are not held can be represented by 0.
         - Non-working capital is labelled 'cash'
         - Example:
            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375

     benchmark : pd.Series
        Benchmark returns computed as the factor universe mean daily returns.

    """

    #
    # Build returns:
    # we don't know the frequency at which the factor returns are computed but
    # pyfolio wants daily returns. So we compute the cumulative returns of the
    # factor, then resample it at 1 day frequency and finally compute daily
    # returns
    #
    freq = factor_data.index.levels[0].freq

    if freq is None:  # TODO: if freq is None, factor_cumulative_returns may create returns with
        freq = BDay()  # different datetimeindex for different periods, it's ugly for following operations
        warnings.warn("'freq' not set, using business day calendar",
                      # such as comparison for returns coming from different periods
                      UserWarning)

    cumrets = factor_cumulative_returns(factor_data=factor_data,
                                        period=period,
                                        long_short=long_short,
                                        group_neutral=group_neutral,
                                        equal_weight=equal_weight,
                                        quantiles=quantiles,
                                        groups=groups,
                                        freq=freq)
    # cumrets = cumrets.resample('1D').last().fillna(method='ffill')             # TODO: not necessary and ambigunous for '1D' instead of '1BD'
    returns = cumrets.pct_change().fillna(0)

    #
    # Build positions. As pyfolio asks for daily position we have to resample
    # the positions returned by 'factor_positions' at 1 day frequency and
    # recompute the weights so that the sum of daily weights is 1.0
    #
    positions, leverage_adj = factor_positions(factor_data=factor_data,
                                               period=period,
                                               long_short=long_short,
                                               group_neutral=group_neutral,
                                               equal_weight=equal_weight,
                                               quantiles=quantiles,
                                               groups=groups,
                                               freq=freq)
    # positions = positions.resample('1D').sum().fillna(method='ffill')           # TODO: fillna() necessary if periods properly formatted ???
    positions = positions.fillna(0)  # TODO: fillna() necessary if periods properly formatted ???
    positions = positions.div(positions.abs().sum(axis=1), axis=0).fillna(
        0)  # TODO: positions.abs().sum(axis=1) naively == 1, multi-periods total portfolio will usually be deleveraged
    positions['cash'] = 1. - positions.sum(
        axis=1)  # TODO: NOTICE: here, positions.sum instead of .abs().sum(axis=1), thus, we have cash

    # transform percentage positions to dollar positions
    if capital is not None:
        positions = positions.mul(  # TODO: positions should be the one corresponding to cumulative_returns
            cumrets.reindex(positions.index) * capital, axis=0)  # compounding capital and then mul with positions

    #
    #
    #
    # Build benchmark returns as the factor universe mean returns traded at
    # 'benchmark_period' frequency
    #
    fwd_ret_cols = utils.get_forward_returns_columns(factor_data.columns)
    if benchmark_period in fwd_ret_cols:
        benchmark_data = factor_data.copy()
        # make sure no negative positions
        benchmark_data['factor'] = benchmark_data[
            'factor'].abs()  # actually, it means changed the sign or align all the sign for each asset
        benchmark_rets = factor_cumulative_returns(factor_data=benchmark_data,
                                                   # TODO: constructed a long-only equal-weight composition, like an index
                                                   period=benchmark_period,
                                                   long_short=False,
                                                   group_neutral=False,  # 此处用的方法，对于逐日计算的意义上，是不准确的，weight 的计算存在这每天的动态平衡
                                                   equal_weight=True,
                                                   freq=freq)  # 如果假设每天确实存在动态平衡，每天的return 也是几何平均的多日收益，也是一层平滑，和固定拿到期末是有区别的，单利复利的区别
        # benchmark_rets = benchmark_rets.resample(
        #     '1D').last().fillna(method='ffill')                               # TODO: same as above
        benchmark_rets = benchmark_rets.pct_change().fillna(0)
        benchmark_rets.name = 'benchmark'
    else:
        benchmark_rets = None

    return returns, positions, benchmark_rets


# %% event studies part
def average_cumulative_return_by_quantile(factor_data,
                                          returns,
                                          periods_before=10,
                                          periods_after=15,
                                          demeaned=True,
                                          group_adjust=False,
                                          by_group=False):
    """
    Plots average cumulative returns by factor quantiles in the period range
    defined by -periods_before to periods_after

    ...

    """
    pass


def common_start_returns(factor,
                         returns,
                         before,
                         after,
                         cumulative=False,
                         mean_by_date=False,
                         demean_by=None):
    """
    A date and equity pair is extracted from each index row in the factor
    dataframe and for each of these pairs a return series is built starting
    from 'before' the date and ending 'after' the date specified in the pair.
    All those returns series are then aligned to a common index (-before to
    after) and returned as a single DataFrame

    ...

    """
    pass


if __name__ == "__main__":
    weights = pd.read_pickle(r'E:\MultiFactor\Test\alphalens\weights.pkl')
    positions, x = positions(weights, period='5D', leverage_up=False)
