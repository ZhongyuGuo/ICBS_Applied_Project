#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/7/16 18:45:02
# @Author  : Michael_Liu @ QTG
# @File    : tears
# @Software: PyCharm

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from . import plotting
from . import performance as perf
from . import utils


# %% class for subplots and allocating plottings
class GridFigure(object):
    """
    It makes life easier with grid plots
    [Note]: if distributed grids is redundant, the redundant grids will not be shown
    [Mechanism]: gridspec.GridSpec only create grids, these grids need subplot to instantiate,
    while uninstantiated grids will be automatically hidden

    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.fig = plt.figure(figsize=(14, rows * 7))
        self.gs = gridspec.GridSpec(rows, cols, wspace=0.4, hspace=0.3)
        self.curr_row = 0
        self.curr_col = 0

    def next_row(self):
        if self.curr_col != 0:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, :])
        self.curr_row += 1
        return subplt

    def next_cell(self):
        if self.curr_col >= self.cols:
            self.curr_row += 1
            self.curr_col = 0
        subplt = plt.subplot(self.gs[self.curr_row, self.curr_col])
        self.curr_col += 1
        return subplt

    def close(self):
        plt.close(self.fig)
        self.fig = None
        self.gs = None


# %% returns related part -------------------------------------------------------------
@plotting.customize
def create_returns_tear_sheet(
        factor_data,
        long_short=True,
        group_neutral=False,
        by_group=False,
        equal_weight=False
):
    """
    [MODIFIED]:
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots

    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots

    by_group : bool
        If True, display graphs separately for each group.

    equal_weight : bool
        factor_weight / equal_weight

    """

    factor_returns = perf.factor_returns(
        factor_data=factor_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight,
        by_asset=False
    )                                             # equal_weight=False as alphalens default
                                                  # by_asset=False as default
                                                  # thus, factor_returns is single index with multi-periods columns

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data=factor_data,
        by_date=False,                            # by_date=False
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]     # average operation to each date
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data=factor_data,
        by_date=True,                              # by date for cumulative and ot
        by_group=False,                            # by_group False as default
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(                    # average operation to each date
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(                                # average operation to each date
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(
        factor_data=factor_data,
        returns=factor_returns,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight
    )                                                                           # equal_weight=False as default
                                                                                # returns inputed, thus, demeaned, group_adjust, equal_weight will be meaningless
                                                                                # these three will be only used for returns missing case
    # TODO: renaming to clarify
    # TODO: std_spread_quant has no usage/applications
    # mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
    mean_rateret_spread_quant, std_rateret_spread_quant = perf.compute_mean_returns_spread(
        mean_returns=mean_quant_rateret_bydate,
        upper_quant=factor_data["factor_quantile"].max(),
        lower_quant=factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )                                                                           # output is rateret sense

    # TODO：extra-added
    mean_ret_spread_quant, std_ret_spread_quant =  perf.compute_mean_returns_spread(
        mean_returns=mean_quant_ret_bydate,
        upper_quant=factor_data["factor_quantile"].max(),
        lower_quant=factor_data["factor_quantile"].min(),
        std_err=std_quant_daily,
    )

    # ------------------------ plotting part start ------------------------
    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 4
    gf = GridFigure(rows=vertical_sections, cols=1)

    notes = ('Factor Weighted ' if not equal_weight else 'Equal Weighted '
            + ('Group Neutral/Adjusted ' if group_neutral else '')
            + ('Long/Short/demeaned ' if long_short else ''))

    plotting.plot_returns_table(
        alpha_beta=alpha_beta,
        mean_ret_quantile= mean_quant_rateret,
        # mean_ret_spread_quantile=mean_ret_spread_quant                        # renaming to clarify
        mean_ret_spread_quantile=mean_rateret_spread_quant,
        notes=notes
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,                          # let func set automatically
        ax=gf.next_row(),
    )

    plotting.plot_quantile_returns_violin(
        return_by_q=mean_quant_rateret_bydate,          # input is rateret, thus, different periods can do cross-comparison
        ylim_percentiles=(1, 99),                       # cut globally
        ax=gf.next_row()
    )

    # TODO: freq operation check
    # if well-prepared, the factor_data.index.levels[0].freq will not be None
    # otherwise, warning popped up for REMIND
    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    # originally in [NEW VERSION]
    # if "1D" in factor_returns:
    #     title = (
    #         "Factor Weighted "
    #         + ("Group Neutral " if group_neutral else "")
    #         + ("Long/Short " if long_short else "")
    #         + "Portfolio Cumulative Return (1D Period)"
    #     )
    #
    #     plotting.plot_cumulative_returns(
    #         factor_returns["1D"], period="1D", title=title, ax=gf.next_row()
    #     )
    #
    #     plotting.plot_cumulative_returns_by_quantile(
    #         mean_quant_ret_bydate["1D"], period="1D", ax=gf.next_row()
    #     )
    #
    # for multi-periods:
    # modified from func-cumulative_returns of previous version
    for p in utils.get_forward_returns_columns(factor_returns.columns, require_exact_day_multiple=True):
        title1 = (('Factor Weighted ' if not equal_weight else 'Equal Weighted ')
                 + ('Group Neutral ' if group_neutral else '')
                 + ('Long/Short ' if long_short else '')
                 + "Portfolio Cumulative Return ({} Period)".format(p))

        plotting.plot_cumulative_returns(
            factor_returns=factor_returns[p],
            period=p,
            freq=trading_calendar,
            title=title1,
            ax=gf.next_row()
        )

        title2 = ('Cumulative Return by Quantile Mean '
                  + ('group_adjusted, ' if group_neutral else '')
                  + ('demeaned ' if long_short else '')
                  + ' ({} Period Forward Return)'.format(p))

        plotting.plot_cumulative_returns_by_quantile(
            quantile_returns=mean_quant_ret_bydate[p],
            period=p,
            freq=trading_calendar,
            title=title2,
            ax=gf.next_row()
        )

        title3 = ('Cumulative Return by Quantile Spread (Top-Bottom) '
                  + ('group_adjusted, ' if group_neutral else '')
                  + ('demeaned ' if long_short else '')
                  + ' ({} Period Forward Return)'.format(p))

        plotting.plot_cumulative_returns(
            factor_returns=mean_ret_spread_quant[p]/2,
            period=p,
            freq=trading_calendar,
            title=title3,
            ax=gf.next_row()
        )

    ax_mean_quantile_returns_spread_ts = [
        gf.next_row() for x in range(fr_cols)
    ]
    plotting.plot_mean_quantile_returns_spread_time_series(
        # mean_returns_spread=mean_ret_spread_quant,
        mean_returns_spread=mean_rateret_spread_quant,                          # renaming to clarify
        std_err=std_rateret_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
    )

    plt.show()
    gf.close()

    # --------------------------- extra by_group part ------------------------------
    if by_group:
        (
            mean_return_quantile_group,
            mean_return_quantile_group_std_err,
        ) = perf.mean_return_by_quantile(
            factor_data=factor_data,
            by_date=False,                                              # by_date=False
            by_group=True,                                              # by_group=True
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret_group = mean_return_quantile_group.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_return_quantile_group.columns[0],
        )

        num_groups = len(
            mean_quant_rateret_group.index.get_level_values("group").unique()
        )

        vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)

        ax_quantile_returns_bar_by_group = [
            gf.next_cell() for _ in range(num_groups)
        ]
        plotting.plot_quantile_returns_bar(                         # theoretically, more plotting can be plotted under by-group case
            mean_ret_by_q=mean_quant_rateret_group,
            by_group=True,
            ylim_percentiles=(5, 95),
            ax=ax_quantile_returns_bar_by_group,
        )
        plt.show()
        gf.close()

@plotting.customize
def create_full_returns_tear_sheet(
        factor_data,
        long_short=True,
        group_neutral=False,
        by_group=False,
        # equal_weight=False
):
    """
    [MODIFIED]:
    Creates a tear sheet for returns analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to,
        and (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.
        Additionally factor values will be demeaned across the factor universe
        when factor weighting the portfolio for cumulative returns plots

    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.
        Additionally each group will weight the same in cumulative returns
        plots

    by_group : bool
        If True, display graphs separately for each group.

    # equal_weight : bool
    #     factor_weight / equal_weight

    """

    # factor weighted
    factor_returns = perf.factor_returns(
        factor_data=factor_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=False,
        by_asset=False
    )                                             # equal_weight=False as alphalens default
                                                  # by_asset=False as default
                                                  # thus, factor_returns is single index with multi-periods columns

    # equal weighted
    factor_returns_eq = perf.factor_returns(
        factor_data=factor_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=True,
        by_asset=False
    )

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
        factor_data=factor_data,
        by_date=False,                            # by_date=False
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]     # average operation to each date
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data=factor_data,
        by_date=True,                              # by date for cumulative and ot
        by_group=False,                            # by_group False as default
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(                    # average operation to each date
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(                                # average operation to each date
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    # factor_weighted
    alpha_beta = perf.factor_alpha_beta(
        factor_data=factor_data,
        returns=factor_returns,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=False
    )                                                                           # equal_weight=False as default
                                                                                # returns inputed, thus, demeaned, group_adjust, equal_weight will be meaningless
                                                                                # these three will be only used for returns missing case

    # equal_weighted
    alpha_beta_eq = perf.factor_alpha_beta(
        factor_data=factor_data,
        returns=factor_returns_eq,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=True
    )

    # TODO: renaming to clarify
    # TODO: std_spread_quant has no usage/applications
    # mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
    mean_rateret_spread_quant, std_rateret_spread_quant = perf.compute_mean_returns_spread(
        mean_returns=mean_quant_rateret_bydate,
        upper_quant=factor_data["factor_quantile"].max(),
        lower_quant=factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )                                                                           # output is rateret sense

    # TODO：extra-added
    mean_ret_spread_quant, std_ret_spread_quant =  perf.compute_mean_returns_spread(
        mean_returns=mean_quant_ret_bydate,
        upper_quant=factor_data["factor_quantile"].max(),
        lower_quant=factor_data["factor_quantile"].min(),
        std_err=std_quant_daily,
    )

    # ------------------------ plotting part start ------------------------
    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 5
    gf = GridFigure(rows=vertical_sections, cols=1)

    notes = ('Factor Weighted and Equal Weighted '
            + ('Group Neutral/Adjusted ' if group_neutral else '')
            + ('Long/Short/demeaned ' if long_short else ''))

    plotting.plot_returns_table(
        alpha_beta=alpha_beta,
        alpha_beta_eq=alpha_beta_eq,
        mean_ret_quantile= mean_quant_rateret,
        # mean_ret_spread_quantile=mean_ret_spread_quant                        # renaming to clarify
        mean_ret_spread_quantile=mean_rateret_spread_quant,
        notes=notes
    )

    plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,                          # let func set automatically
        ax=gf.next_row(),
    )

    plotting.plot_quantile_returns_violin(
        return_by_q=mean_quant_rateret_bydate,          # input is rateret, thus, different periods can do cross-comparison
        ylim_percentiles=(1, 99),                       # cut globally
        ax=gf.next_row()
    )

    # TODO: freq operation check
    # if well-prepared, the factor_data.index.levels[0].freq will not be None
    # otherwise, warning popped up for REMIND
    trading_calendar = factor_data.index.levels[0].freq
    if trading_calendar is None:
        trading_calendar = pd.tseries.offsets.BDay()
        warnings.warn(
            "'freq' not set in factor_data index: assuming business day",
            UserWarning,
        )

    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    # originally in [NEW VERSION]
    # if "1D" in factor_returns:
    #     title = (
    #         "Factor Weighted "
    #         + ("Group Neutral " if group_neutral else "")
    #         + ("Long/Short " if long_short else "")
    #         + "Portfolio Cumulative Return (1D Period)"
    #     )
    #
    #     plotting.plot_cumulative_returns(
    #         factor_returns["1D"], period="1D", title=title, ax=gf.next_row()
    #     )
    #
    #     plotting.plot_cumulative_returns_by_quantile(
    #         mean_quant_ret_bydate["1D"], period="1D", ax=gf.next_row()
    #     )
    #
    # for multi-periods:
    # modified from func-cumulative_returns of previous version
    for p in utils.get_forward_returns_columns(factor_returns.columns, require_exact_day_multiple=True):
        title1 = (('Factor Weighted ')
                 + ('Group Neutral ' if group_neutral else '')
                 + ('Long/Short ' if long_short else '')
                 + "Portfolio Cumulative Return ({} Period)".format(p))

        plotting.plot_cumulative_returns(
            factor_returns=factor_returns[p],
            period=p,
            freq=trading_calendar,
            title=title1,
            ax=gf.next_row()
        )

        title2 = (('Equal Weighted ')
                 + ('Group Neutral ' if group_neutral else '')
                 + ('Long/Short ' if long_short else '')
                 + "Portfolio Cumulative Return ({} Period)".format(p))

        plotting.plot_cumulative_returns(
            factor_returns=factor_returns_eq[p],
            period=p,
            freq=trading_calendar,
            title=title2,
            ax=gf.next_row()
        )

        title3 = ('Cumulative Return by Quantile Mean '
                  + ('group_adjusted, ' if group_neutral else '')
                  + ('demeaned ' if long_short else '')
                  + ' ({} Period Forward Return)'.format(p))

        plotting.plot_cumulative_returns_by_quantile(
            quantile_returns=mean_quant_ret_bydate[p],
            period=p,
            freq=trading_calendar,
            title=title3,
            ax=gf.next_row()
        )

        title4 = ('Cumulative Return by Quantile Spread (Top-Bottom) '
                  + ('group_adjusted, ' if group_neutral else '')
                  + ('demeaned ' if long_short else '')
                  + ' ({} Period Forward Return)'.format(p))

        plotting.plot_cumulative_returns(
            factor_returns=mean_ret_spread_quant[p]/2,
            period=p,
            freq=trading_calendar,
            title=title4,
            ax=gf.next_row()
        )

    ax_mean_quantile_returns_spread_ts = [
        gf.next_row() for x in range(fr_cols)
    ]
    plotting.plot_mean_quantile_returns_spread_time_series(
        # mean_returns_spread=mean_ret_spread_quant,
        mean_returns_spread=mean_rateret_spread_quant,                          # renaming to clarify
        std_err=std_rateret_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
    )

    plt.show()
    gf.close()

    # --------------------------- extra by_group part ------------------------------
    if by_group:
        (
            mean_return_quantile_group,
            mean_return_quantile_group_std_err,
        ) = perf.mean_return_by_quantile(
            factor_data=factor_data,
            by_date=False,                                              # by_date=False
            by_group=True,                                              # by_group=True
            demeaned=long_short,
            group_adjust=group_neutral,
        )

        mean_quant_rateret_group = mean_return_quantile_group.apply(
            utils.rate_of_return,
            axis=0,
            base_period=mean_return_quantile_group.columns[0],
        )

        num_groups = len(
            mean_quant_rateret_group.index.get_level_values("group").unique()
        )

        vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
        gf = GridFigure(rows=vertical_sections, cols=2)

        ax_quantile_returns_bar_by_group = [
            gf.next_cell() for _ in range(num_groups)
        ]
        plotting.plot_quantile_returns_bar(                         # theoretically, more plotting can be plotted under by-group case
            mean_ret_by_q=mean_quant_rateret_group,
            by_group=True,
            ylim_percentiles=(5, 95),
            ax=ax_quantile_returns_bar_by_group,
        )
        plt.show()
        gf.close()


# %% information coefficient part -------------------------------------------------------------
@plotting.customize
def create_information_tear_sheet(
    factor_data, group_neutral=False, by_group=False
):
    """
    [ORIGINAL] with minor [MODIFICATION]
    Creates a tear sheet for information analysis of a factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    group_neutral : bool
        Demean forward returns by group before computing IC.

    by_group : bool
        If True, display graphs separately for each group.
    """

    ic = perf.factor_information_coefficient(factor_data,
                                             group_adjust=group_neutral,
                                             by_group=False)

    plotting.plot_information_table(ic)

    # TODO: display setting
    columns_wide = 2                                                        # setting columns_wide, hard-cored as plot_ic_hist + plot_ic_qq
    fr_cols = len(ic.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    # vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    vertical_sections = fr_cols + fr_cols * rows_when_wide + columns_wide * fr_cols
    # TODO: redundant
    # by_group=False, fr_cols + fr_cols * rows_when_wide + (fr_cols + 1 // 2)
    # by_group=True, fr_cols + fr_cols * rows_when_wide + 1
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]        # each row: plot_ic_hist + plot_ic_qq
    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    # for not by_group application -------------------------------------------------
    if not by_group:

        mean_monthly_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=False,
            by_time="M",
        )
        ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
        plotting.plot_monthly_ic_heatmap(
            mean_monthly_ic=mean_monthly_ic, ax=ax_monthly_ic_heatmap
        )

    # for by_group application ------------------------------------------------------
    if by_group:
        mean_group_ic = perf.mean_information_coefficient(
            factor_data,
            group_adjust=group_neutral,
            by_group=True
        )

        plotting.plot_ic_by_group(ic_group=mean_group_ic, ax=gf.next_row())

    plt.show()
    gf.close()


# %% turnover part  -------------------------------------------------------------
@plotting.customize
def create_turnover_tear_sheet(factor_data, turnover_periods=None):
    """
    [ORIGINAL]
    Creates a tear sheet for analyzing the turnover properties of a factor.
    based on quantile_info

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    turnover_periods : sequence[string], optional
        Periods to compute turnover analysis on. By default periods in
        'factor_data' are used but custom periods can provided instead. This    【IMPORTANT】 TODO: 换手率计算周期，默认为None，但是会被设置成本来的频率
        can be useful when periods in 'factor_data' are not multiples of the
        frequency at which factor values are computed i.e. the periods
        are 2h and 4h and the factor is computed daily and so values like
        ['1D', '2D'] could be used instead
    """

    if turnover_periods is None:
        input_periods = utils.get_forward_returns_columns(
            factor_data.columns, require_exact_day_multiple=True,
        ).values                                                                # MODIFIED
        turnover_periods = utils.timedelta_strings_to_integers(input_periods)
    else:
        turnover_periods = utils.timedelta_strings_to_integers(
            turnover_periods,
        )

    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {                               # {quantile_turnover: DataFrame(index=date, columns=quantiles)}
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    autocorrelation = pd.concat(                            # single indexed by date, columns= multi-periods
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)

    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    # vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    vertical_sections = fr_cols + fr_cols * rows_when_wide + 2 * fr_cols
    # TODO: redundant
    # fr_cols * rows_when_wide is enough
    gf = GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row()
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row()
        )

    plt.show()
    gf.close()


# %% whole tear_sheet part
# full infomation as create_full_tear_sheet
# brief information as create_summary_tear_sheet
@plotting.customize
def create_full_tear_sheet(factor_data,
                           long_short=True,
                           group_neutral=False,
                           by_group=False,
                           # equal_weight=False
                           ):
    """
    [ORIGINAL]
    Creates a full tear sheet for analysis and evaluating single
    return predicting (alpha) factor.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    long_short : bool
        Should this computation happen on a long short portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis

    group_neutral : bool
        Should this computation happen on a group neutral portfolio?
        - See tears.create_returns_tear_sheet for details on how this flag
        affects returns analysis
        - See tears.create_information_tear_sheet for details on how this
        flag affects information analysis

    by_group : bool
        If True, display graphs separately for each group.

    # equal_weight : bool
    # factor_weight / equal_weight

    """

    plotting.plot_quantile_statistics_table(factor_data)

    # create_returns_tear_sheet(
    #     factor_data=factor_data,
    #     long_short=long_short,
    #     group_neutral=group_neutral,
    #     by_group=by_group,
    #     equal_weight=equal_weight,
    #     set_context=False
    # )

    # Both equal and factor weighted analysis
    create_full_returns_tear_sheet(
        factor_data=factor_data,
        long_short=long_short,
        group_neutral=group_neutral,
        by_group=by_group,
        # equal_weight=equal_weight,
        set_context=False
    )

    create_information_tear_sheet(      # Rank IC independent of the demean process, thus, no long_short parameter
        factor_data=factor_data,
        group_neutral=group_neutral,
        by_group=by_group,
        set_context=False
    )

    create_turnover_tear_sheet(         # quantile_turnover
        factor_data=factor_data,
        set_context=False
    )


@plotting.customize
def create_summary_tear_sheet(
    factor_data,
    long_short=True,
    group_neutral=False,
    equal_weight=False
):
    """
    [ORIGINAL]
    [COMMENTS]: subset of the func - create_full_tear_sheet
    Creates a small summary tear sheet with returns, information, and turnover
    analysis.

    - See full explanation in create_full_tear_sheet

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the asset belongs to.
        - See full explanation in utils.get_clean_factor_and_forward_returns

    long_short : bool
        Should this computation happen on a long short portfolio? if so, then
        mean quantile returns will be demeaned across the factor universe.

    group_neutral : bool
        Should this computation happen on a group neutral portfolio? if so,
        returns demeaning will occur on the group level.

    equal_weight : bool
    factor_weight / equal_weight

    """

    # Returns Analysis
    factor_returns = perf.factor_returns(           # covered in "create_returns_tear_sheet()"
        factor_data=factor_data,
        demeaned=long_short,
        group_adjust=group_neutral,
        equal_weight=equal_weight
    )                                             # equal_weight=False as default
                                                  # by_asset=False as default
                                                  # thus, factor_returns is single index with multi-periods columns

    mean_quant_ret, std_quantile = perf.mean_return_by_quantile(            # covered in "create_returns_tear_sheet()"
        factor_data,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret = mean_quant_ret.apply(          # covered in "create_returns_tear_sheet()"
        utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
    )

    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(      # covered in "create_returns_tear_sheet()"
        factor_data,
        by_date=True,
        by_group=False,
        demeaned=long_short,
        group_adjust=group_neutral,
    )

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(        # covered in "create_returns_tear_sheet()"
        utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    )

    compstd_quant_daily = std_quant_daily.apply(                    # covered in "create_returns_tear_sheet()"
        utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    )

    alpha_beta = perf.factor_alpha_beta(                            # covered in "create_returns_tear_sheet()"
        factor_data,
        # returns=None,
        returns=factor_returns,                                     # if returns=None, factor_returns will be calculated inside
                                                                    # same as factor_returns calculated above
        demeaned=long_short,
        group_adjust=group_neutral
    )

    # mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(       # covered in "create_returns_tear_sheet()"
    mean_rateret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(     # rename to "rateret" to clarify
        mean_returns=mean_quant_rateret_bydate,
        upper_quant=factor_data["factor_quantile"].max(),
        lower_quant=factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    periods = utils.get_forward_returns_columns(factor_data.columns)
    periods = list(map(lambda p: pd.Timedelta(p).days, periods))

    fr_cols = len(periods)
    vertical_sections = 2 + fr_cols * 3
    gf = GridFigure(rows=vertical_sections, cols=1)

    plotting.plot_quantile_statistics_table(factor_data)

    plotting.plot_returns_table(                                        # covered in "create_returns_tear_sheet()"
        alpha_beta, mean_quant_rateret, mean_rateret_spread_quant
    )

    plotting.plot_quantile_returns_bar(                                 # covered in "create_returns_tear_sheet()"
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
    )

    # Information Analysis
    ic = perf.factor_information_coefficient(factor_data)               # covered in "create_information_tear_sheet()"
    plotting.plot_information_table(ic)                                 # brief

    # Turnover Analysis
    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {                                               # covered in "create_information_tear_sheet()"
        p: pd.concat(
            [
                perf.quantile_turnover(quantile_factor, q, p)
                for q in range(1, int(quantile_factor.max()) + 1)
            ],
            axis=1,
        )
        for p in periods
    }

    autocorrelation = pd.concat(                                        # covered in "create_information_tear_sheet()"
        [
            perf.factor_rank_autocorrelation(factor_data, period)
            for period in periods
        ],
        axis=1,
    )

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)   # brief

    plt.show()
    gf.close()


# %% event studies part
@plotting.customize
def create_event_returns_tear_sheet(factor_data,
                                    returns,
                                    avgretplot=(5, 15),
                                    long_short=True,
                                    group_neutral=False,
                                    std_bar=True,
                                    by_group=False):
    """
    Creates a tear sheet to view the average cumulative returns for a
    factor within a window (pre and post event).

    ...

    """
    pass


@plotting.customize
def create_event_study_tear_sheet(factor_data,
                                  returns,
                                  avgretplot=(5, 15),
                                  rate_of_ret=True,
                                  n_bars=50):
    """
    Creates an event study tear sheet for analysis of a specific event.

    ...

    """
    pass