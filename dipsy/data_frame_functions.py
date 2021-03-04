"""
functions to process the data frames (e.g. filtering or plotting histograms)
"""
import numpy as np
import matplotlib.pyplot as plt

from .data import T17_Fmm_corr
from .data import Tripathi2017


def get_param_values(df, param_names):
    param_values = {}
    for name in param_names:
        param_values[name] = sorted(list(set(df[name])))
    return param_values


def make_interfaces(param_values):
    """
    Make interfaces from parameter values for the histograms.
    This just puts the interfaces slightly to the left and right of
    the values.
    """
    param_interfaces = {}
    for name, values in param_values.items():
        param_interfaces[name] = np.hstack((0.99 * np.array(values), 1.01 * values[-1]))
    return param_interfaces


def histogram_corner(d, param_values, param_interfaces=None, param_label=None, f=None, vmax=1000., percent=False, n_total=None):
    """
    Produce a "corner plot" where all the parameter combinations
    are shown as 2D histograms.

    param_
    """

    param_names = list(param_values.keys())

    if param_interfaces is None:
        param_interfaces = make_interfaces(param_values)

    if param_label is None:
        param_label = {k: k for k in param_names}

    if f is None:
        f = plt.figure(figsize=(15, 15))

    N = len(param_names)
    for ix, x_name in enumerate(param_names):
        y_names = param_names.copy()
        y_names.remove(x_name)
        for iy, y_name in enumerate(y_names):
            if iy < ix:
                continue

            ax = f.add_subplot(N - 1, N - 1, iy * (N - 1) + ix + 1)

            x = d[x_name].values
            y = d[y_name].values

            # plot the histograms

            H = np.histogram2d(x, y, bins=[param_interfaces[x_name], param_interfaces[y_name]])
            counts = H[0].T

            if percent:
                if percent == 'per_bin':
                    n_bins = len(param_values[x_name]) * len(param_values[y_name])
                elif percent == 'total':
                    n_bins = 1
                else:
                    raise ValueError('percent must be False, \'per_bin\', or \'total\'.')
                if n_total is None:
                    n_total = len(x)

                counts *= 100 / n_total * n_bins

            m = ax.imshow(counts, vmin=0, vmax=vmax, origin='lower')

            # SET X-LABELS

            if iy == N - 2:
                xtick_pos = np.arange(len(param_values[x_name]))
                xtick_val = np.array(param_values[x_name])[xtick_pos]

                ax.set_xlabel(param_label[x_name])
            else:
                xtick_pos = []
                xtick_val = []

            ax.xaxis.set_ticks(xtick_pos)
            ax.xaxis.set_ticklabels([f'{v:.4g}' for v in xtick_val], rotation=45)

            # SET Y-LABELS

            if ix == 0:
                ytick_pos = np.arange(len(param_values[y_name]))
                ytick_val = np.array(param_values[y_name])[ytick_pos]

                ax.set_ylabel(param_label[y_name])
            else:
                ytick_pos = []
                ytick_val = []

            ax.yaxis.set_ticks(ytick_pos)
            ax.yaxis.set_ticklabels([f'{v:.4g}' for v in ytick_val], rotation=45)

    # final figure styling

    f.subplots_adjust(hspace=0.05, wspace=0.05)

    pos0 = f.get_axes()[0].get_position()
    pos1 = f.get_axes()[N].get_position()
    cax = f.add_axes([pos1.x0, pos0.y0, pos0.width / 20, pos0.height])
    cb = plt.colorbar(m, cax=cax)
    cb.set_label('number of simulations')

    return f


def histogram1d_normalized(d, x_name, param_values, param_interfaces=None, param_label=None, n_sims=None, f=None):

    param_names = list(param_values.keys())

    if param_interfaces is None:
        param_interfaces = make_interfaces(param_values)

    if param_label is None:
        param_label = {k: k for k in param_names}

    if f is None:
        f = plt.figure()

    if n_sims is None:
        n_sims = 1

    x = d[x_name].values
    xpos = param_values[x_name]
    pos = np.arange(len(xpos))
    xbins = param_interfaces[x_name]

    H = np.histogram(x, bins=xbins)

    f, ax = plt.subplots()
    ax.bar(pos, H[0] / n_sims * 100, width=0.5)
    ax.set_xticks(pos)
    ax.set_xticklabels(xpos, rotation=45)

    ax.set_xlabel(param_label[x_name])
    ax.set_ylabel('survival fraction [%]')

    ax.set_ylim(0, 100)

    return f


def filter_function(row, i0=0, i1=-1, dex_scatter=0.38, **kwargs):
    """
    Checks if the given entry fulfills the filter criteria.

    `row` can be a `dict` or a `pandas.Series` with parameters and values.

    For each keyword in kwargs, the function checks if the parameter
    with that name is in the given ranges. If that keyword contains
    a single value, the value of `row` needs to be identical. If the
    keyword has two values, they are the lower and upper bounds.

    If `corr` is set, then the distance to the correlation (in units of
    the scatter) needs to be within those limits, e.g. `corr=[-1, 1]`
    means that the flux needs to be within one standard deviation from
    the correlation. This is the same as `corr=1`.

    `i0` and `i1` are the first and last time snapshots that are being considered
    when chekcing the distance to the correlation.

    The function returns true if that parameter is in between (inclusive) the low or the
    high value.

    Arguments
    ---------

    row : dict-like
        contains the parameters and results

    dex_scatter : float
        how many dex of scatter around the correlation
        correspond to 1 sigma; note that this is in
        fluxes, not in effective radii.

    i0, i1 : int
        index of first and last snapshots to consider

    """

    # apply the parameter filters: if the parameter
    # is out of the range, return False

    for key, value in kwargs.items():

        # if the constraint is on the correlation
        # we deal with that later

        if key == 'corr':
            continue

        # make it a 1d list

        value = np.array(value, ndmin=1).tolist()
        if len(value) == 1:
            value = [value[0], value[0]]
        value = sorted(value)

        # check the constraint

        if row[key] < value[0] or row[key] > value[1]:
            return False

    # if the correlation is not limited,
    # we are done

    if 'corr' not in kwargs:
        return True

    # else, we FIRST apply the time filters

    r_eff = row['rf_t'][i0:i1]
    f_mm = row['flux_t'][i0:i1]

    # THEN calculate the distance to the correlation
    # in units of the scatter and check if the are all
    # in the range

    corr = np.array(kwargs['corr'], ndmin=1).tolist()
    if len(corr) == 1:
        corr = [-corr[0], corr[0]]
    corr = sorted(corr)

    distance = np.log10(f_mm / T17_Fmm_corr(r_eff)) / dex_scatter
    return np.all((distance > corr[0]) & (distance < corr[1]))


def histogram2D(d, x_name, y_name, param_values, param_interfaces=None, param_label=None, f=None, ax=None, vmax=1000, percent=False, n_total=None):
    """plot a 2D histogram of the samples in `d`.

    The key `x_name` and `y_name` will be the x and y axis. Their
    nicer names can be given in param_label. By default, counts are
    shown, but this can be translated to percentages per bin where
    the total number of values can also be rescaled with `n_total`.

    Parameters
    ----------
    d : pandas.DataFrame
        the data to be plotted. Each entry should at least have `x_name` and `y_name`.
    x_name : str
        name of the attribute to become the x-axis
    y_name : str
        name of the attribute to become the y-axis
    param_values : dict
        what individual unique values should exist in the attributes `x_name` and `y_name` in `d`.
    param_interfaces : dict, optional
        interfaces for the binning, by default None
    param_label : dict, optional
        contain a nicer string (e.g. using latex) for `x_name` and `y_name`, by default None
    f : figure handle, optional
        plot into this figure if given, else create one, by default None
    ax : axis handle, optional
        plot into this axis if given else create new one, by default None
    vmax : scalar, optional
        maxiumum value for the color scale, by default 1000
    percent : bool | string, optional
        by default False
        if False: plot counts
        if 'per_bin': assume equal distribution over bins, so
            if there are 100 samples, and 2x2 bins, then uniform distribution
            means every bin would have 25. A bin with 25 samples is then showing
            100%.
        if 'total': plot percent out of total number
            if there are 100 samples, and 2x2 bins, then uniform distribution
            means every bin is at 25%
    n_total : int, optional
        number of total samples used to calculate the percentages, by default None
        for example if we have 100 samples equally distributed on 2x2 bins, then 
        each bin would show 100% if `percent=='per_bin'`. But if we are only showing
        a subset of sample out of 200 total samples, we can set `n_total=200` to display
        50% in each bin.

    Returns
    -------
    figure handle
        the handle to the figure into which the histogram was drawn.
    """
    param_names = list(param_values.keys())

    if param_interfaces is None:
        param_interfaces = make_interfaces(param_values)

    if param_label is None:
        param_label = {k: k for k in param_names}

    if f is None:
        f = plt.figure()
    if ax is None:
        ax = f.add_subplot()

    x = d[x_name].values
    y = d[y_name].values

    # get the 2d histogram counts

    H = np.histogram2d(
        x, y,
        bins=[param_interfaces[x_name], param_interfaces[y_name]]
    )
    counts = H[0].T

    # convert to percentages if desired

    if percent:
        if percent == 'per_bin':
            n_bins = len(param_values[x_name]) * len(param_values[y_name])
        elif percent == 'total':
            n_bins = 1
        else:
            raise ValueError('percent must be False, \'per_bin\', or \'total\'.')
        if n_total is None:
            n_total = len(x)
        counts *= 100 / n_total * n_bins

    # plot the counts

    m = ax.imshow(counts, vmin=0, vmax=vmax, origin='lower')
    plt.colorbar(m)

    xvalues = param_values[x_name]
    xpos = np.arange(len(xvalues))
    ax.xaxis.set_ticks(xpos)
    ax.xaxis.set_ticklabels([f'{xvalues[i]:.4g}' for i in range(len(xvalues))], rotation=45)

    yvalues = param_values[y_name]
    ypos = np.arange(len(yvalues))
    ax.yaxis.set_ticks(ypos)
    ax.yaxis.set_ticklabels([f'{yvalues[i]:.4g}' for i in range(len(yvalues))], rotation=45)

    ax.set_xlabel(param_label[x_name])
    ax.set_ylabel(param_label[y_name])

    return f


def heatmap(d, i_snap, f=None, ax=None, observations=True, correlation=True, n_sig=1, **kwargs):
    """Draws a hexbin heatmap on the FL correlation

    Parameters
    ----------
    d : pandas data frame
        frame containing 'rf_t' and 'flux_t' arrays, will plot index i_snap
    i_snap : int
        the time index when to plot it
    f : figure handle, optional
        if figure exists pass it here, by default None
    ax : axis handle, optional
        if axis exists, pass it here, by default None
    observations : bool, optional
        if true, plot the Tripathi sample on top, by default True
    correlation : bool, optional
        if true, plot the Tripathi correlation on top, by default True
    n_sig : float, optional
        numbers of sigma to plot around the correlation, by default 1

    Other keywords are passed to plt.hexbin

    Returns
    -------
    f: figure handle
    ax: axis handle
    """

    if f is None:
        f = plt.figure()
    if ax is None:
        ax = f.add_subplot()

    Tripathi2017().plot_rosotti(ax=ax)

    # make the line solid white

    line = ax.get_lines()[0]
    if correlation:
        line.set_color('w')
        line.set_linestyle('-')
    else:
        line.remove()

    # make the dots white

    for col in ax.findobj(match=lambda o: 'Path' in type(o).__name__):
        if observations:
            col.set_facecolor('w')
        else:
            col.remove()

    # draw the heat map

    extent = kwargs.pop('extent', [*ax.set_xlim(), *ax.set_ylim()])
    kwargs['gridsize'] = kwargs.pop('gridsize', 35)

    _r = [row[i_snap] for row in d['rf_t']]
    _f = [row[i_snap] for row in d['flux_t']]
    ax.hexbin(np.log10(_r), np.log10(_f), zorder=-100, **kwargs, extent=extent)

    # add +/- sigma lines
    if correlation:
        x = np.logspace(1, 2.5, 50)
        ax.plot(np.log10(x), np.log10(T17_Fmm_corr(x, sigma=n_sig)), 'w--')
        ax.plot(np.log10(x), np.log10(T17_Fmm_corr(x, sigma=-n_sig)), 'w--')

    return f, ax
