import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .cgs_constants import k_B, h, c_light


def colored_line(x, y, time, colorbar=False, ax=None, cmap='viridis',
                 label=None, cmap_offset=0, **kwargs):
    """
    Draws a colored line. Time is the progress along the colorbar for a given point.

    Arguments:
    ----------

    x, y : array
        x,y-array

    time : array
        will be normalized from its minimum to maximum.

    colorbar : bool
        whether or not to add a colorbar

    ax : None | axis object
        draw into these axes, use gca else

    cmap : colormap
        which colormap to pass to the LineCollection

    label : None | str
        if None: do not add a label
        if empty string: just add a initial marker
        if string: also add this label to the marker

    text : None | str
        put this text on the marker

    cmap_offset : float
        offset the colormap by this amount (0 ... 1)

    Output:
    -------
    ax : the axis object
    line : the LineCollection
    """

    if 'lw' not in kwargs and 'linewidth' not in kwargs:
        kwargs['lw'] = 6

    text = kwargs.pop('text', None)

    # convert line to segments

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if ax is None:
        ax = plt.gca()

    # Create a continuous norm to map from data points to colors

    norm = plt.Normalize(time.min(), time.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
    lc.set_array(time + cmap_offset * (time.max() - time.min()))
    lc.set_linewidth(2)

    if text is not None:
        ax.text(x[-1], y[-1], text, c='w', horizontalalignment='center', verticalalignment='center', fontsize=6)

    line = ax.add_collection(lc)

    if label is not None:
        if type(cmap) is str:
            cmap = plt.get_cmap(cmap)
        c = kwargs.get('c', kwargs.get('color', cmap(0.75)))
        ax.scatter(x[0], y[0], c=[c], label=label, zorder=lc.get_zorder() + 1, s=50)
        ax.scatter(x[-1], y[-1], c=[c], zorder=lc.get_zorder() + 1, s=50)

    if colorbar:
        ax.figure.colorbar(line, ax=ax)

    return ax, line


def bplanck(freq, temp):
    """
    This function computes the Planck function

                   2 h nu^3 / c^2
       B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                  exp(h nu / kT) - 1

    Arguments:
     freq  [Hz]            = Frequency in Herz (can be array)
     temp  [K]             = Temperature in Kelvin (can be array)
    """
    const1  = h / k_B
    const2  = 2 * h / c_light**2
    const3  = 2 * k_B / c_light**2
    x       = const1 * freq / (temp + 1e-99)
    if np.isscalar(x):
        if x > 500.:
            x = 500.
    else:
        x[np.where(x > 500.)] = 500.
    bpl     = const2 * (freq**3) / ((np.exp(x) - 1.e0) + 1e-99)
    bplrj   = const3 * (freq**2) * temp
    if np.isscalar(x):
        if x < 1.e-3:
            bpl = bplrj
    else:
        ii      = x < 1.e-3
        bpl[ii] = bplrj[ii]
    return bpl


def nuker_profile(rho, rhot, alpha, beta, gamma, N=1):
    """
    Nuker profile used in Tripathi et al. 2017, from Lauer et al. 1995.

    for alpha > 0:

    - at rho<<rhot: profile \\propto rho^-gamma
    - at rho>>rhot: profile \\propto rho^-beta
    - alpha determines transition smoothness: larger alpha is sharper

    Arguments:
    ----------
    rho : array
        radial grid

    rhot : float
        transition radius

    alpha : float
        transition slope

    beta : float
        outer disk index

    gamma :
        inner disk index

    Keywords:
    ---------

    N : float
        normalization constant such that int(2*pi*rho*profile)=N

    Output:
    -------
    profile : array
        normalized Nuker profile on radial grid rho
    """
    profile = (rho / rhot)**-gamma * (1 + (rho / rhot)**alpha)**((gamma - beta) / alpha)
    profile = profile * N / np.trapz(2.0 * np.pi * rho * profile, x=rho)
    return profile
