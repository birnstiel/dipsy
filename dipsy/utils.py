"""
Utility functions that do not fit anywhere else:

- analytical functions like the Nuker profile, Planck function
- helper functions like
    - a colored line to plot tracks
    - function to calculate grid interfaces
"""

import numbers
import sys
from io import StringIO
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import h5py
from dustpylib.radtrans.slab.slab import bplanck

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
        ax.text(x[-1], y[-1], text, c='w', horizontalalignment='center',
                verticalalignment='center', fontsize=6)

    line = ax.add_collection(lc)

    if label is not None:
        if type(cmap) is str:
            cmap = plt.get_cmap(cmap)
        c = kwargs.get('c', kwargs.get('color', cmap(0.75)))
        ax.scatter(x[0], y[0], c=[c], label=label,
                   zorder=lc.get_zorder() + 1, s=50)
        ax.scatter(x[-1], y[-1], c=[c], zorder=lc.get_zorder() + 1, s=50)

    if colorbar:
        ax.figure.colorbar(line, ax=ax)

    return ax, line


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
    profile = (rho / rhot)**-gamma * (1 + (rho / rhot)
                                      ** alpha)**((gamma - beta) / alpha)
    profile = profile * N / np.trapz(2.0 * np.pi * rho * profile, x=rho)
    return profile


def get_interfaces_from_log_cell_centers(x):
    """
    Returns the cell interfaces for an array of logarithmic
    cell centers.

    Arguments:
    ----------

    x : array
    :   Array of logarithmically spaced cell centers for
        which the interfaces should be calculated

    Output:
    -------

    xi : array
    :    Array of length len(x)+1 containing the grid interfaces
         defined such that 0.5*(xi[i]+xi[i+1]) = xi
    """
    x = np.asarray(x)
    B = x[1] / x[0]
    A = (B + 1) / 2.
    xi = np.append(x / A, x[-1] * B / A)
    return xi


def write_to_hdf5(fname, results):
    """writes simulations to hdf5

    Writes one group for each entry in results.

    Parameters
    ----------
    fname : str | path
        hdf5 file name

    results : list of dicts or namedtuples
        each of these will become one group in the file

    Raises
    ------
    ValueError
        if entry in results is not a dict or namedtuple
    """
    with h5py.File(fname, 'w') as f:
        for i, result in enumerate(results):
            hdf5_add_dict(f, i, result)


def hdf5_add_dict(f, i, result):
    """addds dict as group to hdf5 file

    Parameters
    ----------
    f : open, writable h5py file object
        file into which to store the data
    i : int | str
        index -- will be used to create a length-7, zero padded string of that number as index
        if it is already such a sting, it will just use it
    result : dict | namedtuple
        data to store in the hdf5 file

    Raises
    ------
    ValueError
        if dataset is not a dict or namedtuple (or can be converted to a dict with its `_asdict` method)
    """
    # create a group for each simulation
    if type(i) is str:
        key = i
    else:
        key = f'{i:07d}'

    group = f.create_group(key)

    # this should work for namedtuples and dicts

    if not isinstance(result, dict):
        if hasattr(result, '_asdict'):
            result = result._asdict()
        else:
            raise ValueError('result must be dict or namedtuple')

    for key, val in result.items():
        if isinstance(val, (numbers.Number, np.number)):
            group.create_dataset(key, data=val)
        else:
            group.create_dataset(key, data=val, compression='lzf')


def read_from_hdf5(fname, simname=None):
    """Reads simulation from hdf5 file

    Parameters
    ----------
    fname : str | path
        hdf5 file name

    simname : str | None
        - if `None`, then all are read into a list of dicts
        - if `str`, then the group with this name is read

    """
    with h5py.File(fname, 'r') as f:
        simnames = [simname]
        if simname is None:
            simnames = list(f)

        dicts = []

        for name in simnames:
            group = f[name]

            d = dict()

            for key, value in group.items():
                d[key] = group[key][()]

            dicts += [d]

    if simname is not None:
        return dicts[0]
    else:
        return dicts


def is_interactive():
    """
    Function to test if this code is executed interactively (in notebook or console) or if it is run as a script.

    Returns:
    --------
    True if run in notebook or console, False if script.

    """
    import __main__ as main
    return not hasattr(main, '__file__')


class Capturing(list):
    """Context manager capturing standard output of whatever is called in it.

    Keywords
    --------
    stderr : bool
        if True will capture the standard error instead of standard output.
        defaulats to False

    Examples
    --------
    >>> with Capturing() as output:
    >>>     do_something(my_object)

    `output` is now a list containing the lines printed by the function call.

    This can also be concatenated

    >>> with Capturing() as output:
    >>>    print('hello world')
    >>> print('displays on screen')
    displays on screen

    >>> with output:
    >>>     print('hello world2')
    >>> print('output:', output)
    output: ['hello world', 'hello world2']

    >>> import warnings
    >>> with output, Capturing(stderr=True) as err:
    >>>     print('hello world2')
    >>>     warnings.warn('testwarning')
    >>> print(output)
    output: ['hello world', 'hello world2']

    >>> print('error:', err[0].split(':')[-1])
    error:  testwarning

    Mostly copied from [this stackoverflow answer](http://stackoverflow.com/a/16571630/2108771)

    """

    def __init__(self, *args, stderr=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._error = stderr

    def __enter__(self):
        """Start capturing output when entering the context"""
        if self._error:
            self._std = sys.stderr
            sys.stderr = self._stringio = StringIO()
        else:
            self._std = sys.stdout
            sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        """Get & return the collected output when exiting context"""
        self.extend(self._stringio.getvalue().splitlines())
        if self._error:
            sys.stderr = self._std
        else:
            sys.stdout = self._std


def remote_import(filename):
    "imports remote python file `filename`"
    filename = Path(filename)
    module_name = filename.stem
    file_path = str(filename.absolute())
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
