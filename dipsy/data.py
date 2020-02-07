import pickle
import os
import pkg_resources

import numpy as np
import matplotlib.pyplot as plt


class Tripathi2017(object):

    R_eff = None
    L_mm = None

    def __init__(self):
        """
        Data from Tripathi et al. 2017 containing the

        R_eff : array
            effective radius [au]

        L_mm : array
            Luminosity at 857 micron [Jy]
        """
        fname = pkg_resources.resource_filename(__name__, os.path.join('datasets', 'tripathi2017.pickle'))

        with open(fname, 'rb') as fs:
            data = pickle.load(fs)
            self.R_eff = np.array(data['R_eff'])
            self.L_mm = np.array(data['L_mm'])

    def plot(self):
        """
        Creates a plot of the size-luminosity relation.
        Returns the figure handle and axis handle.
        """

        f, ax = plt.subplots()

        ax.scatter(self.L_mm[:, 1], self.R_eff[:, 1], c='k')
        mask = np.isnan(self.R_eff[:, 0])
        ax.scatter(self.L_mm[mask, 1], self.R_eff[mask, 2], marker='v', c='r')
        ax.set_xlim(-3, 0.7)
        ax.set_ylim(0.5, 3.1)
        ax.set_xlabel(r'$\log\,L_\mathrm{mm}/\mathrm{Jy}$')
        ax.set_ylabel(r'$\log\,R_\mathrm{eff}/\mathrm{AU}$')
        x = np.array(ax.get_xlim())
        y = 2.13 + 0.51 * x
        ax.plot(x, y, c='0.5', ls='--')
        return f, ax
