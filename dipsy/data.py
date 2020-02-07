import pickle
import os
import pkg_resources

import numpy as np
import matplotlib.pyplot as plt


class Testi2014(object):

    F_mm = None
    alpha_mm = None
    d_pc = None
    region = None

    def __init__(self):
        """
        Data from Testi et al. 2014 containing

        F_mm : array
            flux at 1 mm [Jy]

        alpha_mm : array
            spectral index between 1 and 3 mm

        region : list
            star forming region

        d_pc : array
            distance in parsec
        """
        fname = pkg_resources.resource_filename(__name__, os.path.join('datasets', 'testi2014.csv'))

        self.F_mm, self.alpha_mm, self.d_pc = \
            np.loadtxt(fname, skiprows=1, delimiter=',', usecols=[0, 1, 3]).T

        with open(fname) as fid:
            _ = fid.readline()
            self.region = [line.split('"')[1] for line in fid.readlines()]

    def plot(self):
        """
        Creates a plot of the flux-alpha observations
        Returns the figure handle and axis handle.
        """

        f, ax = plt.subplots()

        ax.scatter(self.F_mm, self.alpha_mm, c='k')
        ax.set_xlim(10, 1000)
        ax.set_xscale('log')
        ax.set_ylim(1.5, 4.5)
        ax.set_xlabel(r'$F_\mathrm{mm}$ [Jy]')
        ax.set_ylabel(r'$\alpha_\mathrm{1-3mm}$')
        return f, ax



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
