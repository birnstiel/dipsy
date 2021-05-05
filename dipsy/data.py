"""
access to data sets or observational correlations functions.
"""
import pickle
import os
import warnings
import pkg_resources

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
from astroquery.vizier import Vizier

try:
    from lifelines import KaplanMeierFitter
    lifelines_available = True
except (ImportError, ValueError):
    lifelines_available = False

from .dipsy_functions import bplanck


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

    def plot(self, ax=None):
        """
        Creates a plot of the flux-alpha observations
        Returns the figure handle and axis handle.

        ax : None | axis handle
            if given, draw into these axes
        """

        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.figure

        for region in set(self.region):
            mask = np.array(self.region) == region
            ax.scatter(self.F_mm[mask], self.alpha_mm[mask], label=region)

        ax.set_xlim(10, 1000)
        ax.set_xscale('log')
        ax.set_ylim(1.5, 4.5)
        ax.set_xlabel(r'$F_\mathrm{mm}$ [mJy]')
        ax.set_ylabel(r'$\alpha_\mathrm{1-3mm}$')
        ax.legend(fontsize='small').get_frame().set_alpha(0)
        return f, ax


def T17_Reff_corr(Fmm, sigma=0):
    """
    Returns the effective radius according to the Tripathi
    correlation. It adds `sigma` standard deviations, where
    `sigma` can be positive or negative.
    """
    Fmm = np.asarray(Fmm)
    R_eff = 10**(2.12 + sigma * 0.19) * np.sqrt(Fmm)
    return np.squeeze(R_eff)


def T17_Fmm_corr(R_eff, sigma=0):
    """
    Returns the flux according to the Tripathi
    correlation. It adds `sigma` standard deviations, where
    `sigma` can be positive or negative.
    """
    R_eff = np.asarray(R_eff)
    Fmm = R_eff**2 * 10.**(-4.24 + sigma * 0.38)
    return np.squeeze(Fmm)


class Tripathi2017(object):

    R_eff = None
    L_mm = None

    def __init__(self):
        """
        Data from Tripathi et al. 2017 containing the

        R_eff : array
            log10 of effective radius [au]

        L_mm : array
            log10 of Luminosity at 857 micron [Jy]
        """
        fname = pkg_resources.resource_filename(__name__, os.path.join('datasets', 'tripathi2017.pickle'))

        with open(fname, 'rb') as fs:
            data = pickle.load(fs)
            self.R_eff = np.array(data['R_eff'])
            self.L_mm = np.array(data['L_mm'])

    def plot(self, ax=None):
        """
        Creates a plot of the size-luminosity relation.
        Returns the figure handle and axis handle.

        ax : None | axes handle
            if not None: draw into these axes
        """
        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.figure

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

    def plot_rosotti(self, sigma=False, ax=None):
        """
        Creates a plot of the size-luminosity relation with exchanged axes

        Arguments:
        ----------
        sigma : boolean
            if True: fill in between one sigma from the correlation

        ax : None | axes handle
            if not None: draw into these axes

        Output:
        -------

        Returns the figure handle and axis handle.
        """
        if ax is None:
            f, ax = plt.subplots()
        else:
            f = ax.figure

        ax.scatter(self.R_eff[:, 1], self.L_mm[:, 1], c='k', label='Tripathi et al. 2017')
        mask = np.isnan(self.R_eff[:, 0])
        # remove the red v markers and the next line as normal dots
        ax.scatter(self.R_eff[mask, 2], self.L_mm[mask, 1], c='k')

        ax.set_xlim(1, 2.3)
        ax.set_ylim(-2.5, .5)
        ax.set_xlabel(r'$\log\,R_\mathrm{eff}/\mathrm{AU}$')
        ax.set_ylabel(r'$\log\,L_\mathrm{mm}/\mathrm{Jy}$')
        x = np.array(ax.get_ylim())
        y = 2.13 + 0.51 * x

        # add the following lines to fill in between one sigma
        ax.plot(y, x, c='0.', ls='--', label=r'$\mathrm{R_{eff} \propto \sqrt{L_{mm}}}$')
        if sigma:
            # add the standard deviation
            ax.plot(y + 0.19, x, c='0.75', ls='--')
            ax.plot(y - 0.19, x, c='0.75', ls='--')
            ax.fill_between(y, x - 0.38, x + 0.38, alpha=0.3)
        return f, ax


class mm_survey_dataset():

    name = 'Ansdell16'
    paper = 'Ansdell et al. 2016'
    columns = ['Name', 'F890', 'e_F890']
    catalog = 'J/ApJ/828/46/alma'
    d = 150 * u.pc
    lam = 0.133 * u.cm
    unit = u.mJy
    T0 = 20 * u.K

    # Beckwith+1990

    beta = 1
    kappa0 = 3.5 * u.cm**2 / u.g
    lam0 = 870 * u.um

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.get_data()
        self.calculate_masses()
        if lifelines_available:
            self.calculate_KaplanMaier()
        else:
            warnings.warn('lifelines module not available')

    def get_data(self, X=1):
        """Download data from Vizier.

        we want to get flux values (self.values), errors (self.errors),
        and if available lower-limit flags (self.flags). If the latter
        is given or not, we will also dicard everything below X sigma.
        """
        v = Vizier(columns=[c for c in self.columns if c is not None])
        v.ROW_LIMIT = -1
        res = v.get_catalogs(self.catalog)

        # get the fluxes

        self.values = res[0][self.columns[0]].filled(fill_value=0).data

        # get the errors

        if self.columns[1] is None:
            self.errors = None
        else:
            self.errors = res[0][self.columns[1]].filled(fill_value=0).data

        # get or set the flags

        if len(self.columns) > 2:
            self.flag = res[0][self.columns[2]].filled(fill_value='').data
            self.flag = 1.0 - 1.0 * (self.flag.astype(str) == '<')
        else:
            self.flag = np.ones_like(self.values)

        # also set flags for low values
        if self.errors is not None:
            self.flag[self.values < X * self.errors] = 0

    def calculate_masses(self):
        """
        The simple Beckwith scaling from flux to dust mass.

            M = F * d**2 (B_nu(T) * kappa)

        We use an opacity law of

            kappa(lambda) = kappa0 * (lambda/lambda0)**-beta

        Keywords:
        ---------

        F : quantity
            if unitless: returns the conversion factor. If given as flux quantity, will
            return a mass

        lam : quantity
            wavelength

        T0 : quantity
            disk mean temperature

        d : quantity
            distance with Units
        """
        nu = c.c.cgs / self.lam
        Bnu = bplanck(nu.cgs.value, self.T0.cgs.value) * u.erg / (u.cm**2 * u.Hz * u.s)
        kappa = self.kappa0 * (self.lam / self.lam0)**(-self.beta)  # Beckwith 1990

        self.Mearth = (self.values * self.unit * self.d**2 / (Bnu * kappa)).to(c.M_earth).value
        if self.errors is not None:
            self.e_Mearth = (self.errors * self.unit * self.d**2 / (Bnu * kappa)).to(c.M_earth).value

    def calculate_KaplanMaier(self):
        if not lifelines_available:
            warnings.warn('lifelines module not available')
        else:
            values = self.Mearth
            limits = self.flag
            kmf = KaplanMeierFitter()
            kmf.fit_left_censoring(values, limits)

            x_values = kmf.cumulative_density_.index.values
            cdfm1 = 1. - kmf.cumulative_density_.values
            minmax = 1. - kmf.confidence_interval_.values
            minmax[-1] = np.array([0., 0.])

            self.km_x = x_values
            self.km_y_low = minmax[:, 0]
            self.km_y_high = minmax[:, 1]
            self.km_y = cdfm1[:, 0]


class tychoniec_dataset(mm_survey_dataset):
    """
    Note: Tychoniec used 235 pc, but it was updated to 293 pc.
    Note: Tychoniek needs to be filtered using the `final_list`, that he sent me.
    """
    final_list = ['Per-emb-1', 'Per-emb-2', 'Per-emb-3', 'Per-emb-4', 'Per-emb-5', 'Per-emb-6', 'Per-emb-7', 'Per-emb-8', 'Per-emb-9', 'Per-emb-10', 'Per-emb-11-A', 'Per-emb-11-B', 'Per-emb-11-C', 'Per-emb-12-A', 'Per-emb-12-B', 'Per-emb-13', 'Per-emb-14', 'Per-emb-15', 'Per-emb-16', 'Per-emb-17-A', 'Per-emb-17-B', 'Per-emb-18', 'Per-emb-19', 'Per-emb-20', 'Per-emb-21', 'Per-emb-22-A', 'Per-emb-22-B', 'Per-emb-23', 'Per-emb-24', 'Per-emb-25', 'Per-emb-26', 'Per-emb-27-A', 'Per-emb-27-B', 'Per-emb-28', 'Per-emb-29', 'Per-emb-30', 'Per-emb-31', 'Per-emb-32-A', 'Per-emb-32-B', 'Per-emb-33-A', 'Per-emb-33-B', 'Per-emb-33-C', 'Per-emb-34', 'Per-emb-35-B', 'Per-emb-35-A', 'Per-emb-36-A', 'Per-emb-36-B',
                  'Per-emb-37', 'Per-emb-38', 'Per-emb-39', 'Per-emb-40-A', 'Per-emb-40-B', 'Per-emb-41', 'Per-emb-42', 'Per-emb-43', 'Per-emb-44-A', 'Per-emb-44-B', 'Per-emb-45', 'Per-emb-46', 'Per-emb-47', 'Per-emb-48-A', 'Per-emb-48-B', 'Per-emb-49-A', 'Per-emb-49-B', 'Per-emb-50', 'Per-emb-51', 'Per-emb-52', 'Per-emb-53', 'Per-emb-54', 'Per-emb-55-A', 'Per-emb-55-B', 'Per-emb-56', 'Per-emb-57', 'Per-emb-58', 'Per-emb-59', 'Per-emb-60', 'Per-emb-61', 'Per-emb-62', 'Per-emb-63', 'Per-emb-64', 'Per-emb-65', 'Per-emb-66', 'Per-bolo-58', 'Per-bolo-45', 'L1451-MMS', 'L1448IRS2E', 'B1-bN', 'B1-bS', 'L1448IRS1-A', 'L1448IRS1-B', 'L1448NW-A', 'L1448NW-B', 'L1448IRS3A', 'SVS13C', 'SVS13B', 'IRAS03363+3207', "IRAS4B'", 'SVS13A2']

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        self.get_data()

        # get the names from vizier, the 4th row makes it a list of strings

        v = Vizier(columns=['Name'])
        v.ROW_LIMIT = -1
        self.names = v.get_catalogs(self.catalog)[0]
        self.names = [s[0] for s in self.names.as_array()]

        # include only the ones that are in final list

        mask = np.array([n in self.final_list for n in self.names])

        self.values = self.values[mask]
        if self.errors is not None:
            self.errors = self.errors[mask]
        self.flag = self.flag[mask]

        # continue as before

        self.calculate_masses()
        self.calculate_KaplanMaier()


class ansdell17(mm_survey_dataset):
    def __init__(self, **kwargs):
        super().__init__(
            name='Ansdell17',
            paper='Ansdell et al. 2017',
            catalog='J/AJ/153/240/sources',
            columns=['F1.33', 'e_F1.33'],
            unit=u.mJy,
            d=385 * u.pc,
            lam=1.33 * u.mm)


class ansdell16(mm_survey_dataset):
    def __init__(self):
        super().__init__(
            name='Ansdell16',
            paper='Ansdell et al. 2016',
            catalog='J/ApJ/828/46/alma',
            columns=['F890', 'e_F890'],
            unit=u.mJy,
            d=150 * u.pc,
            lam=1.33 * u.mm)


class barenfeld16(mm_survey_dataset):
    def __init__(self):
        super().__init__(
            name='Barenfeld16',
            paper='Barenfeld et al. 2016',
            catalog='J/ApJ/827/142/stars',
            columns=['Snu', 'e_Snu'],
            unit=u.mJy,
            d=145 * u.pc,
            lam=880 * u.um)


class pascucci16(mm_survey_dataset):
    def __init__(self):
        super().__init__(
            name='Pascucci16',
            paper='Pascucci et al. 2016',
            catalog='J/ApJ/831/125/sources',
            columns=['Fnu', 'e_Fnu'],
            unit=u.mJy,
            d=160 * u.pc,
            lam=887 * u.um)


class andrews13(mm_survey_dataset):
    def __init__(self):
        super().__init__(
            name='Andrews13',
            paper='Andrews et al. 2013',
            catalog='J/ApJ/771/129/table2',
            columns=['F1.3', 'e_F1.3', 'l_F1.3'],
            unit=u.Jy,
            d=140 * u.pc,
            lam=1.3 * u.mm)


class tychoniec18(tychoniec_dataset):
    def __init__(self):
        super().__init__(
            name='Tychoniec18',
            paper='Tychoniec et al. 2018',
            catalog='J/ApJS/238/19/protostars',
            columns=['Fc9', None, 'l_Fc9'],
            unit=u.mJy,
            d=293 * u.pc,
            lam=9 * u.mm)


# ansdell17 = mm_survey_dataset(name='Ansdell17',   paper='Ansdell et al. 2017',   catalog='J/AJ/153/240/sources',     columns=['F1.33', 'e_F1.33'],         unit=u.mJy, d=385*u.pc, lam=1.33*u.mm)
# ansdell16 = mm_survey_dataset(name='Ansdell16',   paper='Ansdell et al. 2016',   catalog='J/ApJ/828/46/alma',        columns=['F890', 'e_F890'],           unit=u.mJy, d=150*u.pc, lam=1.33*u.mm)
# barenfeld16 = mm_survey_dataset(name='Barenfeld16', paper='Barenfeld et al. 2016', catalog='J/ApJ/827/142/stars',      columns=['Snu', 'e_Snu'],             unit=u.mJy, d=145*u.pc, lam=880*u.um)
# pascucci16 = mm_survey_dataset(name='Pascucci16',  paper='Pascucci et al. 2016',  catalog='J/ApJ/831/125/sources',    columns=['Fnu', 'e_Fnu'],             unit=u.mJy, d=160*u.pc, lam=887*u.um)
# andrews13 = mm_survey_dataset(name='Andrews13',   paper='Andrews et al. 2013',   catalog='J/ApJ/771/129/table2',     columns=['F1.3', 'e_F1.3', 'l_F1.3'], unit=u.Jy,  d=140*u.pc, lam=1.3*u.mm)
# tychoniec18 = tychoniec_dataset(name='Tychoniec18', paper='Tychoniec et al. 2018', catalog='J/ApJS/238/19/protostars', columns=['Fc9', None, 'l_Fc9'],       unit=u.mJy, d=293*u.pc, lam=9*u.mm)
