import os

import numpy as np
import astropy.constants as c
from scipy.interpolate import interp2d

import dsharp_opac

h = c.h.cgs.value
c_light = c.c.cgs.value
k_B = c.k_B.cgs.value
pc = c.pc.cgs.value


def get_powerlaw_dust_distribution(sigma_d, a_max, q=3.5, na=10, a0=None, a1=None):
    """
    Makes a power-law size distribution up to a_max, normalized to the given surface density.

    Arguments:
    ----------

    sigma_d : array
        dust surface density array

    a_max : array
        maximum particle size array

    Keywords:
    ---------

    q : float
        particle size index, n(a) propto a**-q

    na : int
        number of particle size bins

    a0 : float
        minimum particle size

    a1 : float
        maximum particle size

    Returns:
    --------

    a : array
        particle size grid (centers)

    a_i : array
        particle size grid (interfaces)

    sig_da : array
        particle size distribution of size (len(sigma_d), na)
    """

    if a0 is None:
        a0 = a_max.min()

    if a1 is None:
        a1 = 2 * a_max.max()

    nr = len(sigma_d)
    sig_da = np.zeros([nr, na]) + 1e-100

    a_i = np.logspace(np.log10(a0), np.log10(a1), na + 1)
    a = 0.5 * (a_i[1:] + a_i[:-1])

    for ir in range(nr):

        if a_max[ir] <= a0:
            sig_da[ir, 0] = 1
        else:
            i_up = np.where(a_i < a_max[ir])[0][-1]

            # filling all bins that are strictly below a_max

            for ia in range(i_up):
                sig_da[ir, ia] = a_i[ia + 1]**(4 - q) - a_i[ia]**(4 - q)

            # filling the bin that contains a_max
            sig_da[ir, i_up] = a_max[ir]**(4 - q) - a_i[i_up]**(4 - q)

        # normalize

        sig_da[ir, :] = sig_da[ir, :] / sig_da[ir, :].sum() * sigma_d[ir]

    return a, a_i, sig_da


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


class Opacity(object):
    _filename = None
    _lam      = None
    _a        = None
    _k_abs    = None
    _k_sca    = None
    _g        = None
    _rho_s    = None

    @property
    def rho_s(self):
        "material density in g/cm^3"
        return self._rho_s

    def __init__(self, input=None):

        # set default opacities

        if input is None:
            input = 'default_opacities_smooth.npz'

        if type(input) is str and not os.path.isfile(input):
            input = dsharp_opac.get_datafile('default_opacities_smooth.npz')
            if not os.path.isfile(input):
                raise ValueError('unknown input')

        if type(input) is str and os.path.isfile(input):
            self._filename = input
            with np.load(input) as f:
                self._load_from_dict_like(f)

        elif type(input) is dict:
            self._load_from_dict_like(dict)

    def _load_from_dict_like(self, input):
        for attr in ['a', 'lam', 'k_abs', 'k_sca', 'g', 'rho_s']:
            if attr in input:
                setattr(self, '_' + attr, input[attr])
            else:
                print(f'{attr} not in input')
        if self._rho_s is not None:
            self._rho_s = float(self._rho_s)

        self._interp_k_abs = interp2d(np.log10(self._lam), np.log10(self._a), np.log10(self._k_abs))
        self._interp_k_sca = interp2d(np.log10(self._lam), np.log10(self._a), np.log10(self._k_sca))

    def get_opacities(self, a, lam):
        """
        Returns the absorption and scattering opacities for the given particle
        size a and wavelength lam.

        Arguments:
        ----------

        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm

        Returns:
        --------
        k_abs, k_sca : arrays
            absorption and scattering opacities, each of shape (len(a), len(lam))
        """
        return \
            10.**self._interp_k_abs(np.log10(lam), np.log10(a)), \
            10.**self._interp_k_sca(np.log10(lam), np.log10(a)),


def get_observables(r, sig_g, sig_d, a_max, T, opacity, distance=140 * pc, flux_fraction=0.68):
    """
    Calculates the radial profiles of the (vertical) optical depth and the intensity for a given simulation
    at a given time (using the closest simulation snapshot).

    Arguments:
    ----------

    res : twopoppy.results.results
        twopoppy simulation results object

    time : float
        time at which to calculate the results [s]

    lam : array
        wavelengths at which to calculate the results [cm]

    a_opac : array
        particle size grid on which the opacities are defined [cm]

    k_a : array
        absorption opacity as function of wavelength (grid lam) and
        particle size (grid a_opac) [cm^2/g]

    Keywords:
    ---------

    distance : float
        distance to source [cm]

    flux_fraction : float
        at which fraction of the total flux the effective radius is defined [-]

    Output:
    -------

    rf : array
        effective radii for every wavelength [cm]

    flux_t : array
        integrated flux for every wavelength [Jy]

    tau,Inu : array
        optical depth and intensity profiles at every wavelength [-, Jy/arcsec**2]

    sig_da, : array
        reconstructed particle size distribution on grid (res.a, res.x)

    a_max : array
        maximum particle size [cm]
    """

    # interpolate opacity on the same particle size grid as the size distribution

    kappa = np.array([10.**np.interp(np.log10(res.a), np.log10(a_opac), np.log10(k)) for k in k_a.T]).T

    it = np.abs(res.timesteps - time).argmin()

    if res.T.ndim == 1:
        T = res.T
    else:
        T = res.T[it]

    # reconstruct the size distribution

    sig_da, a_max = get_distri(res, it)

    # calculate planck function at every wavelength and radius

    Bnu = planck_B_nu(c_light / (np.array(lam, ndmin=2).T), np.array(T, ndmin=2))  # shape = (n_lam, n_r)

    # calculate optical depth

    tau = (kappa.T[:, :, np.newaxis] * sig_da[np.newaxis, :, :])  # shape = (n_l, n_a, n_r)
    tau = tau.sum(1)  # shape = (n_l, n_r)

    # calculate intensity at every wavelength and radius for this snapshot
    # here the intensity is still in plain CGS units (per sterad)

    intens = Bnu * (1 - np.exp(-tau))

    # calculate the fluxes

    flux = distance**-2 * cumtrapz(2 * np.pi * res.x * intens, x=res.x, axis=1, initial=0)  # integrated flux density
    flux_t = flux[:, -1] / 1e-23  # store the integrated flux density in Jy (sanity check: TW Hya @ 870 micron and 54 parsec is about 1.5 Jy)

    # converted intensity to Jy/arcsec**2

    Inu = intens * arcsec_sq / 1e-23

    # interpolate radius whithin which >=68% of the dust mass is
    rf = np.array([np.interp(flux_fraction, _f / _f[-1], res.x) for _f in flux])

    return rf, flux_t, tau, Inu, sig_da, a_max
