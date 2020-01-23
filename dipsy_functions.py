import numpy as np

import astropy.constants as c

h = c.h.cgs.value
c_light = c.c.cgs.value
k_B = c.k_B.cgs.value


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
