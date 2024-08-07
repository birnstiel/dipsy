"""
Functions to calculate observables or read data from simulations
"""
from pathlib import Path
from types import SimpleNamespace
import h5py
from dustpylib.radtrans.slab.slab import I_over_B_EB, bplanck

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from collections import namedtuple

from .cgs_constants import year, jy_sas, c_light, pc
from .utils import bplanck

import dsharp_opac

observables = namedtuple(
    'observables', ['rf', 'flux_t', 'tau', 'I_nu', 'a', 'sig_da'])
dustpy_result = namedtuple('dustpy_result', [
                           'r', 'a_max', 'a', 'a_mean', 'sig_d', 'sig_da', 'sig_g', 'time', 'T'])
rosotti_result = namedtuple('rosotti_result', [
                            'a_max', 'time', 'T', 'sig_d', 'sig_g', 'd2g', 'r', 'L_star', 'M_star', 'T_star'])


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

    q : float | array
        particle size index, n(a) propto a**-q
        if array, it has to have the same length as sigma_d

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

    # we want to turn q into an array if it isn't one already
    q = q * np.ones(nr)

    for ir in range(nr):

        if a_max[ir] <= a0:
            sig_da[ir, 0] = 1
        else:
            i_up = np.where(a_i < a_max[ir])[0][-1]

            # filling all bins that are strictly below a_max

            if q[ir] == 4.0:
                for ia in range(i_up):
                    sig_da[ir, ia] = np.log(a_i[ia + 1] / a_i[ia])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = np.log(a_max[ir] / a_i[i_up])
            else:
                for ia in range(i_up):
                    sig_da[ir, ia] = a_i[ia +
                                         1]**(4 - q[ir]) - a_i[ia]**(4 - q[ir])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = a_max[ir]**(4 - q[ir]) - \
                    a_i[i_up]**(4 - q[ir])

        # normalize

        sig_da[ir, :] = sig_da[ir, :] / sig_da[ir, :].sum() * sigma_d[ir]

    return a, a_i, sig_da


def read_rosotti_data(fname):
    """
    Reads Giovanni Rosottis HDF5 file and returns
    a dict with numpy arrays, all in CGS.
    """
    from astropy import constants as c
    from astropy import units as u
    import numpy as np

    au = u.au.to('cm')
    M_sun = c.M_sun.cgs.value
    L_sun = c.L_sun.cgs.value

    with h5py.File(fname) as f:

        dset = f['pop_0']

        snap_indices = []
        for key in dset.keys():
            if key.startswith('time_'):
                snap_indices += [int(key.replace('time_', ''))]

        snap_indices.sort()

        r = dset[f'time_{snap_indices[0]}/yso_0/Disk/Rc'][()] * au
        n_t = len(snap_indices)
        n_r = len(r)

        amax = np.zeros([n_t, n_r])
        d2g = np.zeros([n_t, n_r])
        T = np.zeros([n_t, n_r])
        sig_d = np.zeros([n_t, n_r])
        sig_g = np.zeros([n_t, n_r])
        age = np.zeros([n_t])
        L_star = np.zeros([n_t])
        M_star = np.zeros([n_t])
        T_star = np.zeros([n_t])

        for idx in snap_indices:
            sig = dset[f'time_{idx}/yso_0/Disk/sigma'][()]
            d2g = dset[f'time_{idx}/yso_0/Disk/dust_frac'][()].sum(0)

            sig_g[idx, :] = sig / (1 + d2g)
            sig_d[idx, :] = sig / (1 + d2g) * d2g
            amax[idx, :] = dset[f'time_{idx}/yso_0/Disk/amax'][()]
            T[idx, :] = dset[f'time_{idx}/yso_0/Disk/T'][()]

            age[idx] = dset[f'time_{idx}/yso_0/evolution_time'][()] / \
                (2 * np.pi) * year
            L_star[idx] = dset[f'time_{idx}/yso_0/Star/llum'][()] * L_sun
            M_star[idx] = dset[f'time_{idx}/yso_0/Star/mass'][()] * M_sun
            T_star[idx] = dset[f'time_{idx}/yso_0/Star/teff'][()]

        return rosotti_result(amax, age, T, sig_d, sig_g, d2g, r, L_star, M_star, T_star)


def read_dustpy_data(data_path, time=None):
    """
    Read the dustpy files from the directory data_path, then interpolate the
    densities at the given time snapshots (or take the original time if None is
    passed).

    Arguments:
    ----------

    data_path : str
        path to the output directory where the hdf5 files are

    time : array | None
        if not None: interpolate at those times

    Output:
    -------
    returns dict with these keys:
    - r
    - a_max
    - sig_d
    - sig_g
    - time
    """
    import dustpy
    from scipy.interpolate import interp2d

    reader = dustpy.hdf5writer()
    reader.datadir = str(data_path)

    time_dp = reader.read.sequence("t")

    # Read the radial and mass grid
    r = reader.read.sequence("grid/r")[0]
    # rInt = reader.read.sequence("grid/rInt")
    # m = reader.read.sequence("grid/m")

    # Read the gas and dust densities
    sig_g = reader.read.sequence("gas/Sigma")
    sig_da = reader.read.sequence("dust/Sigma")
    sig_d = sig_da.sum(-1)

    # Read the stokes number and dust size
    # St = reader.read.sequence("dust/St")
    a = reader.read.sequence("dust/a")[0, 0, :]

    # Read the star mass and radius
    # M_star = reader.read.sequence("star/M", files]
    # R_star = reader.read.sequence("star/R", files]

    # Obtain the dust to gas ratio
    # d2g = sig_d / sig_g

    # Read the Gas and Dust scale height
    # Hg = reader.read.sequence("gas/Hp")
    # Hd = reader.read.sequence("dust/h")

    # Read the gas (viscous) and dust velocities
    # Vel_g = reader.read.sequence("gas/v/visc")
    Vel_d = reader.read.sequence("dust/v/rad")

    # Read the alpha parameter and the orbital angular velocity
    # Alpha  = reader.read.sequence("gas/alpha")
    # OmegaK = reader.read.sequence("grid/OmegaK")

    # Read the gas midplane density, sound speed, and eta parameter
    # rho = reader.read.sequence("gas/rho")
    # cs = reader.read.sequence("gas/cs")
    # eta = reader.read.sequence("gas/eta")

    T = reader.read.sequence("gas/T")

    # Obtain the Accretion Rate of dust and gas
    # Acc_g = 2 * np.pi * r * Vel_g * sig_g
    # Acc_d = 2 * np.pi * r * (Vel_d * sig_d).sum(-1)

    # Obtain the alpha-viscosity
    # Visc =  Alpha * cs * cs / OmegaK

    a_mean = (a * sig_da * np.abs(Vel_d)).sum(-1) / \
        (sig_da * np.abs(Vel_d)).sum(-1)
    a_max = a[sig_da.argmax(-1)]

    if time is None:
        time = time_dp
    else:
        f_Td = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(T))
        f_sd = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(sig_d))
        f_sg = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(sig_g))
        f_ax = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(a_max))
        f_am = interp2d(np.log10(r), np.log10(
            time_dp + 1e-100), np.log10(a_mean))

        T = 10.**f_Td(np.log10(r), np.log10(time + 1e-100))
        sig_d = 10.**f_sd(np.log10(r), np.log10(time + 1e-100))
        sig_g = 10.**f_sg(np.log10(r), np.log10(time + 1e-100))
        a_max = 10.**f_ax(np.log10(r), np.log10(time + 1e-100))
        a_mean = 10.**f_am(np.log10(r), np.log10(time + 1e-100))

        sig_da_new = np.zeros([len(time), len(r), len(a)])
        for ia in range(len(a)):
            f = interp2d(np.log10(r), np.log10(time_dp + 1e-100),
                         np.log10(sig_da[:, :, ia]))
            sig_da_new[:, :, ia] = 10.**f(np.log10(r), np.log10(time + 1e-100))

        sig_da = sig_da_new

    dp = dustpy_result(r, a_max, a, a_mean, sig_d, sig_da, sig_g, time, T)

    return dp


def get_observables(r, sig_g, sig_d, a_max, T, opacity, lam, distance=140 * pc,
                    flux_fraction=0.68, a=None, q=3.5, na=50, a0=None, a1=None, scattering=True,
                    inc=0.0):
    """
    Calculates the radial profiles of the (vertical) optical depth and the intensity for a given simulation
    at a given time (using the closest simulation snapshot).

    Arguments:
    ----------

    r : array
        radial array [cm]

    sig_g : array
        gas surface density on r [g/cm^2]

    sig_d : 1d-array | 2d-array
        - 1d: dust surface density on r [g/cm^2]
        - 2d: dust surface density on r and a shape=(len(r), len(a)) [g/cm^2]

    a_max : array
        maximum particle size on r [cm]

    T : array
        temperature grid on r [K]

    opacity : instance of Opacity
        the opacity to use for the calculation

    lam : array
        the wavelengths at which to calculate the observables [cm]

    Keywords:
    ---------

    distance : float
        distance to source [cm]

    flux_fraction : float
        at which fraction of the total flux the effective radius is defined [-]

    a : None | float
        if size distribution information is known (= sig_d is 2D), pass the
        particle size array here

    q : float | array
        size exponent to use: n(a) ~ a^-q, so 3.5=MRN
        if array, it has to have the same length as sig_d

    na : int
        length of the particle size grid

    a0 : float
        minimum particle size to use for the dust size distribution [cm]

    a1 : float
        maximum particle size to use for the dust size distribution [cm]

    scattering : bool
        if True, use the scattering solution, else just absorption

    inc : float
        inclination, default is 0.0 = face-on. This is just treated as
        increasing the path length across the slab model.

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
    """
    from scipy.integrate import cumtrapz

    # get the size distribution
    if (a is not None and sig_d.ndim != 2) or (a is None and sig_d.ndim != 1):
        raise ValueError(
            'either a=None and sig_d.ndim=1 or a!=None and sig_d.ndim=2')

    if a is None:
        a, a_i, sig_da = get_powerlaw_dust_distribution(
            sig_d, a_max, q=q, na=na, a0=a0, a1=a1)
    else:
        sig_da = sig_d

    sig_d_tot = sig_da.sum(-1)

    lam = np.array(lam, ndmin=1)
    n_lam = len(lam)

    I_nu = np.zeros([n_lam, len(r)])
    tau = np.zeros([n_lam, len(r)])

    # get opacities at our wavelength and particle sizes

    if scattering:
        k_a, k_s = opacity.get_opacities(a, lam)
        g = opacity.get_g(a, lam).T
        k_a = k_a.T
        k_s = k_s.T

        k_se = (1.0 - g) * k_s
        k_ext = k_a + k_se
    else:
        k_ext = opacity.get_opacities(a, lam)[0].T

    for ilam, _lam in enumerate(lam):
        freq = c_light / _lam

        # Calculate intensity profile
        # 1. optical depth
        tau[ilam, :] = (sig_da * k_ext[ilam, :].T).sum(-1) / np.cos(inc)
        tau = np.minimum(100., tau)

        if scattering:
            # 2. a size averaged opacity and from that the averaged epsilon
            k_a_mean = (sig_da * k_a[ilam, :].T).sum(-1) / sig_d_tot
            k_s_mean = (sig_da * k_se[ilam, :].T).sum(-1) / sig_d_tot
            eps_avg = k_a_mean / (k_a_mean + k_s_mean)

            # 3. plug those into the solution
            I_nu[ilam, :] = bplanck(freq, T) * \
                I_over_B_EB(tau[ilam, :], eps_avg)
        else:
            dummy = np.where(tau[ilam, :] > 1e-15,
                             (1.0 - np.exp(-tau[ilam, :])),
                             tau[ilam, :])
            I_nu[ilam, :] = bplanck(freq, T) * dummy

    # calculate the fluxes

    flux = np.cos(inc) * distance**-2 * \
        cumtrapz(2 * np.pi * r * I_nu, x=r, axis=1, initial=0)
    # integrated flux density in Jy (sanity check: TW Hya @ 870 micron and 54 parsec is about 1.5 Jy)
    flux_t = flux[:, -1] / 1e-23

    # converted intensity to Jy/arcsec**2

    I_nu = I_nu / jy_sas

    # interpolate radius whithin which >=68% of the dust mass is

    rf = np.array([np.interp(flux_fraction, _f / _f[-1], r) for _f in flux])

    return SimpleNamespace(
        rf=rf,
        flux_t=flux_t,
        tau=tau,
        I_nu=I_nu,
        a=a,
        sig_da=sig_da)


def get_all_observables(d, opac, lam, amax=True, q=3.5, flux_fraction=0.68, scattering=True, inc=0.0):
    """Calculate the radius and total flux for all snapshots of a simulation

    Arguments:
    ----------

    d : namedtuple
        the output of read_dustpy_data, read_rosotti_data, or run_bump_model

    opac : instance of Opacity
        which opacity to use

    lam : float | array
        wavelength(s) of the observations

    amax : bool
        if True, will always use a power-law distribution, even if size distribution
        is available.

    q : float | list of two elements
        size distribution exponent, either a single float to be used everywhere
        or a two-element list to specify q_f and q_d to be used in the fragmentation
        and drift limited regimes, respectively.

    flux_fraction : float
        at which fraction of the total flux the effective radius is defined [-]

    scattering : bool
        whether or not to include scattering

    inc : float
        inclination, default is 0.0 = face-on. This is just treated as
        increasing the path length across the slab model.

    Returns:
    --------
    rf : array
        e.g. 68% effective radii for all snapshots [cm]

    flux : array
        total flux for all snapshots [Jy]
    """
    rf = []
    flux = []
    tau = []
    I_nu = []
    a = []
    sig_da = []

    q_f, q_d = q * np.ones(2)

    if amax is False and hasattr(d, 'sig_da'):
        _a = d.a
        sig_d = d.sig_da
    else:
        _a = None
        sig_d = d.sig_d

    for it in range(len(d.time)):

        # assign the correct q

        q_array = np.where(d.a_max[it, :] > np.minimum(
            d.a_fr[it, :], d.a_df[it, :]), q_f, q_d)
        obs = get_observables(d.r, d.sig_g[it, :], sig_d[it], d.a_max[it, :], d.T[it, :], opac, lam,
                              q=q_array, a=_a, flux_fraction=flux_fraction, scattering=scattering, inc=inc)
        rf += [obs.rf]
        flux += [obs.flux_t]
        tau += [obs.tau]
        I_nu += [obs.I_nu]
        a += [obs.a]
        sig_da += [obs.sig_da]

    rf = np.array(rf)
    flux = np.array(flux)
    tau = np.array(tau)
    I_nu = np.array(I_nu)
    a = np.array(a)
    sig_da = np.array(sig_da)

    return SimpleNamespace(
        rf=rf,
        flux=flux,
        tau=tau,
        I_nu=I_nu,
        a=a,
        sig_da=sig_da,
    )
