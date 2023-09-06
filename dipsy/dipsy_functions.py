"""
Functions to calculate observables or read data from simulations
"""
import os
import h5py

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from collections import namedtuple

from .cgs_constants import year, jy_sas, c_light, pc
from .utils import bplanck

import dsharp_opac

observables = namedtuple('observables', ['rf', 'flux_t', 'tau', 'I_nu', 'a', 'sig_da'])
dustpy_result = namedtuple('dustpy_result', ['r', 'a_max', 'a', 'a_mean', 'sig_d', 'sig_da', 'sig_g', 'time', 'T'])
rosotti_result = namedtuple('rosotti_result', ['a_max', 'time', 'T', 'sig_d', 'sig_g', 'd2g', 'r', 'L_star', 'M_star', 'T_star'])


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

            if q == 4.0:
                for ia in range(i_up):
                    sig_da[ir, ia] = np.log(a_i[ia + 1] / a_i[ia])

                # filling the bin that contains a_max
                sig_da[ir, i_up] = np.log(a_max[ir] / a_i[i_up])
            else:
                for ia in range(i_up):
                    sig_da[ir, ia] = a_i[ia + 1]**(4 - q) - a_i[ia]**(4 - q)

                # filling the bin that contains a_max
                sig_da[ir, i_up] = a_max[ir]**(4 - q) - a_i[i_up]**(4 - q)

        # normalize

        sig_da[ir, :] = sig_da[ir, :] / sig_da[ir, :].sum() * sigma_d[ir]

    return a, a_i, sig_da


class Opacity(object):
    _filename = None
    _lam = None
    _a = None
    _k_abs = None
    _k_sca = None
    _g = None
    _rho_s = None
    _extrapol = False

    @property
    def rho_s(self):
        "material density in g/cm^3"
        return self._rho_s

    def __init__(self, input=None, **kwargs):
        """Object to read and interpolate opacities.

        input : str | path
            the name of the opacity file to be read. If it doesn't
            exist at the given position, it will try to get it from
            dshapr

        kwargs : keyword dict
            they are passed to the interpolation. This way
            it is possible to turn off or change the interpolation method
            (in log-log space for the opacities, in log-linear space
            for g), e.g. by passing keywords like `bounds_error=True`.
        """

        kwargs['fill_value'] = kwargs.get('fill_value', None)
        kwargs['bounds_error'] = kwargs.get('bounds_error', False)

        # set default opacities

        if input is None:
            input = 'default_opacities_smooth.npz'

        if type(input) is str and not os.path.isfile(input):
            input = dsharp_opac.get_datafile(input)
            if not os.path.isfile(input):
                raise ValueError('unknown input')

        if type(input) is str and os.path.isfile(input):
            self._filename = input
            with np.load(input) as f:
                self._load_from_dict_like(f)

        elif type(input) is dict:
            self._load_from_dict_like(dict)

        self._interp_k_abs = RegularGridInterpolator((np.log10(self._lam), np.log10(self._a)), np.log10(self._k_abs).T, **kwargs)
        self._interp_k_sca = RegularGridInterpolator((np.log10(self._lam), np.log10(self._a)), np.log10(self._k_sca).T, **kwargs)
        if self._g is not None:
            self._interp_g = RegularGridInterpolator((np.log10(self._lam), np.log10(self._a)), self._g.T, **kwargs)

    def _load_from_dict_like(self, input):
        for attr in ['a', 'lam', 'k_abs', 'k_sca', 'g', 'rho_s']:
            if attr in input:
                setattr(self, '_' + attr, input[attr])
            else:
                print(f'{attr} not in input')
        if self._rho_s is not None:
            self._rho_s = float(self._rho_s)

    def _check_input(self, a, lam):
        """Checks if the input is asking for extrapolation in a reasonable range

        Parameters
        ----------
        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm
        """
        # either we are in the grid of known opacities
        # OR we are at large enough optical sizes to be
        # properly extrapolating
        mask = \
            (
                ((a <= self._a[-1]) & (a >= self._a[0]))[:, None] &
                ((lam <= self._lam[-1]) & (lam >= self._lam[0]))[None, :]
            ) | (
                a[:, None] > 100 * lam[None, :] / (2 * np.pi)
            )
        if not np.all(mask):
            raise ValueError('extrapolating too close to the interference part of the opacities')

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
        self._check_input(a, lam)
        return \
            10.**self._interp_k_abs(tuple(np.meshgrid(np.log10(lam), np.log10(a)))), \
            10.**self._interp_k_sca(tuple(np.meshgrid(np.log10(lam), np.log10(a)))),

    def get_g(self, a, lam):
        """
        Returns the asymmetry parameter for the given particle
        size a and wavelength lam.

        Arguments:
        ----------

        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm

        Returns:
        --------
        g : arrays
            asymmetry parameter, array of shape (len(a), len(lam))
        """
        if self._g is None:
            return np.zeros([len(np.array(a, ndmin=1)), len(np.array(lam, ndmin=1))]).squeeze()
        else:
            self._check_input(a, lam)
            return self._interp_g(tuple(np.meshgrid(np.log10(lam), np.log10(a))))

    def get_k_ext_eff(self, a, lam):
        """
        Returns the effective extinction opacity for the given particle
        size `a` and wavelength `lam`.

            k_ext_eff = k_abs + (1.0 - g) * k_sca

        Arguments:
        ----------

        a : float | array
            particle size in cm

        lam : float | array
            wavelength in cm

        Returns:
        --------
        k_ext_eff : arrays
            effective extinction opacity, array of shape (len(a), len(lam))
        """
        k_a, k_s = self.get_opacities(a, lam)
        g = self.get_g(a, lam)

        k_se = (1.0 - g) * k_s
        k_ext_eff = k_a + k_se
        return k_ext_eff


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

    q : float
        size exponent to use: n(a) ~ a^-q, so 3.5=MRN

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
        raise ValueError('either a=None and sig_d.ndim=1 or a!=None and sig_d.ndim=2')

    if a is None:
        a, a_i, sig_da = get_powerlaw_dust_distribution(sig_d, a_max, q=q, na=na, a0=a0, a1=a1)
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
            I_nu[ilam, :] = bplanck(freq, T) * I_over_B_EB(tau[ilam, :], eps_avg)
        else:
            I_nu[ilam, :] = bplanck(freq, T) * (1.0 - np.exp(-tau[ilam, :]))

    # calculate the fluxes

    flux = distance**-2 * cumtrapz(2 * np.pi * r * I_nu, x=r, axis=1, initial=0)
    flux_t = flux[:, -1] / 1e-23  # integrated flux density in Jy (sanity check: TW Hya @ 870 micron and 54 parsec is about 1.5 Jy)

    # converted intensity to Jy/arcsec**2

    I_nu = I_nu / jy_sas

    # interpolate radius whithin which >=68% of the dust mass is

    rf = np.array([np.interp(flux_fraction, _f / _f[-1], r) for _f in flux])

    return observables(rf, flux_t, tau, I_nu, a, sig_da)


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

    q : float
        size distribution exponent

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

    if amax is False and hasattr(d, 'sig_da'):
        _a = d.a
        sig_d = d.sig_da
    else:
        _a = None
        sig_d = d.sig_d

    for it in range(len(d.time)):
        obs = get_observables(d.r, d.sig_g[it, :], sig_d[it], d.a_max[it, :], d.T[it, :], opac, lam,
                              q=q, a=_a, flux_fraction=flux_fraction, scattering=scattering, inc=inc)
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

    return observables(rf, flux, tau, I_nu, a, sig_da)


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

            age[idx] = dset[f'time_{idx}/yso_0/evolution_time'][()] / (2 * np.pi) * year
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

    a_mean = (a * sig_da * np.abs(Vel_d)).sum(-1) / (sig_da * np.abs(Vel_d)).sum(-1)
    a_max = a[sig_da.argmax(-1)]

    if time is None:
        time = time_dp
    else:
        f_Td = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(T))
        f_sd = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(sig_d))
        f_sg = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(sig_g))
        f_ax = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(a_max))
        f_am = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(a_mean))

        T = 10.**f_Td(np.log10(r), np.log10(time + 1e-100))
        sig_d = 10.**f_sd(np.log10(r), np.log10(time + 1e-100))
        sig_g = 10.**f_sg(np.log10(r), np.log10(time + 1e-100))
        a_max = 10.**f_ax(np.log10(r), np.log10(time + 1e-100))
        a_mean = 10.**f_am(np.log10(r), np.log10(time + 1e-100))

        sig_da_new = np.zeros([len(time), len(r), len(a)])
        for ia in range(len(a)):
            f = interp2d(np.log10(r), np.log10(time_dp + 1e-100), np.log10(sig_da[:, :, ia]))
            sig_da_new[:, :, ia] = 10.**f(np.log10(r), np.log10(time + 1e-100))

        sig_da = sig_da_new

    dp = dustpy_result(r, a_max, a, a_mean, sig_d, sig_da, sig_g, time, T)

    return dp


def J_over_B(tauz_in, eps_e, tau):
    """Calculate the mean intensity in units of the Planck function value.

    Note: This follows Eq. 15 of Birnstiel et al. 2018. We later found
    that this was already solved in Miyake & Nakagawa 1993 (and used
    in Sierra et al. 2017). The equations are written slightly differently
    but are equivalent.

    Parameters
    ----------
    tauz_in : float | array
        optical depth at which the mean intensity should be returned
    eps_e : float
        effective absorption probability (= 1 - effective albedo)
    tau : float
        total optical depth

    Returns
    -------
    float | array
        mean intensity evaluated at `tauz_in`
    """
    # our tauz goes from 0 to tau
    # while in the paper it goes from -tau/2 to +tau/2
    if isinstance(tauz_in, np.ndarray):
        tauz = tauz_in.copy() - tau / 2
    else:
        tauz = tauz_in - tau / 2

    b = 1.0 / (
        (1.0 - np.sqrt(eps_e)) * np.exp(-np.sqrt(3 * eps_e) * tau) + 1 + np.sqrt(eps_e))

    J = 1.0 - b * (
        np.exp(-np.sqrt(3.0 * eps_e) * (0.5 * tau - tauz)) +
        np.exp(-np.sqrt(3.0 * eps_e) * (0.5 * tau + tauz)))

    return J


def S_over_B(tauz, eps_e, tau):
    """Calculate the source function in units of the Planck function value.

    Note: This follows Eq. 19 of Birnstiel et al. 2018.

    Parameters
    ----------
    tauz : float | array
        optical depth at which the mean source function should be returned
    eps_e : float
        effective absorption probability (= 1 - effective albedo)
    tau : float
        total optical depth

    Returns
    -------
    float | array
        source function evaluated at `tauz`
    """
    return eps_e + (1.0 - eps_e) * J_over_B(tauz, eps_e, tau)


def I_over_B(tau_total, eps_e, mu=1, ntau=300):
    """Integrates the scattering solution of Birnstiel 2018 numerically.

    This integrates Eq. 17 of Birnstiel 2018. See note for `J_over_B`.

    Parameters
    ----------
    tau_total : float
        total optical depth
    eps_e : float
        effective extinction probablility (1-albedo)
    mu : float, optional
        cosine of the incidence angle, by default 1
    ntau : int, optional
        number of grid points, by default 300

    Returns
    -------
    float
        outgoing intensity in units of the planck function.
    """
    tau = np.linspace(0, tau_total, ntau)
    Inu = np.zeros(ntau)
    Jnu = J_over_B(tau, eps_e, tau_total)
    # the first 1.0 here is a placeholder for Bnu, just for reference
    Snu = eps_e * 1.0 + (1.0 - eps_e) * Jnu
    for i in range(1, ntau):
        dtau = (tau[i] - tau[i - 1]) / mu
        expdtray = np.exp(-dtau)
        srcav = 0.5 * (Snu[i] + Snu[i - 1])
        Inu[i] = expdtray * Inu[i - 1] + (1 - expdtray) * srcav
    return Inu[-1]


def I_over_B_EB(tau, eps_e, mu=1):
    """"same as I_over_B but using the Eddington-Barbier approximation.

    This solves Eq. 19 of Birnstiel et al. 2018, but see also notes
    in `J_over_B`.

    Parameters
    ----------
    tau : float
        total optical depth
    eps_e : float
        effective extinction probablility (1-albedo)
    mu : float, optional
        cosine of the incidence angle, by default 1

    Returns
    -------
    float
        outgoing intensity in units of the planck function.
    """
    arg = np.where(tau > 2. / 3. * mu, tau - 2. / 3. * mu, tau)
    return (1.0 - np.exp(-tau / mu)) * S_over_B(arg, eps_e, tau)
