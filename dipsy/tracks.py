import os as _os
import numpy as _np
from pathlib import Path as _Path
from pkg_resources import resource_filename

import astropy.units as _u
import astropy.constants as _c

year = (1.0 * _u.yr).cgs.value
R_sun = _c.R_sun.cgs.value
L_sun = _c.L_sun.cgs.value
M_sun = _c.M_sun.cgs.value


def get_stellar_properties(m, t, z='01', track_dir=None):
    """get stellar properties from evolutionary track.

    This helper function creates two tracks and interpolates in mass space

    Parameters
    ----------
    m : float
        stellar mass [g]
    t : float
        stellar age [g]
    z : float
        metallicity
    track_dir : None | path | str
        path to the data files, by default None

    Returns
    -------
    L : float
        stellar luminosity [erg/s]

    R : float
        effective radius [cm]

    T : float
        effective temperature [K]

    """
    masses = _np.array(track.get_masses(z=z))

    m /= M_sun

    if m < masses[0] or m > masses[-1]:
        raise ValueError(f'mass {m:.2g} is out of bounds [{masses[0]:.2g}, {masses[-1]:.2g}]')

    i_left = masses.searchsorted(m)

    if i_left == len(masses) - 1:
        i_left -= 1

    props_left = track(m=f'{masses[i_left]:.1f}', z=z, track_dir=track_dir).get_stellar_params(t)
    props_right = track(m=f'{masses[i_left + 1]:.1f}', z=z, track_dir=track_dir).get_stellar_params(t)

    eps = (m - masses[i_left]) / (masses[i_left + 1] - masses[i_left])

    return (1 - eps) * props_left + eps * props_right


class track(object):
    """
    Stellar evolutionary track based on Siess L., Dufour E., Forestini M. 2000, A&A, 358, 593.

    Data files need to reside in the folder `tracks/`, if not, the files will be downloaded from
    http://www.astro.ulb.ac.be/~siess/

    Keywords:
    ---------

    m : string
    :  stellar mass in solar masses as string

    z : string
    :  metallicity mass fraction, notation: \'01\' = 0.01. Possible values are {}

    track_dir : string
    :   path to the place where the tracks are located. Defaults to 'tracks/' in the module path.
    """

    _zs = ['01', '02', '03', '04']
    _mask = [2, 4, 6]  # column indices of L, Reff, and Teff in the data file
    _it = -1      # column index of the age in the data file

    __doc__ = __doc__.format(', '.join(_zs))

    track_dir = resource_filename(__name__, 'tracks')
    track_file = None

    @classmethod
    def list_files(self):

        folders = [folder for folder in _Path(self.track_dir).glob('Z*')]

        if len(folders) == 0:
            _ = track()

        folders = [folder for folder in _Path(self.track_dir).glob('Z*')]
        for folder in folders:
            print(f'masses in {folder.name}:')
            masses = self.get_masses(folder_name=folder)
            for mass in masses:
                print(f'    {mass}')

    @classmethod
    def get_masses(self, z=None, folder_name=None, rtype=float):
        """return list of available masses

        Need to pass either ``z`` or ``folder_name``

        Parameters
        ----------
        z : str, optional
            metallicity string like '01'  for '0.01', by default None
        folder_name : str|path, optional
            folder name where the data files resides, by default None

        Returns
        -------
        list
            a list of stellar masses in solar masses

        Raises
        ------
        ValueError
            if ``z`` and ``folder_name`` are both set or both unset
        FileNotFoundError
            if ``folder_name`` is not a folder
        """

        if not((z is None) ^ (folder_name is None)):
            raise ValueError('need to pass ONE OF (z, folder_name) as keyword')

        if folder_name is None:
            folder_name = 'Z' + z

        folder = _Path(self.track_dir) / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f'folder {folder_name} does not exist')

        return sorted([rtype(f.name.split('z')[0][1:]) for f in folder.glob('m*z*')])

    def __init__(self, m='0.5', z='01', track_dir=None):

        from scipy.interpolate import interp1d
        import glob

        if track_dir is not None:
            self.track_dir = track_dir

        self.track_file = _os.path.join(self.track_dir, 'Z' + z, 'm' + m + 'z' + z + '.hrd')

        # download files if respective folder is missing

        if not _os.path.exists(_os.path.join(self.track_dir, 'Z' + z)):
            self._download_files()

        # determine available stellar masses

        self._ms = [_os.path.basename(file_)[1:].split('z')[0] for file_ in glob.glob(_os.path.join(self.track_dir, 'Z' + z, '*.hrd'))]

        # update docstring to show available masses

        doc = self.__doc__.split('\n')
        i, = [i for i, d in enumerate(doc) if d.strip().startswith('m :')]
        doc[i + 1] = doc[i + 1] + ', possible values: ' + ', '.join(self._ms)
        self.__doc__ = '\n'.join(doc)

        if m not in self._ms:
            raise ValueError('Selected mass m={} for z={} not in available masses: {}'.format(m, z, ', '.join(self._ms)))

        # load data and create interpoation function

        self._track_data = _np.loadtxt(self.track_file)
        self._track_function = interp1d(self._track_data[:, self._it], self._track_data[:, self._mask].T)

    def _download_files(self, delete=True):
        """
        Downloads all the grid files from Lionel Siess' website and extracts the evolutionary track data files

        Keywords:
        ---------

        delete : bool
        :   if True, delete the downloaded archives after extracting them
        """
        import sys
        import tarfile
        from urllib import request

        if not _os.path.isdir(self.track_dir):
            _os.mkdir(self.track_dir)

        s = 'Downloading grid files'
        print(s + '\n' + '-' * len(s))

        downloaded_files = []

        for z in self._zs:

            url = 'http://www.astro.ulb.ac.be/~siess/pmwiki/pmwiki.php/StellarModels/Z{0}?action=dirlistget&f=Grid_z{0}.tar.gz'.format(z)

            # open url, get file name and size

            with request.urlopen(url) as remotefile:
                meta = remotefile.info()
                file_name = meta.get_filename()
                file_size = int(meta['Content-Length'])

                downloaded_files += [[file_name, z]]

                # download and write to file with progress bar

                with open(file_name, 'wb') as f:
                    file_size_dl = 0
                    block_sz = 8192
                    while True:
                        buf = remotefile.read(block_sz)
                        if not buf:
                            break

                        file_size_dl += len(buf)
                        f.write(buf)

                        sys.stdout.write("\r{}: {:2.2%}".format(file_name, float(file_size_dl) / file_size))
                        sys.stdout.flush()
                    print("")

        # extracting into the specified directory

        s = 'Extracting tracks'
        print(s + '\n' + '-' * len(s))

        for file_name, z in downloaded_files:
            try:
                print(file_name)
                if file_name.endswith('tar.gz'):
                    mode = 'r:gz'
                elif file_name.endswith('tar'):
                    mode = 'r:'
                else:
                    mode = 'r'
                with tarfile.open(file_name, mode) as tar:
                    for file_ in tar:
                        if file_.isreg() and file_.name.endswith('.hrd'):
                            file_.name = _os.path.join(self.track_dir, 'Z' + z, _os.path.basename(file_.name))
                            if _os.path.isfile(file_.name):
                                _os.unlink(file_.name)
                            tar.extract(file_)

            except Exception:
                import traceback
                print('Could not extract ' + file_name + '\n')
                print('Traceback:')
                print('----------')
                for s in traceback.format_exc().split('\n'):
                    print(4 * ' ' + s)
                print('----------')

            finally:
                if delete:
                    _os.unlink(file_name)

    def get_stellar_params(self, t):
        """
        Returns stellar parameters L, Reff, Teff at the given time [s]

        Arguments:
        ----------

        t : float
        :   age of the star in seconds

        Output:
        -------
        L,Reff,Teff

        L : float
        :   stellar luminosity [erg/s]

        Reff : float
        :   effective radius [cm]

        Teff : float
        :   effective temperature [K]
        """

        # convert to years

        t = t / year

        if t <= self._track_data[0, self._it]:
            r = self._track_data[0, self._mask]
        elif t >= self._track_data[-1, self._it]:
            r = self._track_data[-1, self._mask]
        else:
            r = self._track_function(t)

        return r * [L_sun, R_sun, 1.0]
