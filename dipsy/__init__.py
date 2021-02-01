from . import data
from . import utils
from . import cgs_constants
from .tracks import get_stellar_properties
try:
    from dipsy._fortran_module import fmodule as fortran
except ImportError:
    print('fortran module not available')

__version__ = '0.0.4'

from .dipsy_functions import \
    Opacity, \
    get_all_observables, \
    get_observables, \
    get_powerlaw_dust_distribution, \
    read_dustpy_data, \
    read_rosotti_data

__all__ = [
    'data',
    'utils',
    'cgs_constants',
    'Opacity',
    'get_all_observables',
    'get_observables',
    'get_powerlaw_dust_distribution',
    'read_dustpy_data',
    'read_rosotti_data',
    'get_stellar_properties'
]
