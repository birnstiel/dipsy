"""
Setup file for package `dipsy`.
"""
import setuptools  # noqa
import pathlib
import warnings

from numpy.distutils.core import Extension, setup

PACKAGENAME = 'dipsy'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).parent

# function to parse the version from the init file


def read_version():
    with (HERE / PACKAGENAME / '__init__.py').open() as fid:
        for line in fid:
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":

    extensions = [
        Extension(name='dipsy._fortran_module', sources=['dipsy/fortran_module.f90'])
    ]

    def setup_function(extensions):
        setup(
            name=PACKAGENAME,
            description='Disk Population Synthesis Tools',
            version=read_version(),
            long_description=(HERE / "README.md").read_text(),
            long_description_content_type='text/markdown',
            url='https://github.com/birnstiel/' + PACKAGENAME,
            author='Til Birnstiel',
            author_email='til.birnstiel@lmu.de',
            license='GPLv3',
            packages=[PACKAGENAME],
            package_dir={PACKAGENAME: PACKAGENAME},
            package_data={PACKAGENAME: ['datasets/*.pickle']},
            include_package_data=True,
            ext_modules=extensions,
            install_requires=[
                'numpy',
                'matplotlib',
                'astropy',
                'astroquery',
                'pyarrow'],
            python_requires='>=3.6',
            entry_points={
                'console_scripts': [
                    'run_grid=dipsy.grid.run_grid:main',
                    'analyze_grid=dipsy.grid.analyze_grid:main',
                ],
            }
        )

    try:
        setup_function(extensions)
    except BaseException:
        warnings.warn('fortran routines not available!')
        setup_function([])
