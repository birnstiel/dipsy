"""
Setup file for package `dipsy`.
"""
from setuptools import setup
import pathlib

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
        install_requires=[
            'numpy',
            'matplotlib',
            'astropy',
            'astroquery',
            'lifelines'],
        python_requires='>=3.6',
    )
