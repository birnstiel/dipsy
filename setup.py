"""
Setup file for package `dipsy`.
"""
from setuptools import setup
import pathlib

PACKAGENAME = 'dipsy'

# the directory where this setup.py resides
HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":

    setup(
        name=PACKAGENAME,
        description='Disk Population Synthesis Tools',
        version='0.0.1',
        long_description=(HERE / "README.md").read_text(),
        long_description_content_type='text/markdown',
        url='https://github.com/birnstiel/dipsy',
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
        zip_safe=False,
        )
