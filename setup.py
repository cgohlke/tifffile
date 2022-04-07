# tifffile/setup.py

"""Tifffile package setuptools script."""

import sys
import re

from setuptools import setup

buildnumber = ''

with open('tifffile/tifffile.py') as fh:
    code = fh.read()

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]
version += ('.' + buildnumber) if buildnumber else ''

description = re.search(r'"""(.*)\.(?:\r\n|\r|\n)', code).groups()[0]

readme = re.search(
    r'(?:\r\n|\r|\n){2}r"""(.*)"""(?:\r\n|\r|\n){2}[__version__|from]',
    code,
    re.MULTILINE | re.DOTALL,
).groups()[0]

readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w') as fh:
        fh.write(readme)

    license = re.search(
        r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+r""',
        code,
        re.MULTILINE | re.DOTALL,
    ).groups()[0]

    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

    revisions = (
        re.search(
            r'(?:\r\n|\r|\n){2}(Revisions.*)   \.\.\.',
            readme,
            re.MULTILINE | re.DOTALL,
        )
        .groups()[0]
        .strip()
    )

    with open('CHANGES.rst', 'r') as fh:
        old = fh.read()

    d = revisions.splitlines()[-1]
    old = old.split(d)[-1]
    with open('CHANGES.rst', 'w') as fh:
        fh.write(revisions.strip())
        fh.write(old)

setup(
    name='tifffile',
    version=version,
    description=description,
    long_description=readme,
    author='Christoph Gohlke',
    author_email='cgohlke@uci.edu',
    license='BSD',
    url='https://www.lfd.uci.edu/~gohlke/',
    project_urls={
        'Bug Tracker': 'https://github.com/cgohlke/tifffile/issues',
        'Source Code': 'https://github.com/cgohlke/tifffile',
        # 'Documentation': 'https://',
    },
    packages=['tifffile'],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.2',
        # 'imagecodecs>=2021.11.20',
    ],
    extras_require={
        'all': [
            'imagecodecs>=2021.11.20',
            'matplotlib>=3.3',
            'lxml',
            # 'zarr',
            # 'fsspec'
        ]
    },
    tests_require=[
        'pytest',
        'imagecodecs',
        'czifile',
        'cmapfile',
        'oiffile',
        'lfdfiles',
        'roifile',
        'lxml',
        'zarr',
        'dask',
        'xarray',
        'fsspec',
    ],
    entry_points={
        'console_scripts': [
            'tifffile = tifffile:main',
            'tiffcomment = tifffile.tiffcomment:main',
            'tiff2fsspec = tifffile.tiff2fsspec:main',
            'lsm2bin = tifffile.lsm2bin:main',
        ],
        # 'napari.plugin': ['tifffile = tifffile.napari_tifffile'],
    },
    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
