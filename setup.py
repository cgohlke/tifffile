# -*- coding: utf-8 -*-
# tifffile/setup.py

"""Tifffile package setuptools script."""

import sys
import re

from setuptools import setup

buildnumber = ''

imagecodecs = 'imagecodecs>=2019.1.20'

with open('tifffile/tifffile.py') as fh:
    code = fh.read()

version = re.search(r"__version__ = '(.*?)'", code).groups()[0]
version += ('.' + buildnumber) if buildnumber else ''

description = re.search(r'"""(.*)\.(?:\r\n|\r|\n)', code).groups()[0]

readme = re.search(r'(?:\r\n|\r|\n){2}"""(.*)"""(?:\r\n|\r|\n){2}from', code,
                   re.MULTILINE | re.DOTALL).groups()[0]

readme = '\n'.join([description, '=' * len(description)]
                   + readme.splitlines()[1:])

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w') as fh:
        fh.write(readme)

    license = re.search(r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+""',
                        code, re.MULTILINE | re.DOTALL).groups()[0]

    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w') as fh:
        fh.write(license)

    revisions = re.search(r'(?:\r\n|\r|\n){2}(Revisions.*)   \.\.\.', readme,
                          re.MULTILINE | re.DOTALL).groups()[0].strip()

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
    url='https://www.lfd.uci.edu/~gohlke/',
    license='BSD',
    packages=['tifffile'],
    python_requires='>=2.7',
    install_requires=[
        'numpy>=1.11.3',
        'pathlib;python_version=="2.7"',
        'enum34;python_version=="2.7"',
        'futures;python_version=="2.7"',
        # require imagecodecs on Windows only
        imagecodecs + ';platform_system=="Windows"',
        ],
    extras_require={
        'all': ['matplotlib>=2.2', imagecodecs],
    },
    tests_require=['pytest', imagecodecs,
                   'czifile', 'cmapfile', 'oiffile', 'lfdfiles'],
    entry_points={
        'console_scripts': [
            'tifffile = tifffile:main',
            'lsm2bin = tifffile.lsm2bin:main'
            ]},
    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
)
