# -*- coding: utf-8 -*-
# tiffile.py

"""Proxy module for the tifffile package."""

from tifffile.tifffile import __doc__, __all__, __version__  # noqa
from tifffile.tifffile import lsm2bin, main  # noqa
from tifffile.tifffile import *  # noqa

if __name__ == '__main__':
    import sys
    sys.exit(main())
