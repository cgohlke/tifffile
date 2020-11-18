# tifffile/tests/conftest.py

import os
import sys


if os.environ.get('SKIP_CODECS', None):
    sys.modules['imagecodecs'] = None


def pytest_report_header(config):
    try:
        from numpy import __version__ as numpy
        from tifffile import __version__ as tifffile
        from test_tifffile import config

        try:
            from imagecodecs import __version__ as imagecodecs
        except ImportError:
            imagecodecs = 'N/A'
        return (
            'versions: tifffile-{}, imagecodecs-{}, numpy-{}\n'
            'test config: {}'.format(tifffile, imagecodecs, numpy, config())
        )
    except Exception:
        pass


collect_ignore = ['_tmp', 'data']
