# -*- coding: utf-8 -*-
# tifffile/tests/conftest.py

collect_ignore = ['_tmp', 'data']


def pytest_report_header(config):
    try:
        import numpy
        import tifffile
        import imagecodecs
        return 'versions: tifffile-%s, imagecodecs-%s, numpy-%s' % (
            tifffile.__version__, imagecodecs.__version__, numpy.__version__)
    except Exception:
        pass
