# tifffile/tests/conftest.py

collect_ignore = ['_tmp', 'data']


def pytest_report_header(config):
    try:
        from numpy import __version__ as numpy
        from tifffile import __version__ as tifffile
        from test_tifffile import config
        try:
            from imagecodecs import __version__ as imagecodecs
        except ImportError:
            imagecodecs = 'N/A'
        return ('versions: tifffile-%s, imagecodecs-%s, numpy-%s\n'
                'test config: %s' % (tifffile, imagecodecs, numpy, config()))
    except Exception:
        pass
