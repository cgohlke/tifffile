# -*- coding: utf-8 -*-
# test_tifffile.py

# Copyright (c) 2008-2019, Christoph Gohlke
# Copyright (c) 2008-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the tifffile package.

Data files are not public due to size and copyright restrictions.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:Version: 2019.1.4

"""

from __future__ import division, print_function

import os
import sys
import glob
import json
import math
import struct
import pathlib
import binascii
import datetime
import tempfile
from io import BytesIO

import pytest
import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal

import tifffile

try:
    from tifffile import *  # noqa
    STAR_IMPORTED = (
        imwrite, imsave, imread, imshow,   # noqa
        TiffFile, TiffWriter, TiffSequence,  # noqa
        FileHandle, lazyattr, natural_sorted, stripnull, memmap,  # noqa
        repeat_nd, format_size, product, create_output)  # noqa
except NameError:
    STAR_IMPORTED = None

from tifffile.tifffile import (  # noqa
    TIFF,
    imwrite, imread, imshow, TiffFile, TiffWriter, TiffSequence, FileHandle,
    lazyattr, natural_sorted, stripnull, memmap, format_size,
    repeat_nd, TiffPage, TiffFrame,
    julian_datetime, excel_datetime, squeeze_axes, transpose_axes, unpack_rgb,
    stripascii, sequence, product,
    imagej_description_metadata, imagej_description, imagej_shape,
    json_description, json_description_metadata,
    scanimage_description_metadata, scanimage_artist_metadata,
    svs_description_metadata, pilatus_description_metadata,
    metaseries_description_metadata, fluoview_description_metadata,
    reshape_axes, apply_colormap, askopenfilename,
    reshape_nd, read_scanimage_metadata, matlabstr2py, bytes2str,
    pformat, snipstr, byteorder_isnative,
    lsm2bin, create_output, hexdump, validate_jhove)


# allow to skip large (memory, number, duration) tests
SKIP_HUGE = False
SKIP_EXTENDED = False
SKIP_ROUNDTRIPS = False
SKIP_DATA = False
VALIDATE = False  # validate written files with jhove

MINISBLACK = TIFF.PHOTOMETRIC.MINISBLACK
MINISWHITE = TIFF.PHOTOMETRIC.MINISWHITE
RGB = TIFF.PHOTOMETRIC.RGB
CFA = TIFF.PHOTOMETRIC.CFA
PALETTE = TIFF.PHOTOMETRIC.PALETTE
YCBCR = TIFF.PHOTOMETRIC.YCBCR
CONTIG = TIFF.PLANARCONFIG.CONTIG
SEPARATE = TIFF.PLANARCONFIG.SEPARATE
LZW = TIFF.COMPRESSION.LZW
LZMA = TIFF.COMPRESSION.LZMA
ZSTD = TIFF.COMPRESSION.ZSTD
WEBP = TIFF.COMPRESSION.WEBP
PACKBITS = TIFF.COMPRESSION.PACKBITS
JPEG = TIFF.COMPRESSION.JPEG
APERIO_JP2000_RGB = TIFF.COMPRESSION.APERIO_JP2000_RGB
ADOBE_DEFLATE = TIFF.COMPRESSION.ADOBE_DEFLATE
DEFLATE = TIFF.COMPRESSION.DEFLATE
NONE = TIFF.COMPRESSION.NONE
LSB2MSB = TIFF.FILLORDER.LSB2MSB
ASSOCALPHA = TIFF.EXTRASAMPLE.ASSOCALPHA
UNASSALPHA = TIFF.EXTRASAMPLE.UNASSALPHA
UNSPECIFIED = TIFF.EXTRASAMPLE.UNSPECIFIED

IS_PY2 = sys.version_info[0] == 2
IS_32BIT = sys.maxsize < 2**32

FILE_FLAGS = ['is_' + a for a in TIFF.FILE_FLAGS]
FILE_FLAGS += [name for name in dir(TiffFile) if name.startswith('is_')]
PAGE_FLAGS = [name for name in dir(TiffPage) if name.startswith('is_')]

HERE = os.path.dirname(__file__)
TEMPDIR = os.path.join(HERE, '_tmp')
DATADIR = os.path.join(HERE, 'data')

if not os.path.exists(TEMPDIR):
    TEMPDIR = tempfile.gettempdir()

if not os.path.exists(DATADIR):
    SKIP_DATA = True  # TODO: decorate tests


def data_file(pathname, base=DATADIR):
    """Return path to data file(s)."""
    path = os.path.join(base, *pathname.split('/'))
    if any(i in path for i in '*?'):
        return glob.glob(path)
    return path


def random_data(dtype, shape):
    """Return random numpy array."""
    # TODO: use nd noise
    if dtype == '?':
        return numpy.random.rand(*shape) < 0.5
    data = numpy.random.rand(*shape) * 255
    data = data.astype(dtype)
    return data


def assert_file_flags(tiff_file):
    """Access all flags of TiffFile."""
    for flag in FILE_FLAGS:
        getattr(tiff_file, flag)


def assert_page_flags(tiff_page):
    """Access all flags of TiffPage."""
    for flag in PAGE_FLAGS:
        getattr(tiff_page, flag)


def assert__str__(tif, detail=3):
    """Call the TiffFile.__str__ function."""
    for i in range(detail+1):
        TiffFile.__str__(tif, detail=i)


if VALIDATE:
    def assert_jhove(filename, *args, **kwargs):
        """Validate TIFF file using jhove script."""
        validate_jhove(filename, 'jhove.cmd', *args, **kwargs)
else:
    def assert_jhove(*args, **kwargs):
        """Do not validate TIFF file."""
        return


class TempFileName():
    """Temporary file name context manager."""
    def __init__(self, name=None, ext='.tif', remove=False):
        self.remove = remove or TEMPDIR == tempfile.gettempdir()
        if not name:
            self.name = tempfile.NamedTemporaryFile(prefix='test_').name
        else:
            self.name = os.path.join(TEMPDIR, "test_%s%s" % (name, ext))

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        if self.remove:
            try:
                os.remove(self.name)
            except Exception:
                pass


try:
    numpy.set_printoptions(legacy='1.13')
except TypeError:
    pass


###############################################################################

# Tests for specific issues

def test_issue_star_import():
    """Test from tifffile import *."""
    assert STAR_IMPORTED is not None
    assert lsm2bin not in STAR_IMPORTED


def test_issue_version_mismatch():
    """Test 'tifffile.__version__' matches docstrings."""
    ver = ':Version: ' + tifffile.__version__
    assert ver in __doc__
    assert ver in tifffile.__doc__


def test_issue_specific_pages():
    """Test read second page."""
    data = random_data('uint8', (3, 21, 31))
    with TempFileName('specific_pages') as fname:
        imwrite(fname, data, photometric='MINISBLACK')
        a = imread(fname)
        assert a.shape == (3, 21, 31)
        # UserWarning: can not reshape (21, 31) to (3, 21, 31)
        a = imread(fname, key=1)
        assert a.shape == (21, 31)
        assert_array_equal(a, data[1])
    with TempFileName('specific_pages_bigtiff') as fname:
        imwrite(fname, data, bigtiff=True, photometric='MINISBLACK')
        a = imread(fname)
        assert a.shape == (3, 21, 31)
        # UserWarning: can not reshape (21, 31) to (3, 21, 31)
        a = imread(fname, key=1)
        assert a.shape == (21, 31)
        assert_array_equal(a, data[1])


def test_issue_circular_ifd():
    """Test circular IFD raises error."""
    fname = data_file('Tiff4J/IFD struct/Circular E.tif')
    with pytest.raises(IndexError):
        imread(fname)


@pytest.mark.skipif(IS_PY2, reason='fails on Python 2')
def test_issue_bad_description(caplog):
    """Test page.description is empty when ImageDescription is not ASCII."""
    # ImageDescription is not ASCII but bytes
    fname = data_file('stk/cells in the eye2.stk')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.description == ''
        assert__str__(tif)
    assert 'coercing invalid ASCII to bytes' in caplog.text


@pytest.mark.skipif(IS_PY2, reason='fails on Python 2')
def test_issue_bad_ascii(caplog):
    """Test coercing invalid ASCII to bytes."""
    # ImageID is not ASCII but bytes
    # https://github.com/blink1073/tifffile/pull/38
    fname = data_file('issues/tifffile_013_tagfail.tif')
    with TiffFile(fname) as tif:
        tags = tif.pages[0].tags
        assert tags['ImageID'].value[-8:] == b'rev 2893'
        assert__str__(tif)
    assert 'coercing invalid ASCII to bytes' in caplog.text


def test_issue_sample_format():
    """Test write correct number of SampleFormat values."""
    # https://github.com/ngageoint/geopackage-tiff-java/issues/5
    data = random_data('uint16', (256, 256, 4))
    with TempFileName('sample_format') as fname:
        imwrite(fname, data)
        with TiffFile(fname) as tif:
            tags = tif.pages[0].tags
            assert tags['SampleFormat'].value == (1, 1, 1, 1)
            assert tags['ExtraSamples'].value == 2
            assert__str__(tif)


def test_issue_palette_with_extrasamples():
    """Test read palette with extra samples."""
    # https://github.com/python-pillow/Pillow/issues/1597
    fname = data_file('issues/palette_with_extrasamples.tif')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.photometric == PALETTE
        assert page.compression == LZW
        assert page.imagewidth == 518
        assert page.imagelength == 556
        assert page.bitspersample == 8
        assert page.samplesperpixel == 2
        # assert data
        image = page.asrgb()
        assert image.shape == (556, 518, 3)
        assert image.dtype == 'uint16'
        image = tif.asarray()
        # self.assertEqual(image.shape[-3:], (556, 518, 2))
        assert image.shape == (556, 518, 2)
        assert image.dtype == 'uint8'
        del image
        assert__str__(tif)


def test_issue_incorrect_rowsperstrip_count():
    """Test read incorrect count for rowsperstrip; bitspersample = 4."""
    # https://github.com/python-pillow/Pillow/issues/1544
    fname = data_file('bad/incorrect_count.tiff')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.photometric == PALETTE
        assert page.compression == ADOBE_DEFLATE
        assert page.imagewidth == 32
        assert page.imagelength == 32
        assert page.bitspersample == 4
        assert page.samplesperpixel == 1
        assert page.rowsperstrip == 32
        assert page.databytecounts == (89,)
        # assert data
        image = page.asrgb()
        assert image.shape == (32, 32, 3)
        del image
        assert__str__(tif)


def test_issue_no_bytecounts(caplog):
    """Test read no bytecounts."""
    with TiffFile(data_file('bad/img2_corrupt.tif')) as tif:
        assert not tif.is_bigtiff
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.planarconfig == CONTIG
        assert page.databytecounts == (0,)
        assert page.dataoffsets == (512,)
        # assert data
        image = tif.asarray()
        assert image.shape == (800, 1200)
    assert 'invalid tag value offset' in caplog.text
    assert 'unknown tag data type 31073' in caplog.text
    assert 'invalid page offset (808333686)' in caplog.text


def test_issue_missing_eoi_in_strips():
    """Test read LZW strips without EOI."""
    # 256x256 uint16, lzw, imagej
    # Strips do not contain an EOI code as required by the TIFF spec.
    # File generated by `tiffcp -c lzw Z*.tif stack.tif` from
    # Bars-G10-P15.zip
    # Failed with "series 0 failed: string size must be a multiple of
    # element size"
    # Reported by Kai Wohlfahrt on 3/7/2014
    fname = data_file('issues/stack.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '<'
        assert len(tif.pages) == 128
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 256
        assert page.imagelength == 256
        assert page.bitspersample == 16
        # assert series properties
        series = tif.series[0]
        assert series.shape == (128, 256, 256)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'IYX'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.41e'
        # assert data
        data = tif.asarray()
        assert data.shape == (128, 256, 256)
        assert data.dtype.name == 'uint16'
        assert data[64, 128, 128] == 19226
        del data
        assert__str__(tif)


def test_issue_valueoffset():
    """Test read TiffTag.valueoffsets."""
    unpack = struct.unpack
    data = random_data('uint16', (2, 19, 31))
    software = 'test_tifffile'
    with TempFileName('valueoffset') as fname:
        imwrite(fname, data, software=software, photometric='minisblack')
        with TiffFile(fname, movie=True) as tif:
            with open(fname, 'rb') as fh:
                page = tif.pages[0]
                # inline value
                fh.seek(page.tags['ImageLength'].valueoffset)
                assert page.imagelength == unpack('H', fh.read(2))[0]
                # separate value
                fh.seek(page.tags['Software'].valueoffset)
                assert page.software == bytes2str(fh.read(13))
                # TiffFrame
                page = tif.pages[1].aspage()
                fh.seek(page.tags['StripOffsets'].valueoffset)
                assert page.dataoffsets[0] == unpack('I', fh.read(4))[0]


def test_issue_pages_number():
    """Test number of pages."""
    fname = data_file('large/100000_pages.tif')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 100000
        assert__str__(tif, 0)


def test_issue_pages_iterator():
    """Test iterating over pages in series."""
    data = random_data('int8', (8, 219, 301))
    with TempFileName('page_iterator') as fname:
        imwrite(fname, data[0])
        imwrite(fname, data, photometric='minisblack', append=True,
                metadata={'axes': 'ZYX'})
        imwrite(fname, data[-1], append=True)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 10
            assert len(tif.series) == 3
            page = tif.pages[1]
            assert page.is_contiguous
            assert page.photometric == MINISBLACK
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 1
            # test reading series 1
            series = tif.series[1]
            assert len(series._pages) == 1
            assert len(series.pages) == 8
            image = series.asarray()
            assert_array_equal(data, image)
            for i, page in enumerate(series.pages):
                im = page.asarray()
                assert_array_equal(image[i], im)
            assert__str__(tif)


def test_issue_pathlib():
    """Test support for pathlib.Path."""
    data = random_data('uint16', (219, 301))
    with TempFileName('pathlib') as fname:
        fname = pathlib.Path(fname)
        imwrite(fname, data)
        with TiffFile(fname) as tif:
            with TempFileName('pathlib_out') as outfname:
                outfname = pathlib.Path(outfname)
                im = tif.asarray(out=outfname)
                assert isinstance(im, numpy.core.memmap)
                assert_array_equal(im, data)
                if not IS_PY2:
                    assert os.path.samefile(im.filename, str(outfname))


###############################################################################

# Test specific functions

def test_func_memmap():
    """Test memmap function."""
    with TempFileName('memmap_new') as fname:
        # create new file
        im = memmap(fname, shape=(32, 16), dtype='float32',
                    bigtiff=True, compress=False)
        im[31, 15] = 1.0
        im.flush()
        assert im.shape == (32, 16)
        assert im.dtype == numpy.dtype('float32')
        del im
        im = memmap(fname, page=0, mode='r')
        assert im[31, 15] == 1.0
        del im
        im = memmap(fname, series=0, mode='c')
        assert im[31, 15] == 1.0
        del im
        # append to file
        im = memmap(fname, shape=(3, 64, 64), dtype='uint16',
                    append=True, photometric='MINISBLACK')
        im[2, 63, 63] = 1.0
        im.flush()
        assert im.shape == (3, 64, 64)
        assert im.dtype == numpy.dtype('uint16')
        del im
        im = memmap(fname, page=3, mode='r')
        assert im[63, 63] == 1
        del im
        im = memmap(fname, series=1, mode='c')
        assert im[2, 63, 63] == 1
        del im
        # can not memory-map compressed array
        with pytest.raises(ValueError):
            memmap(fname, shape=(16, 16), dtype='float32',
                   append=True, compress=6)


def test_func_memmap_fail():
    """Test non-native byteorder can not be memory mapped."""
    with TempFileName('memmap_fail') as fname:
        with pytest.raises(ValueError):
            memmap(fname, shape=(16, 16), dtype='float32', byteorder='>')


def test_func_repeat_nd():
    """Test repeat_nd function."""
    a = repeat_nd([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]], (2, 3))
    assert_array_equal(a, [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [0, 0, 0, 1, 1, 1, 2, 2, 2],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [3, 3, 3, 4, 4, 4, 5, 5, 5],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8],
                           [6, 6, 6, 7, 7, 7, 8, 8, 8]])


def test_func_byteorder_isnative():
    """Test byteorder_isnative function."""
    assert not byteorder_isnative('>')
    assert byteorder_isnative('<')
    assert byteorder_isnative('=')
    assert byteorder_isnative(sys.byteorder)


def test_func_reshape_nd():
    """Test reshape_nd function."""
    assert reshape_nd(numpy.empty(0), 2).shape == (1, 0)
    assert reshape_nd(numpy.empty(1), 3).shape == (1, 1, 1)
    assert reshape_nd(numpy.empty((2, 3)), 3).shape == (1, 2, 3)
    assert reshape_nd(numpy.empty((2, 3, 4)), 3).shape == (2, 3, 4)

    assert reshape_nd((0,), 2) == (1, 0)
    assert reshape_nd((1,), 3) == (1, 1, 1)
    assert reshape_nd((2, 3), 3) == (1, 2, 3)
    assert reshape_nd((2, 3, 4), 3) == (2, 3, 4)


def test_func_apply_colormap():
    """Test apply_colormap function."""
    image = numpy.arange(256, dtype='uint8')
    colormap = numpy.vstack([image, image, image]).astype('uint16') * 256
    assert_array_equal(apply_colormap(image, colormap)[-1], colormap[:, -1])


def test_func_reshape_axes():
    """Test reshape_axes function."""
    assert reshape_axes('YXS', (219, 301, 1), (219, 301, 1)) == 'YXS'
    assert reshape_axes('YXS', (219, 301, 3), (219, 301, 3)) == 'YXS'
    assert reshape_axes('YXS', (219, 301, 1), (219, 301)) == 'YX'
    assert reshape_axes('YXS', (219, 301, 1), (219, 1, 1, 301, 1)) == 'YQQXS'
    assert reshape_axes('IYX', (12, 219, 301), (3, 4, 219, 301, 1)) == 'QQYXQ'
    assert reshape_axes('IYX', (12, 219, 301), (3, 4, 219, 1, 301, 1)
                        ) == 'QQYQXQ'
    assert reshape_axes('IYX', (12, 219, 301), (3, 2, 219, 2, 301, 1)
                        ) == 'QQQQXQ'
    with pytest.raises(ValueError):
        reshape_axes('IYX', (12, 219, 301), (3, 4, 219, 2, 301, 1))
    with pytest.raises(ValueError):
        reshape_axes('IYX', (12, 219, 301), (3, 4, 219, 301, 2))


def test_func_julian_datetime():
    """Test julian_datetime function."""
    assert julian_datetime(2451576, 54362783) == (
        datetime.datetime(2000, 2, 2, 15, 6, 2, 783))


def test_func_excel_datetime():
    """Test excel_datetime function."""
    assert excel_datetime(40237.029999999795) == (
        datetime.datetime(2010, 2, 28, 0, 43, 11, 999982))


def test_func_natural_sorted():
    """Test natural_sorted function."""
    assert natural_sorted(['f1', 'f2', 'f10']) == ['f1', 'f2', 'f10']


def test_func_stripnull():
    """Test stripnull function."""
    assert stripnull(b'string\x00') == b'string'


def test_func_stripascii():
    """Test stripascii function."""
    assert stripascii(b'string\x00string\n\x01\x00') == b'string\x00string\n'
    assert stripascii(b'\x00') == b''


def test_func_sequence():
    """Test sequence function."""
    assert sequence(1) == (1,)
    assert sequence([1]) == [1]


def test_func_product():
    """Test product function."""
    assert product([2**8, 2**30]) == 274877906944
    assert product([]) == 1


def test_func_squeeze_axes():
    """Test squeeze_axes function."""
    assert squeeze_axes((5, 1, 2, 1, 1), 'TZYXC') == ((5, 2, 1), 'TYX')


def test_func_transpose_axes():
    """Test transpose_axes function."""
    assert transpose_axes(numpy.zeros((2, 3, 4, 5)), 'TYXC',
                          asaxes='CTZYX').shape == (5, 2, 1, 3, 4)


def test_func_unpack_rgb():
    """Test unpack_rgb function."""
    data = struct.pack('BBBB', 0x21, 0x08, 0xff, 0xff)
    assert_array_equal(unpack_rgb(data, '<B', (5, 6, 5), False),
                       [1, 1, 1, 31, 63, 31])
    assert_array_equal(unpack_rgb(data, '<B', (5, 6, 5)),
                       [8, 4, 8, 255, 255, 255])
    assert_array_equal(unpack_rgb(data, '<B', (5, 5, 5)),
                       [16, 8, 8, 255, 255, 255])


def test_func_json_description():
    """Test json_description function."""
    descr = json_description((256, 256, 3), axes='YXS')
    assert json.loads(descr) == {"shape": [256, 256, 3], "axes": "YXS"}


def test_func_json_description_metadata():
    """Test json_description_metadata function."""
    assert json_description_metadata(
        'shape=(256, 256, 3)') == {'shape': (256, 256, 3)}
    assert json_description_metadata(
        '{"shape": [256, 256, 3], "axes": "YXS"}'
        ) == {'shape': [256, 256, 3], 'axes': 'YXS'}


def test_func_imagej_shape():
    """Test imagej_shape function."""
    for k, v in (((1, None), (1, 1, 1, 1, 1, 1)),
                 ((3, None), (1, 1, 1, 1, 3, 1)),
                 ((4, 3, None), (1, 1, 1, 4, 3, 1)),
                 ((4, 3, True), (1, 1, 1, 1, 4, 3)),
                 ((4, 4, None), (1, 1, 1, 4, 4, 1)),
                 ((4, 4, True), (1, 1, 1, 1, 4, 4)),
                 ((4, 3, 1, None), (1, 1, 1, 4, 3, 1)),
                 ((4, 3, 2, None), (1, 1, 4, 3, 2, 1)),
                 ((4, 3, 3, None), (1, 1, 1, 4, 3, 3)),
                 ((4, 3, 4, None), (1, 1, 1, 4, 3, 4)),
                 ((4, 3, 4, True), (1, 1, 1, 4, 3, 4)),
                 ((4, 3, 4, False), (1, 1, 4, 3, 4, 1)),
                 ((4, 3, 5, None), (1, 1, 4, 3, 5, 1)),
                 ((3, 2, 1, 5, 4, None), (1, 3, 2, 1, 5, 4)),
                 ((3, 2, 1, 4, 5, None), (3, 2, 1, 4, 5, 1)),
                 ((1, 2, 3, 4, 5, None), (1, 2, 3, 4, 5, 1)),
                 ((2, 3, 4, 5, 3, None), (1, 2, 3, 4, 5, 3)),
                 ((2, 3, 4, 5, 3, True), (1, 2, 3, 4, 5, 3)),
                 ((2, 3, 4, 5, 3, False), (2, 3, 4, 5, 3, 1)),
                 ((1, 2, 3, 4, 5, 4, None), (1, 2, 3, 4, 5, 4)),
                 ((6, 5, 4, 3, 2, 1, None), (6, 5, 4, 3, 2, 1)),
                 ):
        assert imagej_shape(k[:-1], k[-1]) == v


def test_func_imagej_description():
    """Test imagej_description function."""
    imagej_str = ('ImageJ=1.11a\nimages=510\nchannels=2\nslices=5\n'
                  'frames=51\nhyperstack=true\nmode=grayscale\nloop=false\n')
    assert imagej_description((51, 5, 2, 196, 171)) == imagej_str
    expected = 'ImageJ=1.11a\nimages=1\nhyperstack=true\nmode=grayscale\n'
    assert imagej_description((196, 171)) == expected
    expected = 'ImageJ=1.11a\nimages=1\nhyperstack=true\n'
    assert imagej_description((196, 171, 3)) == expected


def test_func_imagej_description_metadata():
    """Test imagej_description_metadata function."""
    imagej_str = ('ImageJ=1.11a\nimages=510\nchannels=2\nslices=5\n'
                  'frames=51\nhyperstack=true\nmode=grayscale\nloop=false\n')
    imagej_dict = {'ImageJ': '1.11a', 'images': 510, 'channels': 2,
                   'slices': 5, 'frames': 51, 'hyperstack': True,
                   'mode': 'grayscale', 'loop': False}
    assert imagej_description_metadata(imagej_str) == imagej_dict


def test_func_pilatus_header_metadata():
    """Test pilatus_description_metadata function."""
    header = """
        # Detector: PILATUS 300K, 3-0101
        # 2011-07-22T17:33:22.529
        # Pixel_size 172e-6 m x 172e-6 m
        # Silicon sensor, thickness 0.000320 m
        # Exposure_time 0.0970000 s
        # Exposure_period 0.1000000 s
        # Tau = 383.8e-09 s
        # Count_cutoff 126367 counts
        # Threshold_setting: not set
        # Gain_setting: high gain (vrf = -0.150)
        # N_excluded_pixels = 19
        # Excluded_pixels: badpix_mask.tif
        # Flat_field: (nil)
        # Trim_file: p300k0101_E8048_T4024_vrf_m0p15.bin
        #  Beam_xy (243.12, 309.12) pixels
        # Image_path: /ramdisk/
            Invalid
        # Unknown 1 2 3 4 5""".strip().replace('            ', '')
    attr = pilatus_description_metadata(header)
    assert attr['Detector'] == 'PILATUS 300K 3-0101'
    assert attr['Pixel_size'] == (0.000172, 0.000172)
    assert attr['Silicon'] == 0.000320
    # self.assertEqual(attr['Threshold_setting'], float('nan'))
    assert attr['Beam_xy'] == (243.12, 309.12)
    assert attr['Unknown'] == '1 2 3 4 5'


def test_func_matlabstr2py():
    """Test matlabstr2py function."""
    assert matlabstr2py('1') == 1
    assert matlabstr2py(
        "['x y z' true false; 1 2.0 -3e4; Inf Inf @class;[1;2;3][1 2] 3]"
        ) == [['x y z', True, False],
              [1, 2.0, -30000.0],
              [float('inf'), float('inf'), '@class'],
              [[[1], [2], [3]], [1, 2], 3]]

    assert matlabstr2py(
        "SI.hChannels.channelType = {'stripe' 'stripe'}\n"
        "SI.hChannels.channelsActive = 2"
        )['SI.hChannels.channelType'] == ['stripe', 'stripe']

    p = matlabstr2py("""
        true = true
        false = false
        True = True
        False = False
        Int = 10
        Float = 3.14
        Float.E = 314.0e-02
        Float.NaN = nan
        Float.Inf = inf
        String = 'string'
        String.Empty = ''
        String.Array = ['ab']
        Array = [1 2]
        Array.2D = [1;2]
        Array.Empty = [[]]
        Transform = [1 0 0;0 1 0;0 0 1]
        Zeros = zeros(1,1)
        Zeros.Empty = zeros(1,0)
        Ones = ones(1,0)
        Filename = C:\\Users\\scanimage.cfg
        Cell = {'' ''}
        Class = @class
        StructObject = <nonscalar struct/object>
        Unknown = unknown

        % Comment
        """)
    assert p['Array'] == [1, 2]
    assert p['Array.2D'] == [[1], [2]]
    assert p['Array.Empty'] == []
    assert p['Cell'] == ['', '']
    assert p['Class'] == '@class'
    assert p['False'] is False
    assert p['Filename'] == 'C:\\Users\\scanimage.cfg'
    assert p['Float'] == 3.14
    assert p['Float.E'] == 3.14
    assert p['Float.Inf'] == float('inf')
    # self.assertEqual(p['Float.NaN'], float('nan'))  # can't compare NaN
    assert p['Int'] == 10
    assert p['StructObject'] == '<nonscalar struct/object>'
    assert p['Ones'] == [[]]
    assert p['String'] == 'string'
    assert p['String.Array'] == 'ab'
    assert p['String.Empty'] == ''
    assert p['Transform'] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert p['True'] is True
    assert p['Unknown'] == 'unknown'
    assert p['Zeros'] == [[0.0]]
    assert p['Zeros.Empty'] == [[]]
    assert p['false'] is False
    assert p['true'] is True


def test_func_hexdump():
    """Test hexdump function."""
    # test hexdump function
    data = binascii.unhexlify(
        '49492a00080000000e00fe0004000100'
        '00000000000000010400010000000001'
        '00000101040001000000000100000201'
        '03000100000020000000030103000100')
    # one line
    assert hexdump(data[:16]) == (
        '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............')
    # height=1
    assert hexdump(data, width=64, height=1) == (
        '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............')
    # all lines
    assert hexdump(data) == (
        '00: 49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 '
        'II*.............\n'
        '10: 00 00 00 00 00 00 00 01 04 00 01 00 00 00 00 01 '
        '................\n'
        '20: 00 00 01 01 04 00 01 00 00 00 00 01 00 00 02 01 '
        '................\n'
        '30: 03 00 01 00 00 00 20 00 00 00 03 01 03 00 01 00 '
        '...... .........')
    # skip center
    assert hexdump(data, height=3, snipat=0.5) == (
        '00: 49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 '
        'II*.............\n'
        '...\n'
        '30: 03 00 01 00 00 00 20 00 00 00 03 01 03 00 01 00 '
        '...... .........')
    # skip start
    assert hexdump(data, height=3, snipat=0) == (
        '10: 00 00 00 00 00 00 00 01 04 00 01 00 00 00 00 01 '
        '................\n'
        '20: 00 00 01 01 04 00 01 00 00 00 00 01 00 00 02 01 '
        '................\n'
        '30: 03 00 01 00 00 00 20 00 00 00 03 01 03 00 01 00 '
        '...... .........')
    # skip end
    assert hexdump(data, height=3, snipat=1) == (
        '00: 49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 '
        'II*.............\n'
        '10: 00 00 00 00 00 00 00 01 04 00 01 00 00 00 00 01 '
        '................\n'
        '20: 00 00 01 01 04 00 01 00 00 00 00 01 00 00 02 01 '
        '................')


def test_func_snipstr():
    """Test snipstr function."""
    # cut middle
    assert snipstr(u'abc', 3, ellipsis='...') == u'abc'
    assert snipstr(u'abc', 3, ellipsis='....') == u'abc'
    assert snipstr(u'abcdefg', 4, ellipsis='') == u'abcd'
    assert snipstr(u'abcdefg', 4, ellipsis=None) == u'abc…'
    assert snipstr(b'abcdefg', 4, ellipsis=None) == b'a...'
    assert snipstr(u'abcdefghijklmnop', 8, ellipsis=None) == u'abcd…nop'
    assert snipstr(b'abcdefghijklmnop', 8, ellipsis=None) == b'abc...op'
    assert snipstr(u'abcdefghijklmnop', 9, ellipsis=None) == u'abcd…mnop'
    assert snipstr(b'abcdefghijklmnop', 9, ellipsis=None) == b'abc...nop'
    assert snipstr('abcdefghijklmnop', 8, ellipsis='..') == 'abc..nop'
    assert snipstr('abcdefghijklmnop', 8, ellipsis='....') == 'ab....op'
    assert snipstr('abcdefghijklmnop', 8, ellipsis='......') == 'ab......'
    # cut right
    assert snipstr(u'abc', 3, snipat=1, ellipsis='...') == u'abc'
    assert snipstr(u'abc', 3, snipat=1, ellipsis='....') == u'abc'
    assert snipstr(u'abcdefg', 4, snipat=1, ellipsis='') == u'abcd'
    assert snipstr(u'abcdefg', 4, snipat=1, ellipsis=None) == u'abc…'
    assert snipstr(b'abcdefg', 4, snipat=1, ellipsis=None) == b'a...'
    assert snipstr(
        u'abcdefghijklmnop', 8, snipat=1, ellipsis=None) == u'abcdefg…'
    assert snipstr(
        b'abcdefghijklmnop', 8, snipat=1, ellipsis=None) == b'abcde...'
    assert snipstr(
        u'abcdefghijklmnop', 9, snipat=1, ellipsis=None) == u'abcdefgh…'
    assert snipstr(
        b'abcdefghijklmnop', 9, snipat=1, ellipsis=None) == b'abcdef...'
    assert snipstr(
        'abcdefghijklmnop', 8, snipat=1, ellipsis='..') == 'abcdef..'
    assert snipstr(
        'abcdefghijklmnop', 8, snipat=1, ellipsis='....') == 'abcd....'
    assert snipstr(
        'abcdefghijklmnop', 8, snipat=1, ellipsis='......') == 'ab......'
    # cut left
    assert snipstr(u'abc', 3, snipat=0, ellipsis='...') == u'abc'
    assert snipstr(u'abc', 3, snipat=0, ellipsis='....') == u'abc'
    assert snipstr(u'abcdefg', 4, snipat=0, ellipsis='') == u'defg'
    assert snipstr(u'abcdefg', 4, snipat=0, ellipsis=None) == u'…efg'
    assert snipstr(b'abcdefg', 4, snipat=0, ellipsis=None) == b'...g'
    assert snipstr(
        u'abcdefghijklmnop', 8, snipat=0, ellipsis=None) == u'…jklmnop'
    assert snipstr(
        b'abcdefghijklmnop', 8, snipat=0, ellipsis=None) == b'...lmnop'
    assert snipstr(
        u'abcdefghijklmnop', 9, snipat=0, ellipsis=None) == u'…ijklmnop'
    assert snipstr(
        b'abcdefghijklmnop', 9, snipat=0, ellipsis=None) == b'...klmnop'
    assert snipstr(
        'abcdefghijklmnop', 8, snipat=0, ellipsis='..') == '..klmnop'
    assert snipstr(
        'abcdefghijklmnop', 8, snipat=0, ellipsis='....') == '....mnop'
    assert snipstr(
        'abcdefghijklmnop', 8, snipat=0, ellipsis='......') == '......op'


def test_func_pformat_printable_bytes():
    """Test pformat function with printable bytes."""
    value = (b'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST'
             b'UVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c')

    assert pformat(value, height=1, width=60) == (
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX')

    assert pformat(value, height=8, width=60) == (
        r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!'
        r""""#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")


def test_func_pformat_printable_unicode():
    """Test pformat function with printable unicode."""
    value = (u'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST'
             u'UVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c')

    assert pformat(value, height=1, width=60) == (
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX')

    assert pformat(value, height=8, width=60) == (
        r'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!'
        r""""#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")


def test_func_pformat_hexdump():
    """Test pformat function with unprintable bytes."""
    value = binascii.unhexlify('49492a00080000000e00fe0004000100'
                               '00000000000000010400010000000001'
                               '00000101040001000000000100000201'
                               '03000100000020000000030103000100')

    assert pformat(value, height=1, width=60) == (
        '49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 II*............')

    assert pformat(value, height=8, width=70) == """
00: 49 49 2a 00 08 00 00 00 0e 00 fe 00 04 00 01 00 II*.............
10: 00 00 00 00 00 00 00 01 04 00 01 00 00 00 00 01 ................
20: 00 00 01 01 04 00 01 00 00 00 00 01 00 00 02 01 ................
30: 03 00 01 00 00 00 20 00 00 00 03 01 03 00 01 00 ...... .........
""".strip()


def test_func_pformat_dict():
    """Test pformat function with dict."""
    value = {'GTCitationGeoKey': 'WGS 84 / UTM zone 29N',
             'GTModelTypeGeoKey': 1,
             'GTRasterTypeGeoKey': 1,
             'KeyDirectoryVersion': 1,
             'KeyRevision': 1,
             'KeyRevisionMinor': 2,
             'ModelTransformation': numpy.array([
                 [6.00000e+01, 0.00000e+00, 0.00000e+00, 6.00000e+05],
                 [0.00000e+00, -6.00000e+01, 0.00000e+00, 5.90004e+06],
                 [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00],
                 [0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00]]),
             'PCSCitationGeoKey': 'WGS 84 / UTM zone 29N',
             'ProjectedCSTypeGeoKey': 32629}

    assert pformat(value, height=1, width=60) == (
        "{'GTCitationGeoKey': 'WGS 84 / UTM zone 29N', 'GTModelTypeGe")

    assert pformat(value, height=8, width=60) == (
        """{'GTCitationGeoKey': 'WGS 84 / UTM zone 29N',
 'GTModelTypeGeoKey': 1,
 'GTRasterTypeGeoKey': 1,
 'KeyDirectoryVersion': 1,
...
       [  0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   1.00000000e+00]]),
 'PCSCitationGeoKey': 'WGS 84 / UTM zone 29N',
 'ProjectedCSTypeGeoKey': 32629}""")


@pytest.mark.skipif(IS_PY2, reason='fails on Python 2')
def test_func_pformat_list():
    """Test pformat function with list."""
    value = (60.0, 0.0, 0.0, 600000.0, 0.0, -60.0, 0.0, 5900040.,
             60.0, 0.0, 0.0, 600000.0, 0.0, -60.0, 0.0, 5900040.)

    assert pformat(value, height=1, width=60) == (
        '(60.0, 0.0, 0.0, 600000.0, 0.0, -60.0, 0.0, 5900040.0, 60.0,')

    assert pformat(value, height=8, width=60) == (
        '(60.0, 0.0, 0.0, 600000.0, 0.0, -60.0, 0.0, 5900040.0, 60.0,\n'
        ' 0.0, 0.0, 600000.0, 0.0, -60.0, 0.0, 5900040.0)')


def test_func_pformat_numpy():
    """Test pformat function with numpy array."""
    value = numpy.array(
        (60.0, 0.0, 0.0, 600000.0, 0.0, -60.0, 0.0, 5900040.,
         60.0, 0.0, 0.0, 600000.0, 0.0, -60.0, 0.0, 5900040.))

    assert pformat(value, height=1, width=60) == (
        'array([6.00000000e+01, 0.00000000e+00, 0.00000000e+00, 6.000')

    assert pformat(value, height=8, width=60) == (
        """array([  6.00000000e+01,   0.00000000e+00,   0.00000000e+00,
         6.00000000e+05,   0.00000000e+00,  -6.00000000e+01,
         0.00000000e+00,   5.90004000e+06,   6.00000000e+01,
         0.00000000e+00,   0.00000000e+00,   6.00000000e+05,
         0.00000000e+00,  -6.00000000e+01,   0.00000000e+00,
         5.90004000e+06])""")


def test_func_pformat_xml():
    """Test pformat function with XML."""
    value = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<Dimap_Document name="band2.dim">
  <Metadata_Id>
    <METADATA_FORMAT version="2.12.1">DIMAP</METADATA_FORMAT>
    <METADATA_PROFILE>BEAM-DATAMODEL-V1</METADATA_PROFILE>
  </Metadata_Id>
  <Image_Interpretation>
    <Spectral_Band_Info>
      <BAND_INDEX>0</BAND_INDEX>
    </Spectral_Band_Info>
  </Image_Interpretation>
</Dimap_Document>"""

    assert pformat(value, height=1, width=60) == (
        '<?xml version="1.0" encoding="ISO-8859-1" ?> <Dimap_Document')

    assert pformat(value, height=8, width=60) == (
        """<?xml version='1.0' encoding='ISO-8859-1'?>
<Dimap_Document name="band2.dim">
 <Metadata_Id>
  <METADATA_FORMAT version="2.12.1">DIMAP</METADATA_FORMAT>
...
   <BAND_INDEX>0</BAND_INDEX>
  </Spectral_Band_Info>
 </Image_Interpretation>
</Dimap_Document>""")


@pytest.mark.skipif(IS_32BIT, reason='not enough memory')
@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
def test_func_lsm2bin():
    """Test lsm2bin function."""
    # Converst LSM to BIN
    fname = data_file('lsm/Twoareas_Zstacks54slices_3umintervals_5cycles.lsm')
    # fname = data_file(
    #     'LSM/fish01-wt-t01-10_ForTest-20zplanes10timepoints.lsm')
    lsm2bin(fname, '', verbose=True)


def test_func_create_output():
    """Test create_output function."""
    shape = (16, 17)
    dtype = 'uint16'
    # None
    a = create_output(None, shape, dtype)
    assert_array_equal(a, numpy.zeros(shape, dtype))
    # existing array
    b = create_output(a, a.shape, a.dtype)
    assert a is b.base
    # 'memmap'
    a = create_output('memmap', shape, dtype)
    assert isinstance(a, numpy.core.memmap)
    del a
    # 'memmap:tempdir'
    a = create_output('memmap:%s' % os.path.abspath(TEMPDIR), shape, dtype)
    assert isinstance(a, numpy.core.memmap)
    del a
    # filename
    with TempFileName('nopages') as fname:
        a = create_output(fname, shape, dtype)
        del a


@pytest.mark.parametrize('key', [None, 0, 3, 'series'])
@pytest.mark.parametrize('out', [None, 'empty', 'memmap', 'dir', 'name'])
def test_func_create_output_asarray(out, key):
    """Test create_output function in context of asarray."""
    data = random_data('uint16', (5, 219, 301))

    with TempFileName('out') as fname:
        imwrite(fname, data)
        # assert file
        with TiffFile(fname) as tif:
            tif.pages.useframes = True
            tif.pages.load()

            if key is None:
                # default
                obj = tif
                dat = data
            elif key == 'series':
                # series
                obj = tif.series[0]
                dat = data
            else:
                # single page/frame
                obj = tif.pages[key]
                dat = data[key]
                if key == 0:
                    assert isinstance(obj, TiffPage)
                elif not IS_PY2:
                    assert isinstance(obj, TiffFrame)

            if out is None:
                # new array
                image = obj.asarray(out=None)
                assert_array_equal(dat, image)
                del image
            elif out == 'empty':
                # existing array
                image = numpy.empty_like(dat)
                obj.asarray(out=image)
                assert_array_equal(dat, image)
                del image
            elif out == 'memmap':
                # memmap in temp dir
                image = obj.asarray(out='memmap')
                assert isinstance(image, numpy.core.memmap)
                assert_array_equal(dat, image)
                del image
            elif out == 'dir':
                # memmap in specified dir
                tempdir = os.path.dirname(fname)
                image = obj.asarray(out='memmap:%s' % tempdir)
                assert isinstance(image, numpy.core.memmap)
                assert_array_equal(dat, image)
                del image
            elif out == 'name':
                # memmap in specified file
                with TempFileName('out', ext='.memmap') as fileout:
                    image = obj.asarray(out=fileout)
                    assert isinstance(image, numpy.core.memmap)
                    assert_array_equal(dat, image)
                    del image


###############################################################################

# Test FileHandle class

FILEHANDLE_NAME = data_file('test_FileHandle.bin')
FILEHANDLE_SIZE = 7937381
FILEHANDLE_OFFSET = 333
FILEHANDLE_LENGTH = 7937381 - 666


def create_filehandle_file():
    """Write test_FileHandle.bin file."""
    # array start 999
    # array end 1254
    # recarray start 2253
    # recarray end 6078
    # tiff start 7077
    # tiff end 12821
    # mm offset = 13820
    # mm size = 7936382
    with open(FILEHANDLE_NAME, 'wb') as fh:
        # buffer
        numpy.ones(999, dtype='uint8').tofile(fh)
        # array
        print('array start', fh.tell())
        numpy.arange(255, dtype='uint8').tofile(fh)
        print('array end', fh.tell())
        # buffer
        numpy.ones(999, dtype='uint8').tofile(fh)
        # recarray
        print('recarray start', fh.tell())
        a = numpy.recarray((255, 3),
                           dtype=[('x', 'float32'), ('y', 'uint8')])
        for i in range(3):
            a[:, i].x = numpy.arange(255, dtype='float32')
            a[:, i].y = numpy.arange(255, dtype='uint8')
        a.tofile(fh)
        print('recarray end', fh.tell())
        # buffer
        numpy.ones(999, dtype='uint8').tofile(fh)
        # tiff
        print('tiff start', fh.tell())
        with open('Tests/vigranumpy.tif', 'rb') as tif:
            fh.write(tif.read())
        print('tiff end', fh.tell())
        # buffer
        numpy.ones(999, dtype='uint8').tofile(fh)
        # micromanager
        print('micromanager start', fh.tell())
        with open('Tests/micromanager/micromanager.tif', 'rb') as tif:
            fh.write(tif.read())
        print('micromanager end', fh.tell())
        # buffer
        numpy.ones(999, dtype='uint8').tofile(fh)


def assert_filehandle(fh, offset=0):
    """Assert filehandle can read test_FileHandle.bin."""
    size = FILEHANDLE_SIZE - 2*offset
    pad = 999 - offset
    assert fh.size == size
    assert fh.tell() == 0
    assert fh.read(4) == b'\x01\x01\x01\x01'
    fh.seek(pad-4)
    assert fh.tell() == pad-4
    assert fh.read(4) == b'\x01\x01\x01\x01'
    fh.seek(-4, whence=1)
    assert fh.tell() == pad-4
    assert fh.read(4) == b'\x01\x01\x01\x01'
    fh.seek(-pad, whence=2)
    assert fh.tell() == size-pad
    assert fh.read(4) == b'\x01\x01\x01\x01'
    # assert array
    fh.seek(pad, whence=0)
    assert fh.tell() == pad
    assert_array_equal(fh.read_array('uint8', 255),
                       numpy.arange(255, dtype='uint8'))
    # assert records
    fh.seek(999, whence=1)
    assert fh.tell() == 2253-offset
    records = fh.read_record([('x', 'float32'), ('y', 'uint8')], (255, 3))
    assert_array_equal(records.y[:, 0], range(255))
    assert_array_equal(records.x, records.y)
    # assert memmap
    if fh.is_file:
        assert_array_equal(fh.memmap_array('uint8', 255, pad),
                           numpy.arange(255, dtype='uint8'))


def test_filehandle_seekable():
    """Test FileHandle must be seekable."""
    try:
        from urllib2 import build_opener, HTTPSHandler
    except ImportError:
        from urllib.request import build_opener, HTTPSHandler

    import ssl
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    opener = build_opener(HTTPSHandler(context=context))
    opener.addheaders = [('User-Agent', 'test_tifffile.py')]
    fh = opener.open('https://localhost/tests/test.tif')
    with pytest.raises(ValueError):
        FileHandle(fh)


def test_filehandle_write_bytesio():
    """Test write to FileHandle from BytesIO."""
    value = b'123456789'
    buf = BytesIO()
    with FileHandle(buf) as fh:
        fh.write(value)
    buf.seek(0)
    assert buf.read() == value


def test_filehandle_write_bytesio_offset():
    """Test write to FileHandle from BytesIO with offset."""
    pad = b'abcd'
    value = b'123456789'
    buf = BytesIO()
    buf.write(pad)
    with FileHandle(buf) as fh:
        fh.write(value)
    buf.write(pad)
    # assert buffer
    buf.seek(len(pad))
    assert buf.read(len(value)) == value
    buf.seek(2)
    with FileHandle(buf, offset=len(pad), size=len(value)) as fh:
        assert fh.read(len(value)) == value


def test_filehandle_filename():
    """Test FileHandle from filename."""
    with FileHandle(FILEHANDLE_NAME) as fh:
        assert fh.name == "test_FileHandle.bin"
        assert fh.is_file
        assert_filehandle(fh)


def test_filehandle_filename_offset():
    """Test FileHandle from filename with offset."""
    with FileHandle(FILEHANDLE_NAME, offset=FILEHANDLE_OFFSET,
                    size=FILEHANDLE_LENGTH) as fh:
        assert fh.name == "test_FileHandle.bin"
        assert fh.is_file
        assert_filehandle(fh, FILEHANDLE_OFFSET)


def test_filehandle_bytesio():
    """Test FileHandle from BytesIO."""
    with open(FILEHANDLE_NAME, 'rb') as fh:
        stream = BytesIO(fh.read())
    with FileHandle(stream) as fh:
        assert fh.name == "Unnamed binary stream"
        assert not fh.is_file
        assert_filehandle(fh)


def test_filehandle_bytesio_offset():
    """Test FileHandle from BytesIO with offset."""
    with open(FILEHANDLE_NAME, 'rb') as fh:
        stream = BytesIO(fh.read())
    with FileHandle(stream, offset=FILEHANDLE_OFFSET,
                    size=FILEHANDLE_LENGTH) as fh:
        assert fh.name == "Unnamed binary stream"
        assert not fh.is_file
        assert_filehandle(fh, offset=FILEHANDLE_OFFSET)


def test_filehandle_openfile():
    """Test FileHandle from open file."""
    with open(FILEHANDLE_NAME, 'rb') as fhandle:
        with FileHandle(fhandle) as fh:
            assert fh.name == "test_FileHandle.bin"
            assert fh.is_file
            assert_filehandle(fh)
        assert not fhandle.closed


def test_filehandle_openfile_offset():
    """Test FileHandle from open file with offset."""
    with open(FILEHANDLE_NAME, 'rb') as fhandle:
        with FileHandle(fhandle, offset=FILEHANDLE_OFFSET,
                        size=FILEHANDLE_LENGTH) as fh:
            assert fh.name == "test_FileHandle.bin"
            assert fh.is_file
            assert_filehandle(fh, offset=FILEHANDLE_OFFSET)
        assert not fhandle.closed


def test_filehandle_filehandle():
    """Test FileHandle from other FileHandle."""
    with FileHandle(FILEHANDLE_NAME, 'rb') as fhandle:
        with FileHandle(fhandle) as fh:
            assert fh.name == "test_FileHandle.bin"
            assert fh.is_file
            assert_filehandle(fh)
        assert not fhandle.closed


def test_filehandle_offset():
    """Test FileHandle from other FileHandle with offset."""
    with FileHandle(FILEHANDLE_NAME, 'rb') as fhandle:
        with FileHandle(fhandle, offset=FILEHANDLE_OFFSET,
                        size=FILEHANDLE_LENGTH) as fh:
            assert fh.name == "test_FileHandle@333.bin"
            assert fh.is_file
            assert_filehandle(fh, offset=FILEHANDLE_OFFSET)
        assert not fhandle.closed


def test_filehandle_reopen():
    """Test FileHandle close and open."""
    try:
        fh = FileHandle(FILEHANDLE_NAME)
        assert not fh.closed
        assert fh.is_file
        fh.close()
        assert fh.closed
        fh.open()
        assert not fh.closed
        assert fh.is_file
        assert fh.name == "test_FileHandle.bin"
        assert_filehandle(fh)
    finally:
        fh.close()


def test_filehandle_unc_path():
    """Test FileHandle from UNC path."""
    with FileHandle(r"\\localhost\Data\Data\test_FileHandle.bin") as fh:
        assert fh.name == "test_FileHandle.bin"
        assert fh.dirname == "\\\\localhost\\Data\\Data"
        assert_filehandle(fh)


###############################################################################

# Test reading specific files

if not SKIP_EXTENDED:
    TIGER_FILES = (data_file('Tigers/be/*.tif') +
                   data_file('Tigers/le/*.tif') +
                   data_file('Tigers/bigtiff-be/*.tif') +
                   data_file('Tigers/bigtiff-le/*.tif')
                   )
    TIGER_IDS = ['-'.join(f.split(os.path.sep)[-2:]
                          ).replace('-tiger', '').replace('.tif', '')
                 for f in TIGER_FILES]
else:
    TIGER_FILES = []
    TIGER_IDS = []


@pytest.mark.skipif(SKIP_EXTENDED, reason='many tests')
@pytest.mark.parametrize('fname', TIGER_FILES, ids=TIGER_IDS)
def test_read_tigers(fname):
    """Test tiger images from GraphicsMagick."""
    with TiffFile(fname) as tif:
        byteorder = {'le': '<', 'be': '>'}[os.path.split(fname)[0][-2:]]
        databits = int(fname.rsplit('.tif')[0][-2:])

        # assert file properties
        assert_file_flags(tif)
        assert tif.byteorder == byteorder
        assert tif.is_bigtiff == ('bigtiff' in fname)
        assert len(tif.pages) == 1

        # assert page properties
        page = tif.pages[0]
        assert_page_flags(page)
        assert page.tags['DocumentName'].value == os.path.basename(fname)
        assert page.imagewidth == 73
        assert page.imagelength == 76
        assert page.bitspersample == databits
        assert (page.photometric == RGB) == ('rgb' in fname)
        assert (page.photometric == PALETTE) == ('palette' in fname)
        assert page.is_tiled == ('tile' in fname)
        assert (page.planarconfig == CONTIG) == ('planar' not in fname)
        if 'minisblack' in fname:
            assert page.photometric == MINISBLACK

        # float24 not supported
        if 'float' in fname and databits == 24:
            with pytest.raises(ValueError):
                data = tif.asarray()
            return

        # assert data shapes
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        # if 'palette' in fname:
        #     shape = (76, 73, 3)
        if 'rgb' in fname:
            if 'planar' in fname:
                shape = (3, 76, 73)
            else:
                shape = (76, 73, 3)
        elif 'separated' in fname:
            if 'planar' in fname:
                shape = (4, 76, 73)
            else:
                shape = (76, 73, 4)
        else:
            shape = (76, 73)
        assert data.shape == shape

        # assert data types
        if 'float' in fname:
            dtype = 'float%i' % databits
        # elif 'palette' in fname:
        #     dtype = 'uint16'
        elif databits == 1:
            dtype = 'bool'
        elif databits <= 8:
            dtype = 'uint8'
        elif databits <= 16:
            dtype = 'uint16'
        elif databits <= 32:
            dtype = 'uint32'
        elif databits <= 64:
            dtype = 'uint64'
        assert data.dtype.name == dtype

        assert__str__(tif)


def test_read_exif_paint():
    """Test read EXIF tags."""
    fname = data_file('exif/paint.tif')
    with TiffFile(fname) as tif:
        exif = tif.pages[0].tags['ExifTag'].value
        assert exif['ColorSpace'] == 65535
        assert exif['ExifVersion'] == '0230'
        assert exif['UserComment'] == 'paint'
        assert__str__(tif)


def test_read_hopper_2bit():
    """Test read 2-bit, fillorder=lsb2msb."""
    # https://github.com/python-pillow/Pillow/pull/1789
    fname = data_file('hopper/hopper2.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == MINISBLACK
        assert not page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 128
        assert page.imagelength == 128
        assert page.bitspersample == 2
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (128, 128)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        assert series.offset is None
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (128, 128)
        assert data[50, 63] == 3
        assert__str__(tif)
    # reversed
    fname = data_file('hopper/hopper2R.tif')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.photometric == MINISBLACK
        assert page.fillorder == LSB2MSB
        assert_array_equal(tif.asarray(), data)
        assert__str__(tif)
    # inverted
    fname = data_file('hopper/hopper2I.tif')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.photometric == MINISWHITE
        assert_array_equal(tif.asarray(), 3-data)
        assert__str__(tif)
    # inverted and reversed
    fname = data_file('hopper/hopper2IR.tif')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.photometric == MINISWHITE
        assert_array_equal(tif.asarray(), 3-data)
        assert__str__(tif)


def test_read_hopper_4bit():
    """Test read 4-bit, fillorder=lsb2msb."""
    # https://github.com/python-pillow/Pillow/pull/1789
    fname = data_file('hopper/hopper4.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == MINISBLACK
        assert not page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 128
        assert page.imagelength == 128
        assert page.bitspersample == 4
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (128, 128)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        assert series.offset is None
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (128, 128)
        assert data[50, 63] == 13
    # reversed
    fname = data_file('hopper/hopper4R.tif')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.photometric == MINISBLACK
        assert page.fillorder == LSB2MSB
        assert_array_equal(tif.asarray(), data)
        assert__str__(tif)
    # inverted
    fname = data_file('hopper/hopper4I.tif')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.photometric == MINISWHITE
        assert_array_equal(tif.asarray(), 15-data)
        assert__str__(tif)
    # inverted and reversed
    fname = data_file('hopper/hopper4IR.tif')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.photometric == MINISWHITE
        assert_array_equal(tif.asarray(), 15-data)
        assert__str__(tif)


def test_read_lsb2msb():
    """Test read fillorder=lsb2msb, 2 series."""
    # http://lists.openmicroscopy.org.uk/pipermail/ome-users
    #   /2015-September/005635.html
    fname = data_file('test_lsb2msb.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 2
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 7100
        assert page.imagelength == 4700
        assert page.bitspersample == 16
        assert page.samplesperpixel == 3
        page = tif.pages[1]
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 7100
        assert page.imagelength == 4700
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (4700, 7100, 3)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YXS'
        assert series.offset is None
        series = tif.series[1]
        assert series.shape == (4700, 7100)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YX'
        assert series.offset is None
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (4700, 7100, 3)
        assert data[2350, 3550, 1] == 60457
        data = tif.asarray(series=1)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (4700, 7100)
        assert data[2350, 3550] == 56341
        assert__str__(tif)


def test_read_predictor3():
    """Test read floating point horizontal differencing by OpenImageIO."""
    fname = data_file('predictor3.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == ADOBE_DEFLATE
        assert page.imagewidth == 1500
        assert page.imagelength == 1500
        assert page.bitspersample == 32
        assert page.samplesperpixel == 4
        # assert series properties
        series = tif.series[0]
        assert series.shape == (1500, 1500, 4)
        assert series.dtype.name == 'float32'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (1500, 1500, 4)
        assert data.dtype.name == 'float32'
        assert tuple(data[750, 750]) == (0., 0., 0., 1.)
        assert__str__(tif)


def test_read_gimp16():
    """Test read uint16 with horizontal predictor by GIMP."""
    fname = data_file('GIMP/gimp16.tiff')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == ADOBE_DEFLATE
        assert page.photometric == RGB
        assert page.imagewidth == 333
        assert page.imagelength == 231
        assert page.samplesperpixel == 3
        assert page.predictor == 2
        image = tif.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert tuple(image[110, 110]) == (23308, 17303, 41160)
        assert__str__(tif)


def test_read_gimp32():
    """Test read float32 with horizontal predictor by GIMP."""
    fname = data_file('GIMP/gimp32.tiff')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == ADOBE_DEFLATE
        assert page.photometric == RGB
        assert page.imagewidth == 333
        assert page.imagelength == 231
        assert page.samplesperpixel == 3
        assert page.predictor == 2
        image = tif.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert_array_almost_equal(
            image[110, 110], (0.35565534, 0.26402164, 0.6280674))
        assert__str__(tif)


def test_read_iss_vista():
    """Test read bogus imagedepth tag by ISS Vista."""
    fname = data_file('iss/10um_beads_14stacks_ch1.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 14
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == NONE
        assert page.imagewidth == 256
        assert page.imagelength == 256
        assert page.tags['ImageDepth'].value == 14  # bogus
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (14, 256, 256)
        assert series.dtype.name == 'int16'
        assert series.axes == 'IYX'  # ZYX
        assert__str__(tif)


def test_read_vips():
    """Test read 47x641 RGB, bigtiff, pyramid, tiled, produced by VIPS."""
    fname = data_file('vips.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 4
        assert len(tif.series) == 4
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert page.is_tiled
        assert page.compression == ADOBE_DEFLATE
        assert page.imagewidth == 641
        assert page.imagelength == 347
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (347, 641, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        series = tif.series[3]
        page = series.pages[0]
        assert page.is_reduced
        assert page.is_tiled
        assert series.shape == (43, 80, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (347, 641, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[132, 361]) == (114, 233, 58)
        assert__str__(tif)


def test_read_sgi_depth():
    """Test read 128x128x128, float32, tiled SGI."""
    fname = data_file('sgi/sgi_depth.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_sgi
        assert page.planarconfig == CONTIG
        assert page.is_tiled
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 128
        assert page.imagelength == 128
        assert page.imagedepth == 128
        assert page.tilewidth == 128
        assert page.tilelength == 128
        assert page.tiledepth == 1
        assert page.bitspersample == 32
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == (
            'MFL MeVis File Format Library, TIFF Module')
        # assert series properties
        series = tif.series[0]
        assert series.shape == (128, 128, 128)
        assert series.dtype.name == 'float32'
        assert series.axes == 'ZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (128, 128, 128)
        assert data.dtype.name == 'float32'
        assert data[64, 64, 64] == 0.0
        assert__str__(tif)


def test_read_oxford():
    """Test read 601x81, uint8, PackBits."""
    fname = data_file('oxford.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.planarconfig == SEPARATE
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 601
        assert page.imagelength == 81
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 81, 601)
        assert series.dtype == 'uint8'
        assert series.axes == 'SYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (3, 81, 601)
        assert data.dtype.name == 'uint8'
        assert data[1, 24, 49] == 191
        assert__str__(tif)


def test_read_cramps():
    """Test 800x607 uint8, PackBits."""
    fname = data_file('cramps.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.compression == PACKBITS
        assert page.photometric == MINISWHITE
        assert page.imagewidth == 800
        assert page.imagelength == 607
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (607, 800)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (607, 800)
        assert data.dtype.name == 'uint8'
        assert data[273, 426] == 34
        assert__str__(tif)


def test_read_cramps_tile():
    """Test read 800x607 uint8, raw, sgi, tiled."""
    fname = data_file('cramps-tile.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_tiled
        assert page.is_sgi
        assert page.compression == NONE
        assert page.photometric == MINISWHITE
        assert page.imagewidth == 800
        assert page.imagelength == 607
        assert page.imagedepth == 1
        assert page.tilewidth == 256
        assert page.tilelength == 256
        assert page.tiledepth == 1
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (607, 800)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (607, 800)
        assert data.dtype.name == 'uint8'
        assert data[273, 426] == 34
        assert__str__(tif)


def test_read_jello():
    """Test read 256x192x3, uint16, palette, PackBits."""
    fname = data_file('jello.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == PALETTE
        assert page.planarconfig == CONTIG
        assert page.compression == PACKBITS
        assert page.imagewidth == 256
        assert page.imagelength == 192
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (192, 256)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert data
        data = page.asrgb(uint8=False)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (192, 256, 3)
        assert data.dtype.name == 'uint16'
        assert tuple(data[100, 140, :]) == (48895, 65279, 48895)
        assert__str__(tif)


def test_read_django():
    """Test read 3x480x320, uint16, palette, raw."""
    fname = data_file('django.tiff')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == PALETTE
        assert page.planarconfig == CONTIG
        assert page.compression == NONE
        assert page.imagewidth == 320
        assert page.imagelength == 480
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (480, 320)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert data
        data = page.asrgb(uint8=False)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (480, 320, 3)
        assert data.dtype.name == 'uint16'
        assert tuple(data[64, 64, :]) == (65535, 52171, 63222)
        assert__str__(tif)


def test_read_quad_lzw():
    """Test read 384x512 RGB uint8 old style LZW."""
    fname = data_file('quad-lzw.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_tiled
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 384
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (384, 512, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (384, 512, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[309, 460, :]) == (0, 163, 187)


def test_read_quad_lzw_le():
    """Test read 384x512 RGB uint8 LZW."""
    fname = data_file('quad-lzw_le.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert not page.is_tiled
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 384
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (384, 512, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (384, 512, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[309, 460, :]) == (0, 163, 187)


def test_read_quad_tile():
    """Test read 384x512 RGB uint8 LZW tiled."""
    # Strips and tiles defined in same page
    fname = data_file('quad-tile.tif')
    with TiffFile(fname) as tif:
        assert__str__(tif)
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.is_tiled
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 384
        assert page.imagedepth == 1
        assert page.tilewidth == 128
        assert page.tilelength == 128
        assert page.tiledepth == 1
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (384, 512, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        # assert 'invalid tile data (49153,) (1, 128, 128, 3)' in caplog.text
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (384, 512, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[309, 460, :]) == (0, 163, 187)


def test_read_incomplete_tile_contig():
    """Test read PackBits compressed incomplete tile, contig RGB."""
    fname = data_file('GDAL/contig_tiled.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.compression == PACKBITS
        assert page.imagewidth == 35
        assert page.imagelength == 37
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (37, 35, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = page.asarray()
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (37, 35, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[19, 31]) == (50, 50, 50)
        assert tuple(data[36, 34]) == (70, 70, 70)
        assert__str__(tif)


def test_read_incomplete_tile_separate():
    """Test read PackBits compressed incomplete tile, separate RGB."""
    fname = data_file('GDAL/separate_tiled.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.planarconfig == SEPARATE
        assert page.compression == PACKBITS
        assert page.imagewidth == 35
        assert page.imagelength == 37
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 37, 35)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'SYX'
        # assert data
        data = page.asarray()
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (3, 37, 35)
        assert data.dtype.name == 'uint8'
        assert tuple(data[:, 19, 31]) == (50, 50, 50)
        assert tuple(data[:, 36, 34]) == (70, 70, 70)
        assert__str__(tif)


def test_read_strike():
    """Test read 256x200 RGBA uint8 LZW."""
    fname = data_file('strike.tif')
    with TiffFile(fname) as tif:
        assert__str__(tif)
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 256
        assert page.imagelength == 200
        assert page.bitspersample == 8
        assert page.samplesperpixel == 4
        assert page.extrasamples == ASSOCALPHA
        # assert series properties
        series = tif.series[0]
        assert series.shape == (200, 256, 4)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (200, 256, 4)
        assert data.dtype.name == 'uint8'
        assert tuple(data[65, 139, :]) == (43, 34, 17, 91)
        assert__str__(tif)


def test_read_pygame_icon():
    """Test read 128x128 RGBA uint8 PackBits."""
    fname = data_file('pygame_icon.tiff')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.compression == PACKBITS
        assert page.imagewidth == 128
        assert page.imagelength == 128
        assert page.bitspersample == 8
        assert page.samplesperpixel == 4
        assert page.extrasamples == UNASSALPHA  # ?
        assert page.tags['Software'].value == 'QuickTime 5.0.5'
        assert page.tags['HostComputer'].value == 'MacOS 10.1.2'
        assert page.tags['DateTime'].value == '2001:12:21 04:34:56'
        # assert series properties
        series = tif.series[0]
        assert series.shape == (128, 128, 4)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (128, 128, 4)
        assert data.dtype.name == 'uint8'
        assert tuple(data[22, 112, :]) == (100, 99, 98, 132)
        assert__str__(tif)


def test_read_rgba_wo_extra_samples():
    """Test read 1065x785 RGBA uint8."""
    fname = data_file('rgba_wo_extra_samples.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 1065
        assert page.imagelength == 785
        assert page.bitspersample == 8
        assert page.samplesperpixel == 4
        # with self.assertRaises(AttributeError):
        #     page.extrasamples
        # assert series properties
        series = tif.series[0]
        assert series.shape == (785, 1065, 4)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (785, 1065, 4)
        assert data.dtype.name == 'uint8'
        assert tuple(data[560, 412, :]) == (60, 92, 74, 255)
        assert__str__(tif)


def test_read_rgb565():
    """Test read 64x64 RGB uint8 5,6,5 bitspersample."""
    fname = data_file('rgb565.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.compression == NONE
        assert page.imagewidth == 64
        assert page.imagelength == 64
        assert page.bitspersample == (5, 6, 5)
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (64, 64, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (64, 64, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[56, 32, :]) == (239, 243, 247)
        assert__str__(tif)


def test_read_vigranumpy():
    """Test read 4 series in 6 pages."""
    fname = data_file('vigranumpy.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 6
        assert len(tif.series) == 4
        # assert series 0 properties
        series = tif.series[0]
        assert series.shape == (3, 20, 20)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'IYX'
        page = series.pages[0]
        assert page.compression == LZW
        assert page.imagewidth == 20
        assert page.imagelength == 20
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        data = tif.asarray(series=0)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (3, 20, 20)
        assert data.dtype.name == 'uint8'
        assert tuple(data[:, 9, 9]) == (19, 90, 206)
        # assert series 1 properties
        series = tif.series[1]
        assert series.shape == (10, 10, 3)
        assert series.dtype.name == 'float32'
        assert series.axes == 'YXS'
        page = series.pages[0]
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 10
        assert page.imagelength == 10
        assert page.bitspersample == 32
        assert page.samplesperpixel == 3
        data = tif.asarray(series=1)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (10, 10, 3)
        assert data.dtype.name == 'float32'
        assert round(abs(data[9, 9, 1]-214.5733642578125), 7) == 0
        # assert series 2 properties
        series = tif.series[2]
        assert series.shape == (20, 20, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        page = series.pages[0]
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 20
        assert page.imagelength == 20
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        data = tif.asarray(series=2)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (20, 20, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[9, 9, :]) == (19, 90, 206)
        # assert series 3 properties
        series = tif.series[3]
        assert series.shape == (10, 10)
        assert series.dtype.name == 'float32'
        assert series.axes == 'YX'
        page = series.pages[0]
        assert page.compression == LZW
        assert page.imagewidth == 10
        assert page.imagelength == 10
        assert page.bitspersample == 32
        assert page.samplesperpixel == 1
        data = tif.asarray(series=3)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (10, 10)
        assert data.dtype.name == 'float32'
        assert round(abs(data[9, 9]-223.1648712158203), 7) == 0
        assert__str__(tif)


def test_read_freeimage():
    """Test read 3 series in 3 pages RGB LZW."""
    fname = data_file('freeimage.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 3
        assert len(tif.series) == 3
        for i, shape in enumerate(((100, 600), (379, 574), (689, 636))):
            series = tif.series[i]
            shape = shape + (3, )
            assert series.shape == shape
            assert series.dtype.name == 'uint8'
            assert series.axes == 'YXS'
            page = series.pages[0]
            assert page.photometric == RGB
            assert page.compression == LZW
            assert page.imagewidth == shape[1]
            assert page.imagelength == shape[0]
            assert page.bitspersample == 8
            assert page.samplesperpixel == 3
            data = tif.asarray(series=i)
            assert isinstance(data, numpy.ndarray)
            assert data.flags['C_CONTIGUOUS']
            assert data.shape == shape
            assert data.dtype.name == 'uint8'
            assert__str__(tif)


def test_read_12bit():
    """Test read 12 bit images."""
    fname = data_file('12bit.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1000
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 1024
        assert page.imagelength == 304
        assert page.bitspersample == 12
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (1000, 304, 1024)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'IYX'
        # assert data
        data = tif.asarray(478)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (304, 1024)
        assert data.dtype.name == 'uint16'
        assert round(abs(data[138, 475]-40), 7) == 0
        assert__str__(tif, 0)


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
def test_read_lzw_large_buffer():
    """Test read LZW compression which requires large buffer."""
    # https://github.com/groupdocs-viewer/GroupDocs.Viewer-for-.NET-MVC-App
    # /issues/35
    fname = data_file('lzw/lzw_large_buffer.tiff')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == LZW
        assert page.imagewidth == 5104
        assert page.imagelength == 8400
        assert page.bitspersample == 8
        assert page.samplesperpixel == 4
        # assert data
        image = page.asarray()
        assert image.shape == (8400, 5104, 4)
        assert image.dtype == 'uint8'
        image = tif.asarray()
        assert image.shape == (8400, 5104, 4)
        assert image.dtype == 'uint8'
        assert image[4200, 2550, 0] == 0
        assert image[4200, 2550, 3] == 255
        assert__str__(tif)


def test_read_lzw_ycbcr_subsampling():
    """Test fail LZW compression with subsampling."""
    fname = data_file('lzw/lzw_ycbcr_subsampling.tif')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == LZW
        assert page.photometric == YCBCR
        assert page.planarconfig == CONTIG
        assert page.imagewidth == 39
        assert page.imagelength == 39
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert data
        with pytest.raises(NotImplementedError):
            page.asarray()
        assert__str__(tif)


def test_read_jpeg_baboon():
    """Test JPEG compression."""
    # test that jpeg compression is supported
    fname = data_file('baboon.tiff')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert 'JPEGTables' in page.tags
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == JPEG
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (512, 512, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        # with pytest.raises((ValueError, NotImplementedError)):
        image = tif.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert__str__(tif)


def test_read_jpeg_ycbcr():
    """Test read YCBCR JPEG is returned as RGB."""
    fname = data_file('jpeg/jpeg_ycbcr.tiff')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == JPEG
        assert page.photometric == YCBCR
        assert page.planarconfig == CONTIG
        assert page.imagewidth == 128
        assert page.imagelength == 80
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert data
        image = tif.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (80, 128, 3)
        assert image.dtype == 'uint8'
        assert tuple(image[50, 50, :]) == (177, 149, 210)
        # YCBCR (164, 154, 137)
        assert__str__(tif)


def test_read_jpeg12_mandril():
    """Test read JPEG 12-bit compression."""
    # JPEG 12-bit
    fname = data_file('jpeg/jpeg12_mandril.tif')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == JPEG
        assert page.photometric == YCBCR
        assert page.imagewidth == 512
        assert page.imagelength == 480
        assert page.bitspersample == 12
        assert page.samplesperpixel == 3
        # assert data
        image = tif.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (480, 512, 3)
        assert image.dtype == 'uint16'
        assert tuple(image[128, 128, :]) == (1685, 1859, 1376)
        # YCBCR (1752, 1836, 2000)
        assert__str__(tif)


def test_read_aperio_j2k():
    """Test read SVS slide with J2K compression."""
    fname = data_file('slides/CMU-1-JP2K-33005.tif')
    with TiffFile(fname) as tif:
        assert tif.is_svs
        assert len(tif.pages) == 6
        page = tif.pages[0]
        assert page.compression == APERIO_JP2000_RGB
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.shape == (32893, 46000, 3)
        assert page.dtype == 'uint8'
        page = tif.pages[1]
        assert page.compression == JPEG
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.shape == (732, 1024, 3)
        assert page.dtype == 'uint8'
        page = tif.pages[2]
        assert page.compression == APERIO_JP2000_RGB
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.shape == (8223, 11500, 3)
        assert page.dtype == 'uint8'
        page = tif.pages[3]
        assert page.compression == APERIO_JP2000_RGB
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.shape == (2055, 2875, 3)
        assert page.dtype == 'uint8'
        page = tif.pages[4]
        assert page.is_reduced
        assert page.compression == LZW
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.shape == (463, 387, 3)
        assert page.dtype == 'uint8'
        page = tif.pages[5]
        assert page.is_reduced
        assert page.compression == JPEG
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.shape == (431, 1280, 3)
        assert page.dtype == 'uint8'
        # assert data
        image = tif.pages[3].asarray()
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (2055, 2875, 3)
        assert image.dtype == 'uint8'
        assert image[512, 1024, 0] == 246
        assert image[512, 1024, 1] == 245
        assert image[512, 1024, 2] == 245
        assert__str__(tif)


def test_read_lzma():
    """Test read LZMA compression."""
    # 512x512, uint8, lzma compression
    fname = data_file('lzma.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.compression == LZMA
        assert page.photometric == MINISBLACK
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (512, 512)
        assert series.dtype == 'uint8'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert data.flags['C_CONTIGUOUS']
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (512, 512)
        assert data.dtype.name == 'uint8'
        assert data[273, 426] == 151
        assert__str__(tif)


def test_read_webp():
    """Test read WebP compression."""
    fname = data_file('GDAL/tif_webp.tif')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == WEBP
        assert page.photometric == RGB
        assert page.planarconfig == CONTIG
        assert page.imagewidth == 50
        assert page.imagelength == 50
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert data
        image = tif.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (50, 50, 3)
        assert image.dtype == 'uint8'
        assert image[25, 25, 0] == 92
        assert image[25, 25, 1] == 122
        assert image[25, 25, 2] == 37
        assert__str__(tif)


def test_read_zstd():
    """Test read ZStd compression."""
    fname = data_file('GDAL/byte_zstd.tif')
    with TiffFile(fname) as tif:
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.compression == ZSTD
        assert page.photometric == MINISBLACK
        assert page.planarconfig == CONTIG
        assert page.imagewidth == 20
        assert page.imagelength == 20
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert data
        image = tif.asarray()  # fails with imagecodecs <= 2018.11.8
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (20, 20)
        assert image.dtype == 'uint8'
        assert image[18, 1] == 247
        assert__str__(tif)


def test_read_lena_be_f16_contig():
    """Test read big endian float16 horizontal differencing."""
    fname = data_file('PS/lena_be_f16_contig.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (512, 512, 3)
        assert series.dtype.name == 'float16'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (512, 512, 3)
        assert data.dtype.name == 'float16'
        assert_array_almost_equal(data[256, 256],
                                  (0.4563, 0.052856, 0.064819))
        assert__str__(tif)


def test_read_lena_be_f16_lzw_planar():
    """Test read big endian, float16, LZW, horizontal differencing."""
    fname = data_file('PS/lena_be_f16_lzw_planar.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 512, 512)
        assert series.dtype.name == 'float16'
        assert series.axes == 'SYX'
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (3, 512, 512)
        assert data.dtype.name == 'float16'
        assert_array_almost_equal(data[:, 256, 256],
                                  (0.4563, 0.052856, 0.064819))
        assert__str__(tif)


def test_read_lena_be_f32_deflate_contig():
    """Test read big endian, float32 horizontal differencing, deflate."""
    fname = data_file('PS/lena_be_f32_deflate_contig.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == ADOBE_DEFLATE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 32
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (512, 512, 3)
        assert series.dtype.name == 'float32'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (512, 512, 3)
        assert data.dtype.name == 'float32'
        assert_array_almost_equal(data[256, 256],
                                  (0.456386, 0.052867, 0.064795))
        assert__str__(tif)


def test_read_lena_le_f32_lzw_planar():
    """Test read little endian, LZW, float32 horizontal differencing."""
    fname = data_file('PS/lena_le_f32_lzw_planar.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 32
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 512, 512)
        assert series.dtype.name == 'float32'
        assert series.axes == 'SYX'
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (3, 512, 512)
        assert data.dtype.name == 'float32'
        assert_array_almost_equal(data[:, 256, 256],
                                  (0.456386, 0.052867, 0.064795))
        assert__str__(tif)


def test_read_lena_be_rgb48():
    """Test read little endian, uint16, LZW."""
    fname = data_file('PS/lena_be_rgb48.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert not page.is_reduced
        assert not page.is_tiled
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (512, 512, 3)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray(series=0)
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (512, 512, 3)
        assert data.dtype.name == 'uint16'
        assert_array_equal(data[256, 256], (46259, 16706, 18504))
        assert__str__(tif)


# Test large images

@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
def test_read_huge_ps5_memmap():
    """Test read 30000x30000 float32 contiguous."""
    fname = data_file('large/huge_ps5.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous == (21890, 3600000000)
        assert not page.is_memmappable  # data not aligned!
        assert page.compression == NONE
        assert page.imagewidth == 30000
        assert page.imagelength == 30000
        assert page.bitspersample == 32
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (30000, 30000)
        assert series.dtype.name == 'float32'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray(out='memmap')  # memmap in a temp file
        assert isinstance(data, numpy.core.memmap)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (30000, 30000)
        assert data.dtype.name == 'float32'
        assert data[6597, 8135] == 0.008780896663665771
        del data
        assert not tif.filehandle.closed
        assert__str__(tif)


@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
def test_read_movie():
    """Test read 30000 pages, uint16."""
    fname = data_file('large/movie.tif')
    with TiffFile(fname, movie=True) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 30000
        assert len(tif.series) == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (30000, 64, 64)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'IYX'
        # assert page properties
        page = tif.pages[-1]
        assert isinstance(page, TiffFrame)
        assert page.shape == (64, 64)
        # assert data
        data = tif.pages[29999].asarray()  # last frame
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (64, 64)
        assert data.dtype.name == 'uint16'
        assert data[32, 32] == 460
        del data
        # read selected pages
        # https://github.com/blink1073/tifffile/issues/51
        data = tif.asarray(key=[31, 999, 29999])
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (3, 64, 64)
        assert data[2, 32, 32] == 460
        del data
        assert__str__(tif, 0)


@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
@pytest.mark.skipif(IS_32BIT, reason='might segfault due to low memory')
@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
def test_read_100000_pages_movie():
    """Test read 100000x64x64 big endian in memory."""
    fname = data_file('large/100000_pages.tif')
    with TiffFile(fname, movie=True) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 100000
        assert len(tif.series) == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (100000, 64, 64)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'TYX'
        # assert page properties
        page = tif.pages[100]
        assert isinstance(page, TiffFrame)
        assert page.shape == (64, 64)
        page = tif.pages[0]
        assert page.imagewidth == 64
        assert page.imagelength == 64
        assert page.bitspersample == 16
        assert page.is_contiguous
        # assert ImageJ tags
        tags = tif.imagej_metadata
        assert tags['ImageJ'] == '1.48g'
        assert round(abs(tags['max']-119.0), 7) == 0
        assert round(abs(tags['min']-86.0), 7) == 0
        # assert data
        data = tif.asarray()
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (100000, 64, 64)
        assert data.dtype.name == 'uint16'
        assert round(abs(data[7310, 25, 25]-100), 7) == 0
        del data
        assert__str__(tif, 0)


@pytest.mark.skipif(SKIP_EXTENDED, reason='huge image')
def test_read_movie_memmap():
    """Test read 30000 pages memory-mapped."""
    fname = data_file('large/movie.tif')
    with TiffFile(fname) as tif:
        # assert data
        data = tif.asarray(out='memmap')
        assert isinstance(data, numpy.core.memmap)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (30000, 64, 64)
        assert data.dtype.name == 'uint16'
        assert data[29999, 32, 32] == 460
        del data
        assert not tif.filehandle.closed
        assert__str__(tif, 0)


@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
def test_read_chart_bl():
    """Test read 13228x18710, 1 bit, no bitspersample tag."""
    fname = data_file('large/chart_bl.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.compression == NONE
        assert page.imagewidth == 13228
        assert page.imagelength == 18710
        assert page.bitspersample == 1
        assert page.samplesperpixel == 1
        assert page.rowsperstrip == 18710
        # assert series properties
        series = tif.series[0]
        assert series.shape == (18710, 13228)
        assert series.dtype.name == 'bool'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (18710, 13228)
        assert data.dtype.name == 'bool'
        assert data[0, 0] is numpy.bool_(True)
        assert data[5000, 5000] is numpy.bool_(False)
        assert__str__(tif)


@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
def test_read_srtm_20_13():
    """Test read 6000x6000 int16 GDAL."""
    fname = data_file('large/srtm_20_13.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 6000
        assert page.imagelength == 6000
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        assert page.tags['GDAL_NODATA'].value == "-32768"
        assert page.tags['GeoAsciiParamsTag'].value == "WGS 84|"
        # assert series properties
        series = tif.series[0]
        assert series.shape == (6000, 6000)
        assert series.dtype.name == 'int16'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (6000, 6000)
        assert data.dtype.name == 'int16'
        assert data[5199, 5107] == 1019
        assert data[0, 0] == -32768
        del data
        assert__str__(tif)


@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
def test_read_gel_scan():
    """Test read 6976x4992x3 uint8 LZW."""
    fname = data_file('large/gel_1-scan2.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 4992
        assert page.imagelength == 6976
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (6976, 4992, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (6976, 4992, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[2229, 1080, :]) == (164, 164, 164)
        del data
        assert__str__(tif)


@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
def test_read_caspian():
    """Test read 3x220x279 float64, RGB, deflate, GDAL."""
    fname = data_file('caspian.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.planarconfig == SEPARATE
        assert page.compression == DEFLATE
        assert page.imagewidth == 279
        assert page.imagelength == 220
        assert page.bitspersample == 64
        assert page.samplesperpixel == 3
        assert page.tags['GDAL_METADATA'].value.startswith('<GDALMetadata>')
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 220, 279)
        assert series.dtype.name == 'float64'
        assert series.axes == 'SYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (3, 220, 279)
        assert data.dtype.name == 'float64'
        assert round(abs(data[2, 100, 140]-353.0), 7) == 0
        assert__str__(tif)


def test_read_subifds_array():
    """Test read SubIFDs."""
    fname = data_file('Tiff4J/IFD struct/SubIFDs array E.tif')
    with TiffFile(fname) as tif:
        assert len(tif.series) == 1
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 2000
        assert page.imagelength == 1500
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        assert page.tags['SubIFDs'].value == (14760220, 18614796,
                                              19800716, 18974964)
        # assert subifds
        assert len(page.pages) == 4
        page = tif.pages[0].pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 1600
        assert page.imagelength == 1200
        page = tif.pages[0].pages[1]
        assert page.photometric == RGB
        assert page.imagewidth == 1200
        assert page.imagelength == 900
        page = tif.pages[0].pages[2]
        assert page.photometric == RGB
        assert page.imagewidth == 800
        assert page.imagelength == 600
        page = tif.pages[0].pages[3]
        assert page.photometric == RGB
        assert page.imagewidth == 400
        assert page.imagelength == 300
        # assert data
        image = page.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (300, 400, 3)
        assert image.dtype == 'uint8'
        assert tuple(image[124, 292]) == (236, 109, 95)
        assert__str__(tif)


def test_read_subifd4():
    """Test read BigTIFFSubIFD4."""
    fname = data_file('TwelveMonkeys/bigtiff/BigTIFFSubIFD4.tif')
    with TiffFile(fname) as tif:
        assert len(tif.series) == 1
        assert len(tif.pages) == 2
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 64
        assert page.imagelength == 64
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        assert page.tags['SubIFDs'].value == (3088,)
        # assert subifd
        page = page.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 32
        assert page.imagelength == 32
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert data
        image = page.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (32, 32, 3)
        assert image.dtype == 'uint8'
        assert image[15, 15, 0] == 255
        assert image[16, 16, 2] == 0
        assert__str__(tif)


def test_read_subifd8():
    """Test read BigTIFFSubIFD8."""
    fname = data_file('TwelveMonkeys/bigtiff/BigTIFFSubIFD8.tif')
    with TiffFile(fname) as tif:
        assert len(tif.series) == 1
        assert len(tif.pages) == 2
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 64
        assert page.imagelength == 64
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        assert page.tags['SubIFDs'].value == (3088,)
        # assert subifd
        page = page.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 32
        assert page.imagelength == 32
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert data
        image = page.asarray()
        assert image.flags['C_CONTIGUOUS']
        assert image.shape == (32, 32, 3)
        assert image.dtype == 'uint8'
        assert image[15, 15, 0] == 255
        assert image[16, 16, 2] == 0
        assert__str__(tif)


# @pytest.mark.skipif(SKIP_HUGE, reason='huge image')
@pytest.mark.skipif(IS_32BIT, reason='not enough memory')
def test_read_lsm_mosaic():
    """Test read LSM: PTZCYX (Mosaic mode), two areas, 32 samples, >4 GB."""
    # LSM files are little endian with two series, one of which is reduced RGB
    # Tags may be unordered or contain bogus values
    fname = data_file('lsm/Twoareas_Zstacks54slices_3umintervals_5cycles.lsm')
    with TiffFile(fname) as tif:
        assert tif.is_lsm
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1080
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert page.is_lsm
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 32
        # assert strip offsets are corrected
        page = tif.pages[-2]
        assert page.dataoffsets[0] == 9070895981
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 5, 54, 32, 512, 512)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'PTZCYX'
        if 1:
            series = tif.series[1]
            assert series.shape == (2, 5, 54, 3, 128, 128)
            assert series.dtype.name == 'uint8'
            assert series.axes == 'PTZCYX'
        # assert lsm_info tags
        tags = tif.lsm_metadata
        assert tags['DimensionX'] == 512
        assert tags['DimensionY'] == 512
        assert tags['DimensionZ'] == 54
        assert tags['DimensionTime'] == 5
        assert tags['DimensionChannels'] == 32
        # assert lsm_scan_info tags
        tags = tif.lsm_metadata['ScanInformation']
        assert tags['ScanMode'] == 'Stack'
        assert tags['User'] == 'lfdguest1'
        assert__str__(tif, 0)


@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
def test_read_lsm_carpet():
    """Test read LSM: ZCTYX (time series x-y), 72000 pages."""
    # reads very slowly, ensure colormap is not applied
    fname = data_file('lsm/Cardarelli_carpet_3.lsm')
    with TiffFile(fname) as tif:
        assert tif.is_lsm
        assert tif.byteorder == '<'
        assert len(tif.pages) == 72000
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert page.is_lsm
        assert 'ColorMap' in page.tags
        assert page.photometric == PALETTE
        assert page.compression == NONE
        assert page.imagewidth == 32
        assert page.imagelength == 10
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (1, 1, 36000, 10, 32)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'ZCTYX'
        if 1:
            series = tif.series[1]
            assert series.shape == (1, 1, 36000, 3, 40, 128)
            assert series.dtype.name == 'uint8'
            assert series.axes == 'ZCTCYX'
        # assert lsm_info tags
        tags = tif.lsm_metadata
        assert tags['DimensionX'] == 32
        assert tags['DimensionY'] == 10
        assert tags['DimensionZ'] == 1
        assert tags['DimensionTime'] == 36000
        assert tags['DimensionChannels'] == 1
        # assert lsm_scan_info tags
        tags = tif.lsm_metadata['ScanInformation']
        assert tags['ScanMode'] == 'Plane'
        assert tags['User'] == 'LSM User'
        assert__str__(tif, 0)


def test_read_lsm_take1():
    """Test read LSM: TCZYX (Plane mode), single image, uint8."""
    fname = data_file('lsm/take1.lsm')
    with TiffFile(fname) as tif:
        assert tif.is_lsm
        assert tif.byteorder == '<'
        assert len(tif.pages) == 2
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert page.is_lsm
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        page = tif.pages[1]
        assert page.is_reduced
        assert page.photometric == RGB
        assert page.planarconfig == SEPARATE
        assert page.compression == NONE
        assert page.imagewidth == 128
        assert page.imagelength == 128
        assert page.samplesperpixel == 3
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (1, 1, 1, 512, 512)  # (512, 512)?
        assert series.dtype.name == 'uint8'
        assert series.axes == 'TCZYX'
        if 1:
            series = tif.series[1]
            assert series.shape == (3, 128, 128)
            assert series.dtype.name == 'uint8'
            assert series.axes == 'CYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (1, 1, 1, 512, 512)  # (512, 512)?
        assert data.dtype.name == 'uint8'
        assert data[..., 256, 256] == 101
        if 1:
            data = tif.asarray(series=1)
            assert isinstance(data, numpy.ndarray)
            assert data.shape == (3, 128, 128)
            assert data.dtype.name == 'uint8'
            assert tuple(data[..., 64, 64]) == (89, 89, 89)
        # assert lsm_info tags
        tags = tif.lsm_metadata
        assert tags['DimensionX'] == 512
        assert tags['DimensionY'] == 512
        assert tags['DimensionZ'] == 1
        assert tags['DimensionTime'] == 1
        assert tags['DimensionChannels'] == 1
        # assert lsm_scan_info tags
        tags = tif.lsm_metadata['ScanInformation']
        assert tags['ScanMode'] == 'Plane'
        assert tags['User'] == 'LSM User'
        assert len(tags['Tracks']) == 1
        assert len(tags['Tracks'][0]['DataChannels']) == 1
        track = tags['Tracks'][0]
        assert track['DataChannels'][0]['Name'] == 'Ch1'
        assert track['DataChannels'][0]['BitsPerSample'] == 8
        assert len(track['IlluminationChannels']) == 1
        assert track['IlluminationChannels'][0]['Name'] == '561'
        assert track['IlluminationChannels'][0]['Wavelength'] == 561.0
        assert__str__(tif)


def test_read_lsm_2chzt():
    """Test read LSM: ZCYX (Stack mode) uint8."""
    fname = data_file('lsm/2chzt.lsm')
    with TiffFile(fname) as tif:
        assert tif.is_lsm
        assert tif.byteorder == '<'
        assert len(tif.pages) == 798
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert page.is_lsm
        assert page.is_contiguous
        assert page.photometric == RGB
        assert page.databytecounts[2] == 0  # no strip data
        assert page.dataoffsets[2] == 242632  # bogus offset
        assert page.compression == NONE
        assert page.imagewidth == 400
        assert page.imagelength == 300
        assert page.bitspersample == 8
        assert page.samplesperpixel == 2
        page = tif.pages[1]
        assert page.is_reduced
        assert page.photometric == RGB
        assert page.planarconfig == SEPARATE
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 128
        assert page.imagelength == 96
        assert page.samplesperpixel == 3
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (19, 21, 2, 300, 400)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'TZCYX'
        if 1:
            series = tif.series[1]
            assert series.shape == (19, 21, 3, 96, 128)
            assert series.dtype.name == 'uint8'
            assert series.axes == 'TZCYX'
        # assert data
        data = tif.asarray(out='memmap')
        assert isinstance(data, numpy.core.memmap)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (19, 21, 2, 300, 400)
        assert data.dtype.name == 'uint8'
        assert data[18, 20, 1, 199, 299] == 39
        if 1:
            data = tif.asarray(series=1)
            assert isinstance(data, numpy.ndarray)
            assert data.shape == (19, 21, 3, 96, 128)
            assert data.dtype.name == 'uint8'
            assert tuple(data[18, 20, :, 64, 96]) == (22, 22, 0)
        del data
        # assert lsm_info tags
        tags = tif.lsm_metadata
        assert tags['DimensionX'] == 400
        assert tags['DimensionY'] == 300
        assert tags['DimensionZ'] == 21
        assert tags['DimensionTime'] == 19
        assert tags['DimensionChannels'] == 2
        # assert lsm_scan_info tags
        tags = tif.lsm_metadata['ScanInformation']
        assert tags['ScanMode'] == 'Stack'
        assert tags['User'] == 'zjfhe'
        assert len(tags['Tracks']) == 3
        assert len(tags['Tracks'][0]['DataChannels']) == 1
        track = tags['Tracks'][0]
        assert track['DataChannels'][0]['Name'] == 'Ch3'
        assert track['DataChannels'][0]['BitsPerSample'] == 8
        assert len(track['IlluminationChannels']) == 6
        assert track['IlluminationChannels'][5]['Name'] == '488'
        assert track['IlluminationChannels'][5]['Wavelength'] == 488.0
        assert__str__(tif, 0)


def test_read_lsm_earpax2isl11():
    """Test read LSM: TZCYX (1, 19, 3, 512, 512) uint8, RGB, LZW."""
    fname = data_file('lsm/earpax2isl11.lsm')
    with TiffFile(fname) as tif:
        assert tif.is_lsm
        assert tif.byteorder == '<'
        assert len(tif.pages) == 38
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert page.is_lsm
        assert not page.is_contiguous
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert corrected strip_byte_counts
        assert page.tags['StripByteCounts'].value == (262144, 262144, 262144)
        assert page.databytecounts == (131514, 192933, 167874)
        page = tif.pages[1]
        assert page.is_reduced
        assert page.photometric == RGB
        assert page.planarconfig == SEPARATE
        assert page.compression == NONE
        assert page.imagewidth == 128
        assert page.imagelength == 128
        assert page.samplesperpixel == 3
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (1, 19, 3, 512, 512)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'TZCYX'
        if 1:
            series = tif.series[1]
            assert series.shape == (1, 19, 3, 128, 128)
            assert series.dtype.name == 'uint8'
            assert series.axes == 'TZCYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (1, 19, 3, 512, 512)
        assert data.dtype.name == 'uint8'
        assert tuple(data[0, 18, :, 200, 320]) == (17, 22, 21)
        if 1:
            data = tif.asarray(series=1)
            assert isinstance(data, numpy.ndarray)
            assert data.shape == (1, 19, 3, 128, 128)
            assert data.dtype.name == 'uint8'
            assert tuple(data[0, 18, :, 64, 64]) == (25, 5, 33)
        # assert lsm_info tags
        tags = tif.lsm_metadata
        assert tags['DimensionX'] == 512
        assert tags['DimensionY'] == 512
        assert tags['DimensionZ'] == 19
        assert tags['DimensionTime'] == 1
        assert tags['DimensionChannels'] == 3
        # assert lsm_scan_info tags
        tags = tif.lsm_metadata['ScanInformation']
        assert tags['ScanMode'] == 'Stack'
        assert tags['User'] == 'megason'
        assert__str__(tif)


@pytest.mark.skipif(SKIP_HUGE or IS_32BIT, reason='huge image')
def test_read_lsm_mb231paxgfp_060214():
    """Test read LSM with many LZW compressed pages."""
    # TZCYX (Stack mode), (60, 31, 2, 512, 512), 3720
    fname = data_file('lsm/MB231paxgfp_060214.lsm')
    with TiffFile(fname) as tif:
        assert tif.is_lsm
        assert tif.byteorder == '<'
        assert len(tif.pages) == 3720
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert page.is_lsm
        assert not page.is_contiguous
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 2
        page = tif.pages[1]
        assert page.is_reduced
        assert page.photometric == RGB
        assert page.planarconfig == SEPARATE
        assert page.compression == NONE
        assert page.imagewidth == 128
        assert page.imagelength == 128
        assert page.samplesperpixel == 3
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (60, 31, 2, 512, 512)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'TZCYX'
        if 1:
            series = tif.series[1]
            assert series.shape == (60, 31, 3, 128, 128)
            assert series.dtype.name == 'uint8'
            assert series.axes == 'TZCYX'
        # assert data
        data = tif.asarray(out='memmap', maxworkers=None)
        assert isinstance(data, numpy.core.memmap)
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (60, 31, 2, 512, 512)
        assert data.dtype.name == 'uint16'
        assert data[59, 30, 1, 256, 256] == 222
        del data
        # assert lsm_info tags
        tags = tif.lsm_metadata
        assert tags['DimensionX'] == 512
        assert tags['DimensionY'] == 512
        assert tags['DimensionZ'] == 31
        assert tags['DimensionTime'] == 60
        assert tags['DimensionChannels'] == 2
        # assert some lsm_scan_info tags
        tags = tif.lsm_metadata['ScanInformation']
        assert tags['ScanMode'] == 'Stack'
        assert tags['User'] == 'lfdguest1'
        assert__str__(tif, 0)


def test_read_lsm_lzw_no_eoi():
    """Test read LSM with LZW compressed strip without EOI."""
    # The first LZW compressed strip in page 834 has no EOI
    # such that too much data is returned from the decoder and
    # the data of the 2nd channel was getting corrupted
    fname = data_file('lsm/MB231paxgfp_060214.lsm')
    with TiffFile(fname) as tif:
        assert tif.is_lsm
        assert tif.byteorder == '<'
        assert len(tif.pages) == 3720
        assert len(tif.series) == 2
        # assert page properties
        page = tif.pages[0]
        assert not page.is_contiguous
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 2
        page = tif.pages[834]
        assert isinstance(page, TiffFrame)
        assert page.databytecounts == (454886, 326318)
        assert page.dataoffsets == (344655101, 345109987)
        # assert second channel is not corrupted
        data = page.asarray()
        assert tuple(data[:, 0, 0]) == (288, 238)
        assert__str__(tif, 0)


def test_read_stk_zseries():
    """Test read MetaMorph STK z-series."""
    fname = data_file('stk/zseries.stk')
    with TiffFile(fname) as tif:
        assert tif.is_stk
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 320
        assert page.imagelength == 256
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'MetaMorph'
        assert page.tags['DateTime'].value == '2000:01:02 15:06:33'
        assert page.description.startswith('Acquired from MV-1500')
        # assert uic tags
        tags = tif.stk_metadata
        assert tags['Name'] == 'Z Series'
        assert tags['NumberPlanes'] == 11
        assert ''.join(tags['StageLabel']) == ''
        assert tags['ZDistance'][10] == 2.5
        assert len(tags['Wavelengths']) == 11
        assert tags['Wavelengths'][10] == 490.0
        assert len(tags['AbsoluteZ']) == 11
        assert tags['AbsoluteZ'][10] == 150.0
        assert tuple(tags['StagePosition'][10]) == (0.0, 0.0)
        assert tuple(tags['CameraChipOffset'][10]) == (0.0, 0.0)
        assert tags['PlaneDescriptions'][0].startswith('Acquired from MV-1500')
        assert str(tags['DatetimeCreated'][0]) == (
            '2000-02-02T15:06:02.000783000')
        # assert series properties
        series = tif.series[0]
        assert series.shape == (11, 256, 320)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'ZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (11, 256, 320)
        assert data.dtype.name == 'uint16'
        assert data[8, 159, 255] == 1156
        assert__str__(tif)


def test_read_stk_zser24():
    """Test read MetaMorph STK RGB z-series."""
    fname = data_file('stk/zser24.stk')
    with TiffFile(fname) as tif:
        assert tif.is_stk
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric == RGB
        assert page.compression == NONE
        assert page.imagewidth == 160
        assert page.imagelength == 128
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        assert page.tags['Software'].value == 'MetaMorph'
        assert page.tags['DateTime'].value == '2000:01:02 15:11:23'
        # assert uic tags
        tags = tif.stk_metadata
        assert tags['Name'] == 'Color Encoded'
        assert tags['NumberPlanes'] == 11
        assert ''.join(tags['StageLabel']) == ''
        assert tags['ZDistance'][10] == 2.5
        assert len(tags['Wavelengths']) == 11
        assert tags['Wavelengths'][10] == 510.0
        assert len(tags['AbsoluteZ']) == 11
        assert tags['AbsoluteZ'][10] == 150.0
        assert tuple(tags['StagePosition'][10]) == (0.0, 0.0)
        assert tuple(tags['CameraChipOffset'][10]) == (320., 256.)
        assert str(tags['DatetimeCreated'][0]) == (
            '2000-02-02T15:10:34.000264000')
        # assert series properties
        series = tif.series[0]
        assert series.shape == (11, 128, 160, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'ZYXS'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (11, 128, 160, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[8, 100, 135]) == (70, 63, 0)
        assert__str__(tif)


def test_read_stk_diatoms3d():
    """Test read MetaMorph STK time-series."""
    fname = data_file('stk/diatoms3d.stk')
    with TiffFile(fname) as tif:
        assert tif.is_stk
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 196
        assert page.imagelength == 191
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'MetaMorph'
        assert page.tags['DateTime'].value == '2000:01:04 14:57:22'
        # assert uic tags
        tags = tif.stk_metadata
        assert tags['Name'] == 'diatoms3d'
        assert tags['NumberPlanes'] == 10
        assert ''.join(tags['StageLabel']) == ''
        assert tags['ZDistance'][9] == 3.54545
        assert len(tags['Wavelengths']) == 10
        assert tags['Wavelengths'][9] == 440.0
        assert len(tags['AbsoluteZ']) == 10
        assert tags['AbsoluteZ'][9] == 12898.15
        assert tuple(tags['StagePosition'][9]) == (0.0, 0.0)
        assert tuple(tags['CameraChipOffset'][9]) == (206., 148.)
        assert tags['PlaneDescriptions'][0].startswith(
            'Acquired from Flashbus.')
        assert str(tags['DatetimeCreated'][0]) == (
            '2000-02-04T14:38:37.000738000')
        # assert series properties
        series = tif.series[0]
        assert series.shape == (10, 191, 196)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'ZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (10, 191, 196)
        assert data.dtype.name == 'uint8'
        assert data[8, 100, 135] == 223
        assert__str__(tif)


def test_read_stk_greenbeads():
    """Test read MetaMorph STK time-series, but time_created is corrupt (?)."""
    # 8bit palette is present but should not be applied
    fname = data_file('stk/greenbeads.stk')
    with TiffFile(fname) as tif:
        assert tif.is_stk
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric == PALETTE
        assert page.compression == NONE
        assert page.imagewidth == 298
        assert page.imagelength == 322
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'MetaMorph 7.5.3.0'
        assert page.tags['DateTime'].value == '2008:05:09 17:35:32'
        # assert uic tags
        tags = tif.stk_metadata
        assert tags['Name'] == 'Green'
        assert tags['NumberPlanes'] == 79
        assert tags['ZDistance'][1] == 0.0
        assert len(tags['Wavelengths']) == 79
        assert tuple(tags['CameraChipOffset'][0]) == (0.0, 0.0)
        assert str(tags['DatetimeModified'][0]) == (
            '2008-05-09T17:35:33.000274000')
        assert 'AbsoluteZ' not in tags
        # assert series properties
        series = tif.series[0]
        assert series.shape == (79, 322, 298)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'IYX'  # corrupt time_created
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (79, 322, 298)
        assert data.dtype.name == 'uint8'
        assert data[43, 180, 102] == 205
        assert__str__(tif)


def test_read_stk_10xcalib():
    """Test read MetaMorph STK two planes, not Z or T series."""
    fname = data_file('stk/10xcalib.stk')
    with TiffFile(fname) as tif:
        assert tif.is_stk
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric != PALETTE
        assert page.compression == NONE
        assert page.imagewidth == 640
        assert page.imagelength == 480
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'MetaMorph'
        assert page.tags['DateTime'].value == '2000:03:28 09:24:37'
        # assert uic tags
        tags = tif.stk_metadata
        assert tags['Name'] == '10xcalib'
        assert tags['NumberPlanes'] == 2
        assert tuple(tags['Wavelengths']) == (440.0, 440.0)
        assert tags['XCalibration'] == 1.24975007
        assert tags['YCalibration'] == 1.24975007
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 480, 640)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'IYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (2, 480, 640)
        assert data.dtype.name == 'uint8'
        assert data[1, 339, 579] == 56
        assert__str__(tif)


@pytest.mark.skipif(IS_32BIT, reason='might segfault due to low memory')
@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
def test_read_stk_112508h100():
    """Test read MetaMorph STK large time-series."""
    fname = data_file('stk/112508h100.stk')
    with TiffFile(fname) as tif:
        assert tif.is_stk
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric != PALETTE
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 128
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'MetaMorph 7.5.3.0'
        assert page.tags['DateTime'].value == '2008:11:25 18:59:20'
        # assert uic tags
        tags = tif.stk_metadata
        assert tags['Name'] == 'Photometrics'
        assert tags['NumberPlanes'] == 2048
        assert len(tags['PlaneDescriptions']) == 2048
        assert tags['PlaneDescriptions'][0].startswith(
            'Acquired from Photometrics\r\n')
        assert tags['CalibrationUnits'] == 'pixel'
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2048, 128, 512)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'TYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (2048, 128, 512)
        assert data.dtype.name == 'uint16'
        assert data[2047, 64, 128] == 7132
        assert__str__(tif)


def test_read_ndpi_cmu_1_ndpi():
    """Test read Hamamatsu NDPI slide, JPEG."""
    fname = data_file('HamamatsuNDPI/CMU-1.ndpi')
    with TiffFile(fname) as tif:
        assert tif.is_ndpi
        assert len(tif.pages) == 5
        assert len(tif.series) == 5
        for page in tif.pages:
            assert page.ndpi_tags['Model'] == 'NanoZoomer'
        # first page
        page = tif.pages[0]
        assert page.is_ndpi
        assert page.photometric == YCBCR
        assert page.compression == JPEG
        assert page.shape == (38144, 51200, 3)
        assert page.ndpi_tags['Magnification'] == 20.0
        # page 4
        page = tif.pages[4]
        assert page.is_ndpi
        assert page.photometric == YCBCR
        assert page.compression == JPEG
        assert page.shape == (408, 1191, 3)
        assert page.ndpi_tags['Magnification'] == -1.0
        assert page.asarray()[226, 629, 0] == 167
        assert__str__(tif)


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
def test_read_ndpi_cmu_2():
    """Test read Hamamatsu NDPI slide, JPEG."""
    # JPEG stream too large to be opened with unpatched libjpeg
    fname = data_file('HamamatsuNDPI/CMU-2.ndpi')
    with TiffFile(fname) as tif:
        assert tif.is_ndpi
        assert len(tif.pages) == 6
        assert len(tif.series) == 6
        for page in tif.pages:
            assert page.ndpi_tags['Model'] == 'NanoZoomer'
        # first page
        page = tif.pages[0]
        assert page.is_ndpi
        assert page.photometric == YCBCR
        assert page.compression == JPEG
        assert page.shape == (33792, 79872, 3)
        assert page.ndpi_tags['Magnification'] == 20.0
        # with pytest.raises(RuntimeError):
        data = page.asarray()
        assert data.shape == (33792, 79872, 3)
        assert data.dtype == 'uint8'
        # page 5
        page = tif.pages[-1]
        assert page.is_ndpi
        assert page.photometric == YCBCR
        assert page.compression == JPEG
        assert page.shape == (408, 1191, 3)
        assert page.ndpi_tags['Magnification'] == -1.0
        assert page.asarray()[226, 629, 0] == 181
        assert__str__(tif)


def test_read_svs_cmu_1():
    """Test read Aperio SVS slide, JPEG and LZW."""
    fname = data_file('AperioSVS/CMU-1.svs')
    with TiffFile(fname) as tif:
        assert tif.is_svs
        assert not tif.is_scanimage
        assert len(tif.pages) == 6
        assert len(tif.series) == 6
        for page in tif.pages:
            svs_description_metadata(page.description)
        # first page
        page = tif.pages[0]
        assert page.is_svs
        assert page.is_chroma_subsampled
        assert page.photometric == RGB
        assert page.is_tiled
        assert page.compression == JPEG
        assert page.shape == (32914, 46000, 3)
        metadata = svs_description_metadata(page.description)
        assert metadata['Aperio Image Library'] == 'v10.0.51'
        assert metadata['Originalheight'] == 33014
        # page 4
        page = tif.pages[4]
        assert page.is_svs
        assert page.is_reduced
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.shape == (463, 387, 3)
        metadata = svs_description_metadata(page.description)
        assert metadata[''] == 'label 387x463'
        assert__str__(tif)


def test_read_svs_jp2k_33003_1():
    """Test read Aperio SVS slide, JP2000 and LZW."""
    fname = data_file('AperioSVS/JP2K-33003-1.svs')
    with TiffFile(fname) as tif:
        assert tif.is_svs
        assert not tif.is_scanimage
        assert len(tif.pages) == 6
        assert len(tif.series) == 6
        for page in tif.pages:
            svs_description_metadata(page.description)
        # first page
        page = tif.pages[0]
        assert page.is_svs
        assert not page.is_chroma_subsampled
        assert page.photometric == RGB
        assert page.is_tiled
        assert page.compression.name == 'APERIO_JP2000_YCBC'
        assert page.shape == (17497, 15374, 3)
        metadata = svs_description_metadata(page.description)
        assert metadata['Aperio Image Library'] == 'v10.0.50'
        assert metadata['Originalheight'] == 17597
        # page 4
        page = tif.pages[4]
        assert page.is_svs
        assert page.is_reduced
        assert page.photometric == RGB
        assert page.compression == LZW
        assert page.shape == (422, 415, 3)
        metadata = svs_description_metadata(page.description)
        assert metadata[''] == 'label 415x422'
        assert__str__(tif)


def test_read_scanimage_metadata():
    """Test read ScanImage metadata."""
    fname = data_file('ScanImage/TS_UnitTestImage_BigTIFF.tif')
    with open(fname, 'rb') as fh:
        frame_data, roi_data = read_scanimage_metadata(fh)
    assert frame_data['SI.hChannels.channelType'] == ['stripe', 'stripe']
    assert roi_data['RoiGroups']['imagingRoiGroup']['ver'] == 1


def test_read_scanimage_no_framedata():
    """Test read ScanImage no FrameData."""
    fname = data_file('ScanImage/PSF001_ScanImage36.tif')
    with TiffFile(fname) as tif:
        assert tif.is_scanimage
        assert len(tif.pages) == 100
        assert len(tif.series) == 1
        # no non-tiff scanimage_metadata
        assert 'FrameData' not in tif.scanimage_metadata
        # assert page properties
        page = tif.pages[0]
        assert page.is_scanimage
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 256
        assert page.imagelength == 256
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # description tags
        metadata = scanimage_description_metadata(page.description)
        assert metadata['state.software.version'] == 3.6
        assert__str__(tif)


def test_read_scanimage_bigtiff():
    """Test read ScanImage BigTIFF."""
    fname = data_file('ScanImage/TS_UnitTestImage_BigTIFF.tif')
    with TiffFile(fname) as tif:
        assert tif.is_scanimage
        assert len(tif.pages) == 3
        assert len(tif.series) == 1

        # assert page properties
        page = tif.pages[0]
        assert page.is_scanimage
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # metadata in description, software, artist tags
        metadata = scanimage_description_metadata(page.description)
        assert metadata['frameNumbers'] == 1
        metadata = scanimage_description_metadata(
            page.tags['Software'].value)
        assert metadata['SI.TIFF_FORMAT_VERSION'] == 3
        metadata = scanimage_artist_metadata(page.tags['Artist'].value)
        assert metadata['RoiGroups']['imagingRoiGroup']['ver'] == 1
        metadata = tif.scanimage_metadata
        assert metadata['FrameData']['SI.TIFF_FORMAT_VERSION'] == 3
        assert metadata['RoiGroups']['imagingRoiGroup']['ver'] == 1
        assert metadata['Description']['frameNumbers'] == 1
        assert__str__(tif)


def test_read_ome_single_channel():
    """Test read OME image."""
    # 2D (single image)
    # OME-TIFF reference images from
    # https://www.openmicroscopy.org/site/support/ome-model/ome-tiff
    fname = data_file('OME/single-channel.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (167, 439)
        assert data.dtype.name == 'int8'
        assert data[158, 428] == 91
        assert__str__(tif)


def test_read_ome_multi_channel():
    """Test read OME multi channel image."""
    # 2D (3 channels)
    fname = data_file('OME/multi-channel.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 3
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'CYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (3, 167, 439)
        assert data.dtype.name == 'int8'
        assert data[2, 158, 428] == 91
        assert__str__(tif)


def test_read_ome_z_series():
    """Test read OME volume."""
    # 3D (5 focal planes)
    fname = data_file('OME/z-series.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 5
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (5, 167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'ZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (5, 167, 439)
        assert data.dtype.name == 'int8'
        assert data[4, 158, 428] == 91
        assert__str__(tif)


def test_read_ome_multi_channel_z_series():
    """Test read OME multi-channel volume."""
    # 3D (5 focal planes, 3 channels)
    fname = data_file('OME/multi-channel-z-series.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 15
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 5, 167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'CZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (3, 5, 167, 439)
        assert data.dtype.name == 'int8'
        assert data[2, 4, 158, 428] == 91
        assert__str__(tif)


def test_read_ome_time_series():
    """Test read OME time-series of images."""
    # 3D (7 time points)
    fname = data_file('OME/time-series.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 7
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (7, 167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'TYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (7, 167, 439)
        assert data.dtype.name == 'int8'
        assert data[6, 158, 428] == 91
        assert__str__(tif)


def test_read_ome_multi_channel_time_series():
    """Test read OME time-series of multi-channel images."""
    # 3D (7 time points, 3 channels)
    fname = data_file('OME/multi-channel-time-series.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 21
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (7, 3, 167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'TCYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (7, 3, 167, 439)
        assert data.dtype.name == 'int8'
        assert data[6, 2, 158, 428] == 91
        assert__str__(tif)


def test_read_ome_4d_series():
    """Test read OME time-series of volumes."""
    # 4D (7 time points, 5 focal planes)
    fname = data_file('OME/4D-series.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 35
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (7, 5, 167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'TZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (7, 5, 167, 439)
        assert data.dtype.name == 'int8'
        assert data[6, 4, 158, 428] == 91
        assert__str__(tif)


def test_read_ome_multi_channel_4d_series():
    """Test read OME time-series of multi-channel volumes."""
    # 4D (7 time points, 5 focal planes, 3 channels)
    fname = data_file('OME/multi-channel-4D-series.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 105
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 439
        assert page.imagelength == 167
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (7, 3, 5, 167, 439)
        assert series.dtype.name == 'int8'
        assert series.axes == 'TCZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (7, 3, 5, 167, 439)
        assert data.dtype.name == 'int8'
        assert data[6, 0, 4, 158, 428] == 91
        assert__str__(tif)


def test_read_ome_modulo_flim():
    """Test read OME modulo FLIM."""
    # Two channels each recorded at two timepoints and eight histogram bins
    fname = data_file('OME/FLIM-modulo-sample.ome.tiff')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 32
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 180
        assert page.imagelength == 200
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 8, 2, 200, 180)
        assert series.dtype.name == 'int8'
        assert series.axes == 'THCYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (2, 8, 2, 200, 180)
        assert data.dtype.name == 'int8'
        assert data[1, 7, 1, 190, 161] == 92
        assert__str__(tif)


def test_read_ome_modulo_spim():
    """Test read OME modulo SPIM."""
    # 2x2 tile of planes each recorded at 4 angles
    fname = data_file('OME/SPIM-modulo-sample.ome.tiff')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 192
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value == 'LOCI Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 160
        assert page.imagelength == 220
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 4, 2, 4, 2, 220, 160)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'TRZACYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (3, 4, 2, 4, 2, 220, 160)
        assert data.dtype.name == 'uint8'
        assert data[2, 3, 1, 3, 1, 210, 151] == 92
        assert__str__(tif)


def test_read_ome_modulo_lambda():
    """Test read OME modulo LAMBDA."""
    # Excitation of 5 wavelength [big-lambda] each recorded at 10 emission
    # wavelength ranges [lambda].
    fname = data_file('OME/LAMBDA-modulo-sample.ome.tiff')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 50
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value == 'LOCI Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 200
        assert page.imagelength == 200
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (10, 5, 200, 200)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'EPYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (10, 5, 200, 200)
        assert data.dtype.name == 'uint8'
        assert data[9, 4, 190, 192] == 92
        assert__str__(tif)


def test_read_ome_multi_image_pixels():
    """Test read OME with three image series."""
    fname = data_file('OME/multi-image-pixels.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 86
        assert len(tif.series) == 3
        # assert page properties
        for (i, axes, shape) in ((0, 'CTYX', (2, 7, 555, 431)),
                                 (1, 'TZYX', (6, 2, 461, 348)),
                                 (2, 'TZCYX', (4, 5, 3, 239, 517))):
            series = tif.series[i]
            page = series.pages[0]
            assert page.is_contiguous
            assert page.tags['Software'].value == 'LOCI Bio-Formats'
            assert page.compression == NONE
            assert page.imagewidth == shape[-1]
            assert page.imagelength == shape[-2]
            assert page.bitspersample == 8
            assert page.samplesperpixel == 1
            # assert series properties
            assert series.shape == shape
            assert series.dtype.name == 'uint8'
            assert series.axes == axes
            # assert data
            data = tif.asarray(series=i)
            assert isinstance(data, numpy.ndarray)
            assert data.shape == shape
            assert data.dtype.name == 'uint8'
            assert__str__(tif)


def test_read_ome_zen_2chzt():
    """Test read OME time-series of two-channel volumes by ZEN 2011."""
    fname = data_file('OME/zen_2chzt.ome.tiff')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 798
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value == 'ZEN 2011 (blue edition)'
        assert page.compression == NONE
        assert page.imagewidth == 400
        assert page.imagelength == 300
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 19, 21, 300, 400)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'CTZYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (2, 19, 21, 300, 400)
        assert data.dtype.name == 'uint8'
        assert data[1, 10, 10, 100, 245] == 78
        assert__str__(tif, 0)


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
def test_read_ome_multifile():
    """Test read OME CTZYX series in 86 files."""
    # (2, 43, 10, 512, 512) CTZYX uint8 in 86 files, 10 pages each
    fname = data_file('OME/tubhiswt-4D/tubhiswt_C0_TP10.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 10
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 43, 10, 512, 512)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'CTZYX'
        # assert other files are closed
        for page in tif.series[0].pages:
            assert bool(page.parent.filehandle._fh) == (page.parent == tif)
        # assert data
        data = tif.asarray(out='memmap')
        assert isinstance(data, numpy.core.memmap)
        assert data.shape == (2, 43, 10, 512, 512)
        assert data.dtype.name == 'uint8'
        assert data[1, 42, 9, 426, 272] == 123
        del data
        # assert other files are still closed
        for page in tif.series[0].pages:
            assert bool(page.parent.filehandle._fh) == (page.parent == tif)
        assert__str__(tif)
    # assert all files stay open
    # with TiffFile(fname) as tif:
    #     for page in tif.series[0].pages:
    #         self.assertTrue(page.parent.filehandle._fh)
    #     data = tif.asarray(out='memmap')
    #     for page in tif.series[0].pages:
    #         self.assertTrue(page.parent.filehandle._fh)


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
def test_read_ome_multifile_missing(caplog):
    """Test read OME referencing missing files."""
    # (2, 43, 10, 512, 512) CTZYX uint8, 85 files missing
    fname = data_file('OME/tubhiswt_C1_TP42.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 10
        assert len(tif.series) == 1
        assert 'failed to read' in caplog.text
        # assert page properties
        page = tif.pages[0]
        TiffPage.__str__(page, 4)
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        page = tif.pages[-1]
        TiffPage.__str__(page, 4)
        assert page.shape == (512, 512)
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 43, 10, 512, 512)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'CTZYX'
        # assert data
        data = tif.asarray(out='memmap')
        assert isinstance(data, numpy.core.memmap)
        assert data.shape == (2, 43, 10, 512, 512)
        assert data.dtype.name == 'uint8'
        assert data[1, 42, 9, 426, 272] == 123
        del data
        assert__str__(tif)


def test_read_ome_rgb():
    """Test read OME RGB image."""
    # https://github.com/openmicroscopy/bioformats/pull/1986
    fname = data_file('OME/test_rgb.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 1280
        assert page.imagelength == 720
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 720, 1280)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'CYX'
        assert series.offset == 17524
        # assert data
        data = tif.asarray()
        assert data.shape == (3, 720, 1280)
        assert data.dtype.name == 'uint8'
        assert data[1, 158, 428] == 253
        assert__str__(tif)


def test_read_ome_float_modulo_attributes():
    """Test read OME with floating point modulo attributes."""
    # reported by Start Berg. File by Lorenz Maier.
    fname = data_file('OME/float_modulo_attributes.ome.tiff')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '>'
        assert len(tif.pages) == 2
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 512, 512)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'QYX'
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (2, 512, 512)
        assert data.dtype.name == 'uint16'
        assert data[1, 158, 428] == 51
        assert__str__(tif)


def test_read_ome_cropped(caplog):
    """Test read bad OME by ImageJ cropping."""
    # ImageJ produces invalid ome-xml when cropping
    # http://lists.openmicroscopy.org.uk/pipermail/ome-devel/2013-December
    #  /002631.html
    # Reported by Hadrien Mary on Dec 11, 2013
    fname = data_file('ome/cropped.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 100
        assert len(tif.series) == 1
        assert 'invalid TiffData index' in caplog.text
        # assert page properties
        page = tif.pages[0]
        assert page.tags['Software'].value[:15] == 'OME Bio-Formats'
        assert page.imagewidth == 324
        assert page.imagelength == 249
        assert page.bitspersample == 16
        # assert series properties
        series = tif.series[0]
        assert series.shape == (5, 10, 2, 249, 324)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'TZCYX'
        # assert data
        data = tif.asarray()
        assert data.shape == (5, 10, 2, 249, 324)
        assert data.dtype.name == 'uint16'
        assert data[4, 9, 1, 175, 123] == 9605
        del data
        assert__str__(tif)


def test_read_ome_nikon(caplog):
    """Test read bad OME by Nikon."""
    # OME-XML references only first image
    # received from E. Gratton
    fname = data_file('OME/Nikon-cell011.ome.tif')
    with TiffFile(fname) as tif:
        assert tif.is_ome
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1000
        assert len(tif.series) == 1
        assert 'index out of range' in caplog.text
        # assert page properties
        page = tif.pages[0]
        assert page.photometric != RGB
        assert page.imagewidth == 1982
        assert page.imagelength == 1726
        assert page.bitspersample == 16
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert len(series._pages) == 1
        assert len(series.pages) == 1
        assert series.offset == 16  # contiguous
        assert series.shape == (1726, 1982)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YX'
        assert__str__(tif)

    with TiffFile(fname, is_ome=False) as tif:
        assert not tif.is_ome
        # assert series properties
        series = tif.series[0]
        assert len(series.pages) == 1000
        assert series.offset is None  # not contiguous
        assert series.shape == (1000, 1726, 1982)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'IYX'
        assert__str__(tif)


# Test Andor

def test_read_andor_light_sheet_512p():
    """Test read Andor."""
    # 12113x13453, uint16
    fname = data_file('andor/light sheet 512px.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 100
        assert len(tif.series) == 1
        assert tif.is_andor
        # assert page properties
        page = tif.pages[0]
        assert page.is_andor
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # assert metadata
        t = page.andor_tags
        assert t['SoftwareVersion'] == '4.23.30014.0'
        assert t['Frames'] == 100.0
        # assert series properties
        series = tif.series[0]
        assert series.shape == (100, 512, 512)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'IYX'
        # assert data
        data = tif.asarray()
        assert data.shape == (100, 512, 512)
        assert data.dtype.name == 'uint16'
        assert round(abs(data[50, 256, 256]-703), 7) == 0
        assert__str__(tif, 0)


# Test NIH Image sample images (big endian with nih_image_header)

def test_read_nih_morph():
    """Test read NIH."""
    # 388x252 uint8
    fname = data_file('nihimage/morph.tiff')
    with TiffFile(fname) as tif:
        assert tif.is_nih
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 388
        assert page.imagelength == 252
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (252, 388)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert NIH tags
        tags = tif.nih_metadata
        assert tags['FileID'] == 'IPICIMAG'
        assert tags['PixelsPerLine'] == 388
        assert tags['nLines'] == 252
        assert tags['ForegroundIndex'] == 255
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (252, 388)
        assert data.dtype.name == 'uint8'
        assert data[195, 144] == 41
        assert__str__(tif)


def test_read_nih_silver_lake():
    """Test read NIH palette."""
    # 259x187 16 bit palette
    fname = data_file('nihimage/silver lake.tiff')
    with TiffFile(fname) as tif:
        assert tif.is_nih
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric == PALETTE
        assert page.imagewidth == 259
        assert page.imagelength == 187
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (187, 259)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert NIH tags
        tags = tif.nih_metadata
        assert tags['FileID'] == 'IPICIMAG'
        assert tags['PixelsPerLine'] == 259
        assert tags['nLines'] == 187
        assert tags['ForegroundIndex'] == 109
        # assert data
        data = page.asrgb()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (187, 259, 3)
        assert data.dtype.name == 'uint16'
        assert tuple(data[86, 102, :]) == (26214, 39321, 39321)
        assert__str__(tif)


def test_read_imagej_focal1():
    """Test read ImageJ 205x434x425 uint8."""
    fname = data_file('imagej/focal1.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 205
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric != RGB
        assert page.imagewidth == 425
        assert page.imagelength == 434
        assert page.bitspersample == 8
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert series.offset == 768
        assert series.shape == (205, 434, 425)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'IYX'
        assert len(series._pages) == 1
        assert len(series.pages) == 205
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.34k'
        assert ijtags['images'] == 205
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (205, 434, 425)
        assert data.dtype.name == 'uint8'
        assert data[102, 216, 212] == 120
        assert__str__(tif, 0)


def test_read_imagej_hela_cells():
    """Test read ImageJ 512x672 RGB uint16."""
    fname = data_file('imagej/hela-cells.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 672
        assert page.imagelength == 512
        assert page.bitspersample == 16
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert series.shape == (512, 672, 3)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YXS'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.46i'
        assert ijtags['channels'] == 3
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (512, 672, 3)
        assert data.dtype.name == 'uint16'
        assert tuple(data[255, 336]) == (440, 378, 298)
        assert__str__(tif)


def test_read_imagej_flybrain():
    """Test read ImageJ 57x256x256 RGB."""
    fname = data_file('imagej/flybrain.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 57
        assert len(tif.series) == 1  # hyperstack
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 256
        assert page.imagelength == 256
        assert page.bitspersample == 8
        # assert series properties
        series = tif.series[0]
        assert series.shape == (57, 256, 256, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'ZYXS'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.43d'
        assert ijtags['slices'] == 57
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (57, 256, 256, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[18, 108, 97]) == (165, 157, 0)
        assert__str__(tif)


def test_read_imagej_confocal_series():
    """Test read ImageJ 25x2x400x400 ZCYX."""
    fname = data_file('imagej/confocal-series.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 50
        assert len(tif.series) == 1  # hyperstack
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 400
        assert page.imagelength == 400
        assert page.bitspersample == 8
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert series.shape == (25, 2, 400, 400)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'ZCYX'
        assert len(series._pages) == 1
        assert len(series.pages) == 50
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.43d'
        assert ijtags['images'] == len(tif.pages)
        assert ijtags['channels'] == 2
        assert ijtags['slices'] == 25
        assert ijtags['hyperstack']
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (25, 2, 400, 400)
        assert data.dtype.name == 'uint8'
        assert tuple(data[12, :, 100, 300]) == (6, 66)
        # assert only two pages are loaded
        assert isinstance(tif.pages.pages[0], TiffPage)
        assert isinstance(tif.pages.pages[1], TiffFrame)
        assert tif.pages.pages[2] == 8001073
        assert tif.pages.pages[-1] == 8008687
        assert__str__(tif)


def test_read_imagej_graphite():
    """Test read ImageJ 1024x593 float32."""
    fname = data_file('imagej/graphite1-1024.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 1024
        assert page.imagelength == 593
        assert page.bitspersample == 32
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert len(series._pages) == 1
        assert len(series.pages) == 1
        assert series.shape == (593, 1024)
        assert series.dtype.name == 'float32'
        assert series.axes == 'YX'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.47t'
        assert round(abs(ijtags['max']-1686.10949707), 7) == 0
        assert round(abs(ijtags['min']-852.08605957), 7) == 0
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (593, 1024)
        assert data.dtype.name == 'float32'
        assert round(abs(data[443, 656]-2203.040771484375), 7) == 0
        assert__str__(tif)


def test_read_imagej_bat_cochlea_volume():
    """Test read ImageJ 114 images, no frames, slices, channels specified."""
    fname = data_file('imagej/bat-cochlea-volume.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 114
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric != RGB
        assert page.imagewidth == 121
        assert page.imagelength == 154
        assert page.bitspersample == 8
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert len(series._pages) == 1
        assert len(series.pages) == 114
        assert series.shape == (114, 154, 121)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'IYX'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.20n'
        assert ijtags['images'] == 114
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (114, 154, 121)
        assert data.dtype.name == 'uint8'
        assert data[113, 97, 61] == 255
        assert__str__(tif)


def test_read_imagej_first_instar_brain():
    """Test read ImageJ 56x256x256x3 ZYXS."""
    fname = data_file('imagej/first-instar-brain.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 56
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == RGB
        assert page.imagewidth == 256
        assert page.imagelength == 256
        assert page.bitspersample == 8
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert len(series._pages) == 1
        assert len(series.pages) == 56
        assert series.shape == (56, 256, 256, 3)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'ZYXS'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.44j'
        assert ijtags['images'] == 56
        assert ijtags['slices'] == 56
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (56, 256, 256, 3)
        assert data.dtype.name == 'uint8'
        assert tuple(data[55, 151, 112]) == (209, 8, 58)
        assert__str__(tif)


def test_read_imagej_fluorescentcells():
    """Test read ImageJ three channels."""
    fname = data_file('imagej/FluorescentCells.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 3
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric == PALETTE
        assert page.imagewidth == 512
        assert page.imagelength == 512
        assert page.bitspersample == 8
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert series.shape == (3, 512, 512)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'CYX'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.40c'
        assert ijtags['images'] == 3
        assert ijtags['channels'] == 3
        # assert data
        data = tif.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (3, 512, 512)
        assert data.dtype.name == 'uint8'
        assert tuple(data[:, 256, 256]) == (57, 120, 13)
        assert__str__(tif)


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_EXTENDED, reason='large image')
def test_read_imagej_100000_pages():
    """Test read ImageJ with 100000 pages."""
    # 100000x64x64
    # file is big endian, memory mapped
    fname = data_file('large/100000_pages.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 100000
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 64
        assert page.imagelength == 64
        assert page.bitspersample == 16
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert len(series._pages) == 1
        assert len(series.pages) == 100000
        assert series.shape == (100000, 64, 64)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'TYX'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.48g'
        assert round(abs(ijtags['max']-119.0), 7) == 0
        assert round(abs(ijtags['min']-86.0), 7) == 0
        # assert data
        data = tif.asarray(out='memmap')
        assert isinstance(data, numpy.core.memmap)
        assert data.shape == (100000, 64, 64)
        assert data.dtype.name == 'uint16'
        assert round(abs(data[7310, 25, 25]-100), 7) == 0
        del data
        assert__str__(tif, 0)


def test_read_imagej_invalid_metadata(caplog):
    """Test read bad ImageJ metadata."""
    # file contains 1 page but metadata claims 3500 images
    # memory map big endian data
    fname = data_file('sima/0.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        assert 'invalid metadata or corrupted file' in caplog.text
        # assert page properties
        page = tif.pages[0]
        assert page.photometric != RGB
        assert page.imagewidth == 173
        assert page.imagelength == 173
        assert page.bitspersample == 16
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert series.offset == 8  # 8
        assert series.shape == (173, 173)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YX'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['ImageJ'] == '1.49i'
        assert ijtags['images'] == 3500
        # assert data
        data = tif.asarray(out='memmap')
        assert isinstance(data, numpy.core.memmap)
        assert data.shape == (173, 173)
        assert data.dtype.name == 'uint16'
        assert data[94, 34] == 1257
        del data
        assert__str__(tif)


def test_read_imagej_invalid_hyperstack():
    """Test read bad ImageJ hyperstack."""
    # file claims to be a hyperstack but is not stored as such
    # produced by OME writer
    # reported by Taras Golota on 10/27/2016
    fname = data_file('imagej/X0.ome.CTZ.perm.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '<'
        assert len(tif.pages) == 48  # not a hyperstack
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.photometric != RGB
        assert page.imagewidth == 1392
        assert page.imagelength == 1040
        assert page.bitspersample == 16
        assert page.is_contiguous
        # assert series properties
        series = tif.series[0]
        assert series.offset is None  # not contiguous
        assert series.shape == (2, 4, 6, 1040, 1392)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'TZCYX'
        # assert ImageJ tags
        ijtags = tif.imagej_metadata
        assert ijtags['hyperstack']
        assert ijtags['images'] == 48
        assert__str__(tif)


def test_read_fluoview_lsp1_v_laser():
    """Test read FluoView CTYX."""
    # raises 'UnicodeWarning: Unicode equal comparison failed' on Python 2
    fname = data_file('fluoview/lsp1-V-laser0.3-1.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 100
        assert len(tif.series) == 1
        assert tif.is_fluoview
        # assert page properties
        page = tif.pages[0]
        assert page.is_fluoview
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 256
        assert page.imagelength == 256
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # assert metadata
        m = fluoview_description_metadata(page.description)
        assert m['Version Info']['FLUOVIEW Version'] == (
            'FV10-ASW ,ValidBitColunt=12')
        assert tuple(m['LUT Ch1'][255]) == (255, 255, 255)
        mm = tif.fluoview_metadata
        assert mm['ImageName'] == 'lsp1-V-laser0.3-1.oib'
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2, 50, 256, 256)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'CTYX'
        # assert data
        data = tif.asarray()
        assert data.shape == (2, 50, 256, 256)
        assert data.dtype.name == 'uint16'
        assert round(abs(data[1, 36, 128, 128]-824), 7) == 0
        assert__str__(tif)


@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
@pytest.mark.skipif(IS_32BIT, reason='MemoryError on 32 bit')
def test_read_fluoview_120816_bf_f0000():
    """Test read FluoView TZYX."""
    fname = data_file('fluoview/120816_bf_f0000.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 864
        assert len(tif.series) == 1
        assert tif.is_fluoview
        # assert page properties
        page = tif.pages[0]
        assert page.is_fluoview
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 1024
        assert page.imagelength == 1024
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # assert metadata
        m = fluoview_description_metadata(page.description)
        assert m['Environment']['User'] == 'admin'
        assert m['Region Info (Fields) Field']['Width'] == 1331.2
        m = tif.fluoview_metadata
        assert m['ImageName'] == '120816_bf'
        # assert series properties
        series = tif.series[0]
        assert series.shape == (144, 6, 1024, 1024)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'TZYX'
        # assert data
        data = tif.asarray()
        assert data.shape == (144, 6, 1024, 1024)
        assert data.dtype.name == 'uint16'
        assert round(abs(data[1, 2, 128, 128]-8317), 7) == 0
        assert__str__(tif)


def test_read_metaseries():
    """Test read MetaSeries 1040x1392 uint16, LZW."""
    # Strips do not contain an EOI code as required by the TIFF spec.
    fname = data_file('metaseries/metaseries.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 1392
        assert page.imagelength == 1040
        assert page.bitspersample == 16
        # assert metadata
        assert page.description.startswith('<MetaData>')
        # assert series properties
        series = tif.series[0]
        assert series.shape == (1040, 1392)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert data.shape == (1040, 1392)
        assert data.dtype.name == 'uint16'
        assert data[256, 256] == 1917
        del data
        assert__str__(tif)


def test_read_metaseries_g4d7r():
    """Test read Metamorph/Metaseries."""
    # 12113x13453, uint16
    fname = data_file('metaseries/g4d7r.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        assert tif.is_metaseries
        # assert page properties
        page = tif.pages[0]
        assert page.is_metaseries
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 13453
        assert page.imagelength == 12113
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        # assert metadata
        m = metaseries_description_metadata(page.description)
        assert m['ApplicationVersion'] == '7.8.6.0'
        assert m['PlaneInfo']['pixel-size-x'] == 13453
        assert m['SetInfo']['number-of-planes'] == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (12113, 13453)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray(out='memmap')
        assert isinstance(data, numpy.core.memmap)
        assert data.shape == (12113, 13453)
        assert data.dtype.name == 'uint16'
        assert round(abs(data[512, 2856]-4095), 7) == 0
        del data
        assert__str__(tif)


def test_read_mdgel_rat():
    """Test read Molecular Dynamics GEL."""
    # Second page does not contain data, only private tags
    # TZYX, uint16, OME multifile TIFF
    fname = data_file('mdgel/rat.gel')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 2
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 1528
        assert page.imagelength == 413
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == (
            "ImageQuant Software Release Version 2.0")
        assert page.tags['PageName'].value == r"C:\DATA\RAT.GEL"

        # assert 2nd page properties
        page = tif.pages[1]
        assert page.is_mdgel
        assert page.imagewidth == 0
        assert page.imagelength == 0
        assert page.bitspersample == 1
        assert page.samplesperpixel == 1
        assert page.tags['MDFileTag'].value == 2
        assert page.tags['MDScalePixel'].value == (1, 21025)
        assert len(page.tags['MDColorTable'].value) == 17
        md = tif.mdgel_metadata
        assert md['SampleInfo'] == "Rat slices from Dr. Schweitzer"
        assert md['PrepDate'] == "12 July 90"
        assert md['PrepTime'] == "40hr"
        assert md['FileUnits'] == "Counts"

        # assert series properties
        series = tif.series[0]
        assert series.shape == (413, 1528)
        assert series.dtype.name == 'float32'
        assert series.axes == 'YX'
        # assert data
        data = series.asarray()
        assert isinstance(data, numpy.ndarray)
        assert data.shape == (413, 1528)
        assert data.dtype.name == 'float32'
        assert round(abs(data[260, 740]-399.1728515625), 7) == 0
        assert__str__(tif)


def test_read_mediacy_imagepro():
    """Test read Media Cybernetics SEQ."""
    # TZYX, uint16, OME multifile TIFF
    fname = data_file('mediacy/imagepro.tif')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_mediacy
        assert page.is_contiguous
        assert page.compression == NONE
        assert page.imagewidth == 201
        assert page.imagelength == 201
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'Image-Pro Plus'
        assert page.tags['MC_Id'].value[:-1] == b'MC TIFF 4.0'
        # assert series properties
        series = tif.series[0]
        assert series.shape == (201, 201)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert data
        data = tif.asarray()
        assert data.shape == (201, 201)
        assert data.dtype.name == 'uint8'
        assert round(abs(data[120, 34]-4), 7) == 0
        assert__str__(tif)


def test_read_pilatus_100k():
    """Test read Pilatus."""
    fname = data_file('TvxPilatus/Pilatus100K_scan030_033.tiff')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert tif.is_pilatus
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 487
        assert page.imagelength == 195
        assert page.bitspersample == 32
        assert page.samplesperpixel == 1
        # assert metadata
        assert page.tags['Model'].value == (
            'PILATUS 100K, S/N 1-0230, Cornell University')
        attr = pilatus_description_metadata(page.description)
        assert attr['Tau'] == 1.991e-07
        assert attr['Silicon'] == 0.000320
        assert__str__(tif)


def test_read_pilatus_gibuf2():
    """Test read Pilatus."""
    fname = data_file('TvxPilatus/GIbuf2_A9_18_001_0009.tiff')
    with TiffFile(fname) as tif:
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert tif.is_pilatus
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 487
        assert page.imagelength == 195
        assert page.bitspersample == 32
        assert page.samplesperpixel == 1
        # assert metadata
        assert page.tags['Model'].value == 'PILATUS 100K-S, S/N 1-0299,'
        attr = pilatus_description_metadata(page.description)
        assert attr['Filter_transmission'] == 1.0
        assert attr['Silicon'] == 0.000320
        assert__str__(tif)


def test_read_epics_attrib():
    """Test read EPICS."""
    fname = data_file('epics/attrib.tif')
    with TiffFile(fname) as tif:
        assert tif.is_epics
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (2048, 2048)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YX'
        # assert page properties
        page = tif.pages[0]
        assert page.shape == (2048, 2048)
        assert page.imagewidth == 2048
        assert page.imagelength == 2048
        assert page.bitspersample == 16
        assert page.is_contiguous
        # assert EPICS tags
        tags = tif.epics_metadata
        assert tags['timeStamp'] == datetime.datetime(
            1995, 6, 2, 11, 31, 31, 571414)
        assert tags['uniqueID'] == 15
        assert tags['Focus'] == 0.6778
        assert__str__(tif)


def test_read_tvips_tietz_16bit():
    """Test read TVIPS metadata."""
    # file provided by Marco Oster on 10/26/2016
    fname = data_file('tvips/test_tietz_16bit.tif')
    with TiffFile(fname) as tif:
        assert tif.is_tvips
        tvips = tif.tvips_metadata
        assert tvips['Magic'] == 0xaaaaaaaa
        assert tvips['ImageFolder'] == u'B:\\4Marco\\Images\\Tiling_EMTOOLS\\'
        assert__str__(tif)


def test_read_geotiff_dimapdocument():
    """Test read GeoTIFF with 43 MB XML tag value."""
    # tag 65000  45070067s @487  "<Dimap_Document...
    fname = data_file('geotiff/DimapDocument.tif')
    with TiffFile(fname) as tif:
        assert tif.is_geotiff
        assert tif.byteorder == '>'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (1830, 1830)
        assert series.dtype.name == 'uint16'
        assert series.axes == 'YX'
        # assert page properties
        page = tif.pages[0]
        assert page.shape == (1830, 1830)
        assert page.imagewidth == 1830
        assert page.imagelength == 1830
        assert page.bitspersample == 16
        assert page.is_contiguous
        assert page.tags['65000'].value.startswith(
            '<?xml version="1.0" encoding="ISO-8859-1"?>')
        # assert GeoTIFF tags
        tags = tif.geotiff_metadata
        assert tags['GTCitationGeoKey'] == 'WGS 84 / UTM zone 29N'
        assert tags['ProjectedCSTypeGeoKey'] == 32629
        assert_array_almost_equal(
            tags['ModelTransformation'],
            [[60., 0., 0., 6.e5], [0., -60., 0., 5900040.],
             [0., 0., 0., 0.], [0., 0., 0., 1.]])
        assert__str__(tif)


def test_read_geotiff_spaf27_markedcorrect():
    """Test read GeoTIFF."""
    fname = data_file('geotiff/spaf27_markedcorrect.tif')
    with TiffFile(fname) as tif:
        assert tif.is_geotiff
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert series properties
        series = tif.series[0]
        assert series.shape == (20, 20)
        assert series.dtype.name == 'uint8'
        assert series.axes == 'YX'
        # assert page properties
        page = tif.pages[0]
        assert page.shape == (20, 20)
        assert page.imagewidth == 20
        assert page.imagelength == 20
        assert page.bitspersample == 8
        assert page.is_contiguous
        # assert GeoTIFF tags
        tags = tif.geotiff_metadata
        assert tags['GTCitationGeoKey'] == 'NAD27 / California zone VI'
        assert tags['GeogAngularUnitsGeoKey'] == 9102
        assert tags['ProjFalseOriginLatGeoKey'] == 32.1666666666667
        assert_array_almost_equal(tags['ModelPixelScale'],
                                  [195.509321, 198.32184, 0])
        assert__str__(tif)


def test_read_qpi():
    """Test read PerkinElmer-QPI."""
    fname = data_file('PerkinElmer-QPI/'
                      'LuCa-7color_[13860,52919]_1x1component_data.tiff')
    with TiffFile(fname) as tif:
        assert len(tif.series) == 2
        assert len(tif.pages) == 9
        assert tif.is_qpi
        page = tif.pages[0]
        assert page.compression == LZW
        assert page.photometric == MINISBLACK
        assert page.planarconfig == CONTIG
        assert page.imagewidth == 1868
        assert page.imagelength == 1400
        assert page.bitspersample == 32
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'PerkinElmer-QPI'
        # assert data
        image = tif.asarray()
        assert image.shape == (8, 1400, 1868)
        assert image.dtype == 'float32'
        assert image[7, 1200, 1500] == 2.2132580280303955
        image = tif.asarray(series=1)
        assert image.shape == (350, 467, 3)
        assert image.dtype == 'uint8'
        assert image[300, 400, 1] == 48
        assert__str__(tif)


def test_read_zif():
    """Test read Zoomable Image Format ZIF."""
    fname = data_file('zif/ZoomifyImageExample.zif')
    with TiffFile(fname) as tif:
        # assert tif.is_zif
        assert len(tif.pages) == 5
        assert len(tif.series) == 5
        for page in tif.pages:
            assert page.description == ('Created by Objective '
                                        'Pathology Services')
        # first page
        page = tif.pages[0]
        assert page.photometric == YCBCR
        assert page.compression == JPEG
        assert page.shape == (3120, 2080, 3)
        assert tuple(page.asarray()[3110, 2070, :]) == (27, 45, 59)
        # page 4
        page = tif.pages[-1]
        assert page.photometric == YCBCR
        assert page.compression == JPEG
        assert page.shape == (195, 130, 3)
        assert tuple(page.asarray()[191, 127, :]) == (30, 49, 66)
        assert__str__(tif)


def test_read_sis():
    """Test read Olympus SIS."""
    fname = data_file('sis/4A5IE8EM_F00000409.tif')
    with TiffFile(fname) as tif:
        assert tif.is_sis
        assert tif.byteorder == '<'
        assert len(tif.pages) == 122
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.imagewidth == 353
        assert page.imagelength == 310
        assert page.bitspersample == 16
        assert page.samplesperpixel == 1
        assert page.tags['Software'].value == 'analySIS 5.0'
        # assert data
        data = tif.asarray()
        assert data.shape == (61, 2, 310, 353)
        assert data[30, 1, 256, 256] == 210
        # assert metadata
        sis = tif.sis_metadata
        assert sis['axes'] == 'TC'
        assert sis['shape'] == (61, 2)
        assert sis['Band'][1]['BandName'] == 'Fura380'
        assert sis['Band'][0]['LUT'].shape == (256, 3)
        assert sis['Time']['TimePos'].shape == (61,)
        assert sis['name'] == 'Hela-Zellen'
        assert sis['magnification'] == 60.0
        assert__str__(tif)


def test_read_sis_noini():
    """Test read Olympus SIS without INI tag."""
    fname = data_file('sis/110.tif')
    with TiffFile(fname) as tif:
        assert tif.is_sis
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.imagewidth == 2560
        assert page.imagelength == 1920
        assert page.bitspersample == 8
        assert page.samplesperpixel == 3
        # assert metadata
        sis = tif.sis_metadata
        assert 'axes' not in sis
        assert sis['magnification'] == 20.0
        assert__str__(tif)


def test_read_sem_metadata():
    """Test read Zeiss SEM metadata."""
    # file from hyperspy tests
    fname = data_file('hyperspy/test_tiff_Zeiss_SEM_1k.tif')
    with TiffFile(fname) as tif:
        assert tif.is_sem
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric == PALETTE
        assert page.imagewidth == 1024
        assert page.imagelength == 768
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert data and metadata
        data = page.asrgb()
        assert tuple(data[563, 320]) == (38550, 38550, 38550)
        sem = tif.sem_metadata
        assert sem[''][3] == 2.614514e-06
        assert sem['ap_date'] == ('Date', '23 Dec 2015')
        assert sem['ap_time'] == ('Time', '9:40:32')
        assert sem['dp_image_store'] == ('Store resolution', '1024 * 768')
        if not IS_PY2:
            assert sem['ap_fib_fg_emission_actual'] == (
                'Flood Gun Emission Actual', 0.0, u'µA')
        else:
            assert sem['ap_fib_fg_emission_actual'] == (
                'Flood Gun Emission Actual', 0.0, '\xb5A')
        assert__str__(tif)


def test_read_sem_bad_metadata():
    """Test read Zeiss SEM metadata with wrong length."""
    # reported by Klaus Schwarzburg on 8/27/2018
    fname = data_file('issues/sem_bad_metadata.tif')
    with TiffFile(fname) as tif:
        assert tif.is_sem
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric == PALETTE
        assert page.imagewidth == 1024
        assert page.imagelength == 768
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert data and metadata
        data = page.asrgb()
        assert tuple(data[350, 150]) == (17476, 17476, 17476)
        sem = tif.sem_metadata
        assert sem['sv_version'][1] == 'V05.07.00.00 : 08-Jul-14'
        assert__str__(tif)


def test_read_fei_metadata():
    """Test read Helios FEI metadata."""
    # file from hyperspy tests
    fname = data_file('hyperspy/test_tiff_FEI_SEM.tif')
    with TiffFile(fname) as tif:
        assert tif.is_fei
        assert tif.byteorder == '<'
        assert len(tif.pages) == 1
        assert len(tif.series) == 1
        # assert page properties
        page = tif.pages[0]
        assert page.is_contiguous
        assert page.photometric != PALETTE
        assert page.imagewidth == 1536
        assert page.imagelength == 1103
        assert page.bitspersample == 8
        assert page.samplesperpixel == 1
        # assert data and metadata
        data = page.asarray()
        assert data[563, 320] == 220
        fei = tif.fei_metadata
        assert fei['User']['User'] == 'supervisor'
        assert fei['System']['DisplayHeight'] == 0.324
        assert__str__(tif)


###############################################################################

# Test TiffWriter

WRITE_DATA = numpy.arange(3*219*301).astype('uint16').reshape((3, 219, 301))


@pytest.mark.skipif(SKIP_EXTENDED, reason='generates >2 GB')
@pytest.mark.parametrize('shape', [
    (219, 301),
    (219, 301, 2),
    (219, 301, 3),
    (219, 301, 4),
    (2, 219, 301),
    (3, 219, 301),
    (4, 219, 301),
    (5, 219, 301),
    (4, 3, 219, 301),
    (4, 219, 301, 3),
    (3, 4, 219, 301),
    (3, 4, 219, 301, 1)])
@pytest.mark.parametrize('dtype', list('?bhiqfdBHIQFD'))
@pytest.mark.parametrize('byteorder', ['>', '<'])
@pytest.mark.parametrize('bigtiff', ['plaintiff', 'bigtiff'])
@pytest.mark.parametrize('data', ['random', 'empty'])
def test_write(data, byteorder, bigtiff, dtype, shape):
    """Test TiffWriter with various options."""
    # TODO: test compression ?
    fname = '%s_%s_%s_%s%s' % (
        bigtiff,
        {'<': 'le', '>': 'be'}[byteorder],
        numpy.dtype(dtype).name,
        str(shape).replace(' ', ''),
        '_empty' if data == 'empty' else '')
    bigtiff = bigtiff == 'bigtiff'

    with TempFileName(fname) as fname:
        if data == 'empty':
            with TiffWriter(fname, byteorder=byteorder,
                            bigtiff=bigtiff) as tif:
                tif.save(shape=shape, dtype=dtype)
            with TiffFile(fname) as tif:
                assert__str__(tif)
                image = tif.asarray()
        else:
            data = random_data(dtype, shape)
            imwrite(fname, data, byteorder=byteorder, bigtiff=bigtiff)
            image = imread(fname)
            assert image.flags['C_CONTIGUOUS']
            assert_array_equal(data.squeeze(), image.squeeze())

        assert shape == image.shape
        assert dtype == image.dtype
        if not bigtiff:
            assert_jhove(fname)


def test_write_nopages():
    """Test write TIFF with no pages."""
    with TempFileName('nopages') as fname:
        with TiffWriter(fname) as tif:
            pass
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 0
            tif.asarray()
        if VALIDATE:
            with pytest.raises(ValueError):
                assert_jhove(fname)


def test_write_append_not_exists():
    """Test append to non existing file."""
    with TempFileName('append_not_exists.bin') as fname:
        # with self.assertRaises(ValueError):
        with TiffWriter(fname, append=True):
            pass


def test_write_append_nontif():
    """Test fail to append to non-TIFF file."""
    with TempFileName('append_nontif.bin') as fname:
        with open(fname, 'wb') as fh:
            fh.write(b'not a TIFF file')
        with pytest.raises(ValueError):
            with TiffWriter(fname, append=True):
                pass


def test_write_append_lsm():
    """Test fail to append to LSM file."""
    fname = data_file('lsm/take1.lsm')
    with pytest.raises(ValueError):
        with TiffWriter(fname, append=True):
            pass


def test_write_append_imwrite():
    """Test append using imwrite."""
    data = random_data('uint8', (21, 31))
    with TempFileName('imwrite_append') as fname:
        imwrite(fname, data, metadata=None)
        for _ in range(3):
            imwrite(fname, data, append=True, metadata=None)
        a = imread(fname)
        assert a.shape == (4, 21, 31)
        assert_array_equal(a[3], data)


def test_write_append():
    """Test append to existing TIFF file."""
    data = random_data('uint8', (21, 31))
    with TempFileName('append') as fname:
        with TiffWriter(fname) as tif:
            pass
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 0
            assert__str__(tif)

        with TiffWriter(fname, append=True) as tif:
            tif.save(data)
        with TiffFile(fname) as tif:
            assert len(tif.series) == 1
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.imagewidth == 31
            assert page.imagelength == 21
            assert__str__(tif)

        with TiffWriter(fname, append=True) as tif:
            tif.save(data)
            tif.save(data)
        with TiffFile(fname) as tif:
            assert len(tif.series) == 2
            assert len(tif.pages) == 3
            page = tif.pages[0]
            assert page.imagewidth == 31
            assert page.imagelength == 21
            assert_array_equal(tif.asarray(series=1)[1], data)
            assert__str__(tif)

        assert_jhove(fname)


def test_write_append_bytesio():
    """Test append to existing TIFF file in BytesIO."""
    data = random_data('uint8', (21, 31))
    offset = 11
    file = BytesIO()
    file.write(b'a' * offset)

    with TiffWriter(file) as tif:
        pass
    file.seek(offset)
    with TiffFile(file) as tif:
        assert len(tif.pages) == 0

    file.seek(offset)
    with TiffWriter(file, append=True) as tif:
        tif.save(data)
    file.seek(offset)
    with TiffFile(file) as tif:
        assert len(tif.series) == 1
        assert len(tif.pages) == 1
        page = tif.pages[0]
        assert page.imagewidth == 31
        assert page.imagelength == 21
        assert__str__(tif)

    file.seek(offset)
    with TiffWriter(file, append=True) as tif:
        tif.save(data)
        tif.save(data)
    file.seek(offset)
    with TiffFile(file) as tif:
        assert len(tif.series) == 2
        assert len(tif.pages) == 3
        page = tif.pages[0]
        assert page.imagewidth == 31
        assert page.imagelength == 21
        assert_array_equal(tif.asarray(series=1)[1], data)
        assert__str__(tif)


def test_write_roundtrip_filename():
    """Test write and read using file name."""
    data = imread(data_file('vigranumpy.tif'))
    with TempFileName('roundtrip_filename') as fname:
        imwrite(fname, data)
        assert_array_equal(imread(fname), data)


def test_write_roundtrip_openfile():
    """Test write and read using open file."""
    pad = b'0' * 7
    data = imread(data_file('vigranumpy.tif'))
    with TempFileName('roundtrip_openfile') as fname:
        with open(fname, 'wb') as fh:
            fh.write(pad)
            imwrite(fh, data)
            fh.write(pad)
        with open(fname, 'rb') as fh:
            fh.seek(len(pad))
            assert_array_equal(imread(fh), data)


def test_write_roundtrip_bytesio():
    """Test write and read using BytesIO."""
    pad = b'0' * 7
    data = imread(data_file('vigranumpy.tif'))
    buf = BytesIO()
    buf.write(pad)
    imwrite(buf, data)
    buf.write(pad)
    buf.seek(len(pad))
    assert_array_equal(imread(buf), data)


def test_write_pages():
    """Test write tags for contiguous data in all pages."""
    data = random_data('float32', (17, 219, 301))
    with TempFileName('pages') as fname:
        imwrite(fname, data, photometric='minisblack')
        assert_jhove(fname)
        # assert file
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 17
            for i, page in enumerate(tif.pages):
                assert page.is_contiguous
                assert page.planarconfig == CONTIG
                assert page.photometric == MINISBLACK
                assert page.imagewidth == 301
                assert page.imagelength == 219
                assert page.samplesperpixel == 1
                image = page.asarray()
                assert_array_equal(data[i], image)
            # assert series
            series = tif.series[0]
            assert series.offset is not None
            image = series.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_truncate():
    """Test only one page is written for truncated files."""
    shape = (4, 5, 6, 1)
    with TempFileName('truncate') as fname:
        imwrite(fname, random_data('uint8', shape), truncate=True)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1  # not 4
            page = tif.pages[0]
            assert page.is_shaped
            assert page.shape == (5, 6)
            assert '"shape": [4, 5, 6, 1]' in page.description
            assert '"truncated": true' in page.description
            series = tif.series[0]
            assert series.shape == shape
            assert len(series._pages) == 1
            assert len(series.pages) == 1
            data = tif.asarray()
            assert data.shape == shape
            assert__str__(tif)


def test_write_is_shaped():
    """Test files are written with shape."""
    with TempFileName('is_shaped') as fname:
        imwrite(fname, random_data('uint8', (4, 5, 6, 3)))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 4
            page = tif.pages[0]
            assert page.is_shaped
            assert page.description == '{"shape": [4, 5, 6, 3]}'
            assert__str__(tif)
    with TempFileName('is_shaped_with_description') as fname:
        descr = "test is_shaped_with_description"
        imwrite(fname, random_data('uint8', (5, 6, 3)), description=descr)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_shaped
            assert page.description == descr
            assert__str__(tif)


def test_write_extratags():
    """Test write extratags."""
    data = random_data('uint8', (2, 219, 301))
    description = "Created by TestTiffWriter\nLorem ipsum dolor..."
    pagename = "Page name"
    extratags = [(270, 's', 0, description, True),
                 ('PageName', 's', 0, pagename, False),
                 (50001, 'b', 1, b'1', True),
                 (50002, 'b', 2, b'12', True),
                 (50004, 'b', 4, b'1234', True),
                 (50008, 'B', 8, b'12345678', True),
                 ]
    with TempFileName('extratags') as fname:
        imwrite(fname, data, extratags=extratags)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            assert tif.pages[0].description1 == description
            assert 'ImageDescription' not in tif.pages[1].tags
            assert tif.pages[0].tags['PageName'].value == pagename
            assert tif.pages[1].tags['PageName'].value == pagename
            tags = tif.pages[0].tags
            assert tags['50001'].value == 49
            assert tags['50002'].value == (49, 50)
            assert tags['50004'].value == (49, 50, 51, 52)
            assert_array_equal(tags['50008'].value, b'12345678')
            #                   (49, 50, 51, 52, 53, 54, 55, 56))
            assert__str__(tif)


def test_write_double_tags():
    """Test write single and sequences of doubles."""
    # older versions of tifffile do not use offset to write doubles
    # reported by Eric Prestat on Feb 21, 2016
    data = random_data('uint8', (8, 8))
    value = math.pi
    extratags = [
        (34563, 'd', 1, value, False),
        (34564, 'd', 1, (value,), False),
        (34565, 'd', 2, (value, value), False),
        (34566, 'd', 2, [value, value], False),
        (34567, 'd', 2, numpy.array((value, value)), False),
    ]
    with TempFileName('double_tags') as fname:
        imwrite(fname, data, extratags=extratags)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            tags = tif.pages[0].tags
            assert tags['34563'].value == value
            assert tags['34564'].value == value
            assert tuple(tags['34565'].value) == (value, value)
            assert tuple(tags['34566'].value) == (value, value)
            assert tuple(tags['34567'].value) == (value, value)
            assert__str__(tif)

    with TempFileName('double_tags_bigtiff') as fname:
        imwrite(fname, data, bigtiff=True, extratags=extratags)
        # assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            tags = tif.pages[0].tags
            assert tags['34563'].value == value
            assert tags['34564'].value == value
            assert tuple(tags['34565'].value) == (value, value)
            assert tuple(tags['34566'].value) == (value, value)
            assert tuple(tags['34567'].value) == (value, value)
            assert__str__(tif)


def test_write_short_tags():
    """Test write single and sequences of words."""
    data = random_data('uint8', (8, 8))
    value = 65531
    extratags = [
        (34564, 'H', 1, (value,) * 1, False),
        (34565, 'H', 2, (value,) * 2, False),
        (34566, 'H', 3, (value,) * 3, False),
        (34567, 'H', 4, (value,) * 4, False),
    ]
    with TempFileName('short_tags') as fname:
        imwrite(fname, data, extratags=extratags)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            tags = tif.pages[0].tags
            assert tags['34564'].value == value
            assert tuple(tags['34565'].value) == (value,) * 2
            assert tuple(tags['34566'].value) == (value,) * 3
            assert tuple(tags['34567'].value) == (value,) * 4
            assert__str__(tif)


@pytest.mark.parametrize('subfiletype', [0b1, 0b10, 0b100, 0b1000, 0b1111])
def test_write_subfiletype(subfiletype):
    """Test write subfiletype."""
    data = random_data('uint8', (16, 16))
    if subfiletype & 0b100:
        data = data.astype('bool')
    with TempFileName('subfiletype_%i' % subfiletype) as fname:
        imwrite(fname, data, subfiletype=subfiletype)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            page = tif.pages[0]
            assert page.subfiletype == subfiletype
            assert page.is_reduced == subfiletype & 0b1
            assert page.is_multipage == subfiletype & 0b10
            assert page.is_mask == subfiletype & 0b100
            assert page.is_mrc == subfiletype & 0b1000
            assert_array_equal(data, page.asarray())
            assert__str__(tif)


def test_write_description_tag():
    """Test write two description tags."""
    data = random_data('uint8', (2, 219, 301))
    description = "Created by TestTiffWriter\nLorem ipsum dolor..."
    with TempFileName('description_tag') as fname:
        imwrite(fname, data, description=description)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            assert tif.pages[0].description == description
            assert tif.pages[0].description1 == '{"shape": [2, 219, 301]}'
            assert 'ImageDescription' not in tif.pages[1].tags
            assert__str__(tif)


def test_write_description_tag_nojson():
    """Test no JSON description is written with metatata=None."""
    data = random_data('uint8', (2, 219, 301))
    description = "Created by TestTiffWriter\nLorem ipsum dolor..."
    with TempFileName('description_tag_nojson') as fname:
        imwrite(fname, data, description=description, metadata=None)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            assert tif.pages[0].description == description
            assert 'ImageDescription' not in tif.pages[1].tags
            assert 'ImageDescription1' not in tif.pages[0].tags
            assert__str__(tif)


def test_write_software_tag():
    """Test write Software tag."""
    data = random_data('uint8', (2, 219, 301))
    software = "test_tifffile.py"
    with TempFileName('software_tag') as fname:
        imwrite(fname, data, software=software)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            assert tif.pages[0].software == software
            assert 'Software' not in tif.pages[1].tags
            assert__str__(tif)


def test_write_resolution_float():
    """Test write float Resolution tag."""
    data = random_data('uint8', (2, 219, 301))
    resolution = (92., 92.)
    with TempFileName('resolution_float') as fname:
        imwrite(fname, data, resolution=resolution)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            assert tif.pages[0].tags['XResolution'].value == (92, 1)
            assert tif.pages[0].tags['YResolution'].value == (92, 1)
            assert tif.pages[1].tags['XResolution'].value == (92, 1)
            assert tif.pages[1].tags['YResolution'].value == (92, 1)
            assert__str__(tif)


def test_write_resolution_rational():
    """Test write rational Resolution tag."""
    data = random_data('uint8', (1, 219, 301))
    resolution = ((300, 1), (300, 1))
    with TempFileName('resolution_rational') as fname:
        imwrite(fname, data, resolution=resolution)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            assert tif.pages[0].tags['XResolution'].value == (300, 1)
            assert tif.pages[0].tags['YResolution'].value == (300, 1)


def test_write_resolution_unit():
    """Test write Resolution tag unit."""
    data = random_data('uint8', (219, 301))
    resolution = (92., (9200, 100), None)
    with TempFileName('resolution_unit') as fname:
        imwrite(fname, data, resolution=resolution)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            assert tif.pages[0].tags['XResolution'].value == (92, 1)
            assert tif.pages[0].tags['YResolution'].value == (92, 1)
            assert tif.pages[0].tags['ResolutionUnit'].value == 1
            assert__str__(tif)


def test_write_compress_none():
    """Test write compress=0."""
    data = WRITE_DATA
    with TempFileName('compress_none') as fname:
        imwrite(fname, data, compress=0)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.compression == NONE
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert len(page.dataoffsets) == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_compress_deflate():
    """Test write ZLIB compression."""
    data = WRITE_DATA
    with TempFileName('compress_deflate') as fname:
        imwrite(fname, data, compress=('DEFLATE', 6))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == DEFLATE
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert page.rowsperstrip == 108
            assert len(page.dataoffsets) == 9
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_compress_deflate_level():
    """Test write ZLIB compression with level."""
    data = WRITE_DATA
    with TempFileName('compress_deflate_level') as fname:
        imwrite(fname, data, compress=9)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == ADOBE_DEFLATE
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_compress_lzma():
    """Test write LZMA compression."""
    data = WRITE_DATA
    with TempFileName('compress_lzma') as fname:
        imwrite(fname, data, compress='LZMA')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == LZMA
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert page.rowsperstrip == 108
            assert len(page.dataoffsets) == 9
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


@pytest.mark.skipif(IS_PY2, reason='zstd not available')
def test_write_compress_zstd():
    """Test write ZSTD compression."""
    data = WRITE_DATA
    with TempFileName('compress_zstd') as fname:
        imwrite(fname, data, compress='ZSTD')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == ZSTD
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert page.rowsperstrip == 108
            assert len(page.dataoffsets) == 9
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_compress_webp():
    """Test write WEBP compression."""
    data = WRITE_DATA.astype('uint8').reshape((219, 301, 3))
    with TempFileName('compress_webp') as fname:
        imwrite(fname, data, compress=('WEBP', -1))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == WEBP
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


@pytest.mark.parametrize('dtype', ['i1', 'u1', 'bool'])
def test_write_compress_packbits(dtype):
    """Test write PackBits compression."""
    uncompressed = numpy.frombuffer(
        b'\xaa\xaa\xaa\x80\x00\x2a\xaa\xaa\xaa\xaa\x80\x00'
        b'\x2a\x22\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa\xaa', dtype=dtype)
    shape = 2, 7, uncompressed.size
    data = numpy.empty(shape, dtype=dtype)
    data[..., :] = uncompressed
    with TempFileName('compress_packits_%s' % dtype) as fname:
        imwrite(fname, data, compress='PACKBITS')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == PACKBITS
            assert page.planarconfig == CONTIG
            assert page.imagewidth == uncompressed.size
            assert page.imagelength == 7
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_compress_rowsperstrip():
    """Test write rowsperstrip with compression."""
    data = WRITE_DATA
    with TempFileName('compress_rowsperstrip') as fname:
        imwrite(fname, data, compress=6, rowsperstrip=32)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == ADOBE_DEFLATE
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert page.rowsperstrip == 32
            assert len(page.dataoffsets) == 21
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_compress_tiled():
    """Test write compressed tiles."""
    data = WRITE_DATA
    with TempFileName('compress_tiled') as fname:
        imwrite(fname, data, compress=6, tile=(32, 32))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.is_tiled
            assert page.compression == ADOBE_DEFLATE
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert len(page.dataoffsets) == 210
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_compress_predictor():
    """Test write horizontal differencing."""
    data = WRITE_DATA
    with TempFileName('compress_predictor') as fname:
        imwrite(fname, data, compress=6, predictor=True)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.compression == ADOBE_DEFLATE
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert page.predictor == 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


@pytest.mark.parametrize('dtype', ['u2', 'f4'])
def test_write_compressed_predictor_tiled(dtype):
    """Test write horizontal differencing with tiles."""
    data = WRITE_DATA.astype(dtype)
    with TempFileName('compress_tiled_predictor_%s' % dtype) as fname:
        imwrite(fname, data, compress=6, predictor=True, tile=(32, 32))
        if dtype[0] != 'f':
            assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.is_tiled
            assert page.compression == ADOBE_DEFLATE
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            assert page.predictor == 3 if dtype[0] == 'f' else 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


@pytest.mark.skipif(sys.byteorder == 'big', reason="little endian only")
def test_write_write_bigendian():
    """Test write big endian file."""
    # also test memory mapping non-native byte order
    data = random_data('float32', (2, 3, 219, 301)).newbyteorder()
    with TempFileName('write_bigendian') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            assert len(tif.series) == 1
            assert tif.byteorder == '>'
            # assert not tif.isnative
            assert tif.series[0].offset is not None
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            # test reading data
            image = tif.asarray()
            assert_array_equal(data, image)
            image = page.asarray()
            assert_array_equal(data[0], image)
            # test direct memory mapping; returns big endian array
            image = tif.asarray(out='memmap')
            assert isinstance(image, numpy.core.memmap)
            assert image.dtype == numpy.dtype('>f4')
            assert_array_equal(data, image)
            del image
            image = page.asarray(out='memmap')
            assert isinstance(image, numpy.core.memmap)
            assert image.dtype == numpy.dtype('>f4')
            assert_array_equal(data[0], image)
            del image
            # test indirect memory mapping; returns native endian array
            image = tif.asarray(out='memmap:')
            assert isinstance(image, numpy.core.memmap)
            assert image.dtype == numpy.dtype('=f4')
            assert_array_equal(data, image)
            del image
            image = page.asarray(out='memmap:')
            assert isinstance(image, numpy.core.memmap)
            assert image.dtype == numpy.dtype('=f4')
            assert_array_equal(data[0], image)
            del image
            # test 2nd page
            page = tif.pages[1]
            image = page.asarray(out='memmap')
            assert isinstance(image, numpy.core.memmap)
            assert image.dtype == numpy.dtype('>f4')
            assert_array_equal(data[1], image)
            del image
            image = page.asarray(out='memmap:')
            assert isinstance(image, numpy.core.memmap)
            assert image.dtype == numpy.dtype('=f4')
            assert_array_equal(data[1], image)
            del image
            assert__str__(tif)


def test_write_empty_array():
    """Test write empty array fails."""
    with pytest.raises(ValueError):
        with TempFileName('empty') as fname:
            imwrite(fname, numpy.empty(0))


def test_write_pixel():
    """Test write single pixel."""
    data = numpy.zeros(1, dtype='uint8')
    with TempFileName('pixel') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            assert tif.series[0].axes == 'Y'
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 1
            assert page.imagelength == 1
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_small():
    """Test write small image."""
    data = random_data('uint8', (1, 1))
    with TempFileName('small') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 1
            assert page.imagelength == 1
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_2d_as_rgb():
    """Test write RGB color palette as RGB image."""
    # image length should be 1
    data = numpy.arange(3*256, dtype='uint16').reshape(256, 3) // 3
    with TempFileName('2d_as_rgb_contig') as fname:
        imwrite(fname, data, photometric=RGB)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            assert tif.series[0].axes == 'XS'
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 256
            assert page.imagelength == 1
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_invalid_contig_rgb():
    """Test write planar RGB with 2 samplesperpixel."""
    data = random_data('uint8', (219, 301, 2))
    with pytest.raises(ValueError):
        with TempFileName('invalid_contig_rgb') as fname:
            imwrite(fname, data, photometric=RGB)
    # default to pages
    with TempFileName('invalid_contig_rgb_pages') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 219
            assert tif.series[0].axes == 'QYX'
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 2
            assert page.imagelength == 301
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)
    # better save as contig samples
    with TempFileName('invalid_contig_rgb_samples') as fname:
        imwrite(fname, data, planarconfig='CONTIG')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            assert tif.series[0].axes == 'YXS'
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_invalid_planar_rgb():
    """Test write planar RGB with 2 samplesperpixel."""
    data = random_data('uint8', (2, 219, 301))
    with pytest.raises(ValueError):
        with TempFileName('invalid_planar_rgb') as fname:
            imwrite(fname, data, photometric=RGB, planarconfig='SEPARATE')
    # default to pages
    with TempFileName('invalid_planar_rgb_pages') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            assert tif.series[0].axes == 'QYX'
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)
    # or save as planar samples
    with TempFileName('invalid_planar_rgb_samples') as fname:
        imwrite(fname, data, planarconfig='SEPARATE')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            assert tif.series[0].axes == 'SYX'
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_gray():
    """Test write grayscale with extrasamples contig."""
    data = random_data('uint8', (301, 219, 2))
    with TempFileName('extrasamples_gray') as fname:
        imwrite(fname, data, extrasamples='UNASSALPHA')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.photometric == MINISBLACK
            assert page.planarconfig == CONTIG
            assert page.imagewidth == 219
            assert page.imagelength == 301
            assert page.samplesperpixel == 2
            assert page.extrasamples == 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_gray_planar():
    """Test write planar grayscale with extrasamples."""
    data = random_data('uint8', (2, 301, 219))
    with TempFileName('extrasamples_gray_planar') as fname:
        imwrite(fname, data, planarconfig='SEPARATE',
                extrasamples='UNASSALPHA')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.photometric == MINISBLACK
            assert page.planarconfig == SEPARATE
            assert page.imagewidth == 219
            assert page.imagelength == 301
            assert page.samplesperpixel == 2
            assert page.extrasamples == 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_gray_mix():
    """Test write grayscale with multiple extrasamples."""
    data = random_data('uint8', (301, 219, 4))
    with TempFileName('extrasamples_gray_mix') as fname:
        imwrite(fname, data, photometric='MINISBLACK',
                extrasamples=['ASSOCALPHA', 'UNASSALPHA', 'UNSPECIFIED'])
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.photometric == MINISBLACK
            assert page.imagewidth == 219
            assert page.imagelength == 301
            assert page.samplesperpixel == 4
            assert page.extrasamples == (1, 2, 0)
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_unspecified():
    """Test write RGB with unspecified extrasamples by default."""
    data = random_data('uint8', (301, 219, 5))
    with TempFileName('extrasamples_unspecified') as fname:
        imwrite(fname, data, photometric='RGB')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.photometric == RGB
            assert page.imagewidth == 219
            assert page.imagelength == 301
            assert page.samplesperpixel == 5
            assert page.extrasamples == (0, 0)
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_assocalpha():
    """Test write RGB with assocalpha extrasample."""
    data = random_data('uint8', (219, 301, 4))
    with TempFileName('extrasamples_assocalpha') as fname:
        imwrite(fname, data, photometric='RGB', extrasamples='ASSOCALPHA')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 4
            assert page.extrasamples == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_mix():
    """Test write RGB with mixture of extrasamples."""
    data = random_data('uint8', (219, 301, 6))
    with TempFileName('extrasamples_mix') as fname:
        imwrite(fname, data, photometric='RGB',
                extrasamples=['ASSOCALPHA', 'UNASSALPHA', 'UNSPECIFIED'])
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 6
            assert page.extrasamples == (1, 2, 0)
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_contig():
    """Test write contig grayscale with large number of extrasamples."""
    data = random_data('uint8', (3, 219, 301))
    with TempFileName('extrasamples_contig') as fname:
        imwrite(fname, data, planarconfig='CONTIG')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 219
            assert page.imagelength == 3
            assert page.samplesperpixel == 301
            assert len(page.extrasamples) == 301-1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)
    # better save as RGB planar
    with TempFileName('extrasamples_contig_planar') as fname:
        imwrite(fname, data, planarconfig='SEPARATE')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_contig_rgb2():
    """Test write contig RGB with large number of extrasamples."""
    data = random_data('uint8', (3, 219, 301))
    with TempFileName('extrasamples_contig_rgb2') as fname:
        imwrite(fname, data, photometric=RGB, planarconfig='CONTIG')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 219
            assert page.imagelength == 3
            assert page.samplesperpixel == 301
            assert len(page.extrasamples) == 301-3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)
    # better save as planar
    with TempFileName('extrasamples_contig_rgb2_planar') as fname:
        imwrite(fname, data, photometric=RGB, planarconfig='SEPARATE')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_planar():
    """Test write planar large number of extrasamples."""
    data = random_data('uint8', (219, 301, 3))
    with TempFileName('extrasamples_planar') as fname:
        imwrite(fname, data, planarconfig='SEPARATE')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric != RGB
            assert page.imagewidth == 3
            assert page.imagelength == 301
            assert page.samplesperpixel == 219
            assert len(page.extrasamples) == 219-1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_planar_rgb2():
    """Test write planar RGB with large number of extrasamples."""
    data = random_data('uint8', (219, 301, 3))
    with TempFileName('extrasamples_planar_rgb2') as fname:
        imwrite(fname, data, photometric=RGB, planarconfig='SEPARATE')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 3
            assert page.imagelength == 301
            assert page.samplesperpixel == 219
            assert len(page.extrasamples) == 219-3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_minisblack_planar():
    """Test write planar minisblack."""
    data = random_data('uint8', (3, 219, 301))
    with TempFileName('minisblack_planar') as fname:
        imwrite(fname, data, photometric='MINISBLACK')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 3
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_minisblack_contig():
    """Test write contig minisblack."""
    data = random_data('uint8', (219, 301, 3))
    with TempFileName('minisblack_contig') as fname:
        imwrite(fname, data, photometric='MINISBLACK')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 219
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 3
            assert page.imagelength == 301
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_scalar():
    """Test write 2D grayscale."""
    data = random_data('uint8', (219, 301))
    with TempFileName('scalar') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_scalar_3d():
    """Test write 3D grayscale."""
    data = random_data('uint8', (63, 219, 301))
    with TempFileName('scalar_3d') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 63
            page = tif.pages[62]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert isinstance(image, numpy.ndarray)
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_scalar_4d():
    """Test write 4D grayscale."""
    data = random_data('uint8', (3, 2, 219, 301))
    with TempFileName('scalar_4d') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 6
            page = tif.pages[5]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_contig_extrasample():
    """Test write grayscale with contig extrasamples."""
    data = random_data('uint8', (219, 301, 2))
    with TempFileName('contig_extrasample') as fname:
        imwrite(fname, data, planarconfig='CONTIG')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_planar_extrasample():
    """Test write grayscale with planar extrasamples."""
    data = random_data('uint8', (2, 219, 301))
    with TempFileName('planar_extrasample') as fname:
        imwrite(fname, data, planarconfig='SEPARATE')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 2
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_rgb_contig():
    """Test write auto contig RGB."""
    data = random_data('uint8', (219, 301, 3))
    with TempFileName('rgb_contig') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_rgb_planar():
    """Test write auto planar RGB."""
    data = random_data('uint8', (3, 219, 301))
    with TempFileName('rgb_planar') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_rgba_contig():
    """Test write auto contig RGBA."""
    data = random_data('uint8', (219, 301, 4))
    with TempFileName('rgba_contig') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 4
            assert page.extrasamples == UNASSALPHA
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_rgba_planar():
    """Test write auto planar RGBA."""
    data = random_data('uint8', (4, 219, 301))
    with TempFileName('rgba_planar') as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 4
            assert page.extrasamples == UNASSALPHA
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_contig_rgb():
    """Test write contig RGB with extrasamples."""
    data = random_data('uint8', (219, 301, 8))
    with TempFileName('extrasamples_contig') as fname:
        imwrite(fname, data, photometric=RGB)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 8
            assert len(page.extrasamples) == 5
            assert page.extrasamples[0] == UNSPECIFIED
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_extrasamples_planar_rgb():
    """Test write planar RGB with extrasamples."""
    data = random_data('uint8', (8, 219, 301))
    with TempFileName('extrasamples_planar') as fname:
        imwrite(fname, data, photometric=RGB)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 8
            assert len(page.extrasamples) == 5
            assert page.extrasamples[0] == UNSPECIFIED
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_tiled_compressed():
    """Test write compressed tiles."""
    data = random_data('uint8', (3, 219, 301))
    with TempFileName('tiled_compressed') as fname:
        imwrite(fname, data, compress=5, tile=(96, 64))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_tiled
            assert not page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.tilewidth == 64
            assert page.tilelength == 96
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_tiled():
    """Test write tiled."""
    data = random_data('uint16', (219, 301))
    with TempFileName('tiled') as fname:
        imwrite(fname, data, tile=(96, 64))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_tiled
            assert not page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.tilewidth == 64
            assert page.tilelength == 96
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_tiled_planar():
    """Test write planar tiles."""
    data = random_data('uint8', (4, 219, 301))
    with TempFileName('tiled_planar') as fname:
        imwrite(fname, data, tile=(1, 96, 64))  #
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_tiled
            assert not page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert not page.is_sgi
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.tilewidth == 64
            assert page.tilelength == 96
            assert page.samplesperpixel == 4
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_tiled_contig():
    """Test write contig tiles."""
    data = random_data('uint8', (219, 301, 3))
    with TempFileName('tiled_contig') as fname:
        imwrite(fname, data, tile=(96, 64))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_tiled
            assert not page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.tilewidth == 64
            assert page.tilelength == 96
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_tiled_pages():
    """Test write multiple tiled pages."""
    data = random_data('uint8', (5, 219, 301, 3))
    with TempFileName('tiled_pages') as fname:
        imwrite(fname, data, tile=(96, 64))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 5
            page = tif.pages[0]
            assert page.is_tiled
            assert not page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert not page.is_sgi
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.tilewidth == 64
            assert page.tilelength == 96
            assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_volume():
    """Test write tiled volume."""
    data = random_data('uint8', (253, 64, 96))
    with TempFileName('volume') as fname:
        imwrite(fname, data, tile=(64, 64, 64))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_sgi
            assert page.is_tiled
            assert not page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 96
            assert page.imagelength == 64
            assert page.imagedepth == 253
            assert page.tilewidth == 64
            assert page.tilelength == 64
            assert page.tiledepth == 64
            assert page.samplesperpixel == 1
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_volume_5d_planar_rgb():
    """Test write 5D array as grayscale volumes."""
    shape = (2, 3, 256, 64, 96)
    data = numpy.empty(shape, dtype='uint8')
    data[:] = numpy.arange(256, dtype='uint8').reshape(1, 1, -1, 1, 1)
    with TempFileName('volume_5d_planar_rgb') as fname:
        imwrite(fname, data, tile=(256, 64, 96))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 2
            page = tif.pages[0]
            assert page.is_sgi
            assert page.is_tiled
            assert page.is_contiguous
            assert page.planarconfig == SEPARATE
            assert page.photometric == RGB
            assert page.imagewidth == 96
            assert page.imagelength == 64
            assert page.imagedepth == 256
            assert page.tilewidth == 96
            assert page.tilelength == 64
            assert page.tiledepth == 256
            assert page.samplesperpixel == 3
            series = tif.series[0]
            assert len(series._pages) == 1
            assert len(series.pages) == 2
            assert series.offset is not None
            assert series.shape == shape
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


def test_write_volume_5d_contig_rgb():
    """Test write 6D array as contig RGB volumes."""
    shape = (2, 3, 256, 64, 96, 3)
    data = numpy.empty(shape, dtype='uint8')
    data[:] = numpy.arange(256, dtype='uint8').reshape(1, 1, -1, 1, 1, 1)
    with TempFileName('volume_5d_contig_rgb') as fname:
        imwrite(fname, data, tile=(256, 64, 96))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 6
            page = tif.pages[0]
            assert page.is_sgi
            assert page.is_tiled
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 96
            assert page.imagelength == 64
            assert page.imagedepth == 256
            assert page.tilewidth == 96
            assert page.tilelength == 64
            assert page.tiledepth == 256
            assert page.samplesperpixel == 3
            # self.assertEqual(page.tags['TileOffsets'].value, (352,))
            assert page.tags['TileByteCounts'].value == (4718592,)
            series = tif.series[0]
            assert len(series._pages) == 1
            assert len(series.pages) == 6
            assert series.offset is not None
            assert series.shape == shape
            image = tif.asarray()
            assert_array_equal(data, image)
            # assert iterating over series.pages
            data = data.reshape(6, 256, 64, 96, 3)
            for i, page in enumerate(series.pages):
                image = page.asarray()
                assert_array_equal(data[i], image)
            assert__str__(tif)


@pytest.mark.skipif(SKIP_EXTENDED, reason='large file')
def test_write_volume_5d_contig_rgb_empty():
    """Test write empty 6D array as contig RGB volumes."""
    shape = (2, 3, 256, 64, 96, 3)
    with TempFileName('volume_5d_contig_rgb_empty') as fname:
        with TiffWriter(fname) as tif:
            tif.save(shape=shape, dtype='uint8', tile=(256, 64, 96))
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 6
            page = tif.pages[0]
            assert page.is_sgi
            assert page.is_tiled
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 96
            assert page.imagelength == 64
            assert page.imagedepth == 256
            assert page.tilewidth == 96
            assert page.tilelength == 64
            assert page.tiledepth == 256
            assert page.samplesperpixel == 3
            # self.assertEqual(page.tags['TileOffsets'].value, (352,))
            assert page.tags['TileByteCounts'].value == (4718592,)
            series = tif.series[0]
            assert len(series._pages) == 1
            assert len(series.pages) == 6
            assert series.offset is not None
            assert series.shape == shape
            image = tif.asarray()
            assert_array_equal(image.shape, shape)
            assert__str__(tif)


def test_write_multiple_save():
    """Test append pages."""
    data = random_data('uint8', (5, 4, 219, 301, 3))
    with TempFileName('append') as fname:
        with TiffWriter(fname, bigtiff=True) as tif:
            for i in range(data.shape[0]):
                tif.save(data[i])
        # assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert tif.is_bigtiff
            assert len(tif.pages) == 20
            for page in tif.pages:
                assert page.is_contiguous
                assert page.planarconfig == CONTIG
                assert page.photometric == RGB
                assert page.imagewidth == 301
                assert page.imagelength == 219
                assert page.samplesperpixel == 3
            image = tif.asarray()
            assert_array_equal(data, image)
            assert__str__(tif)


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_HUGE, reason='3 GB image')
def test_write_3gb():
    """Test write 3 GB no-BigTiff file."""
    # https://github.com/blink1073/tifffile/issues/47
    data = numpy.empty((4096-32, 1024, 1024), dtype='uint8')
    with TempFileName('3gb', remove=False) as fname:
        imwrite(fname, data)
        assert_jhove(fname)
        # assert file
        with TiffFile(fname) as tif:
            assert not tif.is_bigtiff


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_HUGE, reason="5 GB image")
def test_write_bigtiff():
    """Test write 5GB BigTiff file."""
    data = numpy.empty((640, 1024, 1024), dtype='float64')
    data[:] = numpy.arange(640, dtype='float64').reshape(-1, 1, 1)
    with TempFileName('bigtiff') as fname:
        # TiffWriter should fail without bigtiff parameter
        with pytest.raises(ValueError):
            with TiffWriter(fname) as tif:
                tif.save(data)
        # imwrite should use bigtiff for large data
        imwrite(fname, data)
        # assert_jhove(fname)
        # assert file
        with TiffFile(fname) as tif:
            assert tif.is_bigtiff
            assert len(tif.pages) == 640
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 1024
            assert page.imagelength == 1024
            assert page.samplesperpixel == 1
            image = tif.asarray(out='memmap')
            assert_array_equal(data, image)
            del image
            assert__str__(tif)


@pytest.mark.parametrize('compress', [0, 6])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16'])
def test_write_palette(dtype, compress):
    """Test write palette images."""
    data = random_data(dtype, (3, 219, 301))
    cmap = random_data('uint16', (3, 2**(data.itemsize*8)))
    with TempFileName('palette_%i%s' % (compress, dtype)) as fname:
        imwrite(fname, data, colormap=cmap, compress=compress)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 3
            page = tif.pages[0]
            assert page.is_contiguous != bool(compress)
            assert page.planarconfig == CONTIG
            assert page.photometric == PALETTE
            assert page.imagewidth == 301
            assert page.imagelength == 219
            assert page.samplesperpixel == 1
            for i, page in enumerate(tif.pages):
                assert_array_equal(apply_colormap(data[i], cmap),
                                   page.asrgb())
            assert__str__(tif)


def test_write_palette_django():
    """Test write palette read from existing file."""
    fname = data_file('django.tiff')
    with TiffFile(fname) as tif:
        page = tif.pages[0]
        assert page.photometric == PALETTE
        assert page.imagewidth == 320
        assert page.imagelength == 480
        data = page.asarray()  # .squeeze()  # UserWarning ...
        cmap = page.colormap
        assert__str__(tif)
    with TempFileName('palette_django') as fname:
        imwrite(fname, data, colormap=cmap, compress=6)
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert not page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == PALETTE
            assert page.imagewidth == 320
            assert page.imagelength == 480
            assert page.samplesperpixel == 1
            image = page.asrgb(uint8=False)
            assert_array_equal(apply_colormap(data, cmap), image)
            assert__str__(tif)


def test_write_multiple_series():
    """Test write multiple data into one file using various options."""
    data1 = imread(data_file('ome/multi-channel-4D-series.ome.tif'))
    image1 = imread(data_file('django.tiff'))
    image2 = imread(data_file('horse-16bit-col-littleendian.tif'))
    with TempFileName('multiple_series') as fname:
        with TiffWriter(fname, bigtiff=False) as tif:
            tif.save(image1, compress=5, description='Django')
            tif.save(image2)
            tif.save(data1[0], metadata=dict(axes='TCZYX'))
            for i in range(1, data1.shape[0]):
                tif.save(data1[i])
            tif.save(data1[0], contiguous=False)
            tif.save(data1[0, 0, 0], tile=(64, 64))
            tif.save(image1, compress='LZMA', description='lzma')
        assert_jhove(fname)
        with TiffFile(fname) as tif:
            assert len(tif.pages) == 124
            assert len(tif.series) == 6
            serie = tif.series[0]
            assert not serie.offset
            assert serie.axes == 'YX'
            assert_array_equal(image1, serie.asarray())
            serie = tif.series[1]
            assert serie.offset
            assert serie.axes == 'YXS'
            assert_array_equal(image2, serie.asarray())
            serie = tif.series[2]
            assert serie.offset
            assert serie.pages[0].is_contiguous
            assert serie.axes == 'TCZYX'
            result = serie.asarray(out='memmap')
            assert_array_equal(data1, result)
            assert tif.filehandle.path == result.filename
            del result
            serie = tif.series[3]
            assert serie.offset
            assert serie.axes == 'QQYX'
            assert_array_equal(data1[0], serie.asarray())
            serie = tif.series[4]
            assert not serie.offset
            assert serie.axes == 'YX'
            assert_array_equal(data1[0, 0, 0], serie.asarray())
            serie = tif.series[5]
            assert not serie.offset
            assert serie.axes == 'YX'
            assert_array_equal(image1, serie.asarray())
            assert__str__(tif)


###############################################################################

# Test ImageJ writing

@pytest.mark.skipif(SKIP_EXTENDED, reason='many tests')
@pytest.mark.parametrize('shape', [
    (219, 301, 1),
    (219, 301, 2),
    (219, 301, 3),
    (219, 301, 4),
    (219, 301, 5),
    (1, 219, 301),
    (2, 219, 301),
    (3, 219, 301),
    (4, 219, 301),
    (5, 219, 301),
    (4, 3, 219, 301),
    (4, 219, 301, 3),
    (3, 4, 219, 301),
    (1, 3, 1, 219, 301),
    (3, 1, 1, 219, 301),
    (1, 3, 4, 219, 301),
    (3, 1, 4, 219, 301),
    (3, 4, 1, 219, 301),
    (3, 4, 1, 219, 301, 3),
    (2, 3, 4, 219, 301),
    (4, 3, 2, 219, 301, 3)])
@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'int16', 'float32'])
@pytest.mark.parametrize('byteorder', ['>', '<'])
def test_write_imagej(byteorder, dtype, shape):
    """Test write ImageJ format."""
    # TODO: test compression and bigtiff ?
    if dtype != 'uint8' and shape[-1] in (3, 4):
        pytest.skip('ImageJ only supports uint8 RGB')
    data = random_data(dtype, shape)
    fname = 'imagej_%s_%s_%s' % (
        {'<': 'le', '>': 'be'}[byteorder], dtype, str(shape).replace(' ', ''))
    with TempFileName(fname) as fname:
        imwrite(fname, data, byteorder=byteorder, imagej=True)
        image = imread(fname)
        assert_array_equal(data.squeeze(), image.squeeze())
        assert_jhove(fname)


def test_write_imagej_voxel_size():
    """Test write ImageJ with xyz voxel size 2.6755x2.6755x3.9474 µm^3."""
    data = numpy.zeros((4, 256, 256), dtype='float32')
    data.shape = 4, 1, 256, 256
    with TempFileName('imagej_voxel_size') as fname:
        imwrite(fname, data, imagej=True,
                resolution=(0.373759, 0.373759),
                metadata={'spacing': 3.947368, 'unit': 'um'})
        with TiffFile(fname) as tif:
            assert tif.is_imagej
            assert 'unit' in tif.imagej_metadata
            assert tif.imagej_metadata['unit'] == 'um'
            series = tif.series[0]
            assert series.axes == 'ZYX'
            assert series.shape == (4, 256, 256)
            assert__str__(tif)
        assert_jhove(fname)


def test_write_imagej_metadata():
    """Test write additional ImageJ metadata."""
    data = numpy.empty((4, 256, 256), dtype='uint16')
    data[:] = numpy.arange(256*256, dtype='uint16').reshape(1, 256, 256)
    with TempFileName('imagej_metadata') as fname:
        imwrite(fname, data, imagej=True, metadata={'unit': 'um'})
        with TiffFile(fname) as tif:
            assert tif.is_imagej
            assert 'unit' in tif.imagej_metadata
            assert tif.imagej_metadata['unit'] == 'um'
            assert__str__(tif)
        assert_jhove(fname)


def test_write_imagej_ijmetadata_tag():
    """Test write and read IJMetadata tag."""
    fname = data_file('imagej/IJMetadata.tif')
    with TiffFile(fname) as tif:
        assert tif.is_imagej
        assert tif.byteorder == '>'
        assert len(tif.pages) == 3
        assert len(tif.series) == 1
        data = tif.asarray()
        ijmetadata = tif.pages[0].tags['IJMetadata'].value

    assert ijmetadata['Info'][:21] == 'FluorescentCells.tif\n'
    assert ijmetadata['ROI'][:5] == b'Iout\x00'
    assert ijmetadata['Overlays'][1][:5] == b'Iout\x00'
    assert ijmetadata['Ranges'] == (0., 255., 0., 255., 0., 255.)
    assert ijmetadata['Labels'] == ['Red', 'Green', 'Blue']
    assert ijmetadata['LUTs'][2][2, 255] == 255
    assert_jhove(fname)

    with TempFileName('imagej_ijmetadata') as fname:
        imwrite(fname, data, byteorder='>', imagej=True,
                metadata={'mode': 'composite'}, ijmetadata=ijmetadata)
        with TiffFile(fname) as tif:
            assert tif.is_imagej
            assert tif.byteorder == '>'
            assert len(tif.pages) == 3
            assert len(tif.series) == 1
            data2 = tif.asarray()
            ijmetadata2 = tif.pages[0].tags['IJMetadata'].value
            assert__str__(tif)

    assert_array_equal(data, data2)
    assert ijmetadata2['Info'] == ijmetadata['Info']
    assert ijmetadata2['ROI'] == ijmetadata['ROI']
    assert ijmetadata2['Overlays'] == ijmetadata['Overlays']
    assert ijmetadata2['Ranges'] == ijmetadata['Ranges']
    assert ijmetadata2['Labels'] == ijmetadata['Labels']
    assert_array_equal(ijmetadata2['LUTs'][2], ijmetadata['LUTs'][2])
    assert_jhove(fname)


def test_write_imagej_hyperstack():
    """Test write truncated ImageJ hyperstack."""
    shape = (5, 6, 7, 49, 61, 3)
    data = numpy.empty(shape, dtype='uint8')
    data[:] = numpy.arange(210, dtype='uint8').reshape(5, 6, 7, 1, 1, 1)

    with TempFileName('imagej_hyperstack') as fname:
        imwrite(fname, data, imagej=True, truncate=True)
        # assert file
        with TiffFile(fname) as tif:
            assert not tif.is_bigtiff
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 61
            assert page.imagelength == 49
            assert page.samplesperpixel == 3
            # assert series properties
            series = tif.series[0]
            assert series.shape == shape
            assert len(series._pages) == 1
            assert len(series.pages) == 1
            assert series.dtype.name == 'uint8'
            assert series.axes == 'TZCYXS'
            # assert data
            image = tif.asarray(out='memmap')
            assert_array_equal(data.squeeze(), image.squeeze())
            del image
            assert__str__(tif)
        assert_jhove(fname)


def test_write_imagej_hyperstack_nontrunc():
    """Test write non-truncated ImageJ hyperstack."""
    shape = (5, 6, 7, 49, 61, 3)
    data = numpy.empty(shape, dtype='uint8')
    data[:] = numpy.arange(210, dtype='uint8').reshape(5, 6, 7, 1, 1, 1)

    with TempFileName('imagej_hyperstack_nontrunc') as fname:
        imwrite(fname, data, imagej=True)
        # assert file
        with TiffFile(fname) as tif:
            assert not tif.is_bigtiff
            assert len(tif.pages) == 210
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric == RGB
            assert page.imagewidth == 61
            assert page.imagelength == 49
            assert page.samplesperpixel == 3
            # assert series properties
            series = tif.series[0]
            assert series.shape == shape
            assert len(series._pages) == 1
            assert len(series.pages) == 210
            assert series.dtype.name == 'uint8'
            assert series.axes == 'TZCYXS'
            # assert data
            image = tif.asarray(out='memmap')
            assert_array_equal(data.squeeze(), image.squeeze())
            del image
            # assert iterating over series.pages
            data = data.reshape(210, 49, 61, 3)
            for i, page in enumerate(series.pages):
                image = page.asarray()
                assert_array_equal(data[i], image)
            assert__str__(tif)
        assert_jhove(fname)


def test_write_imagej_append():
    """Test write ImageJ file consecutively."""
    data = numpy.empty((256, 1, 256, 256), dtype='uint8')
    data[:] = numpy.arange(256, dtype='uint8').reshape(-1, 1, 1, 1)

    with TempFileName('imagej_append') as fname:
        with TiffWriter(fname, imagej=True) as tif:
            for image in data:
                tif.save(image)

        assert_jhove(fname)

        # assert file
        with TiffFile(fname) as tif:
            assert not tif.is_bigtiff
            assert len(tif.pages) == 256
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 256
            assert page.imagelength == 256
            assert page.samplesperpixel == 1
            # assert series properties
            series = tif.series[0]
            assert series.shape == (256, 256, 256)
            assert series.dtype.name == 'uint8'
            assert series.axes == 'ZYX'
            # assert data
            image = tif.asarray(out='memmap')
            assert_array_equal(data.squeeze(), image)
            del image
            assert__str__(tif)


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_HUGE, reason='5 GB image')
def test_write_imagej_raw():
    """Test write ImageJ 5 GB raw file."""
    data = numpy.empty((1280, 1, 1024, 1024), dtype='float32')
    data[:] = numpy.arange(1280, dtype='float32').reshape(-1, 1, 1, 1)

    with TempFileName('imagej_big') as fname:
        with pytest.warns(UserWarning):
            # UserWarning: truncating ImageJ file
            imwrite(fname, data, imagej=True)
        assert_jhove(fname)
        # assert file
        with TiffFile(fname) as tif:
            assert not tif.is_bigtiff
            assert len(tif.pages) == 1
            page = tif.pages[0]
            assert page.is_contiguous
            assert page.planarconfig == CONTIG
            assert page.photometric != RGB
            assert page.imagewidth == 1024
            assert page.imagelength == 1024
            assert page.samplesperpixel == 1
            # assert series properties
            series = tif.series[0]
            assert len(series._pages) == 1
            assert len(series.pages) == 1
            assert series.shape == (1280, 1024, 1024)
            assert series.dtype.name == 'float32'
            assert series.axes == 'ZYX'
            # assert data
            image = tif.asarray(out='memmap')
            assert_array_equal(data.squeeze(), image.squeeze())
            del image
            assert__str__(tif)


###############################################################################

# Test embedded TIFF files

EMBED_NAME = data_file('test_FileHandle.bin')
EMBED_OFFSET = 7077
EMBED_SIZE = 5744
EMBED_OFFSET1 = 13820
EMBED_SIZE1 = 7936382


def assert_embed_tif(tif):
    """Assert embedded TIFF file."""
    # 4 series in 6 pages
    assert tif.byteorder == '<'
    assert len(tif.pages) == 6
    assert len(tif.series) == 4
    # assert series 0 properties
    series = tif.series[0]
    assert series.shape == (3, 20, 20)
    assert series.dtype.name == 'uint8'
    assert series.axes == 'IYX'
    page = series.pages[0]
    assert page.compression == LZW
    assert page.imagewidth == 20
    assert page.imagelength == 20
    assert page.bitspersample == 8
    assert page.samplesperpixel == 1
    data = tif.asarray(series=0)
    assert isinstance(data, numpy.ndarray)
    assert data.shape == (3, 20, 20)
    assert data.dtype.name == 'uint8'
    assert tuple(data[:, 9, 9]) == (19, 90, 206)
    # assert series 1 properties
    series = tif.series[1]
    assert series.shape == (10, 10, 3)
    assert series.dtype.name == 'float32'
    assert series.axes == 'YXS'
    page = series.pages[0]
    assert page.photometric == RGB
    assert page.compression == LZW
    assert page.imagewidth == 10
    assert page.imagelength == 10
    assert page.bitspersample == 32
    assert page.samplesperpixel == 3
    data = tif.asarray(series=1)
    assert isinstance(data, numpy.ndarray)
    assert data.shape == (10, 10, 3)
    assert data.dtype.name == 'float32'
    assert round(abs(data[9, 9, 1]-214.5733642578125), 7) == 0
    # assert series 2 properties
    series = tif.series[2]
    assert series.shape == (20, 20, 3)
    assert series.dtype.name == 'uint8'
    assert series.axes == 'YXS'
    page = series.pages[0]
    assert page.photometric == RGB
    assert page.compression == LZW
    assert page.imagewidth == 20
    assert page.imagelength == 20
    assert page.bitspersample == 8
    assert page.samplesperpixel == 3
    data = tif.asarray(series=2)
    assert isinstance(data, numpy.ndarray)
    assert data.shape == (20, 20, 3)
    assert data.dtype.name == 'uint8'
    assert tuple(data[9, 9, :]) == (19, 90, 206)
    # assert series 3 properties
    series = tif.series[3]
    assert series.shape == (10, 10)
    assert series.dtype.name == 'float32'
    assert series.axes == 'YX'
    page = series.pages[0]
    assert page.compression == LZW
    assert page.imagewidth == 10
    assert page.imagelength == 10
    assert page.bitspersample == 32
    assert page.samplesperpixel == 1
    data = tif.asarray(series=3)
    assert isinstance(data, numpy.ndarray)
    assert data.shape == (10, 10)
    assert data.dtype.name == 'float32'
    assert round(abs(data[9, 9]-223.1648712158203), 7) == 0
    assert__str__(tif)


def assert_embed_micromanager(tif):
    """Assert embedded MicroManager TIFF file."""
    assert tif.is_ome
    assert tif.is_imagej
    assert tif.is_micromanager
    assert tif.byteorder == '<'
    assert len(tif.pages) == 15
    assert len(tif.series) == 1
    # assert non-tiff micromanager_metadata
    tags = tif.micromanager_metadata['Summary']
    assert tags["MicroManagerVersion"] == "1.4.x dev"
    # assert page properties
    page = tif.pages[0]
    assert page.is_contiguous
    assert page.compression == NONE
    assert page.imagewidth == 512
    assert page.imagelength == 512
    assert page.bitspersample == 16
    assert page.samplesperpixel == 1
    # two description tags
    assert page.description.startswith('<?xml')
    assert page.description1.startswith('ImageJ')
    # assert some metadata
    tags = tif.imagej_metadata
    assert tags['frames'] == 5
    assert tags['slices'] == 3
    assert tags['Ranges'] == (706.0, 5846.0)
    tags = tif.micromanager_metadata
    assert tags["FileName"] == "Untitled_1_MMStack.ome.tif"
    assert tags["PixelType"] == "GRAY16"
    # assert series properties
    series = tif.series[0]
    assert series.shape == (5, 3, 512, 512)
    assert series.dtype.name == 'uint16'
    assert series.axes == 'TZYX'
    # assert data
    data = tif.asarray()
    assert data.shape == (5, 3, 512, 512)
    assert data.dtype.name == 'uint16'
    assert data[4, 2, 511, 511] == 1602
    # assert memmap
    data = tif.asarray(out='memmap')
    assert isinstance(data, numpy.core.memmap)
    assert data.shape == (5, 3, 512, 512)
    assert data.dtype.name == 'uint16'
    assert data[4, 2, 511, 511] == 1602
    del data
    assert__str__(tif)


def test_embed_tif_filename():
    """Test embedded TIFF from filename."""
    with TiffFile(EMBED_NAME, offset=EMBED_OFFSET, size=EMBED_SIZE) as tif:
        assert_embed_tif(tif)


def test_embed_tif_openfile():
    """Test embedded TIFF from file handle."""
    with open(EMBED_NAME, 'rb') as fh:
        with TiffFile(fh, offset=EMBED_OFFSET, size=EMBED_SIZE) as tif:
            assert_embed_tif(tif)


def test_embed_tif_openfile_seek():
    """Test embedded TIFF from seeked file handle."""
    with open(EMBED_NAME, 'rb') as fh:
        fh.seek(EMBED_OFFSET)
        with TiffFile(fh, size=EMBED_SIZE) as tif:
            assert_embed_tif(tif)


def test_embed_tif_filehandle():
    """Test embedded TIFF from FileHandle."""
    with FileHandle(EMBED_NAME, offset=EMBED_OFFSET,
                    size=EMBED_SIZE) as fh:
        with TiffFile(fh) as tif:
            assert_embed_tif(tif)


def test_embed_tif_bytesio():
    """Test embedded TIFF from BytesIO."""
    with open(EMBED_NAME, 'rb') as fh:
        fh2 = BytesIO(fh.read())
    with TiffFile(fh2, offset=EMBED_OFFSET, size=EMBED_SIZE) as tif:
        assert_embed_tif(tif)


def test_embed_mm_filename():
    """Test embedded MicroManager TIFF from file name."""
    with TiffFile(EMBED_NAME, offset=EMBED_OFFSET1,
                  size=EMBED_SIZE1) as tif:
        assert_embed_micromanager(tif)


def test_embed_mm_openfile():
    """Test embedded MicroManager TIFF from file handle."""
    with open(EMBED_NAME, 'rb') as fh:
        with TiffFile(fh, offset=EMBED_OFFSET1, size=EMBED_SIZE1) as tif:
            assert_embed_micromanager(tif)


def test_embed_mm_openfile_seek():
    """Test embedded MicroManager TIFF from seeked file handle."""
    with open(EMBED_NAME, 'rb') as fh:
        fh.seek(EMBED_OFFSET1)
        with TiffFile(fh, size=EMBED_SIZE1) as tif:
            assert_embed_micromanager(tif)


def test_embed_mm_filehandle():
    """Test embedded MicroManager TIFF from FileHandle."""
    with FileHandle(EMBED_NAME, offset=EMBED_OFFSET1, size=EMBED_SIZE1) as fh:
        with TiffFile(fh) as tif:
            assert_embed_micromanager(tif)


def test_embed_mm_bytesio():
    """Test embedded MicroManager TIFF from BytesIO."""
    with open(EMBED_NAME, 'rb') as fh:
        fh2 = BytesIO(fh.read())
    with TiffFile(fh2, offset=EMBED_OFFSET1, size=EMBED_SIZE1) as tif:
        assert_embed_micromanager(tif)


###############################################################################

# Test sequence of image files

def test_sequence_stream_list():
    """Test TiffSequence with list of ByteIO streams raises ValueError."""
    data = numpy.random.rand(7, 9)
    files = [BytesIO(), BytesIO()]
    for buffer in files:
        imwrite(buffer, data)
    with pytest.raises(ValueError):
        imread(files)


def test_sequence_glob_pattern():
    """Test TiffSequence with glob pattern without axes pattern."""
    fname = data_file('TiffSequence/*.tif')
    tifs = TiffSequence(fname, pattern=None)
    assert len(tifs) == 10
    assert tifs.shape == (10,)
    assert tifs.axes == 'I'
    data = tifs.asarray()
    assert isinstance(data, numpy.ndarray)
    assert data.flags['C_CONTIGUOUS']
    assert data.shape == (10, 480, 640)
    assert data.dtype == 'uint8'
    assert data[9, 256, 256] == 135


def test_sequence_name_list():
    """Test TiffSequence with list of file names without pattern."""
    fname = [data_file('TiffSequence/AT3_1m4_01.tif'),
             data_file('TiffSequence/AT3_1m4_10.tif')]
    tifs = TiffSequence(fname, pattern=None)
    assert len(tifs) == 2
    assert tifs.shape == (2,)
    assert tifs.axes == 'I'
    data = tifs.asarray()
    assert isinstance(data, numpy.ndarray)
    assert data.flags['C_CONTIGUOUS']
    assert data.shape == (2, 480, 640)
    assert data.dtype == 'uint8'
    assert data[1, 256, 256] == 135


def test_sequence_oif_series():
    """Test TiffSequence with files from Olympus OIF, memory mapped."""
    fname = data_file('oif/MB231cell1_paxgfp_PDMSgasket_'
                      'PMMAflat_30nm_378sli.oif.files/*.tif')
    tifs = TiffSequence(fname, pattern='axes')
    assert len(tifs) == 756
    assert tifs.shape == (2, 378)
    assert tifs.axes == 'CZ'
    # memory map
    data = tifs.asarray(out='memmap')
    assert isinstance(data, numpy.core.memmap)
    assert data.flags['C_CONTIGUOUS']
    assert data.shape == (2, 378, 256, 256)
    assert data.dtype == 'uint16'
    assert data[1, 377, 70, 146] == 29
    del data


def test_sequence_leica_series():
    """Test TiffSequence with Leica TIFF series, memory mapped."""
    fname = data_file('leicaseries/Series019_*.tif')
    tifs = TiffSequence(fname, pattern='axes')
    assert len(tifs) == 46
    assert tifs.shape == (46,)
    assert tifs.axes == 'Y'
    # memory map
    data = tifs.asarray(out='memmap')
    assert isinstance(data, numpy.core.memmap)
    assert data.flags['C_CONTIGUOUS']
    assert data.shape == (46, 512, 512, 3)
    assert data.dtype == 'uint8'
    assert tuple(data[45, 256, 256]) == (93, 56, 56)
    del data


def test_sequence_zip_container():
    """Test TiffSequence with glob pattern without axes pattern."""
    fname = data_file('TiffSequence.zip')
    tifs = TiffSequence('*.tif', container=fname, pattern=None)
    assert len(tifs) == 10
    assert tifs.shape == (10,)
    assert tifs.axes == 'I'
    data = tifs.asarray()
    assert isinstance(data, numpy.ndarray)
    assert data.flags['C_CONTIGUOUS']
    assert data.shape == (10, 480, 640)
    assert data.dtype == 'uint8'
    assert data[9, 256, 256] == 135


###############################################################################

# Test packages depending on tifffile

def test_depend_cmapfile():
    """Test cmapfile.lsm2cmap."""
    from cmapfile import CmapFile, lsm2cmap  # noqa
    filename = data_file('LSM/3d_zfish_onephoton_zoom.lsm')
    with TempFileName('cmapfile', ext='.cmap') as cmapfile:
        lsm2cmap(filename, cmapfile)


def test_depend_czifile():
    """Test czifile.CziFile."""
    # TODO: test LZW compressed czi file
    from czifile import CziFile
    fname = data_file('czi/pollen.czi')
    with CziFile(fname) as czi:
        assert czi.shape == (1, 1, 104, 365, 364, 1)
        assert czi.axes == 'TCZYX0'
        # verify data
        data = czi.asarray()
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (1, 1, 104, 365, 364, 1)
        assert data.dtype == 'uint8'
        assert data[0, 0, 52, 182, 182, 0] == 10


@pytest.mark.skipif(IS_32BIT, reason='requires 64-bit')
@pytest.mark.skipif(SKIP_HUGE, reason='huge image')
def test_depend_czi2tif():
    """Test czifile.czi2tif."""
    from czifile.czifile import czi2tif
    fname = data_file('CZI/AiryscanSRChannel.czi')
    with TempFileName('czi2tif') as tif:
        czi2tif(fname, tif, verbose=True, truncate=True, bigtiff=False)
        im = memmap(tif)
        assert im.shape == (32, 6, 1680, 1680)
        assert tuple(im[17, :, 1500, 1000]) == (95, 109, 3597, 0, 0, 0)
        del im
        assert_jhove(tif)


def test_depend_oiffile():
    """Test oiffile.OifFile."""
    from oiffile import OifFile
    fname = data_file(
        'oib/MB231cell1_paxgfp_PDMSgasket_PMMAflat_30nm_378sli.oib')
    with OifFile(fname) as oib:
        assert oib.is_oib
        tifs = oib.tiffs
        assert len(tifs) == 756
        assert tifs.shape == (2, 378)
        assert tifs.axes == 'CZ'
        # verify data
        data = tifs.asarray(out='memmap')
        assert data.flags['C_CONTIGUOUS']
        assert data.shape == (2, 378, 256, 256)
        assert data.dtype == 'uint16'
        assert data[1, 377, 70, 146] == 29
        del data


###############################################################################

if __name__ == '__main__':
    import warnings
    # warnings.simplefilter('always')  # noqa
    warnings.filterwarnings('ignore', category=ImportWarning)  # noqa
    argv = sys.argv
    argv.append('--verbose')
    pytest.main(argv)
