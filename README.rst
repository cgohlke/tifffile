Read and write TIFF(r) files
============================

Tifffile is a Python library to

(1) store numpy arrays in TIFF (Tagged Image File Format) files, and
(2) read image and metadata from TIFF-like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, SGI,
NIHImage, ImageJ, MicroManager, FluoView, ScanImage, SEQ, GEL, SVS, SCN, SIS,
ZIF (Zoomable Image File Format), QPTIFF (QPI), NDPI, and GeoTIFF files.

Image data can be read as numpy arrays or zarr arrays/groups from strips,
tiles, pages (IFDs), SubIFDs, higher order series, and pyramidal levels.

Numpy arrays can be written to TIFF, BigTIFF, OME-TIFF, and ImageJ hyperstack
compatible files in multi-page, volumetric, pyramidal, memory-mappable, tiled,
predicted, or compressed form.

A subset of the TIFF specification is supported, mainly 8, 16, 32 and 64-bit
integer, 16, 32 and 64-bit float, grayscale and multi-sample images.
Specifically, CCITT and OJPEG compression, chroma subsampling without JPEG
compression, color space transformations, samples with differing types, or
IPTC and XMP metadata are not implemented.

TIFF(r), the Tagged Image File Format, is a trademark and under control of
Adobe Systems Incorporated. BigTIFF allows for files larger than 4 GB.
STK, LSM, FluoView, SGI, SEQ, GEL, QPTIFF, NDPI, SCN, and OME-TIFF, are custom
extensions defined by Molecular Devices (Universal Imaging Corporation),
Carl Zeiss MicroImaging, Olympus, Silicon Graphics International,
Media Cybernetics, Molecular Dynamics, PerkinElmer, Hamamatsu, Leica, and the
Open Microscopy Environment consortium, respectively.

For command line usage run ``python -m tifffile --help``

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2021.1.14

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.7.9, 3.8.7, 3.9.1 64-bit <https://www.python.org>`_
* `Numpy 1.19.5 <https://pypi.org/project/numpy/>`_
* `Imagecodecs 2021.1.11 <https://pypi.org/project/imagecodecs/>`_
  (required only for encoding or decoding LZW, JPEG, etc.)
* `Matplotlib 3.3.3 <https://pypi.org/project/matplotlib/>`_
  (required only for plotting)
* `Lxml 4.6.2 <https://pypi.org/project/lxml/>`_
  (required only for validating and printing XML)
* `Zarr 2.6.1 <https://pypi.org/project/zarr/>`_
  (required only for opening zarr storage)

Revisions
---------
2021.1.14
    Pass 4378 tests.
    Try ImageJ series if OME series fails (#54)
    Add option to use pages as chunks in ZarrFileStore (experimental).
    Fix reading from file objects with no readinto function.
2021.1.11
    Fix test errors on PyPy.
    Fix decoding bitorder with imagecodecs >= 2021.1.11.
2021.1.8
    Decode float24 using imagecodecs >= 2021.1.8.
    Consolidate reading of segments if possible.
2020.12.8
    Fix corrupted ImageDescription in multi shaped series if buffer too small.
    Fix libtiff warning that ImageDescription contains null byte in value.
    Fix reading invalid files using JPEG compression with palette colorspace.
2020.12.4
    Fix reading some JPEG compressed CFA images.
    Make index of SubIFDs a tuple.
    Pass through FileSequence.imread arguments in imread.
    Do not apply regex flags to FileSequence axes patterns (breaking).
2020.11.26
    Add option to pass axes metadata to ImageJ writer.
    Pad incomplete tiles passed to TiffWriter.write (#38).
    Split TiffTag constructor (breaking).
    Change TiffTag.dtype to TIFF.DATATYPES (breaking).
    Add TiffTag.overwrite method.
    Add script to change ImageDescription in files.
    Add TiffWriter.overwrite_description method (WIP).
2020.11.18
    Support writing SEPARATED color space (#37).
    Use imagecodecs.deflate codec if available.
    Fix SCN and NDPI series with Z dimensions.
    Add TiffReader alias for TiffFile.
    TiffPage.is_volumetric returns True if ImageDepth > 1.
    Zarr store getitem returns numpy arrays instead of bytes.
2020.10.1
    Formally deprecate unused TiffFile parameters (scikit-image #4996).
2020.9.30
    Allow to pass additional arguments to compression codecs.
    Deprecate TiffWriter.save method (use TiffWriter.write).
    Deprecate TiffWriter.save compress parameter (use compression).
    Remove multifile parameter from TiffFile (breaking).
    Pass all is_flag arguments from imread to TiffFile.
    Do not byte-swap JPEG2000, WEBP, PNG, JPEGXR segments in TiffPage.decode.
2020.9.29
    Fix reading files produced by ScanImage > 2015 (#29).
2020.9.28
    Derive ZarrStore from MutableMapping.
    Support zero shape ZarrTiffStore.
    Fix ZarrFileStore with non-TIFF files.
    Fix ZarrFileStore with missing files.
    Cache one chunk in ZarrFileStore.
    Keep track of already opened files in FileCache.
    Change parse_filenames function to return zero-based indices.
    Remove reopen parameter from asarray (breaking).
    Rename FileSequence.fromfile to imread (breaking).
2020.9.22
    Add experimental zarr storage interface (WIP).
    Remove unused first dimension from TiffPage.shaped (breaking).
    Move reading of STK planes to series interface (breaking).
    Always use virtual frames for ScanImage files.
    Use DimensionOrder to determine axes order in OmeXml.
    Enable writing striped volumetric images.
    Keep complete dataoffsets and databytecounts for TiffFrames.
    Return full size tiles from Tiffpage.segments.
    Rename TiffPage.is_sgi property to is_volumetric (breaking).
    Rename TiffPageSeries.is_pyramid to is_pyramidal (breaking).
    Fix TypeError when passing jpegtables to non-JPEG decode method (#25).
2020.9.3
    Do not write contiguous series by default (breaking).
    Allow to write to SubIFDs (WIP).
    Fix writing F-contiguous numpy arrays (#24).
2020.8.25
    Do not convert EPICS timeStamp to datetime object.
    Read incompletely written Micro-Manager image file stack header (#23).
    Remove tag 51123 values from TiffFile.micromanager_metadata (breaking).
2020.8.13
    Use tifffile metadata over OME and ImageJ for TiffFile.series (breaking).
    Fix writing iterable of pages with compression (#20).
    Expand error checking of TiffWriter data, dtype, shape, and tile arguments.
2020.7.24
    Parse nested OmeXml metadata argument (WIP).
    Do not lazy load TiffFrame JPEGTables.
    Fix conditionally skipping some tests.
2020.7.22
    Do not auto-enable OME-TIFF if description is passed to TiffWriter.save.
    Raise error writing empty bilevel or tiled images.
    Allow to write tiled bilevel images.
    Allow to write multi-page TIFF from iterable of single page images (WIP).
    Add function to validate OME-XML.
    Correct Philips slide width and length.
2020.7.17
    Initial support for writing OME-TIFF (WIP).
    Return samples as separate dimension in OME series (breaking).
    Fix modulo dimensions for multiple OME series.
    Fix some test errors on big endian systems (#18).
    Fix BytesWarning.
    Allow to pass TIFF.PREDICTOR values to TiffWriter.save.
2020.7.4
    Deprecate support for Python 3.6 (NEP 29).
    Move pyramidal subresolution series to TiffPageSeries.levels (breaking).
    Add parser for SVS, SCN, NDPI, and QPI pyramidal series.
    Read single-file OME-TIFF pyramids.
    Read NDPI files > 4 GB (#15).
    Include SubIFDs in generic series.
    Preliminary support for writing packed integer arrays (#11, WIP).
    Read more LSM info subrecords.
    Fix missing ReferenceBlackWhite tag for YCbCr photometrics.
    Fix reading lossless JPEG compressed DNG files.
2020.6.3
    Support os.PathLike file names (#9).
2020.5.30
    Re-add pure Python PackBits decoder.
2020.5.25
    Make imagecodecs an optional dependency again.
    Disable multi-threaded decoding of small LZW compressed segments.
    Fix caching of TiffPage.decode method.
    Fix xml.etree.cElementTree ImportError on Python 3.9.
    Fix tostring DeprecationWarning.
2020.5.11
    Fix reading ImageJ grayscale mode RGB images (#6).
    Remove napari reader plugin.
2020.5.7
    Add napari reader plugin (tentative).
    Fix writing single tiles larger than image data (#3).
    Always store ExtraSamples values in tuple (breaking).
2020.5.5
    Allow to write tiled TIFF from iterable of tiles (WIP).
    Add method to iterate over decoded segments of TiffPage (WIP).
    Pass chunks of segments to ThreadPoolExecutor.map to reduce memory usage.
    Fix reading invalid files with too many strips.
    Fix writing over-aligned image data.
    Detect OME-XML without declaration (#2).
    Support LERC compression (WIP).
    Delay load imagecodecs functions.
    Remove maxsize parameter from asarray (breaking).
    Deprecate ijmetadata parameter from TiffWriter.save (use metadata).
2020.2.16
    Add method to decode individual strips or tiles.
    Read strips and tiles in order of their offsets.
    Enable multi-threading when decompressing multiple strips.
    Replace TiffPage.tags dictionary with TiffTags (breaking).
    Replace TIFF.TAGS dictionary with TiffTagRegistry.
    Remove TIFF.TAG_NAMES (breaking).
    Improve handling of TiffSequence parameters in imread.
    Match last uncommon parts of file paths to FileSequence pattern (breaking).
    Allow letters in FileSequence pattern for indexing well plate rows.
    Allow to reorder axes in FileSequence.
    Allow to write > 4 GB arrays to plain TIFF when using compression.
    Allow to write zero size numpy arrays to nonconformant TIFF (tentative).
    Fix xml2dict.
    Require imagecodecs >= 2020.1.31.
    Remove support for imagecodecs-lite (breaking).
    Remove verify parameter to asarray method (breaking).
    Remove deprecated lzw_decode functions (breaking).
    Remove support for Python 2.7 and 3.5 (breaking).
2019.7.26
    ...

Refer to the CHANGES file for older revisions.

Notes
-----
The API is not stable yet and might change between revisions.

Tested on little-endian platforms only.

Python 32-bit versions are deprecated. Python <= 3.7 are no longer supported.

Tifffile relies on the `imagecodecs <https://pypi.org/project/imagecodecs/>`_
package for encoding and decoding LZW, JPEG, and other compressed image
segments.

Several TIFF-like formats do not strictly adhere to the TIFF6 specification,
some of which allow file or data sizes to exceed the 4 GB limit:

* *BigTIFF* is identified by version number 43 and uses different file
  header, IFD, and tag structures with 64-bit offsets. It adds more data types.
  Tifffile can read and write BigTIFF files.
* *ImageJ* hyperstacks store all image data, which may exceed 4 GB,
  contiguously after the first IFD. Files > 4 GB contain one IFD only.
  The size (shape and dtype) of the up to 6-dimensional image data can be
  determined from the ImageDescription tag of the first IFD, which is Latin-1
  encoded. Tifffile can read and write ImageJ hyperstacks.
* *OME-TIFF* stores up to 8-dimensional data in one or multiple TIFF of BigTIFF
  files. The 8-bit UTF-8 encoded OME-XML metadata found in the ImageDescription
  tag of the first IFD defines the position of TIFF IFDs in the high
  dimensional data. Tifffile can read OME-TIFF files, except when the OME-XML
  metadata are stored in a separate file. Tifffile can write numpy arrays
  to single-file OME-TIFF.
* *LSM* stores all IFDs below 4 GB but wraps around 32-bit StripOffsets.
  The StripOffsets of each series and position require separate unwrapping.
  The StripByteCounts tag contains the number of bytes for the uncompressed
  data. Tifffile can read large LSM files.
* *STK* (MetaMorph Stack) contains additional image planes stored contiguously
  after the image data of the first page. The total number of planes
  is equal to the counts of the UIC2tag. Tifffile can read STK files.
* *NDPI* uses some 64-bit offsets in the file header, IFD, and tag structures.
  Tag values/offsets can be corrected using high bits stored after IFD
  structures. JPEG compressed segments with dimensions >65536 or missing
  restart markers are not readable with libjpeg. Tifffile can read NDPI
  files > 4 GB. JPEG segments with restart markers and dimensions >65536 can
  be decoded with the imagecodecs library on Windows.
* *Philips* TIFF slides store wrong ImageWidth and ImageLength tag values for
  tiled pages. The values can be corrected using the DICOM_PIXEL_SPACING
  attributes of the XML formatted description of the first page. Tifffile can
  read Philips slides.
* *ScanImage* optionally allows corrupt non-BigTIFF files > 2 GB. The values
  of StripOffsets and StripByteCounts can be recovered using the constant
  differences of the offsets of IFD and tag values throughout the file.
  Tifffile can read such files if the image data are stored contiguously in
  each page.
* *GeoTIFF* sparse files allow strip or tile offsets and byte counts to be 0.
  Such segments are implicitly set to 0 or the NODATA value on reading.
  Tifffile can read GeoTIFF sparse files.

Other libraries for reading scientific TIFF files from Python:

* `Python-bioformats <https://github.com/CellProfiler/python-bioformats>`_
* `Imread <https://github.com/luispedro/imread>`_
* `GDAL <https://github.com/OSGeo/gdal/tree/master/gdal/swig/python>`_
* `OpenSlide-python <https://github.com/openslide/openslide-python>`_
* `PyLibTiff <https://github.com/pearu/pylibtiff>`_
* `SimpleITK <https://github.com/SimpleITK/SimpleITK>`_
* `PyLSM <https://launchpad.net/pylsm>`_
* `PyMca.TiffIO.py <https://github.com/vasole/pymca>`_ (same as fabio.TiffIO)
* `BioImageXD.Readers <http://www.bioimagexd.net/>`_
* `CellCognition <https://cellcognition-project.org/>`_
* `pymimage <https://github.com/ardoi/pymimage>`_
* `pytiff <https://github.com/FZJ-INM1-BDA/pytiff>`_
* `ScanImageTiffReaderPython
  <https://gitlab.com/vidriotech/scanimagetiffreader-python>`_
* `bigtiff <https://pypi.org/project/bigtiff>`_
* `Large Image <https://github.com/girder/large_image>`_

Some libraries are using tifffile to write OME-TIFF files:

* `Zeiss Apeer OME-TIFF library
  <https://github.com/apeer-micro/apeer-ometiff-library>`_
* `Allen Institute for Cell Science imageio
  <https://pypi.org/project/aicsimageio>`_
* `xtiff <https://github.com/BodenmillerGroup/xtiff>`_

Other tools for inspecting and manipulating TIFF files:

* `tifftools <https://github.com/DigitalSlideArchive/tifftools>`_
* `Tyf <https://github.com/Moustikitos/tyf>`_

References
----------
* TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
  https://www.adobe.io/open/standards/TIFF.html
* TIFF File Format FAQ. https://www.awaresystems.be/imaging/tiff/faq.html
* The BigTIFF File Format.
  https://www.awaresystems.be/imaging/tiff/bigtiff.html
* MetaMorph Stack (STK) Image File Format.
  http://mdc.custhelp.com/app/answers/detail/a_id/18862
* Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
  Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
* The OME-TIFF format.
  https://docs.openmicroscopy.org/ome-model/latest/
* UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
  http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
* Micro-Manager File Formats.
  https://micro-manager.org/wiki/Micro-Manager_File_Formats
* ScanImage BigTiff Specification - ScanImage 2016.
  http://scanimage.vidriotechnologies.com/display/SI2016/
  ScanImage+BigTiff+Specification
* ZIF, the Zoomable Image File format. http://zif.photo/
* GeoTIFF File Format https://gdal.org/drivers/raster/gtiff.html
* Cloud optimized GeoTIFF.
  https://github.com/cogeotiff/cog-spec/blob/master/spec.md
* Tags for TIFF and Related Specifications. Digital Preservation.
  https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
* CIPA DC-008-2016: Exchangeable image file format for digital still cameras:
  Exif Version 2.31.
  http://www.cipa.jp/std/documents/e/DC-008-Translation-2016-E.pdf

Examples
--------
Write a numpy array to a single-page RGB TIFF file:

>>> data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb')

Read the image from the TIFF file as numpy array:

>>> image = imread('temp.tif')
>>> image.shape
(256, 256, 3)

Write a 3D numpy array to a multi-page, 16-bit grayscale TIFF file:

>>> data = numpy.random.randint(0, 2**12, (64, 301, 219), 'uint16')
>>> imwrite('temp.tif', data, photometric='minisblack')

Read the whole image stack from the TIFF file as numpy array:

>>> image_stack = imread('temp.tif')
>>> image_stack.shape
(64, 301, 219)
>>> image_stack.dtype
dtype('uint16')

Read the image from the first page in the TIFF file as numpy array:

>>> image = imread('temp.tif', key=0)
>>> image.shape
(301, 219)

Read images from a selected range of pages:

>>> image = imread('temp.tif', key=range(4, 40, 2))
>>> image.shape
(18, 301, 219)

Iterate over all pages in the TIFF file and successively read images:

>>> with TiffFile('temp.tif') as tif:
...     for page in tif.pages:
...         image = page.asarray()

Get information about the image stack in the TIFF file without reading
the image data:

>>> tif = TiffFile('temp.tif')
>>> len(tif.pages)  # number of pages in the file
64
>>> page = tif.pages[0]  # get shape and dtype of the image in the first page
>>> page.shape
(301, 219)
>>> page.dtype
dtype('uint16')
>>> page.axes
'YX'
>>> series = tif.series[0]  # get shape and dtype of the first image series
>>> series.shape
(64, 301, 219)
>>> series.dtype
dtype('uint16')
>>> series.axes
'QYX'
>>> tif.close()

Inspect the "XResolution" tag from the first page in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     tag = tif.pages[0].tags['XResolution']
>>> tag.value
(1, 1)
>>> tag.name
'XResolution'
>>> tag.code
282
>>> tag.count
1
>>> tag.dtype
<DATATYPES.RATIONAL: 5>

Iterate over all tags in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     for page in tif.pages:
...         for tag in page.tags:
...             tag_name, tag_value = tag.name, tag.value

Overwrite the value of an existing tag, e.g. XResolution:

>>> with TiffFile('temp.tif', mode='r+b') as tif:
...     _ = tif.pages[0].tags['XResolution'].overwrite(tif, (96000, 1000))

Write a floating-point ndarray and metadata using BigTIFF format, tiling,
compression, and planar storage:

>>> data = numpy.random.rand(2, 5, 3, 301, 219).astype('float32')
>>> imwrite('temp.tif', data, bigtiff=True, photometric='minisblack',
...         compression='deflate', planarconfig='separate', tile=(32, 32),
...         metadata={'axes': 'TZCYX'})

Write a volume with xyz voxel size 2.6755x2.6755x3.9474 micron^3 to an
ImageJ hyperstack formatted TIFF file:

>>> volume = numpy.random.randn(57, 256, 256).astype('float32')
>>> imwrite('temp.tif', volume, imagej=True, resolution=(1./2.6755, 1./2.6755),
...         metadata={'spacing': 3.947368, 'unit': 'um', 'axes': 'ZYX'})

Read the volume and metadata from the ImageJ file:

>>> with TiffFile('temp.tif') as tif:
...     volume = tif.asarray()
...     axes = tif.series[0].axes
...     imagej_metadata = tif.imagej_metadata
>>> volume.shape
(57, 256, 256)
>>> axes
'ZYX'
>>> imagej_metadata['slices']
57

Create an empty TIFF file and write to the memory-mapped numpy array:

>>> memmap_image = memmap('temp.tif', shape=(3, 256, 256), dtype='float32')
>>> memmap_image[1, 255, 255] = 1.0
>>> memmap_image.flush()
>>> del memmap_image

Memory-map image data of the first page in the TIFF file:

>>> memmap_image = memmap('temp.tif', page=0)
>>> memmap_image[1, 255, 255]
1.0
>>> del memmap_image

Write two numpy arrays to a multi-series TIFF file:

>>> series0 = numpy.random.randint(0, 255, (32, 32, 3), 'uint8')
>>> series1 = numpy.random.randint(0, 1023, (4, 256, 256), 'uint16')
>>> with TiffWriter('temp.tif') as tif:
...     tif.write(series0, photometric='rgb')
...     tif.write(series1, photometric='minisblack')

Read the second image series from the TIFF file:

>>> series1 = imread('temp.tif', series=1)
>>> series1.shape
(4, 256, 256)

Successively write the frames of one contiguous series to a TIFF file:

>>> data = numpy.random.randint(0, 255, (30, 301, 219), 'uint8')
>>> with TiffWriter('temp.tif') as tif:
...     for frame in data:
...         tif.write(frame, contiguous=True)

Append an image series to the existing TIFF file:

>>> data = numpy.random.randint(0, 255, (301, 219, 3), 'uint8')
>>> imwrite('temp.tif', data, append=True)

Create a TIFF file from a generator of tiles:

>>> data = numpy.random.randint(0, 2**12, (31, 33, 3), 'uint16')
>>> def tiles(data, tileshape):
...     for y in range(0, data.shape[0], tileshape[0]):
...         for x in range(0, data.shape[1], tileshape[1]):
...             yield data[y : y + tileshape[0], x : x + tileshape[1]]
>>> imwrite('temp.tif', tiles(data, (16, 16)), tile=(16, 16),
...         shape=data.shape, dtype=data.dtype)

Write two numpy arrays to a multi-series OME-TIFF file:

>>> series0 = numpy.random.randint(0, 255, (32, 32, 3), 'uint8')
>>> series1 = numpy.random.randint(0, 1023, (4, 256, 256), 'uint16')
>>> with TiffWriter('temp.ome.tif') as tif:
...     tif.write(series0, photometric='rgb')
...     tif.write(series1, photometric='minisblack',
...              metadata={'axes': 'ZYX', 'SignificantBits': 10,
...                        'Plane': {'PositionZ': [0.0, 1.0, 2.0, 3.0]}})

Write a tiled, multi-resolution, pyramidal, OME-TIFF file using
JPEG compression. Sub-resolution images are written to SubIFDs:

>>> data = numpy.arange(1024*1024*3, dtype='uint8').reshape((1024, 1024, 3))
>>> with TiffWriter('temp.ome.tif', bigtiff=True) as tif:
...     options = dict(tile=(256, 256), compression='jpeg')
...     tif.write(data, subifds=2, **options)
...     # save pyramid levels to the two subifds
...     # in production use resampling to generate sub-resolutions
...     tif.write(data[::2, ::2], subfiletype=1, **options)
...     tif.write(data[::4, ::4], subfiletype=1, **options)

Access the image levels in the pyramidal OME-TIFF file:

>>> baseimage = imread('temp.ome.tif')
>>> second_level = imread('temp.ome.tif', series=0, level=1)
>>> with TiffFile('temp.ome.tif') as tif:
...     baseimage = tif.series[0].asarray()
...     second_level = tif.series[0].levels[1].asarray()

Iterate over and decode single JPEG compressed tiles in the TIFF file:

>>> with TiffFile('temp.ome.tif') as tif:
...     fh = tif.filehandle
...     for page in tif.pages:
...         for index, (offset, bytecount) in enumerate(
...             zip(page.dataoffsets, page.databytecounts)
...         ):
...             fh.seek(offset)
...             data = fh.read(bytecount)
...             tile, indices, shape = page.decode(
...                 data, index, jpegtables=page.jpegtables
...             )

Use zarr to access the tiled, pyramidal images in the TIFF file:

>>> import zarr
>>> store = imread('temp.ome.tif', aszarr=True)
>>> z = zarr.open(store, mode='r')
>>> z
<zarr.hierarchy.Group '/' read-only>
>>> z[0]  # base layer
<zarr.core.Array '/0' (1024, 1024, 3) uint8 read-only>
>>> store.close()

Read images from a sequence of TIFF files as numpy array:

>>> imwrite('temp_C001T001.tif', numpy.random.rand(64, 64))
>>> imwrite('temp_C001T002.tif', numpy.random.rand(64, 64))
>>> image_sequence = imread(['temp_C001T001.tif', 'temp_C001T002.tif'])
>>> image_sequence.shape
(2, 64, 64)

Read an image stack from a series of TIFF files with a file name pattern
as numpy or zarr arrays:

>>> image_sequence = TiffSequence('temp_C001*.tif', pattern='axes')
>>> image_sequence.shape
(1, 2)
>>> image_sequence.axes
'CT'
>>> data = image_sequence.asarray()
>>> data.shape
(1, 2, 64, 64)
>>> with image_sequence.aszarr() as store:
...     zarr.open(store, mode='r')
<zarr.core.Array (1, 2, 64, 64) float64 read-only>
>>> image_sequence.close()
