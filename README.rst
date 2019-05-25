Read and write TIFF(r) files
============================

Tifffile is a Python library to

(1) store numpy arrays in TIFF (Tagged Image File Format) files, and
(2) read image and metadata from TIFF-like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, SGI,
NIHImage, ImageJ, MicroManager, FluoView, ScanImage, SEQ, GEL, SVS, SCN, SIS,
ZIF, QPI, NDPI, and GeoTIFF files.

Numpy arrays can be written to TIFF, BigTIFF, and ImageJ hyperstack compatible
files in multi-page, memory-mappable, tiled, predicted, or compressed form.

Only a subset of the TIFF specification is supported, mainly uncompressed and
losslessly compressed 1, 8, 16, 32 and 64-bit integer, 16, 32 and 64-bit float,
grayscale and RGB(A) images.
Specifically, reading slices of image data, CCITT and OJPEG compression,
chroma subsampling without JPEG compression, or IPTC and XMP metadata are not
implemented.

TIFF(r), the Tagged Image File Format, is a trademark and under control of
Adobe Systems Incorporated. BigTIFF allows for files greater than 4 GB.
STK, LSM, FluoView, SGI, SEQ, GEL, and OME-TIFF, are custom extensions
defined by Molecular Devices (Universal Imaging Corporation), Carl Zeiss
MicroImaging, Olympus, Silicon Graphics International, Media Cybernetics,
Molecular Dynamics, and the Open Microscopy Environment consortium
respectively.

For command line usage run ``python -m tifffile --help``

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: 3-clause BSD

:Version: 2019.5.22

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 2.7.16, 3.5.4, 3.6.8, 3.7.3, 64-bit <https://www.python.org>`_
* `Numpy 1.15.4 <https://www.numpy.org>`_
* `Imagecodecs 2019.5.22 <https://pypi.org/project/imagecodecs/>`_
  (optional; used for encoding and decoding LZW, JPEG, etc.)
* `Matplotlib 2.2 <https://www.matplotlib.org>`_ (optional; used for plotting)
* Python 2.7 requires 'futures', 'enum34', and 'pathlib'.

Revisions
---------
2019.5.22
    Pass 2783 tests.
    Add optional chroma subsampling for JPEG compression.
    Enable writing PNG, JPEG, JPEGXR, and JPEG2000 compression (tentative).
    Fix writing tiled images with WebP compression.
    Improve handling GeoTIFF sparse files.
2019.3.18
    Fix regression decoding JPEG with RGB photometrics.
    Fix reading OME-TIFF files with corrupted but unused pages.
    Allow to load TiffFrame without specifying keyframe.
    Calculate virtual TiffFrames for non-BigTIFF ScanImage files > 2GB.
    Rename property is_chroma_subsampled to is_subsampled.
    Make more attributes and methods private (WIP).
2019.3.8
    Fix MemoryError when RowsPerStrip > ImageLength.
    Fix SyntaxWarning on Python 3.8.
    Fail to decode JPEG to planar RGB (tentative).
    Separate public from private test files (WIP).
    Allow testing without data files or imagecodecs.
2019.2.22
    Use imagecodecs-lite as a fallback for imagecodecs.
    Simplify reading numpy arrays from file.
    Use TiffFrames when reading arrays from page sequences.
    Support slices and iterators in TiffPageSeries sequence interface.
    Auto-detect uniform series.
    Use page hash to determine generic series.
    Turn off page cache (tentative).
    Pass through more parameters in imread.
    Discontinue movie parameter in imread and TiffFile (breaking).
    Discontinue bigsize parameter in imwrite (breaking).
    Raise TiffFileError in case of issues with TIFF structure.
    Return TiffFile.ome_metadata as XML (breaking).
    Ignore OME series when last dimensions are not stored in TIFF pages.
2019.2.10
    Assemble IFDs in memory to speed-up writing on some slow media.
    Handle discontinued arguments fastij, multifile_close, and pages.
2019.1.30
    Use black background in imshow.
    Do not write datetime tag by default (breaking).
    Fix OME-TIFF with SamplesPerPixel > 1.
    Allow 64-bit IFD offsets for NDPI (files > 4GB still not supported).
2019.1.4
    Fix decoding deflate without imagecodecs.
2019.1.1
    Update copyright year.
    Require imagecodecs >= 2018.12.16.
    Do not use JPEG tables from keyframe.
    Enable decoding large JPEG in NDPI.
    Decode some old-style JPEG.
    Reorder OME channel axis to match PlanarConfiguration storage.
    Return tiled images as contiguous arrays.
    Add decode_lzw proxy function for compatibility with old czifile module.
    Use dedicated logger.
2018.11.28
    Make SubIFDs accessible as TiffPage.pages.
    Make parsing of TiffSequence axes pattern optional (breaking).
    Limit parsing of TiffSequence axes pattern to file names, not path names.
    Do not interpolate in imshow if image dimensions <= 512, else use bilinear.
    Use logging.warning instead of warnings.warn in many cases.
    Fix numpy FutureWarning for out == 'memmap'.
    Adjust ZSTD and WebP compression to libtiff-4.0.10 (WIP).
    Decode old-style LZW with imagecodecs >= 2018.11.8.
    Remove TiffFile.qptiff_metadata (QPI metadata are per page).
    Do not use keyword arguments before variable positional arguments.
    Make either all or none return statements in a function return expression.
    Use pytest parametrize to generate tests.
    Replace test classes with functions.
2018.11.6
    Rename imsave function to imwrite.
    Readd Python implementations of packints, delta, and bitorder codecs.
    Fix TiffFrame.compression AttributeError.
2018.10.18
    Rename tiffile package to tifffile.
2018.10.10
    Read ZIF, the Zoomable Image Format (WIP).
    Decode YCbCr JPEG as RGB (tentative).
    Improve restoration of incomplete tiles.
    Allow to write grayscale with extrasamples without specifying planarconfig.
    Enable decoding of PNG and JXR via imagecodecs.
    Deprecate 32-bit platforms (too many memory errors during tests).
2018.9.27
    Read Olympus SIS (WIP).
    Allow to write non-BigTIFF files up to ~4 GB (fix).
    Fix parsing date and time fields in SEM metadata.
    Detect some circular IFD references.
    Enable WebP codecs via imagecodecs.
    Add option to read TiffSequence from ZIP containers.
    Remove TiffFile.isnative.
    Move TIFF struct format constants out of TiffFile namespace.
2018.8.31
    Fix wrong TiffTag.valueoffset.
    Towards reading Hamamatsu NDPI (WIP).
    Enable PackBits compression of byte and bool arrays.
    Fix parsing NULL terminated CZ_SEM strings.
2018.8.24
    Move tifffile.py and related modules into tiffile package.
    Move usage examples to module docstring.
    Enable multi-threading for compressed tiles and pages by default.
    Add option to concurrently decode image tiles using threads.
    Do not skip empty tiles (fix).
    Read JPEG and J2K compressed strips and tiles.
    Allow floating-point predictor on write.
    Add option to specify subfiletype on write.
    Depend on imagecodecs package instead of _tifffile, lzma, etc modules.
    Remove reverse_bitorder, unpack_ints, and decode functions.
    Use pytest instead of unittest.
2018.6.20
    Save RGBA with unassociated extrasample by default (breaking).
    Add option to specify ExtraSamples values.
2018.6.17 (included with 0.15.1)
    Towards reading JPEG and other compressions via imagecodecs package (WIP).
    Read SampleFormat VOID as UINT.
    Add function to validate TIFF using 'jhove -m TIFF-hul'.
    Save bool arrays as bilevel TIFF.
    Accept pathlib.Path as filenames.
    Move 'software' argument from TiffWriter __init__ to save.
    Raise DOS limit to 16 TB.
    Lazy load LZMA and ZSTD compressors and decompressors.
    Add option to save IJMetadata tags.
    Return correct number of pages for truncated series (fix).
    Move EXIF tags to TIFF.TAG as per TIFF/EP standard.
2018.2.18
    Always save RowsPerStrip and Resolution tags as required by TIFF standard.
    Do not use badly typed ImageDescription.
    Coherce bad ASCII string tags to bytes.
    Tuning of __str__ functions.
    Fix reading 'undefined' tag values.
    Read and write ZSTD compressed data.
    Use hexdump to print byte strings.
    Determine TIFF byte order from data dtype in imsave.
    Add option to specify RowsPerStrip for compressed strips.
    Allow memory-map of arrays with non-native byte order.
    Attempt to handle ScanImage <= 5.1 files.
    Restore TiffPageSeries.pages sequence interface.
    Use numpy.frombuffer instead of fromstring to read from binary data.
    Parse GeoTIFF metadata.
    Add option to apply horizontal differencing before compression.
    Towards reading PerkinElmer QPI (QPTIFF, no test files).
    Do not index out of bounds data in tifffile.c unpackbits and decodelzw.
2017.9.29
    Many backward incompatible changes improving speed and resource usage:
    Add detail argument to __str__ function. Remove info functions.
    Fix potential issue correcting offsets of large LSM files with positions.
    Remove TiffFile sequence interface; use TiffFile.pages instead.
    Do not make tag values available as TiffPage attributes.
    Use str (not bytes) type for tag and metadata strings (WIP).
    Use documented standard tag and value names (WIP).
    Use enums for some documented TIFF tag values.
    Remove 'memmap' and 'tmpfile' options; use out='memmap' instead.
    Add option to specify output in asarray functions.
    Add option to concurrently decode pages using threads.
    Add TiffPage.asrgb function (WIP).
    Do not apply colormap in asarray.
    Remove 'colormapped', 'rgbonly', and 'scale_mdgel' options from asarray.
    Consolidate metadata in TiffFile _metadata functions.
    Remove non-tag metadata properties from TiffPage.
    Add function to convert LSM to tiled BIN files.
    Align image data in file.
    Make TiffPage.dtype a numpy.dtype.
    Add 'ndim' and 'size' properties to TiffPage and TiffPageSeries.
    Allow imsave to write non-BigTIFF files up to ~4 GB.
    Only read one page for shaped series if possible.
    Add memmap function to create memory-mapped array stored in TIFF file.
    Add option to save empty arrays to TIFF files.
    Add option to save truncated TIFF files.
    Allow single tile images to be saved contiguously.
    Add optional movie mode for files with uniform pages.
    Lazy load pages.
    Use lightweight TiffFrame for IFDs sharing properties with key TiffPage.
    Move module constants to 'TIFF' namespace (speed up module import).
    Remove 'fastij' option from TiffFile.
    Remove 'pages' parameter from TiffFile.
    Remove TIFFfile alias.
    Deprecate Python 2.
    Require enum34 and futures packages on Python 2.7.
    Remove Record class and return all metadata as dict instead.
    Add functions to parse STK, MetaSeries, ScanImage, SVS, Pilatus metadata.
    Read tags from EXIF and GPS IFDs.
    Use pformat for tag and metadata values.
    Fix reading some UIC tags.
    Do not modify input array in imshow (fix).
    Fix Python implementation of unpack_ints.
2017.5.23
    Write correct number of SampleFormat values (fix).
    Use Adobe deflate code to write ZIP compressed files.
    Add option to pass tag values as packed binary data for writing.
    Defer tag validation to attribute access.
    Use property instead of lazyattr decorator for simple expressions.
2017.3.17
    Write IFDs and tag values on word boundaries.
    Read ScanImage metadata.
    Remove is_rgb and is_indexed attributes from TiffFile.
    Create files used by doctests.
2017.1.12 (included with scikit-image 0.14.x)
    Read Zeiss SEM metadata.
    Read OME-TIFF with invalid references to external files.
    Rewrite C LZW decoder (5x faster).
    Read corrupted LSM files missing EOI code in LZW stream.
2017.1.1
    ...

Refer to the CHANGES file for older revisions.

Notes
-----
The API is not stable yet and might change between revisions.

Tested on little-endian platforms only.

Python 2.7 and 32-bit versions are deprecated.

Tifffile relies on the `imagecodecs <https://pypi.org/project/imagecodecs/>`_
package for encoding and decoding LZW, JPEG, and other compressed images.
The `imagecodecs-lite <https://pypi.org/project/imagecodecs-lite/>`_ package,
which easier to build, can be used for decoding LZW compressed images instead.

Several TIFF-like formats do not strictly adhere to the TIFF6 specification,
some of which allow file sizes to exceed the 4 GB limit:

* *BigTIFF* is identified by version number 43 and uses different file
  header, IFD, and tag structures with 64-bit offsets. It adds more data types.
  Tifffile can read and write BigTIFF files.
* *ImageJ* hyperstacks store all image data, which may exceed 4 GB,
  contiguously after the first IFD. Files > 4 GB contain one IFD only.
  The size (shape and dtype) of the image data can be determined from the
  ImageDescription of the first IFD. Tifffile can read and write ImageJ
  hyperstacks.
* *LSM* stores all IFDs below 4 GB but wraps around 32-bit StripOffsets.
  The StripOffsets of each series and position require separate unwrapping.
  The StripByteCounts tag contains the number of bytes for the uncompressed
  data. Tifffile can read large LSM files.
* *NDPI* uses some 64-bit offsets in the file header, IFD, and tag structures
  and might require correcting 32-bit offsets found in tags.
  JPEG compressed tiles with dimensions > 65536 are not readable with libjpeg.
  Tifffile can read NDPI files < 4 GB and decompress large JPEG tiles using
  the imagecodecs library on Windows.
* *ScanImage* optionally writes corrupt non-BigTIFF files > 2 GB. The values
  of StripOffsets and StripByteCounts can be recovered using the constant
  differences of the offsets of IFD and tag values throughout the file.
  Tifffile can read such files on Python 3 if the image data is stored
  contiguously in each page.
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

Some libraries are using tifffile to write OME-TIFF files:

* `Zeiss Apeer OME-TIFF library
  <https://github.com/apeer-micro/apeer-ometiff-library>`_
* `Allen Institute for Cell Science imageio
  <https://pypi.org/project/aicsimageio>`_

Acknowledgements
----------------
* Egor Zindy, University of Manchester, for lsm_scan_info specifics.
* Wim Lewis for a bug fix and some LSM functions.
* Hadrien Mary for help on reading MicroManager files.
* Christian Kliche for help writing tiled and color-mapped files.

References
----------
1)  TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
    https://www.adobe.io/open/standards/TIFF.html
2)  TIFF File Format FAQ. https://www.awaresystems.be/imaging/tiff/faq.html
3)  MetaMorph Stack (STK) Image File Format.
    http://mdc.custhelp.com/app/answers/detail/a_id/18862
4)  Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
    Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
5)  The OME-TIFF format.
    https://docs.openmicroscopy.org/ome-model/5.6.4/ome-tiff/
6)  UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
    http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
7)  Micro-Manager File Formats.
    https://micro-manager.org/wiki/Micro-Manager_File_Formats
8)  Tags for TIFF and Related Specifications. Digital Preservation.
    https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
9)  ScanImage BigTiff Specification - ScanImage 2016.
    http://scanimage.vidriotechnologies.com/display/SI2016/
    ScanImage+BigTiff+Specification
10) CIPA DC-008-2016: Exchangeable image file format for digital still cameras:
    Exif Version 2.31.
    http://www.cipa.jp/std/documents/e/DC-008-Translation-2016-E.pdf
11) ZIF, the Zoomable Image File format. http://zif.photo/
12) GeoTIFF File Format https://www.gdal.org/frmt_gtiff.html

Examples
--------
Save a 3D numpy array to a multi-page, 16-bit grayscale TIFF file:

>>> data = numpy.random.randint(0, 2**12, (4, 301, 219), 'uint16')
>>> imwrite('temp.tif', data, photometric='minisblack')

Read the whole image stack from the TIFF file as numpy array:

>>> image_stack = imread('temp.tif')
>>> image_stack.shape
(4, 301, 219)
>>> image_stack.dtype
dtype('uint16')

Read the image from first page in the TIFF file as numpy array:

>>> image = imread('temp.tif', key=0)
>>> image.shape
(301, 219)

Read images from a sequence of TIFF files as numpy array:

>>> image_sequence = imread(['temp.tif', 'temp.tif'])
>>> image_sequence.shape
(2, 4, 301, 219)

Save a numpy array to a single-page RGB TIFF file:

>>> data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb')

Save a floating-point array and metadata, using zlib compression:

>>> data = numpy.random.rand(2, 5, 3, 301, 219).astype('float32')
>>> imwrite('temp.tif', data, compress=6, metadata={'axes': 'TZCYX'})

Save a volume with xyz voxel size 2.6755x2.6755x3.9474 µm^3 to an ImageJ file:

>>> volume = numpy.random.randn(57*256*256).astype('float32')
>>> volume.shape = 1, 57, 1, 256, 256, 1  # dimensions in TZCYXS order
>>> imwrite('temp.tif', volume, imagej=True, resolution=(1./2.6755, 1./2.6755),
...         metadata={'spacing': 3.947368, 'unit': 'um'})

Get the shape and dtype of the images stored in the TIFF file:

>>> tif = TiffFile('temp.tif')
>>> len(tif.pages)  # number of pages in the file
57
>>> page = tif.pages[0]  # get shape and dtype of the image in the first page
>>> page.shape
(256, 256)
>>> page.dtype
dtype('float32')
>>> page.axes
'YX'
>>> series = tif.series[0]  # get shape and dtype of the first image series
>>> series.shape
(57, 256, 256)
>>> series.dtype
dtype('float32')
>>> series.axes
'ZYX'
>>> tif.close()

Read hyperstack and metadata from the ImageJ file:

>>> with TiffFile('temp.tif') as tif:
...     imagej_hyperstack = tif.asarray()
...     imagej_metadata = tif.imagej_metadata
>>> imagej_hyperstack.shape
(57, 256, 256)
>>> imagej_metadata['slices']
57

Read the "XResolution" tag from the first page in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     tag = tif.pages[0].tags['XResolution']
>>> tag.value
(2000, 5351)
>>> tag.name
'XResolution'
>>> tag.code
282
>>> tag.count
1
>>> tag.dtype
'2I'
>>> tag.valueoffset
360

Read images from a selected range of pages:

>>> image = imread('temp.tif', key=range(4, 40, 2))
>>> image.shape
(18, 256, 256)

Create an empty TIFF file and write to the memory-mapped numpy array:

>>> memmap_image = memmap('temp.tif', shape=(256, 256), dtype='float32')
>>> memmap_image[255, 255] = 1.0
>>> memmap_image.flush()
>>> memmap_image.shape, memmap_image.dtype
((256, 256), dtype('float32'))
>>> del memmap_image

Memory-map image data of the first page in the TIFF file:

>>> memmap_image = memmap('temp.tif', page=0)
>>> memmap_image[255, 255]
1.0
>>> del memmap_image

Successively append images to a BigTIFF file, which can exceed 4 GB:

>>> data = numpy.random.randint(0, 255, (5, 2, 3, 301, 219), 'uint8')
>>> with TiffWriter('temp.tif', bigtiff=True) as tif:
...     for i in range(data.shape[0]):
...         tif.save(data[i], compress=6, photometric='minisblack')

Iterate over pages and tags in the TIFF file and successively read images:

>>> with TiffFile('temp.tif') as tif:
...     image_stack = tif.asarray()
...     for page in tif.pages:
...         for tag in page.tags.values():
...             tag_name, tag_value = tag.name, tag.value
...         image = page.asarray()

Save two image series to a TIFF file:

>>> data0 = numpy.random.randint(0, 255, (301, 219, 3), 'uint8')
>>> data1 = numpy.random.randint(0, 255, (5, 301, 219), 'uint16')
>>> with TiffWriter('temp.tif') as tif:
...     tif.save(data0, compress=6, photometric='rgb')
...     tif.save(data1, compress=6, photometric='minisblack', contiguous=False)

Read the second image series from the TIFF file:

>>> series1 = imread('temp.tif', series=1)
>>> series1.shape
(5, 301, 219)

Read an image stack from a sequence of TIFF files with a file name pattern:

>>> imwrite('temp_C001T001.tif', numpy.random.rand(64, 64))
>>> imwrite('temp_C001T002.tif', numpy.random.rand(64, 64))
>>> image_sequence = TiffSequence('temp_C001*.tif', pattern='axes')
>>> image_sequence.shape
(1, 2)
>>> image_sequence.axes
'CT'
>>> data = image_sequence.asarray()
>>> data.shape
(1, 2, 64, 64)
