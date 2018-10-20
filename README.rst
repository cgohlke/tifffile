Read and write TIFF(r) files
============================

Tifffile is a Python library to

(1) store numpy arrays in TIFF (Tagged Image File Format) files, and
(2) read image and metadata from TIFF like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH,
SGI, ImageJ, MicroManager, FluoView, ScanImage, SEQ, GEL, SVS, SCN, SIS, ZIF,
and GeoTIFF files.

Numpy arrays can be written to TIFF, BigTIFF, and ImageJ hyperstack compatible
files in multi-page, memory-mappable, tiled, predicted, or compressed form.

Only a subset of the TIFF specification is supported, mainly uncompressed and
losslessly compressed 1, 8, 16, 32 and 64-bit integer, 16, 32 and 64-bit float,
grayscale and RGB(A) images.
Specifically, reading slices of image data, image trees defined via SubIFDs,
CCITT and OJPEG compression, chroma subsampling without JPEG compression,
or IPTC and XMP metadata are not implemented.

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

:Version: 2018.10.18

Requirements
------------
* `CPython 2.7 or 3.5+ 64-bit <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* `Imagecodecs 2018.10.18 <https://www.lfd.uci.edu/~gohlke/>`_
  (optional for decoding LZW, JPEG, etc.)
* `Matplotlib 2.2 <https://www.matplotlib.org>`_ (optional for plotting)
* Python 2 requires 'futures', 'enum34', and 'pathlib'.

Revisions
---------
2018.10.18
    Rename tiffile package to tifffile.
2018.10.10
    Pass 2710 tests.
    Read ZIF, the Zoomable Image Format (WIP).
    Decode YCbCr JPEG as RGB (tentative).
    Improve restoration of incomplete tiles.
    Allow to write grayscale with extrasamples without specifying planarconfig.
    Enable decoding of PNG and JXR via imagecodecs.
    Deprecate 32-bit platforms (too many memory errors during tests).
2018.9.27
    Read Olympus SIS (WIP).
    Allow to write non-BigTIFF files up to ~4 GB (bug fix).
    Fix parsing date and time fields in SEM metadata (bug fix).
    Detect some circular IFD references.
    Enable WebP codecs via imagecodecs.
    Add option to read TiffSequence from ZIP containers.
    Remove TiffFile.isnative.
    Move TIFF struct format constants out of TiffFile namespace.
2018.8.31
    Pass 2699 tests.
    Fix wrong TiffTag.valueoffset (bug fix).
    Towards reading Hamamatsu NDPI (WIP).
    Enable PackBits compression of byte and bool arrays.
    Fix parsing NULL terminated CZ_SEM strings.
2018.8.24
    Move tifffile.py and related modules into tiffile package.
    Move usage examples to module docstring.
    Enable multi-threading for compressed tiles and pages by default.
    Add option to concurrently decode image tiles using threads.
    Do not skip empty tiles (bug fix).
    Read JPEG and J2K compressed strips and tiles.
    Allow floating point predictor on write.
    Add option to specify subfiletype on write.
    Depend on imagecodecs package instead of _tifffile, lzma, etc modules.
    Remove reverse_bitorder, unpack_ints, and decode functions.
    Use pytest instead of unittest.
2018.6.20
    Save RGBA with unassociated extrasample by default (backward incompatible).
    Add option to specify ExtraSamples values.
2018.6.17
    Pass 2680 tests.
    Towards reading JPEG and other compressions via imagecodecs package (WIP).
    Read SampleFormat VOID as UINT.
    Add function to validate TIFF using 'jhove -m TIFF-hul'.
    Save bool arrays as bilevel TIFF.
    Accept pathlib.Path as filenames.
    Move 'software' argument from TiffWriter __init__ to save.
    Raise DOS limit to 16 TB.
    Lazy load lzma and zstd compressors and decompressors.
    Add option to save IJMetadata tags.
    Return correct number of pages for truncated series (bug fix).
    Move EXIF tags to TIFF.TAG as per TIFF/EP standard.
2018.2.18
    Pass 2293 tests.
    Always save RowsPerStrip and Resolution tags as required by TIFF standard.
    Do not use badly typed ImageDescription.
    Coherce bad ASCII string tags to bytes.
    Tuning of __str__ functions.
    Fix reading 'undefined' tag values (bug fix).
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
    Towards reading PerkinElmer QPTIFF (no test files).
    Do not index out of bounds data in tifffile.c unpackbits and decodelzw.
2017.9.29 (tentative)
    Many backward incompatible changes improving speed and resource usage:
    Pass 2268 tests.
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
    Fix reading some UIC tags (bug fix).
    Do not modify input array in imshow (bug fix).
    Fix Python implementation of unpack_ints.
2017.5.23
    Pass 1961 tests.
    Write correct number of SampleFormat values (bug fix).
    Use Adobe deflate code to write ZIP compressed files.
    Add option to pass tag values as packed binary data for writing.
    Defer tag validation to attribute access.
    Use property instead of lazyattr decorator for simple expressions.
2017.3.17
    Write IFDs and tag values on word boundaries.
    Read ScanImage metadata.
    Remove is_rgb and is_indexed attributes from TiffFile.
    Create files used by doctests.
2017.1.12
    Read Zeiss SEM metadata.
    Read OME-TIFF with invalid references to external files.
    Rewrite C LZW decoder (5x faster).
    Read corrupted LSM files missing EOI code in LZW stream.
2017.1.1
    Add option to append images to existing TIFF files.
    Read files without pages.
    Read S-FEG and Helios NanoLab tags created by FEI software.
    Allow saving Color Filter Array (CFA) images.
    Add info functions returning more information about TiffFile and TiffPage.
    Add option to read specific pages only.
    Remove maxpages argument (backward incompatible).
    Remove test_tifffile function.
2016.10.28
    Pass 1944 tests.
    Improve detection of ImageJ hyperstacks.
    Read TVIPS metadata created by EM-MENU (by Marco Oster).
    Add option to disable using OME-XML metadata.
    Allow non-integer range attributes in modulo tags (by Stuart Berg).
2016.6.21
    Do not always memmap contiguous data in page series.
2016.5.13
    Add option to specify resolution unit.
    Write grayscale images with extra samples when planarconfig is specified.
    Do not write RGB color images with 2 samples.
    Reorder TiffWriter.save keyword arguments (backward incompatible).
2016.4.18
    Pass 1932 tests.
    TiffWriter, imread, and imsave accept open binary file streams.
2016.04.13
    Correctly handle reversed fill order in 2 and 4 bps images (bug fix).
    Implement reverse_bitorder in C.
2016.03.18
    Fix saving additional ImageJ metadata.
2016.2.22
    Pass 1920 tests.
    Write 8 bytes double tag values using offset if necessary (bug fix).
    Add option to disable writing second image description tag.
    Detect tags with incorrect counts.
    Disable color mapping for LSM.
2015.11.13
    Read LSM 6 mosaics.
    Add option to specify directory of memory-mapped files.
    Add command line options to specify vmin and vmax values for colormapping.
2015.10.06
    New helper function to apply colormaps.
    Renamed is_palette attributes to is_indexed (backward incompatible).
    Color-mapped samples are now contiguous (backward incompatible).
    Do not color-map ImageJ hyperstacks (backward incompatible).
    Towards reading Leica SCN.
2015.9.25
    Read images with reversed bit order (FillOrder is LSB2MSB).
2015.9.21
    Read RGB OME-TIFF.
    Warn about malformed OME-XML.
2015.9.16
    Detect some corrupted ImageJ metadata.
    Better axes labels for 'shaped' files.
    Do not create TiffTag for default values.
    Chroma subsampling is not supported.
    Memory-map data in TiffPageSeries if possible (optional).
2015.8.17
    Pass 1906 tests.
    Write ImageJ hyperstacks (optional).
    Read and write LZMA compressed data.
    Specify datetime when saving (optional).
    Save tiled and color-mapped images (optional).
    Ignore void bytecounts and offsets if possible.
    Ignore bogus image_depth tag created by ISS Vista software.
    Decode floating point horizontal differencing (not tiled).
    Save image data contiguously if possible.
    Only read first IFD from ImageJ files if possible.
    Read ImageJ 'raw' format (files larger than 4 GB).
    TiffPageSeries class for pages with compatible shape and data type.
    Try to read incomplete tiles.
    Open file dialog if no filename is passed on command line.
    Ignore errors when decoding OME-XML.
    Rename decoder functions (backward incompatible).
2014.8.24
    TiffWriter class for incremental writing images.
    Simplify examples.
2014.8.19
    Add memmap function to FileHandle.
    Add function to determine if image data in TiffPage is memory-mappable.
    Do not close files if multifile_close parameter is False.
2014.8.10
    Pass 1730 tests.
    Return all extrasamples by default (backward incompatible).
    Read data from series of pages into memory-mapped array (optional).
    Squeeze OME dimensions (backward incompatible).
    Workaround missing EOI code in strips.
    Support image and tile depth tags (SGI extension).
    Better handling of STK/UIC tags (backward incompatible).
    Disable color mapping for STK.
    Julian to datetime converter.
    TIFF ASCII type may be NULL separated.
    Unwrap strip offsets for LSM files greater than 4 GB.
    Correct strip byte counts in compressed LSM files.
    Skip missing files in OME series.
    Read embedded TIFF files.
2014.2.05
    Save rational numbers as type 5 (bug fix).
2013.12.20
    Keep other files in OME multi-file series closed.
    FileHandle class to abstract binary file handle.
    Disable color mapping for bad OME-TIFF produced by bio-formats.
    Read bad OME-XML produced by ImageJ when cropping.
2013.11.3
    Allow zlib compress data in imsave function (optional).
    Memory-map contiguous image data (optional).
2013.10.28
    Read MicroManager metadata and little-endian ImageJ tag.
    Save extra tags in imsave function.
    Save tags in ascending order by code (bug fix).
2012.10.18
    Accept file like objects (read from OIB files).
2012.8.21
    Rename TIFFfile to TiffFile and TIFFpage to TiffPage.
    TiffSequence class for reading sequence of TIFF files.
    Read UltraQuant tags.
    Allow float numbers as resolution in imsave function.
2012.8.3
    Read MD GEL tags and NIH Image header.
2012.7.25
    Read ImageJ tags.
    ...

Notes
-----
The API is not stable yet and might change between revisions.

Tested on little-endian platforms only.

Python 2.7, 3.4, and 32-bit versions are deprecated.

Other libraries for reading scientific TIFF files from Python:

*  `Python-bioformats <https://github.com/CellProfiler/python-bioformats>`_
*  `Imread <https://github.com/luispedro/imread>`_
*  `GDAL <https://github.com/OSGeo/gdal/tree/master/gdal/swig/python>`_
*  `OpenSlide-python <https://github.com/openslide/openslide-python>`_
*  `PyLibTiff <https://github.com/pearu/pylibtiff>`_
*  `SimpleITK <http://www.simpleitk.org>`_
*  `PyLSM <https://launchpad.net/pylsm>`_
*  `PyMca.TiffIO.py <https://github.com/vasole/pymca>`_ (same as fabio.TiffIO)
*  `BioImageXD.Readers <http://www.bioimagexd.net/>`_
*  `Cellcognition.io <http://cellcognition.org/>`_
*  `pymimage <https://github.com/ardoi/pymimage>`_
*  `pytiff <https://github.com/FZJ-INM1-BDA/pytiff>`_

Acknowledgements
----------------
*   Egor Zindy, University of Manchester, for lsm_scan_info specifics.
*   Wim Lewis for a bug fix and some LSM functions.
*   Hadrien Mary for help on reading MicroManager files.
*   Christian Kliche for help writing tiled and color-mapped files.

References
----------
1)  TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
    http://partners.adobe.com/public/developer/tiff/
2)  TIFF File Format FAQ. http://www.awaresystems.be/imaging/tiff/faq.html
3)  MetaMorph Stack (STK) Image File Format.
    http://support.meta.moleculardevices.com/docs/t10243.pdf
4)  Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
    Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
5)  The OME-TIFF format.
    http://www.openmicroscopy.org/site/support/file-formats/ome-tiff
6)  UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
    http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
7)  Micro-Manager File Formats.
    http://www.micro-manager.org/wiki/Micro-Manager_File_Formats
8)  Tags for TIFF and Related Specifications. Digital Preservation.
    http://www.digitalpreservation.gov/formats/content/tiff_tags.shtml
9)  ScanImage BigTiff Specification - ScanImage 2016.
    http://scanimage.vidriotechnologies.com/display/SI2016/
    ScanImage+BigTiff+Specification
10) CIPA DC-008-2016: Exchangeable image file format for digital still cameras:
    Exif Version 2.31.
    http://www.cipa.jp/std/documents/e/DC-008-Translation-2016-E.pdf
11) ZIF, the Zoomable Image File format. http://zif.photo/

Examples
--------
Save a 3D numpy array to a multi-page, 16-bit grayscale TIFF file:

>>> data = numpy.random.randint(0, 2**16, (4, 301, 219), 'uint16')
>>> imsave('temp.tif', data, photometric='minisblack')

Read the whole image stack from the TIFF file as numpy array:

>>> image_stack = imread('temp.tif')
>>> image_stack.shape
(4, 301, 219)
>>> image_stack.dtype
dtype('uint16')

Read the image from first page (IFD) in the TIFF file:

>>> image = imread('temp.tif', key=0)
>>> image.shape
(301, 219)

Read images from a sequence of TIFF files as numpy array:

>>> image_sequence = imread(['temp.tif', 'temp.tif'])
>>> image_sequence.shape
(2, 4, 301, 219)

Save a numpy array to a single-page RGB TIFF file:

>>> data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
>>> imsave('temp.tif', data, photometric='rgb')

Save a floating-point array and metadata, using zlib compression:

>>> data = numpy.random.rand(2, 5, 3, 301, 219).astype('float32')
>>> imsave('temp.tif', data, compress=6, metadata={'axes': 'TZCYX'})

Save a volume with xyz voxel size 2.6755x2.6755x3.9474 µm^3 to ImageJ file:

>>> volume = numpy.random.randn(57*256*256).astype('float32')
>>> volume.shape = 1, 57, 1, 256, 256, 1  # dimensions in TZCYXS order
>>> imsave('temp.tif', volume, imagej=True, resolution=(1./2.6755, 1./2.6755),
...        metadata={'spacing': 3.947368, 'unit': 'um'})

Read hyperstack and metadata from ImageJ file:

>>> with TiffFile('temp.tif') as tif:
...     imagej_hyperstack = tif.asarray()
...     imagej_metadata = tif.imagej_metadata
>>> imagej_hyperstack.shape
(57, 256, 256)
>>> imagej_metadata['slices']
57

Create an empty TIFF file and write to the memory-mapped numpy array:

>>> memmap_image = memmap('temp.tif', shape=(256, 256), dtype='float32')
>>> memmap_image[255, 255] = 1.0
>>> memmap_image.flush()
>>> memmap_image.shape, memmap_image.dtype
((256, 256), dtype('float32'))
>>> del memmap_image

Memory-map image data in the TIFF file:

>>> memmap_image = memmap('temp.tif', page=0)
>>> memmap_image[255, 255]
1.0
>>> del memmap_image

Successively append images to a BigTIFF file:

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
...     tif.save(data1, compress=6, photometric='minisblack')

Read the second image series from the TIFF file:

>>> series1 = imread('temp.tif', series=1)
>>> series1.shape
(5, 301, 219)

Read a image stack from a sequence of TIFF files with a file name pattern:

>>> imsave('temp_C001T001.tif', numpy.random.rand(64, 64))
>>> imsave('temp_C001T002.tif', numpy.random.rand(64, 64))
>>> image_sequence = TiffSequence('temp_C001*.tif')
>>> image_sequence.shape
(1, 2)
>>> image_sequence.axes
'CT'
>>> data = image_sequence.asarray()
>>> data.shape
(1, 2, 64, 64)
