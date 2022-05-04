Read and write TIFF files
=========================

Tifffile is a Python library to

(1) store numpy arrays in TIFF (Tagged Image File Format) files, and
(2) read image and metadata from TIFF-like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, SGI,
NIHImage, ImageJ, MicroManager, FluoView, ScanImage, SEQ, GEL, SVS, SCN, SIS,
BIF, ZIF (Zoomable Image File Format), QPTIFF (QPI), NDPI, and GeoTIFF files.

Image data can be read as numpy arrays or zarr arrays/groups from strips,
tiles, pages (IFDs), SubIFDs, higher order series, and pyramidal levels.

Numpy arrays can be written to TIFF, BigTIFF, OME-TIFF, and ImageJ hyperstack
compatible files in multi-page, volumetric, pyramidal, memory-mappable, tiled,
predicted, or compressed form.

A subset of the TIFF specification is supported, mainly 8, 16, 32 and 64-bit
integer, 16, 32 and 64-bit float, grayscale and multi-sample images.
Specifically, CCITT and OJPEG compression, chroma subsampling without JPEG
compression, color space transformations, samples with differing types, or
IPTC, ICC, and XMP metadata are not implemented.

TIFF, the Tagged Image File Format, was created by the Aldus Corporation and
Adobe Systems Incorporated. BigTIFF allows for files larger than 4 GB.
STK, LSM, FluoView, SGI, SEQ, GEL, QPTIFF, NDPI, SCN, SVS, ZIF, BIF, and
OME-TIFF, are custom extensions defined by Molecular Devices (Universal Imaging
Corporation), Carl Zeiss MicroImaging, Olympus, Silicon Graphics International,
Media Cybernetics, Molecular Dynamics, PerkinElmer, Hamamatsu, Leica,
ObjectivePathology, Roche Digital Pathology, and the Open Microscopy
Environment consortium, respectively.

For command line usage run ``python -m tifffile --help``

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:License: BSD 3-Clause

:Version: 2022.5.4

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 3.8.10, 3.9.12, 3.10.4, 64-bit <https://www.python.org>`_
* `Numpy 1.21.5 <https://pypi.org/project/numpy/>`_
* `Imagecodecs 2022.2.22 <https://pypi.org/project/imagecodecs/>`_
  (required only for encoding or decoding LZW, JPEG, etc.)
* `Matplotlib 3.4.3 <https://pypi.org/project/matplotlib/>`_
  (required only for plotting)
* `Lxml 4.8.0 <https://pypi.org/project/lxml/>`_
  (required only for validating and printing XML)
* `Zarr 2.11.3 <https://pypi.org/project/zarr/>`_
  (required only for opening zarr storage)

Revisions
---------
2022.5.4
    Pass 4887 tests.
    Allow to write NewSubfileType=0 (#132).
    Support writing iterators of strip or tile bytes.
    Convert iterables (not iterators) to numpy arrays when writing.
    Explicitly specify optional keyword parameters for imread and imwrite.
    Return number of written bytes from FileHandle write functions.
2022.4.28
    Add option to specify fsspec version 1 url template name (#131).
    Ignore invalid dates in UIC tags (#129).
    Fix zlib_encode and lzma_encode to work with non-contiguous arrays (#128).
    Fix delta_encode to preserve byteorder of ndarrays.
    Move imagecodecs fallback functions to private module and add tests.
2022.4.26
    Fix AttributeError in TiffFile.shaped_metadata (#127).
    Fix TiffTag.overwrite with pre-packed binary value.
    Write sparse TIFF if tile iterator contains None.
    Raise ValueError when writing photometric mode with too few samples.
    Improve test coverage.
2022.4.22
    Add type hints for Python 3.10 (WIP).
    Fix mypy errors (breaking).
    Mark many parameters positional-only or keyword-only (breaking).
    Remove deprecated 'pages' parameter from imread (breaking).
    Remove deprecated 'compress' and 'ijmetadata' write parameters (breaking).
    Remove deprecated 'fastij' and 'movie' parameters from TiffFile (breaking).
    Remove deprecated 'multifile' parameters from TiffFile (breaking).
    Remove deprecated 'tif' parameter from TiffTag.overwrite (breaking).
    Remove deprecated 'file' parameter from FileSequence.asarray (breaking).
    Remove option to pass imread class to FileSequence (breaking).
    Remove optional parameters from '__str__' functions (breaking).
    Rename TiffPageSeries.offset to dataoffset (breaking)
    Change TiffPage.pages to None if no SubIFDs are present (breaking).
    Change TiffPage.index to int (breaking).
    Change TiffPage.is_contiguous, is_imagej, and is_shaped to bool (breaking).
    Add TiffPage imagej_description and shaped_description properties.
    Add TiffFormat abstract base class.
    Deprecate 'lazyattr' and use functools.cached_property instead (breaking).
    Julian_datetime raises ValueError for dates before year 1 (breaking).
    Regressed import time due to typing.
2022.4.8
    Add _ARRAY_DIMENSIONS attributes to ZarrTiffStore.
    Allow C instead of S axis when writing OME-TIFF.
    Fix writing OME-TIFF with separate samples.
    Fix reading unsqueezed pyramidal OME-TIFF series.
2022.3.25
    Fix another ValueError using ZarrStore with zarr >= 2.11.0 (tiffslide #25).
    Add parser for Hamamatsu streak metadata.
    Improve hexdump.
2022.3.16
    Use multi-threading to compress strips and tiles.
    Raise TiffFileError when reading corrupted strips and tiles (#122).
    Fix ScanImage single channel count (#121).
    Add parser for AstroTIFF FITS metadata.
2022.2.9
    Fix ValueError using multiscale ZarrStore with zarr >= 2.11.0.
    Raise KeyError if ZarrStore does not contain key.
    Limit number of warnings for missing files in multifile series.
    Allow to save colormap to 32-bit ImageJ files (#115).
2022.2.2
    Fix TypeError when second ImageDescription tag contains non-ASCII (#112).
    Fix parsing IJMetadata with many IJMetadataByteCounts (#111).
    Detect MicroManager NDTiffv2 header (not tested).
    Remove cache from ZarrFileSequenceStore (use zarr.LRUStoreCache).
    Raise limit on maximum number of pages.
    Use J2K format when encoding JPEG2000 segments.
    Formally deprecate imsave and TiffWriter.save.
    Drop support for Python 3.7 and numpy < 1.19 (NEP29).
2021.11.2
    Lazy-load non-essential tag values (breaking).
    Warn when reading from closed file.
    Support ImageJ 'prop' metadata type (#103).
    Support writing indexed ImageJ format.
    Fix multi-threaded access of multi-page Zarr stores with chunkmode 2.
    Raise error if truncate is used with compression, packints, or tile.
    Read STK metadata without UIC2tag.
    Improve log and warning messages (WIP).
    Improve string representation of large tag values.
2021.10.12
    Revert renaming of 'file' parameter in FileSequence.asarray (breaking).
    Deprecate 'file' parameter in FileSequence.asarray.
2021.10.10
    Disallow letters as indices in FileSequence; use categories (breaking).
    Do not warn of missing files in FileSequence; use files_missing property.
    Support predictors in ZarrTiffStore.write_fsspec.
    Add option to specify zarr group name in write_fsspec.
    Add option to specify categories for FileSequence patterns (#76).
    Add option to specify chunk shape and dtype for ZarrFileSequenceStore.
    Add option to tile ZarrFileSequenceStore and FileSequence.asarray.
    Add option to pass additional zattrs to Zarr stores.
    Detect Roche BIF files.
2021.8.30
    Fix horizontal differencing with non-native byte order.
    Fix multi-threaded access of memory-mappable, multi-page Zarr stores (#67).
2021.8.8
    Fix tag offset and valueoffset for NDPI > 4 GB (#96).
2021.7.30
    Deprecate first parameter to TiffTag.overwrite (no longer required).
    TiffTag init API change (breaking).
    Detect Ventana BIF series and warn that tiles are not stitched.
    Enable reading PreviewImage from RAW formats (#93, #94).
    Work around numpy.ndarray.tofile is very slow for non-contiguous arrays.
    Fix issues with PackBits compression (requires imagecodecs 2021.7.30).
2021.7.2
    Decode complex integer images found in SAR GeoTIFF.
    Support reading NDPI with JPEG-XR compression.
    Deprecate TiffWriter RGB auto-detection, except for RGB24/48 and RGBA32/64.
2021.6.14
    Set stacklevel for deprecation warnings (#89).
    Fix svs_description_metadata for SVS with double header (#88, breaking).
    Fix reading JPEG compressed CMYK images.
    Support ALT_JPEG and JPEG_2000_LOSSY compression found in Bio-Formats.
    Log warning if TiffWriter auto-detects RGB mode (specify photometric).
2021.6.6
    Fix TIFF.COMPESSOR typo (#85).
    Round resolution numbers that do not fit in 64-bit rationals (#81).
    Add support for JPEG XL compression.
    Add numcodecs compatible TIFF codec.
    Rename ZarrFileStore to ZarrFileSequenceStore (breaking).
    Add method to export fsspec ReferenceFileSystem from ZarrFileStore.
    Fix fsspec ReferenceFileSystem v1 for multifile series.
    Fix creating OME-TIFF with micron character in OME-XML.
2021.4.8
    Fix reading OJPEG with wrong photometric or samplesperpixel tags (#75).
    Fix fsspec ReferenceFileSystem v1 and JPEG compression.
    Use TiffTagRegistry for NDPI_TAGS, EXIF_TAGS, GPS_TAGS, IOP_TAGS constants.
    Make TIFF.GEO_KEYS an Enum (breaking).
2021.3.31
    Use JPEG restart markers as tile offsets in NDPI.
    Support version 1 and more codecs in fsspec ReferenceFileSystem (untested).
2021.3.17
    Fix regression reading multi-file OME-TIFF with missing files (#72).
    Fix fsspec ReferenceFileSystem with non-native byte order (#56).
2021.3.16
    TIFF is no longer a defended trademark.
    Add method to export fsspec ReferenceFileSystem from ZarrTiffStore (#56).
2021.3.5
    Preliminary support for EER format (#68).
    Do not warn about unknown compression (#68).
2021.3.4
    Fix reading multi-file, multi-series OME-TIFF (#67).
    Detect ScanImage 2021 files (#46).
    Shape new version ScanImage series according to metadata (breaking).
    Remove Description key from TiffFile.scanimage_metadata dict (breaking).
    Also return ScanImage version from read_scanimage_metadata (breaking).
    Fix docstrings.
2021.2.26
    Squeeze axes of LSM series by default (breaking).
    Add option to preserve single dimensions when reading from series (WIP).
    Do not allow appending to OME-TIFF files.
    Fix reading STK files without name attribute in metadata.
    Make TIFF constants multi-thread safe and pickleable (#64).
    Add detection of NDTiffStorage MajorVersion to read_micromanager_metadata.
    Support ScanImage v4 files in read_scanimage_metadata.
2021.2.1
    Fix multi-threaded access of ZarrTiffStores using same TiffFile instance.
    Use fallback zlib and lzma codecs with imagecodecs lite builds.
    Open Olympus and Panasonic RAW files for parsing, albeit not supported.
    Support X2 and X4 differencing found in DNG.
    Support reading JPEG_LOSSY compression found in DNG.
2021.1.14
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
* *ImageJ hyperstacks* store all image data, which may exceed 4 GB,
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
* *Hamamatsu NDPI* uses some 64-bit offsets in the file header, IFD, and tag
  structures. Tag values/offsets can be corrected using high bits stored after
  IFD structures. Tifffile can read NDPI files > 4 GB.
  JPEG compressed segments with dimensions >65530 or missing restart markers
  are not decodable with libjpeg. Tifffile works around this limitation by
  separately decoding the MCUs between restart markers.
  BitsPerSample, SamplesPerPixel, and PhotometricInterpretation tags may
  contain wrong values, which can be corrected using the value of tag 65441.
* *Philips TIFF* slides store wrong ImageWidth and ImageLength tag values for
  tiled pages. The values can be corrected using the DICOM_PIXEL_SPACING
  attributes of the XML formatted description of the first page. Tifffile can
  read Philips slides.
* *Ventana/Roche BIF* slides store tiles and metadata in a BigTIFF container.
  Tiles may overlap and require stitching based on the TileJointInfo elements
  in the XMP tag. Volumetric scans are stored using the ImageDepth extension.
  Tifffile can read BIF and decode individual tiles, but does not perform
  stitching.
* *ScanImage* optionally allows corrupted non-BigTIFF files > 2 GB. The values
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
* `Slideio <https://gitlab.com/bioslide/slideio>`_
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
* `tiffslide <https://github.com/bayer-science-for-a-better-life/tiffslide>`_
* `opentile <https://github.com/imi-bigpicture/opentile>`_

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
* ScanImage BigTiff Specification - ScanImage 2019.
  http://scanimage.vidriotechnologies.com/display/SI2019/
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
* The EER (Electron Event Representation) file format.
  https://github.com/fei-company/EerReaderLib
* Digital Negative (DNG) Specification. Version 1.5.0.0, June 2012.
  https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/
  dng_spec_1.5.0.0.pdf
* Roche Digital Pathology. BIF image file format for digital pathology.
  https://diagnostics.roche.com/content/dam/diagnostics/Blueprint/en/pdf/rmd/
  Roche-Digital-Pathology-BIF-Whitepaper.pdf
* Astro-TIFF specification. https://astro-tiff.sourceforge.io/

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

>>> images = imread('temp.tif', key=range(4, 40, 2))
>>> images.shape
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
...     _ = tif.pages[0].tags['XResolution'].overwrite((96000, 1000))

Write a floating-point ndarray and metadata using BigTIFF format, tiling,
compression, and planar storage:

>>> data = numpy.random.rand(2, 5, 3, 301, 219).astype('float32')
>>> imwrite('temp.tif', data, bigtiff=True, photometric='minisblack',
...         compression='zlib', planarconfig='separate', tile=(32, 32),
...         metadata={'axes': 'TZCYX'})

Write a 10 fps time series of volumes with xyz voxel size 2.6755x2.6755x3.9474
micron^3 to an ImageJ hyperstack formatted TIFF file:

>>> volume = numpy.random.randn(6, 57, 256, 256).astype('float32')
>>> imwrite('temp.tif', volume, imagej=True, resolution=(1./2.6755, 1./2.6755),
...         metadata={'spacing': 3.947368, 'unit': 'um', 'finterval': 1/10,
...                   'axes': 'TZYX'})

Read the volume and metadata from the ImageJ file:

>>> with TiffFile('temp.tif') as tif:
...     volume = tif.asarray()
...     axes = tif.series[0].axes
...     imagej_metadata = tif.imagej_metadata
>>> volume.shape
(6, 57, 256, 256)
>>> axes
'TZYX'
>>> imagej_metadata['slices']
57
>>> imagej_metadata['frames']
6

Create a TIFF file containing an empty image and write to the memory-mapped
numpy array:

>>> memmap_image = memmap(
...     'temp.tif', shape=(256, 256, 3), dtype='float32', photometric='rgb'
... )
>>> type(memmap_image)
<class 'numpy.memmap'>
>>> memmap_image[255, 255, 1] = 1.0
>>> memmap_image.flush()
>>> del memmap_image

Memory-map and read contiguous image data in the TIFF file:

>>> memmap_image = memmap('temp.tif')
>>> memmap_image.shape
(256, 256, 3)
>>> memmap_image[255, 255, 1]
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
>>> imwrite('temp.tif', data, photometric='rgb', append=True)

Create a TIFF file from a generator of tiles:

>>> data = numpy.random.randint(0, 2**12, (31, 33, 3), 'uint16')
>>> def tiles(data, tileshape):
...     for y in range(0, data.shape[0], tileshape[0]):
...         for x in range(0, data.shape[1], tileshape[1]):
...             yield data[y : y + tileshape[0], x : x + tileshape[1]]
>>> imwrite('temp.tif', tiles(data, (16, 16)), tile=(16, 16),
...         shape=data.shape, dtype=data.dtype, photometric='rgb')

Write two numpy arrays to a multi-series OME-TIFF file:

>>> series0 = numpy.random.randint(0, 255, (32, 32, 3), 'uint8')
>>> series1 = numpy.random.randint(0, 1023, (4, 256, 256), 'uint16')
>>> with TiffWriter('temp.ome.tif') as tif:
...     tif.write(series0, photometric='rgb')
...     tif.write(series1, photometric='minisblack',
...               metadata={'axes': 'ZYX', 'SignificantBits': 10,
...                         'Plane': {'PositionZ': [0.0, 1.0, 2.0, 3.0]}})

Write a multi-dimensional, multi-resolution (pyramidal) OME-TIFF file using
JPEG compressed tiles. Sub-resolution images are written to SubIFDs:

>>> data = numpy.random.randint(0, 2**12, (8, 512, 512, 3), 'uint16')
>>> with TiffWriter('temp.ome.tif', bigtiff=True) as tif:
...     options = dict(photometric='rgb', tile=(128, 128), compression='jpeg',
...                    metadata={'axes': 'TYXS'})
...     tif.write(data, subifds=2, **options)
...     # save pyramid levels to the two subifds
...     # in production use resampling to generate sub-resolutions
...     tif.write(data[:, ::2, ::2], subfiletype=1, **options)
...     tif.write(data[:, ::4, ::4], subfiletype=1, **options)

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
...             _ = fh.seek(offset)
...             data = fh.read(bytecount)
...             tile, indices, shape = page.decode(
...                 data, index, jpegtables=page.jpegtables
...             )

Use zarr to read parts of the tiled, pyramidal images in the TIFF file:

>>> import zarr
>>> store = imread('temp.ome.tif', aszarr=True)
>>> z = zarr.open(store, mode='r')
>>> z
<zarr.hierarchy.Group '/' read-only>
>>> z[0]  # base layer
<zarr.core.Array '/0' (8, 512, 512, 3) uint16 read-only>
>>> z[0][2, 128:384, 256:].shape  # read a tile from the base layer
(256, 256, 3)
>>> store.close()

Read images from a sequence of TIFF files as numpy array:

>>> imwrite('temp_C001T001.tif', numpy.random.rand(64, 64))
>>> imwrite('temp_C001T002.tif', numpy.random.rand(64, 64))
>>> image_sequence = imread(['temp_C001T001.tif', 'temp_C001T002.tif'])
>>> image_sequence.shape
(2, 64, 64)
>>> image_sequence.dtype
dtype('float64')

Read an image stack from a series of TIFF files with a file name pattern
as numpy or zarr arrays:

>>> image_sequence = TiffSequence('temp_C0*.tif', pattern=r'_(C)(\d+)(T)(\d+)')
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

Write the zarr store to a fsspec ReferenceFileSystem in JSON format:

>>> with image_sequence.aszarr() as store:
...     store.write_fsspec('temp.json', url='file://')

Open the fsspec ReferenceFileSystem as a zarr array:

>>> import fsspec
>>> import tifffile.numcodecs
>>> tifffile.numcodecs.register_codec()
>>> mapper = fsspec.get_mapper(
...     'reference://', fo='temp.json', target_protocol='file')
>>> zarr.open(mapper, mode='r')
<zarr.core.Array (1, 2, 64, 64) float64 read-only>
