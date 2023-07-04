Read and write TIFF files
=========================

Tifffile is a Python library to

(1) store NumPy arrays in TIFF (Tagged Image File Format) files, and
(2) read image and metadata from TIFF-like files used in bioimaging.

Image and metadata can be read from TIFF, BigTIFF, OME-TIFF, DNG, STK, LSM,
SGI, NIHImage, ImageJ, MMStack, NDTiff, FluoView, ScanImage, SEQ, GEL,
SVS, SCN, SIS, BIF, ZIF (Zoomable Image File Format), QPTIFF (QPI, PKI), NDPI,
and GeoTIFF formatted files.

Image data can be read as NumPy arrays or Zarr arrays/groups from strips,
tiles, pages (IFDs), SubIFDs, higher order series, and pyramidal levels.

Image data can be written to TIFF, BigTIFF, OME-TIFF, and ImageJ hyperstack
compatible files in multi-page, volumetric, pyramidal, memory-mappable,
tiled, predicted, or compressed form.

Many compression and predictor schemes are supported via the imagecodecs
library, including LZW, PackBits, Deflate, PIXTIFF, LZMA, LERC, Zstd,
JPEG (8 and 12-bit, lossless), JPEG 2000, JPEG XR, JPEG XL, WebP, PNG, Jetraw,
24-bit floating-point, and horizontal differencing.

Tifffile can also be used to inspect TIFF structures, read image data from
multi-dimensional file sequences, write fsspec ReferenceFileSystem for
TIFF files and image file sequences, patch TIFF tag values, and parse
many proprietary metadata formats.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD 3-Clause
:Version: 2023.7.4
:DOI: `10.5281/zenodo.6795860 <https://doi.org/10.5281/zenodo.6795860>`_

Quickstart
----------

Install the tifffile package and all dependencies from the
`Python Package Index <https://pypi.org/project/tifffile/>`_::

    python -m pip install -U tifffile[all]

Tifffile is also available in other package repositories such as Anaconda,
Debian, and MSYS2.

The tifffile library is type annotated and documented via docstrings::

    python -c "import tifffile; help(tifffile)"

Tifffile can be used as a console script to inspect and preview TIFF files::

    python -m tifffile --help

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/tifffile>`_.

Support is also provided on the
`image.sc <https://forum.image.sc/tag/tifffile>`_ forum.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.9.13, 3.10.11, 3.11.4, 3.12.0b3, 64-bit
- `NumPy <https://pypi.org/project/numpy/>`_ 1.25.0
- `Imagecodecs <https://pypi.org/project/imagecodecs/>`_ 2023.7.4
  (required for encoding or decoding LZW, JPEG, etc. compressed segments)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.7.1
  (required for plotting)
- `Lxml <https://pypi.org/project/lxml/>`_ 4.9.2
  (required only for validating and printing XML)
- `Zarr <https://pypi.org/project/zarr/>`_ 2.15.0
  (required only for opening Zarr stores)
- `Fsspec <https://pypi.org/project/fsspec/>`_ 2023.6.0
  (required only for opening ReferenceFileSystem files)

Revisions
---------

2023.7.4

- Pass 4992 tests.
- Add option to return selection from imread (#200).
- Fix reading OME series with missing trailing frames (#199).
- Fix fsspec reference for WebP compressed segments missing alpha channel.
- Fix linting issues.
- Detect files written by Agilent Technologies.
- Drop support for Python 3.8 and numpy < 1.21 (NEP29).

2023.4.12

- Do not write duplicate ImageDescription tags from extratags (breaking).
- Support multifocal SVS files (#193).
- Log warning when filtering out extratags.
- Fix writing OME-TIFF with image description in extratags.
- Ignore invalid predictor tag value if prediction is not used.
- Raise KeyError if ZarrStore is missing requested chunk.

2023.3.21

- Fix reading MMstack with missing data (#187).

2023.3.15

- Fix corruption using tile generators with prediction/compression (#185).
- Add parser for Micro-Manager MMStack series (breaking).
- Return micromanager_metadata IndexMap as numpy array (breaking).
- Revert optimizations for Micro-Manager OME series.
- Do not use numcodecs zstd in write_fsspec (kerchunk issue 317).
- More type annotations.

2023.2.28

- Fix reading some Micro-Manager metadata from corrupted files.
- Speed up reading Micro-Manager indexmap for creation of OME series.

2023.2.27

- Use Micro-Manager indexmap offsets to create virtual TiffFrames.
- Fixes for future imagecodecs.

2023.2.3

- Fix overflow in calculation of databytecounts for large NDPI files.

2023.2.2

- Fix regression reading layered NDPI files.
- Add option to specify offset in FileHandle.read_array.

2023.1.23

- Support reading NDTiffStorage.
- Support reading PIXTIFF compression.
- Support LERC with Zstd or Deflate compression.
- Do not write duplicate and select extratags.
- Allow to write uncompressed image data beyond 4 GB in classic TIFF.
- Add option to specify chunkshape and dtype in FileSequence.asarray.
- Add option for imread to write to output in FileSequence.asarray (#172).
- Add function to read GDAL structural metadata.
- Add function to read NDTiff.index files.
- Fix IndexError accessing TiffFile.mdgel_metadata in non-MDGEL files.
- Fix unclosed file ResourceWarning in TiffWriter.
- Fix non-bool predictor arguments (#167).
- Relax detection of OME-XML (#173).
- Rename some TiffFrame parameters (breaking).
- Deprecate squeeze_axes (will change signature).
- Use defusexml in xml2dict.

2022.10.10

- Fix RecursionError in peek_iterator.
- Fix reading NDTiffv3 summary settings.
- Fix svs_description_metadata parsing (#149).
- Fix ImportError if Python was built without zlib or lzma.
- Fix bool of COMPRESSION and PREDICTOR instances.
- Deprecate non-sequence extrasamples arguments.
- Parse SCIFIO metadata as ImageJ.

2022.8.12

- Fix writing ImageJ format with hyperstack argument.
- Fix writing description with metadata disabled.
- Add option to disable writing shaped metadata in TiffWriter.

2022.8.8

- Fix regression using imread out argument (#147).
- Fix imshow show argument.
- Support fsspec OpenFile.

2022.8.3

- Fix regression writing default resolutionunit (#145).
- Add strptime function parsing common datetime formats.

2022.7.31

- Fix reading corrupted WebP compressed segments missing alpha channel (#122).
- Fix regression reading compressed ImageJ files.

2022.7.28

- Rename FileSequence.labels attribute to dims (breaking).
- Rename tifffile_geodb module to geodb (breaking).
- Rename TiffFile._astuple method to astuple (breaking).
- Rename noplots command line argument to maxplots (breaking).
- Fix reading ImageJ hyperstacks with non-TZC order.
- Fix colorspace of JPEG segments encoded by Bio-Formats.
- Fix fei_metadata for HELIOS FIB-SEM (#141, needs test).
- Add xarray style properties to TiffPage (WIP).
- Add option to specify OME-XML for TiffFile.
- Add option to control multiscales in ZarrTiffStore.
- Support writing to uncompressed ZarrTiffStore.
- Support writing empty images with tiling.
- Support overwriting some tag values in NDPI (#137).
- Support Jetraw compression (experimental).
- Standardize resolution parameter and property.
- Deprecate third resolution argument on write (use resolutionunit).
- Deprecate tuple type compression argument on write (use compressionargs).
- Deprecate enums in TIFF namespace (use enums from module).
- Improve default number of threads to write compressed segments (#139).
- Parse metaseries time values as datetime objects (#143).
- Increase internal read and write buffers to 256 MB.
- Convert some warnings to debug messages.
- Declare all classes final.
- Add script to generate documentation via Sphinx.
- Convert docstrings to Google style with Sphinx directives.

2022.5.4

- …

Refer to the CHANGES file for older revisions.

Notes
-----

TIFF, the Tagged Image File Format, was created by the Aldus Corporation and
Adobe Systems Incorporated. STK, LSM, FluoView, SGI, SEQ, GEL, QPTIFF, NDPI,
SCN, SVS, ZIF, BIF, and OME-TIFF, are custom extensions defined by Molecular
Devices (Universal Imaging Corporation), Carl Zeiss MicroImaging, Olympus,
Silicon Graphics International, Media Cybernetics, Molecular Dynamics,
PerkinElmer, Hamamatsu, Leica, ObjectivePathology, Roche Digital Pathology,
and the Open Microscopy Environment consortium, respectively.

Tifffile supports a subset of the TIFF6 specification, mainly 8, 16, 32, and
64-bit integer, 16, 32 and 64-bit float, grayscale and multi-sample images.
Specifically, CCITT and OJPEG compression, chroma subsampling without JPEG
compression, color space transformations, samples with differing types, or
IPTC, ICC, and XMP metadata are not implemented.

Besides classic TIFF, tifffile supports several TIFF-like formats that do not
strictly adhere to the TIFF6 specification. Some formats allow file and data
sizes to exceed the 4 GB limit of the classic TIFF:

- **BigTIFF** is identified by version number 43 and uses different file
  header, IFD, and tag structures with 64-bit offsets. The format also adds
  64-bit data types. Tifffile can read and write BigTIFF files.
- **ImageJ hyperstacks** store all image data, which may exceed 4 GB,
  contiguously after the first IFD. Files > 4 GB contain one IFD only.
  The size and shape of the up to 6-dimensional image data can be determined
  from the ImageDescription tag of the first IFD, which is Latin-1 encoded.
  Tifffile can read and write ImageJ hyperstacks.
- **OME-TIFF** files store up to 8-dimensional image data in one or multiple
  TIFF or BigTIFF files. The UTF-8 encoded OME-XML metadata found in the
  ImageDescription tag of the first IFD defines the position of TIFF IFDs in
  the high dimensional image data. Tifffile can read OME-TIFF files (except
  multi-file pyramidal) and write NumPy arrays to single-file OME-TIFF.
- **Micro-Manager NDTiff** stores multi-dimensional image data in one
  or more classic TIFF files. Metadata contained in a separate NDTiff.index
  binary file defines the position of the TIFF IFDs in the image array.
  Each TIFF file also contains metadata in a non-TIFF binary structure at
  offset 8. Downsampled image data of pyramidal datasets are stored in
  separate folders. Tifffile can read NDTiff files. Version 0 and 1 series,
  tiling, stitching, and multi-resolution pyramids are not supported.
- **Micro-Manager MMStack** stores 6-dimensional image data in one or more
  classic TIFF files. Metadata contained in non-TIFF binary structures and
  JSON strings define the image stack dimensions and the position of the image
  frame data in the file and the image stack. The TIFF structures and metadata
  are often corrupted or wrong. Tifffile can read MMStack files.
- **Carl Zeiss LSM** files store all IFDs below 4 GB and wrap around 32-bit
  StripOffsets pointing to image data above 4 GB. The StripOffsets of each
  series and position require separate unwrapping. The StripByteCounts tag
  contains the number of bytes for the uncompressed data. Tifffile can read
  LSM files of any size.
- **MetaMorph Stack, STK** files contain additional image planes stored
  contiguously after the image data of the first page. The total number of
  planes is equal to the count of the UIC2tag. Tifffile can read STK files.
- **ZIF**, the Zoomable Image File format, is a subspecification of BigTIFF
  with SGI's ImageDepth extension and additional compression schemes.
  Only little-endian, tiled, interleaved, 8-bit per sample images with
  JPEG, PNG, JPEG XR, and JPEG 2000 compression are allowed. Tifffile can
  read and write ZIF files.
- **Hamamatsu NDPI** files use some 64-bit offsets in the file header, IFD,
  and tag structures. Single, LONG typed tag values can exceed 32-bit.
  The high bytes of 64-bit tag values and offsets are stored after IFD
  structures. Tifffile can read NDPI files > 4 GB.
  JPEG compressed segments with dimensions >65530 or missing restart markers
  cannot be decoded with common JPEG libraries. Tifffile works around this
  limitation by separately decoding the MCUs between restart markers, which
  performs poorly. BitsPerSample, SamplesPerPixel, and
  PhotometricInterpretation tags may contain wrong values, which can be
  corrected using the value of tag 65441.
- **Philips TIFF** slides store wrong ImageWidth and ImageLength tag values
  for tiled pages. The values can be corrected using the DICOM_PIXEL_SPACING
  attributes of the XML formatted description of the first page. Tifffile can
  read Philips slides.
- **Ventana/Roche BIF** slides store tiles and metadata in a BigTIFF container.
  Tiles may overlap and require stitching based on the TileJointInfo elements
  in the XMP tag. Volumetric scans are stored using the ImageDepth extension.
  Tifffile can read BIF and decode individual tiles but does not perform
  stitching.
- **ScanImage** optionally allows corrupted non-BigTIFF files > 2 GB.
  The values of StripOffsets and StripByteCounts can be recovered using the
  constant differences of the offsets of IFD and tag values throughout the
  file. Tifffile can read such files if the image data are stored contiguously
  in each page.
- **GeoTIFF sparse** files allow strip or tile offsets and byte counts to be 0.
  Such segments are implicitly set to 0 or the NODATA value on reading.
  Tifffile can read GeoTIFF sparse files.
- **Tifffile shaped** files store the array shape and user-provided metadata
  of multi-dimensional image series in JSON format in the ImageDescription tag
  of the first page of the series. The format allows for multiple series,
  SubIFDs, sparse segments with zero offset and byte count, and truncated
  series, where only the first page of a series is present, and the image data
  are stored contiguously. No other software besides Tifffile supports the
  truncated format.

Other libraries for reading, writing, inspecting, or manipulating scientific
TIFF files from Python are
`aicsimageio <https://pypi.org/project/aicsimageio>`_,
`apeer-ometiff-library
<https://github.com/apeer-micro/apeer-ometiff-library>`_,
`bigtiff <https://pypi.org/project/bigtiff>`_,
`fabio.TiffIO <https://github.com/silx-kit/fabio>`_,
`GDAL <https://github.com/OSGeo/gdal/>`_,
`imread <https://github.com/luispedro/imread>`_,
`large_image <https://github.com/girder/large_image>`_,
`openslide-python <https://github.com/openslide/openslide-python>`_,
`opentile <https://github.com/imi-bigpicture/opentile>`_,
`pylibtiff <https://github.com/pearu/pylibtiff>`_,
`pylsm <https://launchpad.net/pylsm>`_,
`pymimage <https://github.com/ardoi/pymimage>`_,
`python-bioformats <https://github.com/CellProfiler/python-bioformats>`_,
`pytiff <https://github.com/FZJ-INM1-BDA/pytiff>`_,
`scanimagetiffreader-python
<https://gitlab.com/vidriotech/scanimagetiffreader-python>`_,
`SimpleITK <https://github.com/SimpleITK/SimpleITK>`_,
`slideio <https://gitlab.com/bioslide/slideio>`_,
`tiffslide <https://github.com/bayer-science-for-a-better-life/tiffslide>`_,
`tifftools <https://github.com/DigitalSlideArchive/tifftools>`_,
`tyf <https://github.com/Moustikitos/tyf>`_,
`xtiff <https://github.com/BodenmillerGroup/xtiff>`_, and
`ndtiff <https://github.com/micro-manager/NDTiffStorage>`_.

References
----------

- TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
  https://www.adobe.io/open/standards/TIFF.html
- TIFF File Format FAQ. https://www.awaresystems.be/imaging/tiff/faq.html
- The BigTIFF File Format.
  https://www.awaresystems.be/imaging/tiff/bigtiff.html
- MetaMorph Stack (STK) Image File Format.
  http://mdc.custhelp.com/app/answers/detail/a_id/18862
- Image File Format Description LSM 5/7 Release 6.0 (ZEN 2010).
  Carl Zeiss MicroImaging GmbH. BioSciences. May 10, 2011
- The OME-TIFF format.
  https://docs.openmicroscopy.org/ome-model/latest/
- UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
  http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
- Micro-Manager File Formats.
  https://micro-manager.org/wiki/Micro-Manager_File_Formats
- ScanImage BigTiff Specification.
  https://docs.scanimage.org/Appendix/ScanImage+BigTiff+Specification.html
- ZIF, the Zoomable Image File format. https://zif.photo/
- GeoTIFF File Format https://gdal.org/drivers/raster/gtiff.html
- Cloud optimized GeoTIFF.
  https://github.com/cogeotiff/cog-spec/blob/master/spec.md
- Tags for TIFF and Related Specifications. Digital Preservation.
  https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml
- CIPA DC-008-2016: Exchangeable image file format for digital still cameras:
  Exif Version 2.31.
  http://www.cipa.jp/std/documents/e/DC-008-Translation-2016-E.pdf
- The EER (Electron Event Representation) file format.
  https://github.com/fei-company/EerReaderLib
- Digital Negative (DNG) Specification. Version 1.5.0.0, June 2012.
  https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/
  dng_spec_1.5.0.0.pdf
- Roche Digital Pathology. BIF image file format for digital pathology.
  https://diagnostics.roche.com/content/dam/diagnostics/Blueprint/en/pdf/rmd/
  Roche-Digital-Pathology-BIF-Whitepaper.pdf
- Astro-TIFF specification. https://astro-tiff.sourceforge.io/
- Aperio Technologies, Inc. Digital Slides and Third-Party Data Interchange.
  Aperio_Digital_Slides_and_Third-party_data_interchange.pdf
- PerkinElmer image format.
  https://downloads.openmicroscopy.org/images/Vectra-QPTIFF/perkinelmer/
  PKI_Image%20Format.docx
- NDTiffStorage. https://github.com/micro-manager/NDTiffStorage

Examples
--------

Write a NumPy array to a single-page RGB TIFF file:

>>> data = numpy.random.randint(0, 255, (256, 256, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb')

Read the image from the TIFF file as NumPy array:

>>> image = imread('temp.tif')
>>> image.shape
(256, 256, 3)

Use the `photometric` and `planarconfig` arguments to write a 3x3x3 NumPy
array to an interleaved RGB, a planar RGB, or a 3-page grayscale TIFF:

>>> data = numpy.random.randint(0, 255, (3, 3, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb')
>>> imwrite('temp.tif', data, photometric='rgb', planarconfig='separate')
>>> imwrite('temp.tif', data, photometric='minisblack')

Use the `extrasamples` argument to specify how extra components are
interpreted, for example, for an RGBA image with unassociated alpha channel:

>>> data = numpy.random.randint(0, 255, (256, 256, 4), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb', extrasamples=['unassalpha'])

Write a 3-dimensional NumPy array to a multi-page, 16-bit grayscale TIFF file:

>>> data = numpy.random.randint(0, 2**12, (64, 301, 219), 'uint16')
>>> imwrite('temp.tif', data, photometric='minisblack')

Read the whole image stack from the multi-page TIFF file as NumPy array:

>>> image_stack = imread('temp.tif')
>>> image_stack.shape
(64, 301, 219)
>>> image_stack.dtype
dtype('uint16')

Read the image from the first page in the TIFF file as NumPy array:

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
any image data:

>>> tif = TiffFile('temp.tif')
>>> len(tif.pages)  # number of pages in the file
64
>>> page = tif.pages[0]  # get shape and dtype of image in first page
>>> page.shape
(301, 219)
>>> page.dtype
dtype('uint16')
>>> page.axes
'YX'
>>> series = tif.series[0]  # get shape and dtype of first image series
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
<DATATYPE.RATIONAL: 5>

Iterate over all tags in the TIFF file:

>>> with TiffFile('temp.tif') as tif:
...     for page in tif.pages:
...         for tag in page.tags:
...             tag_name, tag_value = tag.name, tag.value

Overwrite the value of an existing tag, for example, XResolution:

>>> with TiffFile('temp.tif', mode='r+') as tif:
...     _ = tif.pages[0].tags['XResolution'].overwrite((96000, 1000))

Write a 5-dimensional floating-point array using BigTIFF format, separate
color components, tiling, Zlib compression level 8, horizontal differencing
predictor, and additional metadata:

>>> data = numpy.random.rand(2, 5, 3, 301, 219).astype('float32')
>>> imwrite(
...     'temp.tif',
...     data,
...     bigtiff=True,
...     photometric='rgb',
...     planarconfig='separate',
...     tile=(32, 32),
...     compression='zlib',
...     compressionargs={'level': 8},
...     predictor=True,
...     metadata={'axes': 'TZCYX'}
... )

Write a 10 fps time series of volumes with xyz voxel size 2.6755x2.6755x3.9474
micron^3 to an ImageJ hyperstack formatted TIFF file:

>>> volume = numpy.random.randn(6, 57, 256, 256).astype('float32')
>>> image_labels = [f'{i}' for i in range(volume.shape[0] * volume.shape[1])]
>>> imwrite(
...     'temp.tif',
...     volume,
...     imagej=True,
...     resolution=(1./2.6755, 1./2.6755),
...     metadata={
...         'spacing': 3.947368,
...         'unit': 'um',
...         'finterval': 1/10,
...         'fps': 10.0,
...         'axes': 'TZYX',
...         'Labels': image_labels,
...     }
... )

Read the volume and metadata from the ImageJ hyperstack file:

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

Memory-map the contiguous image data in the ImageJ hyperstack file:

>>> memmap_volume = memmap('temp.tif')
>>> memmap_volume.shape
(6, 57, 256, 256)
>>> del memmap_volume

Create a TIFF file containing an empty image and write to the memory-mapped
NumPy array (note: this does not work with compression or tiling):

>>> memmap_image = memmap(
...     'temp.tif',
...     shape=(256, 256, 3),
...     dtype='float32',
...     photometric='rgb'
... )
>>> type(memmap_image)
<class 'numpy.memmap'>
>>> memmap_image[255, 255, 1] = 1.0
>>> memmap_image.flush()
>>> del memmap_image

Write two NumPy arrays to a multi-series TIFF file (note: other TIFF readers
will not recognize the two series; use the OME-TIFF format for better
interoperability):

>>> series0 = numpy.random.randint(0, 255, (32, 32, 3), 'uint8')
>>> series1 = numpy.random.randint(0, 255, (4, 256, 256), 'uint16')
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

Append an image series to the existing TIFF file (note: this does not work
with ImageJ hyperstack or OME-TIFF files):

>>> data = numpy.random.randint(0, 255, (301, 219, 3), 'uint8')
>>> imwrite('temp.tif', data, photometric='rgb', append=True)

Create a TIFF file from a generator of tiles:

>>> data = numpy.random.randint(0, 2**12, (31, 33, 3), 'uint16')
>>> def tiles(data, tileshape):
...     for y in range(0, data.shape[0], tileshape[0]):
...         for x in range(0, data.shape[1], tileshape[1]):
...             yield data[y : y + tileshape[0], x : x + tileshape[1]]
>>> imwrite(
...     'temp.tif',
...     tiles(data, (16, 16)),
...     tile=(16, 16),
...     shape=data.shape,
...     dtype=data.dtype,
...     photometric='rgb'
... )

Write a multi-dimensional, multi-resolution (pyramidal), multi-series OME-TIFF
file with metadata. Sub-resolution images are written to SubIFDs. Write a
thumbnail image as a separate image series:

>>> data = numpy.random.randint(0, 255, (8, 2, 512, 512, 3), 'uint8')
>>> subresolutions = 2
>>> pixelsize = 0.29  # micrometer
>>> with TiffWriter('temp.ome.tif', bigtiff=True) as tif:
...     metadata={
...         'axes': 'TCYXS',
...         'SignificantBits': 10,
...         'TimeIncrement': 0.1,
...         'TimeIncrementUnit': 's',
...         'PhysicalSizeX': pixelsize,
...         'PhysicalSizeXUnit': 'µm',
...         'PhysicalSizeY': pixelsize,
...         'PhysicalSizeYUnit': 'µm',
...         'Channel': {'Name': ['Channel 1', 'Channel 2']},
...         'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['µm'] * 16}
...     }
...     options = dict(
...         photometric='rgb',
...         tile=(128, 128),
...         compression='jpeg',
...         resolutionunit='CENTIMETER'
...     )
...     tif.write(
...         data,
...         subifds=subresolutions,
...         resolution=(1e4 / pixelsize, 1e4 / pixelsize),
...         metadata=metadata,
...         **options
...     )
...     # write pyramid levels to the two subifds
...     # in production use resampling to generate sub-resolution images
...     for level in range(subresolutions):
...         mag = 2**(level + 1)
...         tif.write(
...             data[..., ::mag, ::mag, :],
...             subfiletype=1,
...             resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
...             **options
...         )
...     # add a thumbnail image as a separate series
...     # it is recognized by QuPath as an associated image
...     thumbnail = (data[0, 0, ::8, ::8] >> 2).astype('uint8')
...     tif.write(thumbnail, metadata={'Name': 'thumbnail'})

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

Use Zarr to read parts of the tiled, pyramidal images in the TIFF file:

>>> import zarr
>>> store = imread('temp.ome.tif', aszarr=True)
>>> z = zarr.open(store, mode='r')
>>> z
<zarr.hierarchy.Group '/' read-only>
>>> z[0]  # base layer
<zarr.core.Array '/0' (8, 2, 512, 512, 3) uint8 read-only>
>>> z[0][2, 0, 128:384, 256:].shape  # read a tile from the base layer
(256, 256, 3)
>>> store.close()

Load the base layer from the Zarr store as a dask array:

>>> import dask.array
>>> store = imread('temp.ome.tif', aszarr=True)
>>> dask.array.from_zarr(store, 0)
dask.array<...shape=(8, 2, 512, 512, 3)...chunksize=(1, 1, 128, 128, 3)...
>>> store.close()

Write the Zarr store to a fsspec ReferenceFileSystem in JSON format:

>>> store = imread('temp.ome.tif', aszarr=True)
>>> store.write_fsspec('temp.ome.tif.json', url='file://')
>>> store.close()

Open the fsspec ReferenceFileSystem as a Zarr group:

>>> import fsspec
>>> import imagecodecs.numcodecs
>>> imagecodecs.numcodecs.register_codecs()
>>> mapper = fsspec.get_mapper(
...     'reference://', fo='temp.ome.tif.json', target_protocol='file'
... )
>>> z = zarr.open(mapper, mode='r')
>>> z
<zarr.hierarchy.Group '/' read-only>

Create an OME-TIFF file containing an empty, tiled image series and write
to it via the Zarr interface (note: this does not work with compression):

>>> imwrite(
...     'temp.ome.tif',
...     shape=(8, 800, 600),
...     dtype='uint16',
...     photometric='minisblack',
...     tile=(128, 128),
...     metadata={'axes': 'CYX'}
... )
>>> store = imread('temp.ome.tif', mode='r+', aszarr=True)
>>> z = zarr.open(store, mode='r+')
>>> z
<zarr.core.Array (8, 800, 600) uint16>
>>> z[3, 100:200, 200:300:2] = 1024
>>> store.close()

Read images from a sequence of TIFF files as NumPy array:

>>> imwrite('temp_C001T001.tif', numpy.random.rand(64, 64))
>>> imwrite('temp_C001T002.tif', numpy.random.rand(64, 64))
>>> image_sequence = imread(['temp_C001T001.tif', 'temp_C001T002.tif'])
>>> image_sequence.shape
(2, 64, 64)
>>> image_sequence.dtype
dtype('float64')

Read an image stack from a series of TIFF files with a file name pattern
as NumPy or Zarr arrays:

>>> image_sequence = TiffSequence(
...     'temp_C0*.tif', pattern=r'_(C)(\d+)(T)(\d+)'
... )
>>> image_sequence.shape
(1, 2)
>>> image_sequence.axes
'CT'
>>> data = image_sequence.asarray()
>>> data.shape
(1, 2, 64, 64)
>>> store = image_sequence.aszarr()
>>> zarr.open(store, mode='r')
<zarr.core.Array (1, 2, 64, 64) float64 read-only>
>>> image_sequence.close()

Write the Zarr store to a fsspec ReferenceFileSystem in JSON format:

>>> store = image_sequence.aszarr()
>>> store.write_fsspec('temp.json', url='file://')

Open the fsspec ReferenceFileSystem as a Zarr array:

>>> import fsspec
>>> import tifffile.numcodecs
>>> tifffile.numcodecs.register_codec()
>>> mapper = fsspec.get_mapper(
...     'reference://', fo='temp.json', target_protocol='file'
... )
>>> zarr.open(mapper, mode='r')
<zarr.core.Array (1, 2, 64, 64) float64 read-only>

Inspect the TIFF file from the command line::

    $ python -m tifffile temp.ome.tif
