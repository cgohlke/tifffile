Revisions
---------

2025.10.16

- Pass 5124 tests.
- Add option to decode EER super-resolution sub-pixels (breaking, #313).
- Parse EER metadata to dict (breaking).

2025.10.4

- Fix parsing SVS description ending with "|".

2025.9.30

- Fix reading NDTiff series with unordered axes in index (#311).

2025.9.20

- Derive TiffFileError from ValueError.
- Natural-sort files in glob pattern passed to imread by default (breaking).
- Fix optional sorting of list of files passed to FileSequence and imread.

2025.9.9

- Consolidate Nuvu camera metadata.

2025.8.28

- Support DNG DCP files (#306).

2025.6.11

- Fix reading images with dimension length 1 through Zarr (#303).

2025.6.1

- Add experimental option to write iterator of bytes and bytecounts (#301).

2025.5.26

- Use threads in Zarr stores.

2025.5.24

- Fix incorrect tags created by Philips DP v1.1 (#299).
- Make Zarr stores partially listable.

2025.5.21

- Move Zarr stores to tifffile.zarr namespace (breaking).
- Require Zarr 3 for Zarr stores and remove support for Zarr 2 (breaking).
- Drop support for Python 3.10.

2025.5.10

- Raise ValueError when using Zarr 3 (#296).
- Fall back to compression.zstd on Python >= 3.14 if no imagecodecs.
- Remove doctest command line option.
- Support Python 3.14.

2025.3.30

- Fix for imagecodecs 2025.3.30.

2025.3.13

- Change bytes2str to decode only up to first NULL character (breaking).
- Remove stripnull function calls to reduce overhead (#285).
- Deprecate stripnull function.

2025.2.18

- Fix julian_datetime milliseconds (#283).
- Remove deprecated dtype arguments from imread and FileSequence (breaking).
- Remove deprecated imsave and TiffWriter.save function/method (breaking).
- Remove deprecated option to pass multiple values to compression (breaking).
- Remove deprecated option to pass unit to resolution (breaking).
- Remove deprecated enums from TIFF namespace (breaking).
- Remove deprecated lazyattr and squeeze_axes functions (breaking).

2025.1.10

- Improve type hints.
- Deprecate Python 3.10.

2024.12.12

- Read PlaneProperty from STK UIC1Tag (#280).
- Allow 'None' as alias for COMPRESSION.NONE and PREDICTOR.NONE (#274).
- Zarr 3 is not supported (#272).

2024.9.20

- Fix writing colormap to ImageJ files (breaking).
- Improve typing.
- Drop support for Python 3.9.

2024.8.30

- Support writing OME Dataset and some StructuredAnnotations elements.

2024.8.28

- Fix LSM scan types and dimension orders (#269, breaking).
- Use IO[bytes] instead of BinaryIO for typing (#268).

2024.8.24

- Do not remove trailing length-1 dimension writing non-shaped file (breaking).
- Fix writing OME-TIFF with certain modulo axes orders.
- Make imshow NaN aware.

2024.8.10

- Relax bitspersample check for JPEG, JPEG2K, and JPEGXL compression (#265).

2024.7.24

- Fix reading contiguous multi-page series via Zarr store (#67).

2024.7.21

- Fix integer overflow in product function caused by numpy types.
- Allow tag reader functions to fail.

2024.7.2

- Enable memmap to create empty files with non-native byte order.
- Deprecate Python 3.9, support Python 3.13.

2024.6.18

- Ensure TiffPage.nodata is castable to dtype (breaking, #260).
- Support Argos AVS slides.

2024.5.22

- Derive TiffPages, TiffPageSeries, FileSequence, StoredShape from Sequence.
- Truncate circular IFD chain, do not raise TiffFileError (breaking).
- Deprecate access to TiffPages.pages and FileSequence.files.
- Enable DeprecationWarning for enums in TIFF namespace.
- Remove some deprecated code (breaking).
- Add iccprofile property to TiffPage and parameter to TiffWriter.write.
- Do not detect VSI as SIS format.
- Limit length of logged exception messages.
- Fix docstring examples not correctly rendered on GitHub (#254, #255).

2024.5.10

- Support reading JPEGXL compression in DNG 1.7.
- Read invalid TIFF created by IDEAS software.

2024.5.3

- Fix reading incompletely written LSM.
- Fix reading Philips DP with extra rows of tiles (#253, breaking).

2024.4.24

- Fix compatibility issue with numpy 2 (#252).

2024.4.18

- Fix write_fsspec when last row of tiles is missing in Philips slide (#249).
- Add option not to quote file names in write_fsspec.
- Allow compressing bilevel images with deflate, LZMA, and Zstd.

2024.2.12

- Deprecate dtype, add chunkdtype parameter in FileSequence.asarray.
- Add imreadargs parameters passed to FileSequence.imread.

2024.1.30

- Fix compatibility issue with numpy 2 (#238).
- Enable DeprecationWarning for tuple compression argument.
- Parse sequence of numbers in xml2dict.

2023.12.9

- Read 32-bit Indica Labs TIFF as float32.
- Fix UnboundLocalError reading big LSM files without time axis.
- Use os.sched_getaffinity, if available, to get the number of CPUs (#231).
- Limit the number of default worker threads to 32.

2023.9.26

- Lazily convert dask array to ndarray when writing.
- Allow specifying buffersize for reading and writing.
- Fix IndexError reading some corrupted files with ZarrTiffStore (#227).

2023.9.18

- Raise exception when writing non-volume data with volumetric tiles (#225).
- Improve multi-threaded writing of compressed multi-page files.
- Fix fsspec reference for big-endian files with predictors.

2023.8.30

- Support exclusive file creation mode (#221, #223).

2023.8.25

- Verify shaped metadata is compatible with page shape.
- Support out parameter when returning selection from imread (#222).

2023.8.12

- Support decompressing EER frames.
- Facilitate filtering logged warnings (#216).
- Read more tags from UIC1Tag (#217).
- Fix premature closing of files in main (#218).
- Don't force matplotlib backend to tkagg in main (#219).
- Add py.typed marker.
- Drop support for imagecodecs < 2023.3.16.

2023.7.18

- Limit threading via TIFFFILE_NUM_THREADS environment variable (#215).
- Remove maxworkers parameter from tiff2fsspec (breaking).

2023.7.10

- Increase default strip size to 256 KB when writing with compression.
- Fix ZarrTiffStore with non-default chunkmode.

2023.7.4

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
- Allow writing uncompressed image data beyond 4 GB in classic TIFF.
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

- Allow writing NewSubfileType=0 (#132).
- Support writing iterators of strip or tile bytes.
- Convert iterables (not iterators) to NumPy arrays when writing (breaking).
- Explicitly specify optional keyword parameters for imread and imwrite.
- Return number of written bytes from FileHandle write functions.

2022.4.28

- Add option to specify fsspec version 1 URL template name (#131).
- Ignore invalid dates in UIC tags (#129).
- Fix zlib_encode and lzma_encode to work with non-contiguous arrays (#128).
- Fix delta_encode to preserve byteorder of ndarrays.
- Move Imagecodecs fallback functions to private module and add tests.

2022.4.26

- Fix AttributeError in TiffFile.shaped_metadata (#127).
- Fix TiffTag.overwrite with pre-packed binary value.
- Write sparse TIFF if tile iterator contains None.
- Raise ValueError when writing photometric mode with too few samples.
- Improve test coverage.

2022.4.22

- Add type hints for Python 3.10 (WIP).
- Fix Mypy errors (breaking).
- Mark many parameters positional-only or keyword-only (breaking).
- Remove deprecated pages parameter from imread (breaking).
- Remove deprecated compress and ijmetadata write parameters (breaking).
- Remove deprecated fastij and movie parameters from TiffFile (breaking).
- Remove deprecated multifile parameters from TiffFile (breaking).
- Remove deprecated tif parameter from TiffTag.overwrite (breaking).
- Remove deprecated file parameter from FileSequence.asarray (breaking).
- Remove option to pass imread class to FileSequence (breaking).
- Remove optional parameters from __str__ functions (breaking).
- Rename TiffPageSeries.offset to dataoffset (breaking)
- Change TiffPage.pages to None if no SubIFDs are present (breaking).
- Change TiffPage.index to int (breaking).
- Change TiffPage.is_contiguous, is_imagej, and is_shaped to bool (breaking).
- Add TiffPage imagej_description and shaped_description properties.
- Add TiffFormat abstract base class.
- Deprecate lazyattr and use functools.cached_property instead (breaking).
- Julian_datetime raises ValueError for dates before year 1 (breaking).
- Regressed import time due to typing.

2022.4.8

- Add _ARRAY_DIMENSIONS attributes to ZarrTiffStore.
- Allow C instead of S axis when writing OME-TIFF.
- Fix writing OME-TIFF with separate samples.
- Fix reading unsqueezed pyramidal OME-TIFF series.

2022.3.25

- Fix another ValueError using ZarrStore with zarr >= 2.11.0 (tiffslide #25).
- Add parser for Hamamatsu streak metadata.
- Improve hexdump.

2022.3.16

- Use multi-threading to compress strips and tiles.
- Raise TiffFileError when reading corrupted strips and tiles (#122).
- Fix ScanImage single channel count (#121).
- Add parser for AstroTIFF FITS metadata.

2022.2.9

- Fix ValueError using multiscale ZarrStore with zarr >= 2.11.0.
- Raise KeyError if ZarrStore does not contain key.
- Limit number of warnings for missing files in multifile series.
- Allow saving colormap to 32-bit ImageJ files (#115).

2022.2.2

- Fix TypeError when second ImageDescription tag contains non-ASCII (#112).
- Fix parsing IJMetadata with many IJMetadataByteCounts (#111).
- Detect MicroManager NDTiffv2 header (not tested).
- Remove cache from ZarrFileSequenceStore (use zarr.LRUStoreCache).
- Raise limit on maximum number of pages.
- Use J2K format when encoding JPEG2000 segments.
- Formally deprecate imsave and TiffWriter.save.
- Drop support for Python 3.7 and NumPy < 1.19 (NEP29).

2021.11.2

- Lazy-load non-essential tag values (breaking).
- Warn when reading from closed file.
- Support ImageJ prop metadata type (#103).
- Support writing indexed ImageJ format.
- Fix multi-threaded access of multi-page Zarr stores with chunkmode 2.
- Raise error if truncate is used with compression, packints, or tile.
- Read STK metadata without UIC2tag.
- Improve log and warning messages (WIP).
- Improve string representation of large tag values.

2021.10.12

- Revert renaming of file parameter in FileSequence.asarray (breaking).
- Deprecate file parameter in FileSequence.asarray.

2021.10.10

- Disallow letters as indices in FileSequence; use categories (breaking).
- Do not warn about missing files in FileSequence; use files_missing property.
- Support predictors in ZarrTiffStore.write_fsspec.
- Add option to specify Zarr group name in write_fsspec.
- Add option to specify categories for FileSequence patterns (#76).
- Add option to specify chunk shape and dtype for ZarrFileSequenceStore.
- Add option to tile ZarrFileSequenceStore and FileSequence.asarray.
- Add option to pass additional zattrs to Zarr stores.
- Detect Roche BIF files.

2021.8.30

- Fix horizontal differencing with non-native byte order.
- Fix multi-threaded access of memory-mappable, multi-page Zarr stores (#67).

2021.8.8

- Fix tag offset and valueoffset for NDPI > 4 GB (#96).

2021.7.30

- Deprecate first parameter to TiffTag.overwrite (no longer required).
- TiffTag init API change (breaking).
- Detect Ventana BIF series and warn that tiles are not stitched.
- Enable reading PreviewImage from RAW formats (#93, #94).
- Work around numpy.ndarray.tofile is very slow for non-contiguous arrays.
- Fix issues with PackBits compression (requires imagecodecs 2021.7.30).

2021.7.2

- Decode complex integer images found in SAR GeoTIFF.
- Support reading NDPI with JPEG-XR compression.
- Deprecate TiffWriter RGB auto-detection, except for RGB24/48 and RGBA32/64.

2021.6.14

- Set stacklevel for deprecation warnings (#89).
- Fix svs_description_metadata for SVS with double header (#88, breaking).
- Fix reading JPEG compressed CMYK images.
- Support ALT_JPEG and JPEG_2000_LOSSY compression found in Bio-Formats.
- Log warning if TiffWriter auto-detects RGB mode (specify photometric).

2021.6.6

- Fix TIFF.COMPESSOR typo (#85).
- Round resolution numbers that do not fit in 64-bit rationals (#81).
- Add support for JPEG XL compression.
- Add Numcodecs compatible TIFF codec.
- Rename ZarrFileStore to ZarrFileSequenceStore (breaking).
- Add method to export fsspec ReferenceFileSystem from ZarrFileStore.
- Fix fsspec ReferenceFileSystem v1 for multifile series.
- Fix creating OME-TIFF with micron character in OME-XML.

2021.4.8

- Fix reading OJPEG with wrong photometric or samplesperpixel tags (#75).
- Fix fsspec ReferenceFileSystem v1 and JPEG compression.
- Use TiffTagRegistry for NDPI_TAGS, EXIF_TAGS, GPS_TAGS, IOP_TAGS constants.
- Make TIFF.GEO_KEYS an Enum (breaking).

2021.3.31

- Use JPEG restart markers as tile offsets in NDPI.
- Support version 1 and more codecs in fsspec ReferenceFileSystem (untested).

2021.3.17

- Fix regression reading multi-file OME-TIFF with missing files (#72).
- Fix fsspec ReferenceFileSystem with non-native byte order (#56).

2021.3.16

- TIFF is no longer a defended trademark.
- Add method to export fsspec ReferenceFileSystem from ZarrTiffStore (#56).

2021.3.5

- Preliminary support for EER format (#68).
- Do not warn about unknown compression (#68).

2021.3.4

- Fix reading multi-file, multi-series OME-TIFF (#67).
- Detect ScanImage 2021 files (#46).
- Shape new version ScanImage series according to metadata (breaking).
- Remove Description key from TiffFile.scanimage_metadata dict (breaking).
- Also return ScanImage version from read_scanimage_metadata (breaking).
- Fix docstrings.

2021.2.26

- Squeeze axes of LSM series by default (breaking).
- Add option to preserve single dimensions when reading from series (WIP).
- Do not allow appending to OME-TIFF files.
- Fix reading STK files without name attribute in metadata.
- Make TIFF constants multi-thread safe and pickleable (#64).
- Add detection of NDTiffStorage MajorVersion to read_micromanager_metadata.
- Support ScanImage v4 files in read_scanimage_metadata.

2021.2.1

- Fix multi-threaded access of ZarrTiffStores using same TiffFile instance.
- Use fallback zlib and lzma codecs with imagecodecs lite builds.
- Open Olympus and Panasonic RAW files for parsing, albeit not supported.
- Support X2 and X4 differencing found in DNG.
- Support reading JPEG_LOSSY compression found in DNG.

2021.1.14

- Try ImageJ series if OME series fails (#54)
- Add option to use pages as chunks in ZarrFileStore (experimental).
- Fix reading from file objects with no readinto function.

2021.1.11

- Fix test errors on PyPy.
- Fix decoding bitorder with imagecodecs >= 2021.1.11.

2021.1.8

- Decode float24 using imagecodecs >= 2021.1.8.
- Consolidate reading of segments if possible.

2020.12.8

- Fix corrupted ImageDescription in multi shaped series if buffer too small.
- Fix libtiff warning that ImageDescription contains null byte in value.
- Fix reading invalid files using JPEG compression with palette colorspace.

2020.12.4

- Fix reading some JPEG compressed CFA images.
- Make index of SubIFDs a tuple.
- Pass through FileSequence.imread arguments in imread.
- Do not apply regex flags to FileSequence axes patterns (breaking).

2020.11.26

- Add option to pass axes metadata to ImageJ writer.
- Pad incomplete tiles passed to TiffWriter.write (#38).
- Split TiffTag constructor (breaking).
- Change TiffTag.dtype to TIFF.DATATYPES (breaking).
- Add TiffTag.overwrite method.
- Add script to change ImageDescription in files.
- Add TiffWriter.overwrite_description method (WIP).

2020.11.18

- Support writing SEPARATED color space (#37).
- Use imagecodecs.deflate codec if available.
- Fix SCN and NDPI series with Z dimensions.
- Add TiffReader alias for TiffFile.
- TiffPage.is_volumetric returns True if ImageDepth > 1.
- Zarr store getitem returns NumPy arrays instead of bytes.

2020.10.1

- Formally deprecate unused TiffFile parameters (scikit-image #4996).

2020.9.30

- Allow passing additional arguments to compression codecs.
- Deprecate TiffWriter.save method (use TiffWriter.write).
- Deprecate TiffWriter.save compress parameter (use compression).
- Remove multifile parameter from TiffFile (breaking).
- Pass all is_flag arguments from imread to TiffFile.
- Do not byte-swap JPEG2000, WEBP, PNG, JPEGXR segments in TiffPage.decode.

2020.9.29

- Fix reading files produced by ScanImage > 2015 (#29).

2020.9.28

- Derive ZarrStore from MutableMapping.
- Support zero shape ZarrTiffStore.
- Fix ZarrFileStore with non-TIFF files.
- Fix ZarrFileStore with missing files.
- Cache one chunk in ZarrFileStore.
- Keep track of already opened files in FileCache.
- Change parse_filenames function to return zero-based indices.
- Remove reopen parameter from asarray (breaking).
- Rename FileSequence.fromfile to imread (breaking).

2020.9.22

- Add experimental Zarr storage interface (WIP).
- Remove unused first dimension from TiffPage.shaped (breaking).
- Move reading of STK planes to series interface (breaking).
- Always use virtual frames for ScanImage files.
- Use DimensionOrder to determine axes order in OmeXml.
- Enable writing striped volumetric images.
- Keep complete dataoffsets and databytecounts for TiffFrames.
- Return full size tiles from Tiffpage.segments.
- Rename TiffPage.is_sgi property to is_volumetric (breaking).
- Rename TiffPageSeries.is_pyramid to is_pyramidal (breaking).
- Fix TypeError when passing jpegtables to non-JPEG decode method (#25).

2020.9.3

- Do not write contiguous series by default (breaking).
- Allow writing to SubIFDs (WIP).
- Fix writing F-contiguous NumPy arrays (#24).

2020.8.25

- Do not convert EPICS timeStamp to datetime object.
- Read incompletely written Micro-Manager image file stack header (#23).
- Remove tag 51123 values from TiffFile.micromanager_metadata (breaking).

2020.8.13

- Use tifffile metadata over OME and ImageJ for TiffFile.series (breaking).
- Fix writing iterable of pages with compression (#20).
- Expand error checking of TiffWriter data, dtype, shape, and tile arguments.

2020.7.24

- Parse nested OmeXml metadata argument (WIP).
- Do not lazy load TiffFrame JPEGTables.
- Fix conditionally skipping some tests.

2020.7.22

- Do not auto-enable OME-TIFF if description is passed to TiffWriter.save.
- Raise error writing empty bilevel or tiled images.
- Allow writing tiled bilevel images.
- Allow writing multi-page TIFF from iterable of single page images (WIP).
- Add function to validate OME-XML.
- Correct Philips slide width and length.

2020.7.17

- Initial support for writing OME-TIFF (WIP).
- Return samples as separate dimension in OME series (breaking).
- Fix modulo dimensions for multiple OME series.
- Fix some test errors on big endian systems (#18).
- Fix BytesWarning.
- Allow passing TIFF.PREDICTOR values to TiffWriter.save.

2020.7.4

- Deprecate support for Python 3.6 (NEP 29).
- Move pyramidal subresolution series to TiffPageSeries.levels (breaking).
- Add parser for SVS, SCN, NDPI, and QPI pyramidal series.
- Read single-file OME-TIFF pyramids.
- Read NDPI files > 4 GB (#15).
- Include SubIFDs in generic series.
- Preliminary support for writing packed integer arrays (#11, WIP).
- Read more LSM info subrecords.
- Fix missing ReferenceBlackWhite tag for YCbCr photometrics.
- Fix reading lossless JPEG compressed DNG files.

2020.6.3

- Support os.PathLike file names (#9).

2020.5.30

- Re-add pure Python PackBits decoder.

2020.5.25

- Make imagecodecs an optional dependency again.
- Disable multi-threaded decoding of small LZW compressed segments.
- Fix caching of TiffPage.decode method.
- Fix xml.etree.cElementTree ImportError on Python 3.9.
- Fix tostring DeprecationWarning.

2020.5.11

- Fix reading ImageJ grayscale mode RGB images (#6).
- Remove napari reader plugin.

2020.5.7

- Add napari reader plugin (tentative).
- Fix writing single tiles larger than image data (#3).
- Always store ExtraSamples values in tuple (breaking).

2020.5.5

- Allow writing tiled TIFF from iterable of tiles (WIP).
- Add method to iterate over decoded segments of TiffPage (WIP).
- Pass chunks of segments to ThreadPoolExecutor.map to reduce memory usage.
- Fix reading invalid files with too many strips.
- Fix writing over-aligned image data.
- Detect OME-XML without declaration (#2).
- Support LERC compression (WIP).
- Delay load imagecodecs functions.
- Remove maxsize parameter from asarray (breaking).
- Deprecate ijmetadata parameter from TiffWriter.save (use metadata).

2020.2.16

- Add method to decode individual strips or tiles.
- Read strips and tiles in order of their offsets.
- Enable multi-threading when decompressing multiple strips.
- Replace TiffPage.tags dictionary with TiffTags (breaking).
- Replace TIFF.TAGS dictionary with TiffTagRegistry.
- Remove TIFF.TAG_NAMES (breaking).
- Improve handling of TiffSequence parameters in imread.
- Match last uncommon parts of file paths to FileSequence pattern (breaking).
- Allow letters in FileSequence pattern for indexing well plate rows.
- Allow reordering of axes in FileSequence.
- Allow writing > 4 GB arrays to plain TIFF when using compression.
- Allow writing zero size NumPy arrays to nonconformant TIFF (tentative).
- Fix xml2dict.
- Require imagecodecs >= 2020.1.31.
- Remove support for imagecodecs-lite (breaking).
- Remove verify parameter to asarray method (breaking).
- Remove deprecated lzw_decode functions (breaking).
- Remove support for Python 2.7 and 3.5 (breaking).

2019.7.26

- Fix infinite loop reading more than two tags of same code in IFD.
- Delay import of logging module.

2019.7.20

- Fix OME-XML detection for files created by Imaris.
- Remove or replace assert statements.

2019.7.2

- Do not write SampleFormat tag for unsigned data types.
- Write ByteCount tag values as SHORT or LONG if possible.
- Allow specifying axes in FileSequence pattern via group names.
- Add option to read FileSequence concurrently using threads.
- Derive TiffSequence from FileSequence.
- Use str(datetime.timedelta) to format Timer duration.
- Use perf_counter for Timer if possible.

2019.6.18

- Fix reading planar RGB ImageJ files created by Bio-Formats.
- Fix reading single-file, multi-image OME-TIFF without UUID.
- Presume LSM stores uncompressed images contiguously per page.
- Reformat some complex expressions.

2019.5.30

- Ignore invalid frames in OME-TIFF.
- Set default subsampling to (2, 2) for RGB JPEG compression.
- Fix reading and writing planar RGB JPEG compression.
- Replace buffered_read with FileHandle.read_segments.
- Include page or frame numbers in exceptions and warnings.
- Add Timer class.

2019.5.22

- Add optional chroma subsampling for JPEG compression.
- Enable writing PNG, JPEG, JPEGXR, and JPEG2K compression (WIP).
- Fix writing tiled images with WebP compression.
- Improve handling GeoTIFF sparse files.

2019.3.18

- Fix regression decoding JPEG with RGB photometrics.
- Fix reading OME-TIFF files with corrupted but unused pages.
- Allow loading TiffFrame without specifying keyframe.
- Calculate virtual TiffFrames for non-BigTIFF ScanImage files > 2GB.
- Rename property is_chroma_subsampled to is_subsampled (breaking).
- Make more attributes and methods private (WIP).

2019.3.8

- Fix MemoryError when RowsPerStrip > ImageLength.
- Fix SyntaxWarning on Python 3.8.
- Fail to decode JPEG to planar RGB (tentative).
- Separate public from private test files (WIP).
- Allow testing without data files or imagecodecs.

2019.2.22

- Use imagecodecs-lite as fallback for imagecodecs.
- Simplify reading NumPy arrays from file.
- Use TiffFrames when reading arrays from page sequences.
- Support slices and iterators in TiffPageSeries sequence interface.
- Auto-detect uniform series.
- Use page hash to determine generic series.
- Turn off TiffPages cache (tentative).
- Pass through more parameters in imread.
- Discontinue movie parameter in imread and TiffFile (breaking).
- Discontinue bigsize parameter in imwrite (breaking).
- Raise TiffFileError in case of issues with TIFF structure.
- Return TiffFile.ome_metadata as XML (breaking).
- Ignore OME series when last dimensions are not stored in TIFF pages.

2019.2.10

- Assemble IFDs in memory to speed-up writing on some slow media.
- Handle discontinued arguments fastij, multifile_close, and pages.

2019.1.30

- Use black background in imshow.
- Do not write datetime tag by default (breaking).
- Fix OME-TIFF with SamplesPerPixel > 1.
- Allow 64-bit IFD offsets for NDPI (files > 4GB still not supported).

2019.1.4

- Fix decoding deflate without imagecodecs.

2019.1.1

- Update copyright year.
- Require imagecodecs >= 2018.12.16.
- Do not use JPEG tables from keyframe.
- Enable decoding large JPEG in NDPI.
- Decode some old-style JPEG.
- Reorder OME channel axis to match PlanarConfiguration storage.
- Return tiled images as contiguous arrays.
- Add decode_lzw proxy function for compatibility with old czifile module.
- Use dedicated logger.

2018.11.28

- Make SubIFDs accessible as TiffPage.pages.
- Make parsing of TiffSequence axes pattern optional (breaking).
- Limit parsing of TiffSequence axes pattern to file names, not path names.
- Do not interpolate in imshow if image dimensions <= 512, else use bilinear.
- Use logging.warning instead of warnings.warn in many cases.
- Fix NumPy FutureWarning for out == 'memmap'.
- Adjust ZSTD and WebP compression to libtiff-4.0.10 (WIP).
- Decode old-style LZW with imagecodecs >= 2018.11.8.
- Remove TiffFile.qptiff_metadata (QPI metadata are per page).
- Do not use keyword arguments before variable positional arguments.
- Make either all or none return statements in function return expression.
- Use pytest parametrize to generate tests.
- Replace test classes with functions.

2018.11.6

- Rename imsave function to imwrite.
- Re-add Python implementations of packints, delta, and bitorder codecs.
- Fix TiffFrame.compression AttributeError.

2018.10.18

- Rename tiffile package to tifffile.

2018.10.10

- Read ZIF, the Zoomable Image Format (WIP).
- Decode YCbCr JPEG as RGB (tentative).
- Improve restoration of incomplete tiles.
- Allow writing grayscale with extrasamples without specifying planarconfig.
- Enable decoding of PNG and JXR via imagecodecs.
- Deprecate 32-bit platforms (too many memory errors during tests).

2018.9.27

- Read Olympus SIS (WIP).
- Fix writing non-BigTIFF files up to ~4 GB.
- Fix parsing date and time fields in SEM metadata.
- Detect some circular IFD references.
- Enable WebP codecs via imagecodecs.
- Add option to read TiffSequence from ZIP containers.
- Remove TiffFile.isnative.
- Move TIFF struct format constants out of TiffFile namespace.

2018.8.31

- Fix wrong TiffTag.valueoffset.
- Towards reading Hamamatsu NDPI (WIP).
- Enable PackBits compression of byte and bool arrays.
- Fix parsing NULL terminated CZ_SEM strings.

2018.8.24

- Move tifffile.py and related modules into tiffile package.
- Move usage examples to module docstring.
- Enable multi-threading for compressed tiles and pages by default.
- Add option to concurrently decode image tiles using threads.
- Do not skip empty tiles (fix).
- Read JPEG and J2K compressed strips and tiles.
- Allow floating-point predictor on write.
- Add option to specify subfiletype on write.
- Depend on imagecodecs package instead of _tifffile, lzma, etc modules.
- Remove reverse_bitorder, unpack_ints, and decode functions.
- Use pytest instead of unittest.

2018.6.20

- Save RGBA with unassociated extrasample by default (breaking).
- Add option to specify ExtraSamples values.

2018.6.17 (included with 0.15.1)

- Towards reading JPEG and other compressions via imagecodecs package (WIP).
- Read SampleFormat VOID as UINT.
- Add function to validate TIFF using ``jhove -m TIFF-hul``.
- Save bool arrays as bilevel TIFF.
- Accept pathlib.Path as filenames.
- Move software argument from TiffWriter __init__ to save.
- Raise DOS limit to 16 TB.
- Lazy load LZMA and ZSTD compressors and decompressors.
- Add option to save IJMetadata tags.
- Return correct number of pages for truncated series (fix).
- Move EXIF tags to TIFF.TAG as per TIFF/EP standard.

2018.2.18

- Always save RowsPerStrip and Resolution tags as required by TIFF standard.
- Do not use badly typed ImageDescription.
- Coerce bad ASCII string tags to bytes.
- Tuning of __str__ functions.
- Fix reading undefined tag values.
- Read and write ZSTD compressed data.
- Use hexdump to print bytes.
- Determine TIFF byte order from data dtype in imsave.
- Add option to specify RowsPerStrip for compressed strips.
- Allow memory-map of arrays with non-native byte order.
- Attempt to handle ScanImage <= 5.1 files.
- Restore TiffPageSeries.pages sequence interface.
- Use numpy.frombuffer instead of fromstring to read from binary data.
- Parse GeoTIFF metadata.
- Add option to apply horizontal differencing before compression.
- Towards reading PerkinElmer QPI (QPTIFF, no test files).
- Do not index out of bounds data in tifffile.c unpackbits and decodelzw.

2017.9.29

- Many backward incompatible changes improving speed and resource usage:
- Add detail argument to __str__ function. Remove info functions.
- Fix potential issue correcting offsets of large LSM files with positions.
- Remove TiffFile sequence interface; use TiffFile.pages instead.
- Do not make tag values available as TiffPage attributes.
- Use str (not bytes) type for tag and metadata strings (WIP).
- Use documented standard tag and value names (WIP).
- Use enums for some documented TIFF tag values.
- Remove memmap and tmpfile options; use out='memmap' instead.
- Add option to specify output in asarray functions.
- Add option to concurrently decode pages using threads.
- Add TiffPage.asrgb function (WIP).
- Do not apply colormap in asarray.
- Remove colormapped, rgbonly, and scale_mdgel options from asarray.
- Consolidate metadata in TiffFile _metadata functions.
- Remove non-tag metadata properties from TiffPage.
- Add function to convert LSM to tiled BIN files.
- Align image data in file.
- Make TiffPage.dtype a numpy.dtype.
- Add ndim and size properties to TiffPage and TiffPageSeries.
- Allow imsave to write non-BigTIFF files up to ~4 GB.
- Only read one page for shaped series if possible.
- Add memmap function to create memory-mapped array stored in TIFF file.
- Add option to save empty arrays to TIFF files.
- Add option to save truncated TIFF files.
- Allow single tile images to be saved contiguously.
- Add optional movie mode for files with uniform pages.
- Lazy load pages.
- Use lightweight TiffFrame for IFDs sharing properties with key TiffPage.
- Move module constants to TIFF namespace (speed up module import).
- Remove fastij option from TiffFile.
- Remove pages parameter from TiffFile.
- Remove TIFFfile alias.
- Deprecate Python 2.
- Require enum34 and futures packages on Python 2.7.
- Remove Record class and return all metadata as dict instead.
- Add functions to parse STK, MetaSeries, ScanImage, SVS, Pilatus metadata.
- Read tags from EXIF and GPS IFDs.
- Use pformat for tag and metadata values.
- Fix reading some UIC tags.
- Do not modify input array in imshow (fix).
- Fix Python implementation of unpack_ints.

2017.5.23

- Write correct number of SampleFormat values (fix).
- Use Adobe deflate code to write ZIP compressed files.
- Add option to pass tag values as packed binary data for writing.
- Defer tag validation to attribute access.
- Use property instead of lazyattr decorator for simple expressions.

2017.3.17

- Write IFDs and tag values on word boundaries.
- Read ScanImage metadata.
- Remove is_rgb and is_indexed attributes from TiffFile.
- Create files used by doctests.

2017.1.12 (included with scikit-image 0.14.x)

- Read Zeiss SEM metadata.
- Read OME-TIFF with invalid references to external files.
- Rewrite C LZW decoder (5x faster).
- Read corrupted LSM files missing EOI code in LZW stream.

2017.1.1

- Add option to append images to existing TIFF files.
- Read files without pages.
- Read S-FEG and Helios NanoLab tags created by FEI software.
- Allow saving Color Filter Array (CFA) images.
- Add info functions returning more information about TiffFile and TiffPage.
- Add option to read specific pages only.
- Remove maxpages argument (breaking).
- Remove test_tifffile function.

2016.10.28

- Improve detection of ImageJ hyperstacks.
- Read TVIPS metadata created by EM-MENU (by Marco Oster).
- Add option to disable using OME-XML metadata.
- Allow non-integer range attributes in modulo tags (by Stuart Berg).

2016.6.21

- Do not always memmap contiguous data in page series.

2016.5.13

- Add option to specify resolution unit.
- Write grayscale images with extra samples when planarconfig is specified.
- Do not write RGB color images with 2 samples.
- Reorder TiffWriter.save keyword arguments (breaking).

2016.4.18

- TiffWriter, imread, and imsave accept open binary file streams.

2016.04.13

- Fix reversed fill order in 2 and 4 bps images.
- Implement reverse_bitorder in C.

2016.03.18

- Fix saving additional ImageJ metadata.

2016.2.22

- Write 8 bytes double tag values using offset if necessary (bug fix).
- Add option to disable writing second image description tag.
- Detect tags with incorrect counts.
- Disable color mapping for LSM.

2015.11.13

- Read LSM 6 mosaics.
- Add option to specify directory of memory-mapped files.
- Add command line options to specify vmin and vmax values for colormapping.

2015.10.06

- New helper function to apply colormaps.
- Renamed is_palette attributes to is_indexed (breaking).
- Color-mapped samples are now contiguous (breaking).
- Do not color-map ImageJ hyperstacks (breaking).
- Towards reading Leica SCN.

2015.9.25

- Read images with reversed bit order (FillOrder is LSB2MSB).

2015.9.21

- Read RGB OME-TIFF.
- Warn about malformed OME-XML.

2015.9.16

- Detect some corrupted ImageJ metadata.
- Better axes labels for shaped files.
- Do not create TiffTag for default values.
- Chroma subsampling is not supported.
- Memory-map data in TiffPageSeries if possible (optional).

2015.8.17

- Write ImageJ hyperstacks (optional).
- Read and write LZMA compressed data.
- Specify datetime when saving (optional).
- Save tiled and color-mapped images (optional).
- Ignore void bytecounts and offsets if possible.
- Ignore bogus image_depth tag created by ISS Vista software.
- Decode floating-point horizontal differencing (not tiled).
- Save image data contiguously if possible.
- Only read first IFD from ImageJ files if possible.
- Read ImageJ raw format (files larger than 4 GB).
- TiffPageSeries class for pages with compatible shape and data type.
- Try to read incomplete tiles.
- Open file dialog if no filename is passed on command line.
- Ignore errors when decoding OME-XML.
- Rename decoder functions (breaking).

2014.8.24

- TiffWriter class for incremental writing images.
- Simplify examples.

2014.8.19

- Add memmap function to FileHandle.
- Add function to determine if image data in TiffPage is memory-mappable.
- Do not close files if multifile_close parameter is False.

2014.8.10

- Return all extrasamples by default (breaking).
- Read data from series of pages into memory-mapped array (optional).
- Squeeze OME dimensions (breaking).
- Workaround missing EOI code in strips.
- Support image and tile depth tags (SGI extension).
- Better handling of STK/UIC tags (breaking).
- Disable color mapping for STK.
- Julian to datetime converter.
- TIFF ASCII type may be NULL separated.
- Unwrap strip offsets for LSM files greater than 4 GB.
- Correct strip byte counts in compressed LSM files.
- Skip missing files in OME series.
- Read embedded TIFF files.

2014.2.05

- Save rational numbers as type 5 (bug fix).

2013.12.20

- Keep other files in OME multi-file series closed.
- FileHandle class to abstract binary file handle.
- Disable color mapping for bad OME-TIFF produced by bio-formats.
- Read bad OME-XML produced by ImageJ when cropping.

2013.11.3

- Allow zlib compress data in imsave function (optional).
- Memory-map contiguous image data (optional).

2013.10.28

- Read MicroManager metadata and little-endian ImageJ tag.
- Save extra tags in imsave function.
- Save tags in ascending order by code (bug fix).

2012.10.18

- Accept file like objects (read from OIB files).

2012.8.21

- Rename TIFFfile to TiffFile and TIFFpage to TiffPage.
- TiffSequence class for reading sequence of TIFF files.
- Read UltraQuant tags.
- Allow float numbers as resolution in imsave function.

2012.8.3

- Read MD GEL tags and NIH Image header.

2012.7.25

- Read ImageJ tags.
- …
