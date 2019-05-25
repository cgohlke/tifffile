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
    Add option to append images to existing TIFF files.
    Read files without pages.
    Read S-FEG and Helios NanoLab tags created by FEI software.
    Allow saving Color Filter Array (CFA) images.
    Add info functions returning more information about TiffFile and TiffPage.
    Add option to read specific pages only.
    Remove maxpages argument (breaking).
    Remove test_tifffile function.
2016.10.28
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
    Reorder TiffWriter.save keyword arguments (breaking).
2016.4.18
    TiffWriter, imread, and imsave accept open binary file streams.
2016.04.13
    Fix reversed fill order in 2 and 4 bps images.
    Implement reverse_bitorder in C.
2016.03.18
    Fix saving additional ImageJ metadata.
2016.2.22
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
    Renamed is_palette attributes to is_indexed (breaking).
    Color-mapped samples are now contiguous (breaking).
    Do not color-map ImageJ hyperstacks (breaking).
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
    Write ImageJ hyperstacks (optional).
    Read and write LZMA compressed data.
    Specify datetime when saving (optional).
    Save tiled and color-mapped images (optional).
    Ignore void bytecounts and offsets if possible.
    Ignore bogus image_depth tag created by ISS Vista software.
    Decode floating-point horizontal differencing (not tiled).
    Save image data contiguously if possible.
    Only read first IFD from ImageJ files if possible.
    Read ImageJ 'raw' format (files larger than 4 GB).
    TiffPageSeries class for pages with compatible shape and data type.
    Try to read incomplete tiles.
    Open file dialog if no filename is passed on command line.
    Ignore errors when decoding OME-XML.
    Rename decoder functions (breaking).
2014.8.24
    TiffWriter class for incremental writing images.
    Simplify examples.
2014.8.19
    Add memmap function to FileHandle.
    Add function to determine if image data in TiffPage is memory-mappable.
    Do not close files if multifile_close parameter is False.
2014.8.10
    Return all extrasamples by default (breaking).
    Read data from series of pages into memory-mapped array (optional).
    Squeeze OME dimensions (breaking).
    Workaround missing EOI code in strips.
    Support image and tile depth tags (SGI extension).
    Better handling of STK/UIC tags (breaking).
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