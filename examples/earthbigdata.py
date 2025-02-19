# tifffile/examples/earthbigdata.py

# Copyright (c) 2021-2025, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

# This file uses VSCode Jupyter-like code cells
# https://code.visualstudio.com/docs/python/jupyter-support-py

# %% [markdown]
"""
# Create a fsspec ReferenceFileSystem for a large set of remote GeoTIFF files

by [Christoph Gohlke](https://www.cgohlke.com)

Published Oct 9, 2021. Last updated Feb 18, 2025.

This Python script uses the [tifffile](https://github.com/cgohlke/tifffile) and
[imagecodecs](https://github.com/cgohlke/imagecodecs) packages to create a
[fsspec ReferenceFileSystem](https://github.com/fsspec/kerchunk) file in
JSON format for the [Earthbigdata](
http://sentinel-1-global-coherence-earthbigdata.s3-website-us-west-2.amazonaws.com
) set, which consists of 1,033,422 GeoTIFF files stored on AWS.
The ReferenceFileSystem is used to create a multi-dimensional Xarray dataset.

Refer to the discussion at [kerchunk/issues/78](
https://github.com/fsspec/kerchunk/issues/78).
"""

# %%
import base64
import os

import fsspec
import imagecodecs.numcodecs
import matplotlib.pyplot
import numcodecs
import tifffile
import xarray
import zarr  # < 3

assert zarr.__version__[0] == '2'

# %% [markdown]
"""
## Get a list of all remote TIFF files

Call the AWS command line app to recursively list all files in the Earthbigdata
set. Cache the output in a local file. Filter the list for TIFF files and
remove the common path.
"""

# %%
if not os.path.exists('earthbigdata.txt'):
    os.system(
        'aws s3 ls sentinel-1-global-coherence-earthbigdata/data/tiles'
        ' --recursive > earthbigdata.txt'
    )

with open('earthbigdata.txt', encoding='utf-8') as fh:
    tiff_files = [
        line.split()[-1][11:] for line in fh.readlines() if '.tif' in line
    ]
print('Number of TIFF files:', len(tiff_files))


# %% [markdown]
"""
## Define metadata to describe the dataset

Define labels, coordinate arrays, file name regular expression patterns, and
categories for all dimensions in the Earthbigdata set.
"""

# %%
baseurl = (
    'https://'
    'sentinel-1-global-coherence-earthbigdata.s3.us-west-2.amazonaws.com'
    '/data/tiles/'
)

chunkshape = (1200, 1200)
fillvalue = 0

latitude_label = 'latitude'
latitude_pattern = rf'(?P<{latitude_label}>[NS]\d+)'
latitude_coordinates = [
    (j * -0.00083333333 - 0.000416666665 + i)
    for i in range(82, -79, -1)
    for j in range(1200)
]
latitude_category = {}
i = 0
for j in range(82, -1, -1):
    latitude_category[f'N{j:-02}'] = i
    i += 1
for j in range(1, 79):
    latitude_category[f'S{j:-02}'] = i
    i += 1

longitude_label = 'longitude'
longitude_pattern = rf'(?P<{longitude_label}>[EW]\d+)'
longitude_coordinates = [
    (j * 0.00083333333 + 0.000416666665 + i)
    for i in range(-180, 180)
    for j in range(1200)
]
longitude_category = {}
i = 0
for j in range(180, 0, -1):
    longitude_category[f'W{j:-03}'] = i
    i += 1
for j in range(180):
    longitude_category[f'E{j:-03}'] = i
    i += 1

season_label = 'season'
season_category = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}
season_coordinates = list(season_category.keys())
season_pattern = rf'_(?P<{season_label}>{"|".join(season_category)})'

polarization_label = 'polarization'
polarization_category = {'vv': 0, 'vh': 1, 'hv': 2, 'hh': 3}
polarization_coordinates = list(polarization_category.keys())
polarization_pattern = (
    rf'_(?P<{polarization_label}>{"|".join(polarization_category)})'
)

coherence_label = 'coherence'
coherence_category = {
    '06': 0,
    '12': 1,
    '18': 2,
    '24': 3,
    '36': 4,
    '48': 5,
}
coherence_coordinates = list(int(i) for i in coherence_category.keys())
coherence_pattern = (
    rf'_COH(?P<{coherence_label}>{"|".join(coherence_category)})'
)

orbit_label = 'orbit'
orbit_coordinates = list(range(1, 176))
orbit_pattern = rf'_(?P<{orbit_label}>\d+)'

flightdirection_label = 'flightdirection'
flightdirection_category = {'A': 0, 'D': 1}
flightdirection_coordinates = list(flightdirection_category.keys())
flightdirection_pattern = (
    rf'(?P<{flightdirection_label}>[{"|".join(flightdirection_category)}])_'
)


# %% [markdown]
"""
## Open a file for writing the fsspec ReferenceFileSystem in JSON format
"""

# %%
jsonfile = open('earthbigdata.json', 'w', encoding='utf-8', newline='\n')

# %% [markdown]
"""
## Write the coordinate arrays

Add the coordinate arrays to a Zarr group, convert it to a fsspec
ReferenceFileSystem JSON string, and write it to the open file.
"""

# %%
coordinates = {}  # type: ignore[var-annotated]
zarrgroup = zarr.open_group(coordinates)
zarrgroup.array(
    longitude_label,
    data=longitude_coordinates,
    dtype='float32',
    # compression='zlib',
).attrs['_ARRAY_DIMENSIONS'] = [longitude_label]

zarrgroup.array(
    latitude_label,
    data=latitude_coordinates,
    dtype='float32',
    # compression='zlib',
).attrs['_ARRAY_DIMENSIONS'] = [latitude_label]

zarrgroup.array(
    season_label,
    data=season_coordinates,
    dtype=object,
    object_codec=numcodecs.VLenUTF8(),
    compression=None,
).attrs['_ARRAY_DIMENSIONS'] = [season_label]

zarrgroup.array(
    polarization_label,
    data=polarization_coordinates,
    dtype=object,
    object_codec=numcodecs.VLenUTF8(),
    compression=None,
).attrs['_ARRAY_DIMENSIONS'] = [polarization_label]

zarrgroup.array(
    coherence_label,
    data=coherence_coordinates,
    dtype='uint8',
    compression=None,
).attrs['_ARRAY_DIMENSIONS'] = [coherence_label]

zarrgroup.array(orbit_label, data=orbit_coordinates, dtype='int32').attrs[
    '_ARRAY_DIMENSIONS'
] = [orbit_label]

zarrgroup.array(
    flightdirection_label,
    data=flightdirection_coordinates,
    dtype=object,
    object_codec=numcodecs.VLenUTF8(),
    compression=None,
).attrs['_ARRAY_DIMENSIONS'] = [flightdirection_label]

# base64 encode any values containing non-ascii characters
for k, v in coordinates.items():
    try:
        coordinates[k] = v.decode()
    except UnicodeDecodeError:
        coordinates[k] = 'base64:' + base64.b64encode(v).decode()

coordinates_json = tifffile.ZarrStore._json(coordinates).decode()

jsonfile.write(coordinates_json[:-2])  # skip the last newline and brace

# %% [markdown]
"""
## Create a TiffSequence from a list of file names

Filter the list of GeoTIFF files for files containing coherence 'COH' data.
The regular expression pattern and categories are used to parse the file names
for chunk indices.

Note: the created TiffSequence cannot be used to access any files. The file
names do not refer to existing files. The `baseurl` is later used to get
the real location of the files.
"""

# %%
mode = 'COH'
fileseq = tifffile.TiffSequence(
    [file for file in tiff_files if '_' + mode in file],
    pattern=(
        latitude_pattern
        + longitude_pattern
        + season_pattern
        + polarization_pattern
        + coherence_pattern
    ),
    categories={
        latitude_label: latitude_category,
        longitude_label: longitude_category,
        season_label: season_category,
        polarization_label: polarization_category,
        coherence_label: coherence_category,
    },
)
assert len(fileseq) == 444821
assert fileseq.files_missing == 5119339
assert fileseq.shape == (161, 360, 4, 4, 6)
assert fileseq.dims == (
    'latitude',
    'longitude',
    'season',
    'polarization',
    'coherence',
)
print(fileseq)


# %% [markdown]
"""
## Create a ZarrTiffStore from the TiffSequence

Define `axestiled` to tile the latitude and longitude dimensions of the
TiffSequence with the first and second image/chunk dimensions.
Define extra `zattrs` to create a Xarray compatible store.
"""

# %%
store = fileseq.aszarr(
    chunkdtype='uint8',
    chunkshape=chunkshape,
    fillvalue=fillvalue,
    axestiled={0: 0, 1: 1},
    zattrs={
        '_ARRAY_DIMENSIONS': [
            season_label,
            polarization_label,
            coherence_label,
            latitude_label,
            longitude_label,
        ]
    },
)
print(store)

# %% [markdown]
"""
## Append the ZarrTiffStore to the open ReferenceFileSystem file

Use the mode name to create a Zarr subgroup.
Use the `imagecodecs_tiff` Numcodecs compatible codec for decoding TIFF files.
"""

# %%
store.write_fsspec(
    jsonfile,
    baseurl,
    groupname=mode,
    codec_id='imagecodecs_tiff',
    _append=True,
    _close=False,
)

# %% [markdown]
"""
## Repeat for the other modes

Repeat the `TiffSequence->aszarr->write_fsspec` workflow for the other modes.
"""

# %%
for mode in (
    'AMP',
    'tau',
    'rmse',
    'rho',
):
    fileseq = tifffile.TiffSequence(
        [file for file in tiff_files if '_' + mode in file],
        pattern=(
            latitude_pattern
            + longitude_pattern
            + season_pattern
            + polarization_pattern
        ),
        categories={
            latitude_label: latitude_category,
            longitude_label: longitude_category,
            season_label: season_category,
            polarization_label: polarization_category,
        },
    )
    print(fileseq)
    with fileseq.aszarr(
        chunkdtype='uint16',
        chunkshape=chunkshape,
        fillvalue=fillvalue,
        axestiled={0: 0, 1: 1},
        zattrs={
            '_ARRAY_DIMENSIONS': [
                season_label,
                polarization_label,
                latitude_label,
                longitude_label,
            ]
        },
    ) as store:
        print(store)
        store.write_fsspec(
            jsonfile,
            baseurl,
            groupname=mode,
            codec_id='imagecodecs_tiff',
            _append=True,
            _close=False,
        )


for mode in ('inc', 'lsmap'):
    fileseq = tifffile.TiffSequence(
        [file for file in tiff_files if '_' + mode in file],
        pattern=(
            latitude_pattern
            + longitude_pattern
            + orbit_pattern
            + flightdirection_pattern
        ),
        categories={
            latitude_label: latitude_category,
            longitude_label: longitude_category,
            # orbit has no category
            flightdirection_label: flightdirection_category,
        },
    )
    print(fileseq)
    with fileseq.aszarr(
        chunkdtype='uint8',
        chunkshape=chunkshape,
        fillvalue=fillvalue,
        axestiled={0: 0, 1: 1},
        zattrs={
            '_ARRAY_DIMENSIONS': [
                orbit_label,
                flightdirection_label,
                latitude_label,
                longitude_label,
            ]
        },
    ) as store:
        print(store)
        store.write_fsspec(
            jsonfile,
            baseurl,
            groupname=mode,
            codec_id='imagecodecs_tiff',
            _append=True,
            _close=mode == 'lsmap',  # close after last store
        )

# %% [markdown]
"""
## Close the JSON file
"""

# %%
jsonfile.close()

# %% [markdown]
"""
## Use the fsspec ReferenceFileSystem file to create a Xarray dataset

Register `imagecodecs.numcodecs` before using the ReferenceFileSystem.
"""

# %%
imagecodecs.numcodecs.register_codecs()

# %% [markdown]
"""
### Create a fsspec mapper instance from the ReferenceFileSystem file

Specify the `target_protocol` to load a local file.
"""

# %%
mapper = fsspec.get_mapper(
    'reference://',
    fo='earthbigdata.json',
    target_protocol='file',
    remote_protocol='https',
)

# %% [markdown]
"""
### Create a Xarray dataset from the mapper

Use `mask_and_scale` to disable conversion to floating point.
"""

# %%
dataset = xarray.open_dataset(
    mapper,
    engine='zarr',
    mask_and_scale=False,
    backend_kwargs={'consolidated': False},
)
print(dataset)

# %% [markdown]
"""
### Select the Southern California region in the dataset
"""

# %%
socal = dataset.sel(latitude=slice(36, 32.5), longitude=slice(-121, -115))
print(socal)

# %% [markdown]
"""
### Plot a selection of the dataset

The few GeoTIFF files comprising the selection are transparently downloaded,
decoded, and stitched to an in-memory NumPy array and plotted using Matplotlib.
"""

# %%
image = socal['COH'].loc['winter', 'vv', 12]
assert image[100, 100] == 53
xarray.plot.imshow(image, size=6, aspect=1.8)
matplotlib.pyplot.show()

# %% [markdown]
"""
## System information

Print information about the software used to run this script.
"""


# %%
def system_info() -> str:
    """Print information about Python environment."""
    import datetime
    import sys

    import fsspec
    import imagecodecs
    import matplotlib
    import numcodecs
    import numpy
    import tifffile
    import xarray
    import zarr

    return '\n'.join(
        (
            sys.executable,
            f'Python {sys.version}',
            '',
            f'numpy {numpy.__version__}',
            f'matplotlib {matplotlib.__version__}',
            f'tifffile {tifffile.__version__}',
            f'imagecodecs {imagecodecs.__version__}',
            f'numcodecs {numcodecs.__version__}',
            f'fsspec {fsspec.__version__}',
            f'xarray {xarray.__version__}',
            f'zarr {zarr.__version__}',
            '',
            str(datetime.datetime.now()),
        )
    )


print(system_info())
