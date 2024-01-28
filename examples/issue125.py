# tifffile/examples/issues125.py

"""Create a Fsspec ReferenceFileSystem for a sequence of TIFF files on S3

This Python script uses the Tifffile and Fsspec libraries to create a
multiscale ReferenceFileSystem JSON file for a sequence of cloud optimized
GeoTIFF (COG) files stored on S3. The tiles of the COG files are used as
chunks. No additional Numcodecs codec needs to be registered since the COG
files use Zlib compression. A Xarray dataset is created from the
ReferenceFileSystem file and a subset of the dataset is plotted.

See https://github.com/cgohlke/tifffile/issues/125

"""

import fsspec
import tifffile
import xarray
from matplotlib import pyplot

# get a list of cloud optimized GeoTIFF files stored on S3
remote_options = {
    'anon': True,
    'client_kwargs': {'endpoint_url': 'https://mghp.osn.xsede.org'},
}
fs = fsspec.filesystem('s3', **remote_options)
files = [f's3://{f}' for f in fs.ls('/rsignellbucket1/lcmap/cog')]

# write the ReferenceFileSystem of each file to a JSON file
with open('issue125.json', 'w', encoding='utf-8', newline='\n') as jsonfile:
    for i, filename in enumerate(tifffile.natural_sorted(files)):
        url, name = filename.rsplit('/', 1)
        with fs.open(filename) as fh:
            with tifffile.TiffFile(fh, name=name) as tif:
                print(tif.geotiff_metadata)
                with tif.series[0].aszarr() as store:
                    store.write_fsspec(
                        jsonfile,
                        url=url,
                        # using an experimental API:
                        _shape=[len(files)],  # shape of file sequence
                        _axes='T',  # axes of file sequence
                        _index=[i],  # index of this file in sequence
                        _append=i != 0,  # if True, only write index keys+value
                        _close=i == len(files) - 1,  # if True, no more appends
                        # groupname='0',  # required for non-pyramidal series
                    )

# create a fsspec mapper instance from the ReferenceFileSystem file
mapper = fsspec.get_mapper(
    'reference://',
    fo='issue125.json',
    target_protocol='file',
    remote_protocol='s3',
    remote_options=remote_options,
)

# create a xarray dataset from the mapper
dataset = xarray.open_zarr(mapper, consolidated=False)
print(dataset)

# plot a slice of the 5th pyramidal level of the dataset
xarray.plot.imshow(dataset['5'][0, 32:-32, 32:-32])
pyplot.show()
