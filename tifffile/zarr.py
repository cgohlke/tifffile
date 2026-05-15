# tifffile/zarr.py

# Copyright (c) 2008-2026, Christoph Gohlke
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

"""Zarr 3 TIFF codec, TIFF and file sequence stores."""

from __future__ import annotations

__all__ = [
    'Tiff',
    'ZarrFileSequenceStore',
    'ZarrStore',
    'ZarrTiffStore',
    'register_codec',
]

import asyncio
import base64
import contextlib
import enum
import json
import sys
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Literal, override

import numpy
import zarr

try:
    from zarr.abc.codec import ArrayBytesCodec
    from zarr.abc.store import ByteRequest, Store
    from zarr.core.buffer.cpu import NDBuffer as NDBufferCPU
    from zarr.core.chunk_grids import ChunkGrid
    from zarr.core.common import parse_named_configuration
    from zarr.core.indexing import BasicIndexer
except ImportError as exc:
    msg = f'zarr {zarr.__version__} < 3 is not supported'
    raise ValueError(msg) from exc

from .tifffile import (
    CHUNKMODE,
    COMPRESSION,
    EXTRASAMPLE,
    METADATA_DEFAULT,
    PHOTOMETRIC,
    PLANARCONFIG,
    PREDICTOR,
    ByteOrder,
    FileCache,
    FileSequence,
    NullContext,
    TagTuple,
    TiffFile,
    TiffFrame,
    TiffPage,
    TiffPageSeries,
    TiffWriter,
    TiledSequence,
    create_output,
    enumarg,
    imread,
    jpeg_decode_colorspace,
    product,
)

if TYPE_CHECKING:
    import os
    import threading
    from collections.abc import (
        AsyncIterator,
        Callable,
        Iterable,
        Iterator,
        Sequence,
    )
    from typing import Any, Self, TextIO

    from numpy.typing import DTypeLike, NDArray
    from zarr.core.array_spec import ArraySpec
    from zarr.core.buffer import Buffer, BufferPrototype, NDBuffer
    from zarr.core.common import JSON
    from zarr.core.indexing import BasicSelection

    from .tifffile import ByteOrder, OutputType


@dataclass(frozen=True)
class Tiff(ArrayBytesCodec):
    """TIFF codec for Zarr 3."""

    is_fixed_size = False

    # TiffFile.asarray
    key: int | slice | Sequence[int] | None = None
    series: int | None = None
    kind: Literal['generic', 'imagej', 'ome', 'shaped'] | None = None
    level: int | None = None
    squeeze: bool | None = None
    buffersize: int | None = None
    # TiffWriter
    bigtiff: bool = False
    byteorder: ByteOrder | None = None
    # TiffWriter.write
    photometric: str | None = None
    planarconfig: str | None = None
    extrasamples: tuple[str, ...] | None = None
    volumetric: bool = False
    tile: tuple[int, ...] | None = None
    rowsperstrip: int | None = None
    bitspersample: int | None = None
    compression: str | None = None
    compressionargs: dict[str, Any] | None = None
    predictor: str | bool | None = None
    subsampling: tuple[int, int] | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)
    extratags: Sequence[TagTuple] | None = None
    truncate: bool = False
    maxworkers: int | None = None

    def __init__(
        self,
        *,
        key: int | slice | Sequence[int] | None = None,
        series: int | None = None,
        kind: Literal['generic', 'imagej', 'ome', 'shaped'] | None = None,
        level: int | None = None,
        squeeze: bool | None = None,
        buffersize: int | None = None,
        bigtiff: bool = False,
        byteorder: ByteOrder | None = None,
        photometric: PHOTOMETRIC | int | str | None = None,
        planarconfig: PLANARCONFIG | int | str | None = None,
        extrasamples: Sequence[EXTRASAMPLE | int | str] | None = None,
        volumetric: bool = False,
        tile: Sequence[int] | None = None,
        rowsperstrip: int | None = None,
        bitspersample: int | None = None,
        compression: COMPRESSION | int | str | None = None,
        compressionargs: dict[str, Any] | None = None,
        predictor: PREDICTOR | int | str | bool | None = None,
        subsampling: tuple[int, int] | None = None,
        metadata: dict[str, Any] | None = METADATA_DEFAULT,
        extratags: Sequence[TagTuple] | None = None,
        truncate: bool = False,
        maxworkers: int | None = None,
    ) -> None:
        _setattrs(
            self,
            key=key,
            series=int(series) if series is not None else None,
            kind=kind,
            level=int(level) if level is not None else None,
            squeeze=bool(squeeze) if squeeze is not None else None,
            buffersize=int(buffersize) if buffersize is not None else None,
            bigtiff=bool(bigtiff),
            byteorder=byteorder,
            photometric=_enum_name(photometric, PHOTOMETRIC),
            planarconfig=_enum_name(planarconfig, PLANARCONFIG),
            extrasamples=(
                tuple(_enum_name(e, EXTRASAMPLE) for e in extrasamples)
                if extrasamples is not None
                else None
            ),
            volumetric=bool(volumetric),
            tile=tuple(int(x) for x in tile) if tile is not None else None,
            rowsperstrip=(
                int(rowsperstrip) if rowsperstrip is not None else None
            ),
            bitspersample=(
                int(bitspersample) if bitspersample is not None else None
            ),
            compression=_enum_name(compression, COMPRESSION),
            compressionargs=compressionargs,
            predictor=(
                predictor
                if isinstance(predictor, bool)
                else _enum_name(predictor, PREDICTOR)
            ),
            subsampling=(
                (int(subsampling[0]), int(subsampling[1]))
                if subsampling is not None
                else None
            ),
            metadata=metadata,
            extratags=extratags,
            truncate=bool(truncate),
            maxworkers=int(maxworkers) if maxworkers is not None else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Create instance of model from dictionary."""
        _, cfg = parse_named_configuration(
            data, 'tifffile', require_configuration=False
        )
        return cls(**(cfg if cfg is not None else {}))  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, JSON]:
        """Convert instance of model to dictionary."""
        cfg: dict[str, JSON] = {}
        if self.key is not None:
            cfg['key'] = self.key  # type: ignore[assignment]
        if self.series is not None:
            cfg['series'] = self.series
        if self.kind is not None:
            cfg['kind'] = self.kind
        if self.level is not None:
            cfg['level'] = self.level
        if self.squeeze is not None:
            cfg['squeeze'] = self.squeeze
        if self.buffersize is not None:
            cfg['buffersize'] = self.buffersize
        if self.bigtiff:
            cfg['bigtiff'] = self.bigtiff
        if self.byteorder is not None:
            cfg['byteorder'] = self.byteorder
        if self.photometric is not None:
            cfg['photometric'] = self.photometric
        if self.planarconfig is not None:
            cfg['planarconfig'] = self.planarconfig
        if self.extrasamples is not None:
            cfg['extrasamples'] = list(self.extrasamples)
        if self.volumetric:
            cfg['volumetric'] = self.volumetric
        if self.tile is not None:
            cfg['tile'] = list(self.tile)
        if self.rowsperstrip is not None:
            cfg['rowsperstrip'] = self.rowsperstrip
        if self.bitspersample is not None:
            cfg['bitspersample'] = self.bitspersample
        if self.compression is not None:
            cfg['compression'] = self.compression
        if self.compressionargs is not None:
            cfg['compressionargs'] = self.compressionargs
        if self.predictor is not None:
            cfg['predictor'] = self.predictor
        if self.subsampling is not None:
            cfg['subsampling'] = list(self.subsampling)
        if self.metadata is not METADATA_DEFAULT:
            cfg['metadata'] = self.metadata
        if self.extratags is not None:
            cfg['extratags'] = list(self.extratags)
        if self.truncate:
            cfg['truncate'] = self.truncate
        if self.maxworkers is not None:
            cfg['maxworkers'] = self.maxworkers
        if cfg:
            return {'name': 'tifffile', 'configuration': cfg}
        return {'name': 'tifffile'}

    def _decode_sync(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        with BytesIO(chunk_bytes.as_buffer_like()) as fh, TiffFile(fh) as tif:
            decoded = tif.asarray(
                key=self.key,
                series=self.series,
                kind=self.kind,
                level=self.level,
                squeeze=self.squeeze,
                maxworkers=self.maxworkers,
                buffersize=self.buffersize,
            )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(
            decoded.reshape(chunk_spec.shape)
        )

    async def _decode_single(
        self, chunk_bytes: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return await asyncio.to_thread(
            self._decode_sync, chunk_bytes, chunk_spec
        )

    def _encode_sync(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        arr = numpy.atleast_2d(numpy.squeeze(chunk_array.as_numpy_array()))
        with BytesIO() as fh:
            with TiffWriter(
                fh,
                bigtiff=self.bigtiff,
                byteorder=self.byteorder,
                kind=self.kind,
            ) as tif:
                tif.write(
                    arr,
                    photometric=self.photometric,
                    planarconfig=self.planarconfig,
                    extrasamples=self.extrasamples,
                    volumetric=self.volumetric,
                    tile=self.tile,
                    rowsperstrip=self.rowsperstrip,
                    bitspersample=self.bitspersample,
                    compression=self.compression,
                    compressionargs=self.compressionargs,
                    predictor=self.predictor,
                    subsampling=self.subsampling,
                    metadata=self.metadata,
                    extratags=self.extratags,
                    truncate=self.truncate,
                    maxworkers=self.maxworkers,
                )
            encoded = fh.getvalue()
        return chunk_spec.prototype.buffer.from_bytes(encoded)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        return await asyncio.to_thread(
            self._encode_sync, chunk_array, chunk_spec
        )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        """Compute size of encoded chunk in bytes."""
        raise NotImplementedError


class ZarrStore(Store):
    """Zarr 3 store base class.

    Parameters:
        fillvalue:
            Value to use for missing chunks of Zarr store.
            The default is 0.
        chunkmode:
            Specifies how to chunk data.
        read_only:
            Passed to :py:class:`zarr.abc.store.Store`.

    References:
        1. https://zarr.readthedocs.io/en/stable/api/zarr/abc/store/
        2. https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
        3. https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
        4. https://ngff.openmicroscopy.org/specifications/0.5/

    """

    _read_only: bool
    _store: dict[str, Any]
    _fillvalue: float
    _chunkmode: int

    def __init__(
        self,
        /,
        *,
        fillvalue: float | None = None,
        chunkmode: CHUNKMODE | int | str | None = None,
        read_only: bool = True,
    ) -> None:
        super().__init__(read_only=read_only)

        self._store = {}
        self._fillvalue = 0 if fillvalue is None else fillvalue
        if chunkmode is None:
            self._chunkmode = CHUNKMODE(0)
        else:
            self._chunkmode = enumarg(CHUNKMODE, chunkmode)

    def __hash__(self) -> int:
        return hash(
            (tuple(self._store.items()), self._fillvalue, self._chunkmode)
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether objects are equal."""
        return (
            isinstance(other, type(self))
            and self._store == other._store
            and self._fillvalue == other._fillvalue
            and self._chunkmode == other._chunkmode
        )

    @override
    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        """Return possibly partial values from given key_ranges."""
        # print(f'get_partial_values({key_ranges=})')
        return [
            await self.get(key, prototype, byte_range)
            for key, byte_range in key_ranges
        ]

    @override
    @property
    def supports_writes(self) -> bool:
        """Store supports writes."""
        return not self._read_only

    def _set(self, key: str, value: Buffer, /) -> None:
        """Store (key, value) pair."""
        raise NotImplementedError

    @override
    async def set(self, key: str, value: Buffer) -> None:
        """Store (key, value) pair."""
        self._set(key, value)

    @override
    @property
    def supports_deletes(self) -> bool:
        """Store supports deletes."""
        return False

    @override
    async def delete(self, key: str) -> None:
        """Remove key from store."""
        msg = 'ZarrStore does not support deletes'
        raise PermissionError(msg)

    @override
    @property
    def supports_listing(self) -> bool:
        """Store supports listing."""
        return True

    @override
    async def list(self) -> AsyncIterator[str]:
        """Return all keys in store."""
        for key in self._store:
            yield key

    @override
    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """Return all keys in store that begin with prefix.

        Keys are returned relative to the root of the store.

        """
        for key in list(self._store):
            if key.startswith(prefix):
                yield key

    @override
    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """Return all keys and prefixes with prefix.

        Keys and prefixes do not contain the character "/" after the given
        prefix.

        """
        prefix = prefix.rstrip('/')
        if prefix == '':
            keys_unique = {k.split('/')[0] for k in self._store}
        else:
            keys_unique = {
                key.removeprefix(prefix + '/').split('/')[0]
                for key in self._store
                if key.startswith(prefix + '/') and key != prefix
            }
        for key in keys_unique:
            yield key

    @property
    def is_multiscales(self) -> bool:
        """Return whether ZarrStore contains multiscales."""
        for key in ('zarr.json', '.zattrs'):
            if b'multiscales' in self._store.get(key, b''):
                return True
        return False

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'

    # async def _get_many(
    #     self,
    #     requests: Iterable[tuple[str, BufferPrototype, ByteRequest | None]]
    # ) -> AsyncGenerator[tuple[str, Buffer | None], None]:
    #     print(f'_get_many({requests=})')
    #     return super()._get_many(requests)

    # async def getsize_prefix(self, prefix: str) -> int:
    #     print(f'getsize_prefix({prefix=})')
    #     return super().getsize_prefix(prefix)


class ZarrTiffStore(ZarrStore):
    """Zarr 3 store interface to image array in TiffPage or TiffPageSeries.

    The store uses Zarr v3 format. Pyramidal series use OME-Zarr v0.5
    multiscales metadata.

    ZarrTiffStore is using a TiffFile instance for reading and decoding chunks.
    Therefore, ZarrTiffStore instances cannot be pickled.

    For writing, image data must be stored in uncompressed, unpredicted,
    and unpacked form. Sparse strips and tiles are not written.

    Parameters:
        arg:
            TIFF page or series to wrap as Zarr store.
        level:
            Pyramidal level to wrap. The default is 0.
        chunkmode:
            Use strips or tiles (0) or whole page data (2) as chunks.
            The default is 0.
        fillvalue:
            Value to use for missing chunks. The default is 0.
        dimension_names:
            Names of dimensions in image array.
            Overrides dimension names derived from series axes.
        zattrs:
            Additional attributes to store in `.zattrs`.
        multiscales:
            Create a multiscales-compatible Zarr group store.
            By default, create a Zarr array store for pages and non-pyramidal
            series.
            If *True*, encode coordinate metadata (pixel sizes, units, offsets)
            using the NGFF 0.5 multiscales structure.
        lock:
            Reentrant lock to synchronize seeks and reads from file.
            By default, the lock of the parent's file handle is used.
        maxworkers:
            If `chunkmode=0`, asynchronously run chunk decode function
            in separate thread if greater than 1.
            If `chunkmode=2`, maximum number of threads to concurrently decode
            strips or tiles.
            If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS` or
            asyncio assigned threads.
        buffersize:
            Approximate number of bytes to read from file in one pass
            if `chunkmode=2`. The default is :py:attr:`_TIFF.BUFFERSIZE`.
        read_only:
            Passed to :py:class:`zarr.abc.store.Store`.
        _openfiles:
            Internal API.

    """

    _data: list[TiffPageSeries]
    _filecache: FileCache
    _transform: Callable[[NDArray[Any]], NDArray[Any]] | None
    _maxworkers: int
    _buffersize: int | None
    _multiscales: bool

    def __init__(
        self,
        arg: TiffPage | TiffFrame | TiffPageSeries,
        /,
        *,
        level: int | None = None,
        chunkmode: CHUNKMODE | int | str | None = None,
        fillvalue: float | None = None,
        dimension_names: Sequence[str] | None = None,
        zattrs: dict[str, Any] | None = None,
        multiscales: bool | None = None,
        lock: threading.RLock | NullContext | None = None,
        maxworkers: int | None = None,
        buffersize: int | None = None,
        read_only: bool | None = None,
        _openfiles: int | None = None,
    ) -> None:
        if chunkmode is None:
            chunkmode = CHUNKMODE(0)
        else:
            chunkmode = enumarg(CHUNKMODE, chunkmode)

        if chunkmode not in {0, 2}:
            msg = f'{chunkmode!r} not implemented'
            raise NotImplementedError(msg)

        self._buffersize = buffersize

        if isinstance(arg, TiffPageSeries):
            self._data = arg.levels
            self._transform = arg.transform
            if multiscales is False:
                level = 0
            if level is not None:
                self._data = [self._data[level]]
            name = arg.name
        else:
            self._data = [TiffPageSeries([arg])]
            self._transform = None
            name = 'Unnamed'

        if not maxworkers:
            maxworkers = self._data[0].keyframe.maxworkers
            if maxworkers < 3 and chunkmode == 0:
                maxworkers = 1
        self._maxworkers = maxworkers

        fh = self._data[0].keyframe.parent.root.filehandle

        if read_only is None:
            read_only = not fh.writable() or chunkmode != 0

        super().__init__(
            fillvalue=fillvalue, chunkmode=chunkmode, read_only=read_only
        )

        if lock is None:
            fh.set_lock(True)
            lock = fh.lock
        self._filecache = FileCache(size=_openfiles, lock=lock)

        zattrs = {} if zattrs is None else dict(zattrs)
        # TODO: Zarr Encoding Specification
        # https://xarray.pydata.org/en/stable/internals/zarr-encoding-spec.html

        if multiscales or (multiscales is None and len(self._data) > 1):
            # multiscales: NGFF 0.5 + Zarr v3
            self._multiscales = True
            dimension_names = tuple(dimension_names or self._data[0].axes)
            series0 = self._data[0]
            shape0 = series0.shape
            coord_units = series0.coord_units
            coord_offsets = series0.coord_offsets

            # NGFF 0.5 axis type from TIFF axis character code
            axis_type = {
                'X': 'space',
                'Y': 'space',
                'Z': 'space',
                'T': 'time',
                'C': 'channel',
                'S': 'channel',
            }

            ngff_axes: list[dict[str, Any]] = []
            for ax in dimension_names:
                ngff_axis: dict[str, Any] = {'name': ax}
                ax_type = axis_type.get(ax.upper())
                if ax_type:
                    ngff_axis['type'] = ax_type
                unit = coord_units.get(ax)
                if unit:
                    ngff_axis['unit'] = unit
                ngff_axes.append(ngff_axis)

            datasets: list[dict[str, Any]] = []
            for ilevel, series in enumerate(self._data):
                level_scales = series.coord_scales
                scale = [
                    float(
                        level_scales[ax]
                        if ax in level_scales and s > 0
                        else s0 / s if s > 0 else 1.0
                    )
                    for ax, s0, s in zip(
                        dimension_names, shape0, series.shape, strict=True
                    )
                ]
                coord_transforms: list[dict[str, Any]] = [
                    {'type': 'scale', 'scale': scale}
                ]
                if ilevel == 0:
                    offsets = [coord_offsets.get(ax) for ax in dimension_names]
                    if any(o is not None for o in offsets):
                        coord_transforms.append(
                            {
                                'type': 'translation',
                                'translation': [
                                    float(o) if o is not None else 0.0
                                    for o in offsets
                                ],
                            }
                        )
                datasets.append(
                    {
                        'path': str(ilevel),
                        'coordinateTransformations': coord_transforms,
                    }
                )

            self._store['zarr.json'] = _json_dumps(
                {
                    'zarr_format': 3,
                    'node_type': 'group',
                    'attributes': {
                        'ome': {
                            'version': '0.5',
                            'multiscales': [
                                {
                                    'name': name,
                                    'axes': ngff_axes,
                                    'datasets': datasets,
                                }
                            ],
                        },
                        **zattrs,
                    },
                }
            )
            for ilevel, series in enumerate(self._data):
                # use level-suffixed axis names for axes that differ in size
                # from the base level (xarray requires unique dim names per
                # dataset that have different lengths)
                if ilevel == 0:
                    level_dims = dimension_names
                else:
                    level_dims = tuple(
                        (f'{ax}{ilevel}' if i != j else ax)
                        for ax, i, j in zip(
                            dimension_names,
                            series.shape,
                            shape0,
                            strict=True,
                        )
                    )
                fillvalue, _ = self._init_zarray(
                    series,
                    f'{ilevel}/zarr.json',
                    fillvalue,
                    level_dims,
                )
        else:
            self._multiscales = False
            series = self._data[0]
            dimension_names = tuple(dimension_names or series.axes)
            fillvalue, _ = self._init_zarray(
                series, 'zarr.json', fillvalue, dimension_names
            )
            if zattrs:
                zarr_json = json.loads(self._store['zarr.json'])
                zarr_json['attributes'].update(zattrs)
                self._store['zarr.json'] = _json_dumps(zarr_json)

    def _init_zarray(
        self,
        series: TiffPageSeries,
        key: str,
        fillvalue: float | None,
        /,
        dimension_names: tuple[str, ...] | None = None,
    ) -> tuple[float | None, tuple[int, ...]]:
        """Store zarr.json for series; return updated fillvalue and shape."""
        keyframe = series.keyframe
        keyframe.decode  # noqa: B018 - cache decode function
        shape = series.shape
        dtype = series.dtype
        if fillvalue is None:
            fillvalue = keyframe.nodata
            self._fillvalue = fillvalue
        if keyframe._dtype is None:
            # empty page (e.g. C2PA), consistent with TiffPage.asarray()
            shape = (0, 0)
            dtype = numpy.dtype(numpy.bool_)
            dimension_names = None  # reset: forced 2D shape, names invalid
        chunks = keyframe.shape if self._chunkmode else keyframe.chunks
        chunk_shape = list(_chunks(chunks, shape, keyframe.shaped))
        # get() returns already-decoded, native-endian numpy data, so the
        # bytes codec must use native byte order regardless of TIFF byteorder
        if dtype.itemsize == 1:
            bytes_codec: dict[str, Any] = {'name': 'bytes'}
        elif sys.byteorder == 'big':
            bytes_codec = {
                'name': 'bytes',
                'configuration': {'endian': 'big'},
            }
        else:
            bytes_codec = {
                'name': 'bytes',
                'configuration': {'endian': 'little'},
            }
        self._store[key] = _json_dumps(
            {
                'zarr_format': 3,
                'node_type': 'array',
                'shape': list(shape),
                'data_type': dtype.name,
                'chunk_grid': {
                    'name': 'regular',
                    'configuration': {'chunk_shape': chunk_shape},
                },
                'chunk_key_encoding': {
                    'name': 'default',
                    'configuration': {'separator': '/'},
                },
                'fill_value': _json_value(fillvalue, dtype),
                'codecs': [bytes_codec],
                'dimension_names': dimension_names,
                'attributes': {},
            }
        )
        if not self._read_only:
            self._read_only = not (
                keyframe.compression == 1
                and keyframe.fillorder == 1
                and keyframe.sampleformat in {1, 2, 3, 6}
                and keyframe.bitspersample in {8, 16, 32, 64, 128}
                # and (
                #     keyframe.rowsperstrip == 0
                #     or keyframe.imagelength % keyframe.rowsperstrip == 0
                # )
            )
        return fillvalue, shape

    @override
    def close(self) -> None:
        """Close store."""
        super().close()
        self._filecache.clear()

    def write_fsspec(
        self,
        jsonfile: str | os.PathLike[Any] | TextIO,
        /,
        url: str | None,
        *,
        groupname: str | None = None,
        templatename: str | None = None,
        compressors: dict[COMPRESSION | int, str | None] | None = None,
        zarr_format: int | None = None,
        version: int | None = None,
        _shape: Sequence[int] | None = None,
        _axes: Sequence[str] | None = None,
        _index: Sequence[int] | None = None,
        _append: bool = False,
        _close: bool = True,
    ) -> None:
        """Write fsspec ReferenceFileSystem as JSON to file.

        Parameters:
            jsonfile:
                Name or open file handle of output JSON file.
            url:
                Remote location of TIFF file(s) without file name(s).
            groupname:
                Zarr group name.
            templatename:
                Version 1 URL template name. The default is 'u'.
            compressors:
                Mapping of :py:class:`COMPRESSION` codes to Numcodecs codec
                names (zarr format 2) or imagecodecs.zarr codec names
                (zarr format 3).
            zarr_format:
                Version of Zarr array format to write.
                The default is 2. If 3, write Zarr version 3 format using
                :py:mod:`imagecodecs.zarr` native codec specifications.
                Chunk keys use 'c/' prefix with '/' separator and
                :py:meth:`imagecodecs.zarr.register_codecs` must be called
                before reading the resulting store.
            version:
                Version of fsspec file to write. The default is 0.
            _shape:
                Shape of file sequence (experimental).
            _axes:
                Axes of file sequence (experimental).
            _index:
                Index of file in sequence (experimental).
            _append:
                If *True*, only write index keys and values (experimental).
            _close:
                If *True*, no more appends (experimental).

        Raises:
            ValueError:
                ZarrTiffStore cannot be represented as ReferenceFileSystem
                due to features that are not supported by Zarr, Numcodecs,
                or Imagecodecs:

                - compressors, such as CCITT
                - filters, such as bitorder reversal, packed integers
                - dtypes, such as float24, complex integers
                - JPEGTables in multi-page series
                - incomplete chunks, such as `imagelength % rowsperstrip != 0`

                Files containing incomplete tiles may fail at runtime.

        Notes:
            Parameters ``_shape``,  ``_axes``, ``_index``, ``_append``, and
            ``_close`` are an experimental API for joining the
            ReferenceFileSystems of multiple files of a TiffSequence.

            Multiscales metadata for pyramidal series uses OME-Zarr v0.5
            when ``zarr_format=3`` and OME-Zarr v0.4 when ``zarr_format=2``.

        References:
            - `fsspec ReferenceFileSystem format
              <https://github.com/fsspec/kerchunk>`_

        """
        compressors = {
            1: None,
            # 2: 'imagecodecs_ccittrle',
            # 3: 'imagecodecs_ccittfax3',
            # 4: 'imagecodecs_ccittfax4',
            5: 'imagecodecs_lzw',
            7: 'imagecodecs_jpeg',
            8: 'imagecodecs_zlib',
            22610: 'imagecodecs_jpegxr',
            32773: 'imagecodecs_packbits',
            32946: 'imagecodecs_zlib',
            33003: 'imagecodecs_jpeg2k',
            33004: 'imagecodecs_jpeg2k',
            33005: 'imagecodecs_jpeg2k',
            33007: 'imagecodecs_jpeg',
            34712: 'imagecodecs_jpeg2k',
            34887: 'imagecodecs_lerc',
            34892: 'imagecodecs_jpeg',
            34925: 'imagecodecs_lzma',
            34933: 'imagecodecs_png',
            34934: 'imagecodecs_jpegxr',
            # 48124: 'imagecodecs_jetraw',  # not supported by imagecodecs.zarr
            50000: 'imagecodecs_zstd',
            50001: 'imagecodecs_webp',
            50002: 'imagecodecs_jpegxl',
            50013: 'imagecodecs_zlib',  # pixtiff
            52546: 'imagecodecs_jpegxl',
            **({} if compressors is None else compressors),
        }

        for series in self._data:
            errormsg = ' not supported by the fsspec ReferenceFileSystem'
            keyframe = series.keyframe
            if (
                keyframe.compression in {65000, 65001, 65002}
                and keyframe.parent.is_eer
            ):
                compressors[keyframe.compression] = 'imagecodecs_eer'
            if keyframe.compression not in compressors:
                raise ValueError(f'{keyframe.compression!r} is' + errormsg)
            if keyframe.fillorder != 1:
                raise ValueError(f'{keyframe.fillorder!r} is' + errormsg)
            if keyframe.sampleformat not in {1, 2, 3, 6}:
                # TODO: support float24 and cint via filters?
                raise ValueError(f'{keyframe.sampleformat!r} is' + errormsg)
            if (
                keyframe.bitspersample
                not in {
                    8,
                    16,
                    32,
                    64,
                    128,
                }
                and keyframe.compression
                not in {
                    # JPEG
                    7,
                    33007,
                    34892,
                }
                and compressors[keyframe.compression] != 'imagecodecs_eer'
            ):
                raise ValueError(
                    f'BitsPerSample {keyframe.bitspersample} is' + errormsg
                )
            if (
                not self._chunkmode
                and not keyframe.is_tiled
                and keyframe.imagelength % keyframe.rowsperstrip
            ):
                raise ValueError('incomplete chunks are' + errormsg)
            if self._chunkmode and not keyframe.is_final:
                raise ValueError(f'{self._chunkmode!r} is' + errormsg)
            if keyframe.jpegtables is not None and len(series) > 1:
                raise ValueError(
                    'JPEGTables in multi-page files are' + errormsg
                )

        if url is None:
            url = ''
        elif url and url[-1] != '/':
            url += '/'
        url = url.replace('\\', '/')

        if groupname is None:
            groupname = ''
        elif groupname and groupname[-1] != '/':
            groupname += '/'

        byteorder: ByteOrder | None = '<' if sys.byteorder == 'big' else '>'
        if (
            self._data[0].keyframe.parent.byteorder != byteorder
            or self._data[0].keyframe.dtype is None
            or self._data[0].keyframe.dtype.itemsize == 1
        ):
            byteorder = None

        index: str
        _shape = [] if _shape is None else list(_shape)
        _axes = [] if _axes is None else list(_axes)
        if len(_shape) != len(_axes):
            msg = 'len(_shape) != len(_axes)'
            raise ValueError(msg)
        if _index is None:
            index = ''
        elif len(_shape) != len(_index):
            msg = 'len(_shape) != len(_index)'
            raise ValueError(msg)
        elif _index:
            index = '.'.join(str(i) for i in _index)
            index += '.'

        if zarr_format is None:
            zarr_format = 2
        elif zarr_format not in {2, 3}:
            msg = f'invalid {zarr_format=!r} not in {{2, 3}}'
            raise ValueError(msg)

        refs: dict[str, Any] = {}
        refzarr: dict[str, Any]
        if version == 1:
            if _append:
                msg = 'cannot append to version 1'
                raise ValueError(msg)
            if templatename is None:
                templatename = 'u'
            refs['version'] = 1
            refs['templates'] = {}
            refs['gen'] = []
            templates = {}
            if self._data[0].is_multifile:
                i = 0
                for page in self._data[0]:
                    if page is None or page.keyframe is None:
                        continue
                    filename = page.keyframe.parent.filehandle.name
                    if filename in templates:
                        continue
                    key = f'{templatename}{i}'
                    templates[filename] = f'{{{{{key}}}}}'
                    refs['templates'][key] = url + filename
                    i += 1
            else:
                filename = self._data[0].keyframe.parent.filehandle.name
                key = f'{templatename}'
                templates[filename] = f'{{{{{key}}}}}'
                refs['templates'][key] = url + filename

            refs['refs'] = refzarr = {}
        else:
            refzarr = refs

        if not _append:
            if zarr_format == 3:
                _write_fsspec_v3_metadata(
                    self._store,
                    self._data,
                    refzarr,
                    groupname,
                    compressors,
                    _shape,
                    _axes,
                )
            else:
                _write_fsspec_v2_metadata(
                    self._store,
                    self._data,
                    refzarr,
                    groupname,
                    byteorder,
                    compressors,
                    _shape,
                    _axes,
                )

        fh: TextIO
        with contextlib.ExitStack() as stack:
            if hasattr(jsonfile, 'write'):
                fh = jsonfile  # type: ignore[assignment]
            else:
                fh = stack.enter_context(open(jsonfile, 'w', encoding='utf-8'))

            if version == 1:
                fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
                indent = '  '
            elif _append:
                indent = ' '
            else:
                fh.write(json.dumps(refs, indent=1)[:-2])
                indent = ' '

            offset: int | None
            chunk_sep = '/' if zarr_format == 3 else '.'
            chunk_prefix = 'c/' if zarr_format == 3 else ''
            for key, value_bytes in self._store.items():
                if not key.endswith('zarr.json'):
                    continue
                value = json.loads(value_bytes)
                if value.get('node_type') != 'array':
                    continue
                shape = value['shape']
                chunks = value['chunk_grid']['configuration']['chunk_shape']
                levelstr = (key.split('/')[0] + '/') if '/' in key else ''
                for chunkindex in _ndindex(shape, chunks):
                    # internal_key uses '.' for _parse_key/_indices
                    internal_key = levelstr + chunkindex
                    fsspec_key = (
                        levelstr
                        + chunk_prefix
                        + chunkindex.replace('.', chunk_sep)
                        if zarr_format == 3
                        else internal_key
                    )
                    keyframe, page, _, offset, bytecount = self._parse_key(
                        internal_key
                    )
                    if page and self._chunkmode and offset is None:
                        offset = page.dataoffsets[0]
                        bytecount = keyframe.nbytes
                    if offset and bytecount:
                        filename = keyframe.parent.filehandle.name
                        if version == 1:
                            filename = templates[filename]
                        else:
                            filename = f'{url}{filename}'
                        fh.write(
                            f',\n{indent}"{groupname}{fsspec_key}": '
                            f'["{filename}", {offset}, {bytecount}]'
                        )

            # TODO: support nested groups
            if version == 1:
                fh.write('\n }\n}')
            elif _close:
                fh.write('\n}')

    @override
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Return value associated with key."""
        # print(f'get({key=}, {byte_range=})')
        if byte_range is not None:
            msg = f'{byte_range=!r} not supported'
            raise NotImplementedError(msg)

        if key in self._store:
            return prototype.buffer.from_bytes(self._store[key])

        if key.endswith(('.zmetadata', '.zarray', '.zgroup', '.zattrs')):
            return None

        keyframe, page, chunkindex, offset, bytecount = self._parse_key(key)

        if page is None or offset == 0 or bytecount == 0:
            return None

        fh = page.parent.filehandle

        if self._chunkmode:
            if offset is not None:
                # contiguous image data in page or series
                # create virtual frame instead of loading page from file
                assert bytecount is not None
                page = TiffFrame(
                    page.parent,
                    index=0,
                    keyframe=keyframe,
                    dataoffsets=(offset,),
                    databytecounts=(bytecount,),
                )
            # TODO: use asyncio.to_thread ?
            self._filecache.open(fh)
            chunk = page.asarray(
                lock=self._filecache.lock,
                maxworkers=self._maxworkers,
                buffersize=self._buffersize,
            )
            self._filecache.close(fh)
            if self._transform is not None:
                chunk = self._transform(chunk)
            return prototype.buffer.from_array_like(
                chunk.reshape(-1).view('B')
            )

        assert offset is not None
        assert bytecount is not None
        chunk_bytes = self._filecache.read(fh, offset, bytecount)

        decodeargs: dict[str, Any] = {'_fullsize': True}
        if page.jpegtables is not None:
            decodeargs['jpegtables'] = page.jpegtables
        if keyframe.jpegheader is not None:
            decodeargs['jpegheader'] = keyframe.jpegheader

        assert chunkindex is not None
        keyframe.decode  # noqa: B018 - cache decode function
        if self._maxworkers > 1:
            decoded = await asyncio.to_thread(
                keyframe.decode, chunk_bytes, chunkindex, **decodeargs
            )
        else:
            decoded = keyframe.decode(chunk_bytes, chunkindex, **decodeargs)
        chunk = decoded[0]  # type: ignore[assignment]
        assert chunk is not None
        if self._transform is not None:
            chunk = self._transform(chunk)

        chunks = keyframe.chunks
        if chunk.size != product(chunks):
            msg = f'{chunk.size=} != {product(chunks)=}'
            raise RuntimeError(msg)
        return prototype.buffer.from_array_like(chunk.reshape(-1).view('B'))

    @override
    async def exists(self, key: str) -> bool:
        """Return whether key exists in store."""
        # print(f'exists({key=})')
        if key in self._store:
            return True
        try:
            _kf, page, _ci, offset, bytecount = self._parse_key(key)
        except (KeyError, IndexError):
            return False
        if self._chunkmode and offset is None:
            return True
        return (
            page is not None
            and offset is not None
            and bytecount is not None
            and offset > 0
            and bytecount > 0
        )

    @override
    async def set(self, key: str, value: Buffer) -> None:
        """Store (key, value) pair."""
        if self._read_only:
            msg = 'ZarrTiffStore is read-only'
            raise PermissionError(msg)

        if key in self._store or key.endswith(
            ('zarr.json', '.zmetadata', '.zarray', '.zgroup', '.zattrs')
        ):
            return

        _keyframe, page, _chunkindex, offset, bytecount = self._parse_key(key)
        if (
            page is None
            or offset is None
            or offset == 0
            or bytecount is None
            or bytecount == 0
        ):
            return
        data = value.to_bytes()
        if bytecount < len(data):
            data = data[:bytecount]
        self._filecache.write(page.parent.filehandle, offset, data)

    def _parse_key(self, key: str, /) -> tuple[
        TiffPage,
        TiffPage | TiffFrame | None,
        int | None,
        int | None,
        int | None,
    ]:
        """Return keyframe, page, index, offset, and bytecount from key.

        Raise KeyError if key is not valid.

        """
        if self._multiscales:
            try:
                level, chunk_key = key.split('/', 1)
                series = self._data[int(level)]
            except (ValueError, IndexError) as exc:
                raise KeyError(key) from exc
            key = chunk_key
        else:
            series = self._data[0]
        # Normalize Zarr v3 chunk key: strip 'c/' prefix, convert '/' to '.'
        key = key.removeprefix('c/')
        if '/' in key:
            key = key.replace('/', '.')
        keyframe = series.keyframe
        page: TiffPage | TiffFrame | None = None
        offset: int | None = None
        pageindex, chunkindex = self._indices(key, series)
        if series.dataoffset is not None:
            # contiguous or truncated
            page = series[0]
            if page is None or page.dtype is None or page.keyframe is None:
                return keyframe, None, chunkindex, 0, 0
            offset = pageindex * page.size * page.dtype.itemsize
            try:
                offset += page.dataoffsets[chunkindex]
            except IndexError as exc:
                raise KeyError(key) from exc
            if self._chunkmode:
                bytecount = page.size * page.dtype.itemsize
                return page.keyframe, page, chunkindex, offset, bytecount
        elif self._chunkmode:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None or page.keyframe is None:
                return keyframe, None, None, 0, 0
            return page.keyframe, page, None, None, None
        else:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None or page.keyframe is None:
                return keyframe, None, chunkindex, 0, 0
            try:
                offset = page.dataoffsets[chunkindex]
            except IndexError:
                # raise KeyError(key) from exc
                # issue #249: Philips may be missing last row of tiles
                return page.keyframe, page, chunkindex, 0, 0
        try:
            bytecount = page.databytecounts[chunkindex]
        except IndexError as exc:
            raise KeyError(key) from exc
        return page.keyframe, page, chunkindex, offset, bytecount

    def _indices(self, key: str, series: TiffPageSeries, /) -> tuple[int, int]:
        """Return page and strile indices from Zarr chunk index."""
        keyframe = series.keyframe
        shape = series.shape
        try:
            indices = [int(i) for i in key.split('.')]
        except ValueError as exc:
            raise KeyError(key) from exc
        assert len(indices) == len(shape)
        if self._chunkmode:
            chunked = (1,) * len(keyframe.shape)
        else:
            chunked = keyframe.chunked
        p = 1
        for j, s in enumerate(shape[::-1]):
            p *= s
            if p == keyframe.size:
                i = len(indices) - j - 1
                frames_indices = indices[:i]
                strile_indices = indices[i:]
                frames_chunked = shape[:i]
                strile_chunked = list(shape[i:])  # updated later
                break
        else:
            msg = 'shape does not match keyframe size'
            raise RuntimeError(msg)
        if len(strile_chunked) == len(keyframe.shape):
            strile_chunked = list(chunked)
        else:
            # get strile_chunked including singleton dimensions
            i = len(strile_indices) - 1
            j = len(keyframe.shape) - 1
            while True:
                if strile_chunked[i] == keyframe.shape[j]:
                    strile_chunked[i] = chunked[j]
                    i -= 1
                    j -= 1
                elif strile_chunked[i] == 1:
                    i -= 1
                else:
                    msg = 'shape does not match page shape'
                    raise RuntimeError(msg)
                if i < 0 or j < 0:
                    break
            assert product(strile_chunked) == product(chunked)
        frameindex = 0
        strileindex = 0
        if frames_indices:
            frameindex = int(
                numpy.ravel_multi_index(frames_indices, frames_chunked)
            )
        if strile_indices:
            strileindex = int(
                numpy.ravel_multi_index(strile_indices, strile_chunked)
            )
        return frameindex, strileindex


class ZarrFileSequenceStore(ZarrStore):
    """Zarr 3 store interface to image array in FileSequence.

    The store uses Zarr v3 format.

    Parameters:
        filesequence:
            FileSequence instance to wrap as Zarr store.
            Files in containers are not supported.
        fillvalue:
            Value to use for missing chunks. The default is 0.
        chunkmode:
            Currently only one chunk per file is supported.
        chunkshape:
            Shape of chunk in each file.
            Must match ``FileSequence.imread(file, **imreadargs).shape``.
        chunkdtype:
            Data type of chunk in each file.
            Must match ``FileSequence.imread(file, **imreadargs).dtype``.
        axestiled:
            Axes to be tiled. Map stacked sequence axis to chunk axis.
        dimension_names:
            Names of dimensions in image array.
            If *None* and all chunk axes are tiled, derive from
            ``filesequence.dims``. Otherwise not set in the store.
        zattrs:
            Additional attributes to store in `.zattrs`.
        ioworkers:
            If not 1, asynchronously run `imread` function in separate thread.
            If enabled, internal threading for the `imread` function
            should be disabled.
        read_only:
            Passed to :py:class:`zarr.abc.store.Store`.
        imreadargs:
            Arguments passed to :py:attr:`FileSequence.imread`.
        **kwargs:
            Arguments passed to :py:attr:`FileSequence.imread` in addition
            to `imreadargs`.

    Notes:
        If `chunkshape` or `chunkdtype` are *None* (default), their values
        are determined by reading the first file with
        ``FileSequence.imread(arg.files[0], **imreadargs)``.

    """

    _imread: Callable[..., NDArray[Any]]
    _lookup: dict[tuple[int, ...], str]
    _chunks: tuple[int, ...]
    _dtype: numpy.dtype[Any]
    _tiled: TiledSequence
    _commonpath: str
    _ioworkers: int
    _kwargs: dict[str, Any]

    def __init__(
        self,
        filesequence: FileSequence,
        /,
        *,
        fillvalue: float | None = None,
        chunkmode: CHUNKMODE | int | str | None = None,
        chunkshape: Sequence[int] | None = None,
        chunkdtype: DTypeLike | None = None,
        axestiled: dict[int, int] | Sequence[tuple[int, int]] | None = None,
        dimension_names: Sequence[str] | None = None,
        zattrs: dict[str, Any] | None = None,
        ioworkers: int | None = 1,
        imreadargs: dict[str, Any] | None = None,
        read_only: bool = True,
        **kwargs: Any,
    ) -> None:
        if not isinstance(filesequence, FileSequence):
            msg = 'not a FileSequence'  # type: ignore[unreachable]
            raise TypeError(msg)

        if filesequence._container:
            msg = 'cannot open container as Zarr store'
            raise NotImplementedError(msg)

        if len(filesequence) == 0:
            msg = 'filesequence is empty'
            raise ValueError(msg)

        super().__init__(
            fillvalue=fillvalue, chunkmode=chunkmode, read_only=read_only
        )

        if self._chunkmode not in {0, 3}:
            msg = f'invalid chunkmode {self._chunkmode!r}'
            raise ValueError(msg)

        # TODO: deprecate kwargs?
        if imreadargs is not None:
            kwargs |= imreadargs

        self._ioworkers = 1 if ioworkers is None else ioworkers

        self._kwargs = kwargs
        self._imread = filesequence.imread
        self._commonpath = filesequence.commonpath()

        if chunkshape is None or chunkdtype is None:
            chunk = filesequence.imread(filesequence[0], **kwargs)
            self._chunks = chunk.shape
            self._dtype = chunk.dtype
        else:
            self._chunks = tuple(chunkshape)
            self._dtype = numpy.dtype(chunkdtype)

        self._tiled = TiledSequence(
            filesequence.shape, self._chunks, axestiled=axestiled
        )
        self._lookup = dict(
            zip(
                self._tiled.indices(filesequence.indices),
                filesequence,
                strict=True,
            )
        )

        if (
            dimension_names is None
            and filesequence.dims
            and len(self._tiled._axestiled) == self._tiled._chunkdims
        ):
            # Auto-derive dimension names when all chunk axes are tiled.
            # The tiled shape drops tiled stack axes and appends chunk axes,
            # so reorder: non-tiled stack dims (in order) + tiled dims (in
            # chunk-axis order).
            tiled_stack_axes = {ax0 for ax0, _ in self._tiled._axestiled}
            dimension_names = (
                *(
                    name
                    for i, name in enumerate(filesequence.dims)
                    if i not in tiled_stack_axes
                ),
                *(
                    filesequence.dims[ax0]
                    for ax0, _ in sorted(
                        self._tiled._axestiled, key=lambda t: t[1]
                    )
                ),
            )

        zattrs = {} if zattrs is None else dict(zattrs)

        bytes_codec: dict[str, Any] = {'name': 'bytes'}
        if self._dtype.itemsize == 1:
            bytes_codec['configuration'] = {'endian': sys.byteorder}

        self._store['zarr.json'] = _json_dumps(
            {
                'zarr_format': 3,
                'node_type': 'array',
                'shape': self._tiled.shape,
                'data_type': self._dtype.name,
                'chunk_grid': {
                    'name': 'regular',
                    'configuration': {'chunk_shape': self._tiled.chunks},
                },
                'chunk_key_encoding': {
                    'name': 'default',
                    'configuration': {'separator': '/'},
                },
                'fill_value': _json_value(self._fillvalue, self._dtype),
                'codecs': [bytes_codec],
                'dimension_names': dimension_names,
                'attributes': zattrs,
            }
        )

    @override
    async def exists(self, key: str) -> bool:
        """Return whether key exists in store."""
        # print(f'exists({key=})')
        if key in self._store:
            return True
        assert isinstance(key, str)
        try:
            indices = tuple(int(i) for i in key.removeprefix('c/').split('/'))
        except ValueError:
            return False
        return indices in self._lookup

    @override
    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        """Return value associated with key."""
        if byte_range is not None:
            msg = f'{byte_range=!r} not supported'
            raise NotImplementedError(msg)

        if key in self._store:
            return prototype.buffer.from_bytes(self._store[key])

        if key.endswith(('.zmetadata', '.zarray', '.zgroup', '.zattrs')):
            # catch legacy zarr v2 keys
            return None

        try:
            indices = tuple(int(i) for i in key.removeprefix('c/').split('/'))
        except ValueError:
            return None
        filename = self._lookup.get(indices)
        if filename is None:
            return None
        if self._ioworkers != 1:
            chunk = await asyncio.to_thread(
                self._imread, filename, **self._kwargs
            )
        else:
            chunk = self._imread(filename, **self._kwargs)
        return prototype.buffer.from_array_like(chunk.reshape(-1).view('B'))

    def write_fsspec(
        self,
        jsonfile: str | os.PathLike[Any] | TextIO,
        /,
        url: str | None,
        *,
        quote: bool | None = None,
        groupname: str | None = None,
        templatename: str | None = None,
        codec_id: str | None = None,
        zarr_format: int | None = None,
        version: int | None = None,
        _append: bool = False,
        _close: bool = True,
    ) -> None:
        """Write fsspec ReferenceFileSystem as JSON to file.

        Parameters:
            jsonfile:
                Name or open file handle of output JSON file.
            url:
                Remote location of TIFF file(s) without file name(s).
            quote:
                Quote file names, that is, replace ' ' with '%20'.
                The default is True.
            groupname:
                Zarr group name.
            templatename:
                Version 1 URL template name. The default is 'u'.
            codec_id:
                Name of Numcodecs (zarr format 2) or imagecodecs.zarr
                (zarr format 3) codec to decode files or chunks.
            zarr_format:
                Version of Zarr array format to write.
                The default is 2. If 3, write Zarr version 3 format using
                :py:mod:`imagecodecs.zarr` native codec specifications.
                Chunk keys use 'c/' prefix with '/' separator and
                :py:meth:`imagecodecs.zarr.register_codecs` must be called
                before reading the resulting store.
            version:
                Version of fsspec file to write. The default is 0.
            _append, _close:
                Experimental API.

        References:
            - `fsspec ReferenceFileSystem format
              <https://github.com/fsspec/kerchunk>`_

        """
        from urllib.parse import quote as quote_

        kwargs = self._kwargs.copy()

        if zarr_format is None:
            zarr_format = 2
        elif zarr_format not in {2, 3}:
            msg = f'invalid {zarr_format=!r} not in {{2, 3}}'
            raise ValueError(msg)

        if codec_id is not None:
            pass
        elif self._imread is imread:
            codec_id = 'tifffile'
        elif 'imagecodecs' in self._imread.__module__:
            if (
                self._imread.__name__ != 'imread'
                or 'codec' not in self._kwargs
            ):
                msg = 'cannot determine codec_id'
                raise ValueError(msg)
            codec = kwargs.pop('codec')
            if isinstance(codec, (list, tuple)):
                codec = codec[0]
            if callable(codec):
                codec = codec.__name__.split('_')[0]
            try:
                codec_id = {
                    'apng': 'imagecodecs_apng',
                    'avif': 'imagecodecs_avif',
                    'bmp': 'imagecodecs_bmp',
                    'dds': 'imagecodecs_dds',
                    'exr': 'imagecodecs_exr',
                    'gif': 'imagecodecs_gif',
                    'heif': 'imagecodecs_heif',
                    'htj2k': 'imagecodecs_htj2k',
                    'jpeg': 'imagecodecs_jpeg',
                    'jpeg8': 'imagecodecs_jpeg',
                    'jpeg12': 'imagecodecs_jpeg',
                    'jpeg2k': 'imagecodecs_jpeg2k',
                    'jpegls': 'imagecodecs_jpegls',
                    'jpegxl': 'imagecodecs_jpegxl',
                    'jpegxr': 'imagecodecs_jpegxr',
                    'ljpeg': 'imagecodecs_ljpeg',
                    'lerc': 'imagecodecs_lerc',
                    # 'npy': 'imagecodecs_npy',
                    'pcx': 'imagecodecs_pcx',
                    'png': 'imagecodecs_png',
                    'qoi': 'imagecodecs_qoi',
                    'rgbe': 'imagecodecs_rgbe',
                    'tga': 'imagecodecs_tga',
                    'tiff': 'imagecodecs_tiff',
                    'ultrahdr': 'imagecodecs_ultrahdr',
                    'webp': 'imagecodecs_webp',
                    'wic': 'imagecodecs_wic',
                    'zfp': 'imagecodecs_zfp',
                }[codec]
            except KeyError:
                msg = f'unknown {codec=!r}'
                raise ValueError(msg) from None
        else:
            # TODO: choose codec from filename
            msg = 'cannot determine codec_id'
            raise ValueError(msg)

        if url is None:
            url = ''
        elif url and url[-1] != '/':
            url += '/'

        if groupname is None:
            groupname = ''
        elif groupname and groupname[-1] != '/':
            groupname += '/'

        refs: dict[str, Any] = {}
        if version == 1:
            if _append:
                msg = 'cannot append to version 1 files'
                raise ValueError(msg)
            if templatename is None:
                templatename = 'u'
            refs['version'] = 1
            refs['templates'] = {templatename: url}
            refs['gen'] = []
            refs['refs'] = refzarr = {}
            url = f'{{{{{templatename}}}}}'
        else:
            refzarr = refs

        if groupname and not _append:
            if zarr_format == 3:
                refzarr['zarr.json'] = _json_dumps(
                    {'zarr_format': 3, 'node_type': 'group', 'attributes': {}}
                ).decode()
            else:
                refzarr['.zgroup'] = _json_dumps({'zarr_format': 2}).decode()

        if zarr_format == 3:
            for key, value_bytes in self._store.items():
                if not key.endswith('zarr.json'):
                    continue
                value = json.loads(value_bytes)
                if value.get('node_type') != 'array':
                    refzarr[groupname + key] = value_bytes.decode()
                    continue
                # inject actual codec into zarr v3 codec pipeline
                # all file-format codecs are array-bytes codecs
                seq_codecs: list[dict[str, Any]] = [
                    {'name': codec_id, **kwargs}
                ]
                refzarr[groupname + key] = _json_dumps(
                    {
                        **value,
                        'codecs': seq_codecs,
                        'storage_transformers': [],
                    }
                ).decode()
        else:
            # zarr_format == 2
            for key, value_bytes in self._store.items():
                if not key.endswith('zarr.json'):
                    continue
                value = json.loads(value_bytes)
                if value.get('node_type') != 'array':
                    continue
                # Convert zarr v3 zarr.json -> zarr v2 .zarray + .zattrs
                dtype = numpy.dtype(value['data_type'])
                zarray_dict: dict[str, Any] = {
                    'zarr_format': 2,
                    'shape': value['shape'],
                    'chunks': value['chunk_grid']['configuration'][
                        'chunk_shape'
                    ],
                    'dtype': _dtype_str(dtype),
                    'compressor': None,
                    'fill_value': value['fill_value'],
                    'order': 'C',
                    'filters': None,
                }
                if codec_id is not None:
                    # TODO: make kwargs serializable
                    zarray_dict['compressor'] = {'id': codec_id, **kwargs}
                zarray_key = key.replace('zarr.json', '.zarray')
                refzarr[groupname + zarray_key] = _json_dumps(
                    zarray_dict
                ).decode()
                seq_zattrs: dict[str, Any] = {}
                dim_names = value.get('dimension_names')
                if dim_names is not None:
                    seq_zattrs['_ARRAY_DIMENSIONS'] = dim_names
                seq_attrs = value.get('attributes', {})
                if seq_attrs:
                    seq_zattrs.update(seq_attrs)
                if seq_zattrs:
                    zattrs_key = key.replace('zarr.json', '.zattrs')
                    refzarr[groupname + zattrs_key] = _json_dumps(
                        seq_zattrs
                    ).decode()

        fh: TextIO
        with contextlib.ExitStack() as stack:
            if hasattr(jsonfile, 'write'):
                fh = jsonfile  # type: ignore[assignment]
            else:
                fh = stack.enter_context(open(jsonfile, 'w', encoding='utf-8'))

            if version == 1:
                fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
                indent = '  '
            elif _append:
                fh.write(',\n')
                fh.write(json.dumps(refs, indent=1)[2:-2])
                indent = ' '
            else:
                fh.write(json.dumps(refs, indent=1)[:-2])
                indent = ' '

            for key, value_bytes in self._store.items():
                if not key.endswith('zarr.json'):
                    continue
                value = json.loads(value_bytes)
                if value.get('node_type') != 'array':
                    continue
                for index, fname in sorted(
                    self._lookup.items(), key=lambda x: x[0]
                ):
                    filename = fname.removeprefix(self._commonpath)
                    filename = filename.replace('\\', '/')
                    if quote is None or quote:
                        filename = quote_(filename)
                    if filename and filename[0] == '/':
                        filename = filename[1:]
                    if zarr_format == 3:
                        indexstr = 'c/' + '/'.join(str(i) for i in index)
                    else:
                        indexstr = '.'.join(str(i) for i in index)
                    fh.write(
                        f',\n{indent}"{groupname}{indexstr}": '
                        f'["{url}{filename}"]'
                    )

            if version == 1:
                fh.write('\n }\n}')
            elif _close:
                fh.write('\n}')


def zarr_selection(
    store: ZarrStore,
    selection: BasicSelection,
    /,
    *,
    groupindex: str | None = None,
    close: bool = True,
    out: OutputType = None,
) -> NDArray[Any]:
    """Return selection from Zarr store.

    Parameters:
        store:
            ZarrStore instance to read selection from.
        selection:
            Subset of image to be extracted and returned.
            Refer to the Zarr documentation for valid selections.
        groupindex:
            Index of array if store is Zarr group.
        close:
            Close store before returning.
        out:
            Specifies how image array is returned.
            By default, create a new array.
            If a *numpy.ndarray*, a writable array to which the images
            are copied.
            If *'memmap'*, create a memory-mapped array in a temporary
            file.
            If a *string* or *open file*, the file used to create a
            memory-mapped array.

    """
    zarray: zarr.Array[Any]

    try:
        z = zarr.open(store, mode='r', zarr_format=None)
        if isinstance(z, zarr.Group):
            if groupindex is None:
                groupindex = '0'
            zarray = z[groupindex]  # type: ignore[assignment]
        else:
            zarray = z
        if out is not None:
            shape = BasicIndexer(
                selection,
                shape=zarray.shape,
                chunk_grid=ChunkGrid.from_sizes(zarray.shape, zarray.chunks),
            ).shape
            ndbuffer = NDBufferCPU.from_numpy_array(
                create_output(out, shape, zarray.dtype)
            )
        else:
            ndbuffer = None
        result = zarray.get_basic_selection(selection, out=ndbuffer)
        del zarray
    finally:
        if close:
            store.close()
    return result  # type: ignore[return-value]


def _write_fsspec_v3_metadata(
    store: dict[str, Any],
    pages: list[TiffPageSeries],
    refzarr: dict[str, Any],
    groupname: str,
    compressors: dict[COMPRESSION | int, str | None],
    _shape: list[int],
    _axes: list[str],
    /,
) -> None:
    """Write Zarr v3 metadata to refzarr dict based on store and pages."""
    # Zarr v3: internal store is already zarr.json-based (NGFF 0.5)
    if groupname:
        # root wrapper group for groupname prefix
        refzarr['zarr.json'] = _json_dumps(
            {
                'zarr_format': 3,
                'node_type': 'group',
                'attributes': {},
            }
        ).decode()

    # imagecodecs.zarr ArrayBytesCodecs that appear as TIFF compression codecs
    array_byte_codecs = {
        'imagecodecs_ccittfax3',
        'imagecodecs_ccittfax4',
        'imagecodecs_ccittrle',
        'imagecodecs_eer',
        'imagecodecs_float24',
        'imagecodecs_jpeg',
        'imagecodecs_jpeg2k',
        'imagecodecs_jpegxl',
        'imagecodecs_jpegxr',
        'imagecodecs_lerc',
        'imagecodecs_pixarlog',
        'imagecodecs_png',
        'imagecodecs_webp',
    }

    for key, value_bytes in store.items():
        value = json.loads(value_bytes)
        node_type = value.get('node_type')
        if node_type == 'group':
            # Group zarr.json: inject _axes into NGFF multiscales
            attrs = value.get('attributes', {})
            if _axes:
                ome = attrs.get('ome', {})
                for ms in ome.get('multiscales', []):
                    ms['axes'] = [{'name': ax} for ax in _axes] + ms.get(
                        'axes', []
                    )
                    for ds in ms.get('datasets', []):
                        for ct in ds.get('coordinateTransformations', []):
                            if ct['type'] == 'scale':
                                ct['scale'] = [1.0] * len(_axes) + ct['scale']
                            elif ct['type'] == 'translation':
                                ct['translation'] = [0.0] * len(_axes) + ct[
                                    'translation'
                                ]
            refzarr[groupname + key] = _json_dumps(
                {
                    'zarr_format': 3,
                    'node_type': 'group',
                    'attributes': attrs,
                }
            ).decode()
        elif node_type == 'array':
            # Array zarr.json: replace placeholder codec with
            # actual codec pipeline for the TIFF compression
            level = int(key.split('/')[0]) if '/' in key else 0
            levelstr = f'{level}/' if '/' in key else ''
            keyframe = pages[level].keyframe
            shape = list(value['shape'])
            chunk_shape = list(
                value['chunk_grid']['configuration']['chunk_shape']
            )
            dim_names = value.get('dimension_names')
            if _shape:
                shape = list(_shape) + shape
                chunk_shape = [1] * len(_shape) + chunk_shape
            if _axes and dim_names is not None:
                dim_names = list(_axes) + dim_names
            # build zarr v3 codec pipeline
            codec_id = compressors[keyframe.compression]
            dtype = numpy.dtype(value['data_type'])
            tiff_byteorder = keyframe.parent.byteorder
            if dtype.itemsize == 1 or tiff_byteorder is None:
                bytes_codec: dict[str, Any] = {'name': 'bytes'}
            elif tiff_byteorder == '>':
                bytes_codec = {
                    'name': 'bytes',
                    'configuration': {'endian': 'big'},
                }
            else:
                bytes_codec = {
                    'name': 'bytes',
                    'configuration': {'endian': 'little'},
                }
            codecs: list[dict[str, Any]] = []
            if keyframe.predictor > 1:
                # array-array filter codec
                if keyframe.predictor in {2, 34892, 34893}:
                    filter_id = 'imagecodecs_delta'
                else:
                    filter_id = 'imagecodecs_floatpred'
                if keyframe.predictor <= 3:
                    dist = 1
                elif keyframe.predictor in {34892, 34894}:
                    dist = 2
                else:
                    dist = 4
                if keyframe.planarconfig == 1 and keyframe.samplesperpixel > 1:
                    axis = -2
                else:
                    axis = -1
                codecs.append(
                    {
                        'name': filter_id,
                        'configuration': {
                            'axis': axis,
                            'dist': dist,
                        },
                    }
                )
            if codec_id is None:
                # no compression: just bytes array-bytes codec
                codecs.append(bytes_codec)
            elif codec_id in array_byte_codecs:
                # array-bytes codec: handles array-bytes directly
                if codec_id == 'imagecodecs_jpeg':
                    # TODO: handle JPEG color spaces
                    jpegtables = keyframe.jpegtables
                    tables = (
                        None
                        if jpegtables is None
                        else base64.b64encode(jpegtables).decode()
                    )
                    jpegheader = keyframe.jpegheader
                    header = (
                        None
                        if jpegheader is None
                        else base64.b64encode(jpegheader).decode()
                    )
                    (
                        colorspace_jpeg,
                        colorspace_data,
                    ) = jpeg_decode_colorspace(
                        keyframe.photometric,
                        keyframe.planarconfig,
                        keyframe.extrasamples,
                        keyframe.is_jfif,
                    )
                    cfg: dict[str, Any] = {
                        'bitspersample': keyframe.bitspersample,
                        'colorspace_jpeg': colorspace_jpeg,
                        'colorspace_data': colorspace_data,
                    }
                    if tables is not None:
                        cfg['tables'] = tables
                    if header is not None:
                        cfg['header'] = header
                    codecs.append(
                        {
                            'name': codec_id,
                            'configuration': cfg,
                        }
                    )
                elif (
                    codec_id == 'imagecodecs_webp'
                    and keyframe.samplesperpixel == 4
                ):
                    codecs.append(
                        {
                            'name': codec_id,
                            'configuration': {'hasalpha': True},
                        }
                    )
                elif codec_id == 'imagecodecs_eer':
                    horzbits = vertbits = 2
                    if keyframe.compression == 65002:
                        skipbits = int(keyframe.tags.valueof(65007, 7))
                        horzbits = int(keyframe.tags.valueof(65008, 2))
                        vertbits = int(keyframe.tags.valueof(65009, 2))
                    elif keyframe.compression == 65001:
                        skipbits = 7
                    else:
                        skipbits = 8
                    eer_cfg: dict[str, Any] = {
                        'shape': list(keyframe.chunks),
                        'skipbits': skipbits,
                        'horzbits': horzbits,
                        'vertbits': vertbits,
                    }
                    if keyframe.parent._superres:
                        eer_cfg['superres'] = keyframe.parent._superres
                    codecs.append(
                        {
                            'name': codec_id,
                            'configuration': eer_cfg,
                        }
                    )
                else:
                    codecs.append({'name': codec_id})
            else:
                # bytes-bytes codec: needs bytes array-bytes first
                codecs.append(bytes_codec)
                codecs.append({'name': codec_id})

            refzarr[groupname + levelstr + 'zarr.json'] = _json_dumps(
                {
                    'zarr_format': 3,
                    'node_type': 'array',
                    'shape': shape,
                    'data_type': dtype.name,
                    'chunk_grid': {
                        'name': 'regular',
                        'configuration': {'chunk_shape': chunk_shape},
                    },
                    'chunk_key_encoding': {
                        'name': 'default',
                        'configuration': {'separator': '/'},
                    },
                    'fill_value': value['fill_value'],
                    'codecs': codecs,
                    'dimension_names': dim_names,
                    'attributes': value.get('attributes', {}),
                    'storage_transformers': [],
                }
            ).decode()


def _write_fsspec_v2_metadata(
    store: dict[str, Any],
    pages: list[TiffPageSeries],
    refzarr: dict[str, Any],
    groupname: str,
    byteorder: ByteOrder | None,
    compressors: dict[COMPRESSION | int, str | None],
    _shape: list[int],
    _axes: list[str],
    /,
) -> None:
    """Write Zarr v2 metadata to refzarr dict based on store and pages."""
    # Zarr v2 format: convert internal zarr.json to .zarray/.zattrs
    if groupname:
        # TODO: support nested groups
        refzarr['.zgroup'] = _json_dumps({'zarr_format': 2}).decode()

    for key, value_bytes in store.items():
        value = json.loads(value_bytes)
        node_type = value.get('node_type')
        if node_type == 'group':
            # Group zarr.json -> .zgroup + .zattrs
            # expose ome namespace at top level for NGFF v2 compat
            attrs = dict(value.get('attributes', {}))
            ome_attrs = attrs.pop('ome', {})
            v2_attrs: dict[str, Any] = {}
            if 'multiscales' in ome_attrs:
                multiscales = ome_attrs['multiscales']
                if _axes:
                    for ms in multiscales:
                        ms['axes'] = [{'name': ax} for ax in _axes] + ms.get(
                            'axes', []
                        )
                        for ds in ms.get('datasets', []):
                            for ct in ds.get('coordinateTransformations', []):
                                if ct['type'] == 'scale':
                                    ct['scale'] = [1.0] * len(_axes) + ct[
                                        'scale'
                                    ]
                                elif ct['type'] == 'translation':
                                    ct['translation'] = [0.0] * len(
                                        _axes
                                    ) + ct['translation']
                for ms in multiscales:
                    ms.setdefault('version', '0.4')
                v2_attrs['multiscales'] = multiscales
            v2_attrs.update(attrs)
            zgroup_key = key.replace('zarr.json', '.zgroup')
            zattrs_key = key.replace('zarr.json', '.zattrs')
            refzarr[groupname + zgroup_key] = _json_dumps(
                {'zarr_format': 2}
            ).decode()
            if v2_attrs:
                refzarr[groupname + zattrs_key] = _json_dumps(
                    v2_attrs
                ).decode()
        elif node_type == 'array':
            # Array zarr.json -> .zarray + .zattrs
            level = int(key.split('/')[0]) if '/' in key else 0
            keyframe = pages[level].keyframe
            dim_names = value.get('dimension_names')
            shape = list(value['shape'])
            chunk_shape = list(
                value['chunk_grid']['configuration']['chunk_shape']
            )
            if _shape:
                shape = list(_shape) + shape
                chunk_shape = [1] * len(_shape) + chunk_shape
            if _axes and dim_names is not None:
                dim_names = list(_axes) + dim_names
            dtype = numpy.dtype(value['data_type'])
            dtype_str = (
                byteorder + dtype.str[1:]
                if byteorder is not None
                else _dtype_str(dtype)
            )
            zarray: dict[str, Any] = {
                'zarr_format': 2,
                'shape': shape,
                'chunks': chunk_shape,
                'dtype': dtype_str,
                'compressor': None,
                'fill_value': value['fill_value'],
                'order': 'C',
                'filters': None,
            }
            codec_id = compressors[keyframe.compression]
            if codec_id == 'imagecodecs_jpeg':
                # TODO: handle JPEG color spaces
                jpegtables = keyframe.jpegtables
                jpegtables_b64 = (
                    None
                    if jpegtables is None
                    else base64.b64encode(jpegtables).decode()
                )
                jpegheader = keyframe.jpegheader
                jpegheader_b64 = (
                    None
                    if jpegheader is None
                    else base64.b64encode(jpegheader).decode()
                )
                (
                    colorspace_jpeg,
                    colorspace_data,
                ) = jpeg_decode_colorspace(
                    keyframe.photometric,
                    keyframe.planarconfig,
                    keyframe.extrasamples,
                    keyframe.is_jfif,
                )
                zarray['compressor'] = {
                    'id': codec_id,
                    'tables': jpegtables_b64,
                    'header': jpegheader_b64,
                    'bitspersample': keyframe.bitspersample,
                    'colorspace_jpeg': colorspace_jpeg,
                    'colorspace_data': colorspace_data,
                }
            elif (
                codec_id == 'imagecodecs_webp'
                and keyframe.samplesperpixel == 4
            ):
                zarray['compressor'] = {
                    'id': codec_id,
                    'hasalpha': True,
                }
            elif codec_id == 'imagecodecs_eer':
                horzbits = vertbits = 2
                if keyframe.compression == 65002:
                    skipbits = int(keyframe.tags.valueof(65007, 7))
                    horzbits = int(keyframe.tags.valueof(65008, 2))
                    vertbits = int(keyframe.tags.valueof(65009, 2))
                elif keyframe.compression == 65001:
                    skipbits = 7
                else:
                    skipbits = 8
                zarray['compressor'] = {
                    'id': codec_id,
                    'shape': keyframe.chunks,
                    'skipbits': skipbits,
                    'horzbits': horzbits,
                    'vertbits': vertbits,
                    'superres': keyframe.parent._superres,
                }
            elif codec_id is not None:
                codec_id = {
                    # use numcodecs built-in codecs
                    'imagecodecs_zlib': 'zlib',
                    'imagecodecs_lzma': 'lzma',
                    # 'imagecodecs_zstd': 'zstd',
                }.get(codec_id, codec_id)
                zarray['compressor'] = {'id': codec_id}
            if keyframe.predictor > 1:
                # predictors need access to chunk shape and dtype
                # requires imagecodecs > 2021.8.26 to read
                if keyframe.predictor in {2, 34892, 34893}:
                    filter_id = 'imagecodecs_delta'
                else:
                    filter_id = 'imagecodecs_floatpred'
                if keyframe.predictor <= 3:
                    dist = 1
                elif keyframe.predictor in {34892, 34894}:
                    dist = 2
                else:
                    dist = 4
                if keyframe.planarconfig == 1 and keyframe.samplesperpixel > 1:
                    axis = -2
                else:
                    axis = -1
                zarray['filters'] = [
                    {
                        'id': filter_id,
                        'axis': axis,
                        'dist': dist,
                        'shape': chunk_shape,
                        'dtype': dtype_str,
                    }
                ]
            zarray_key = key.replace('zarr.json', '.zarray')
            refzarr[groupname + zarray_key] = _json_dumps(zarray).decode()
            if dim_names is not None:
                zattrs_key = key.replace('zarr.json', '.zattrs')
                refzarr[groupname + zattrs_key] = _json_dumps(
                    {'_ARRAY_DIMENSIONS': dim_names}
                ).decode()


def _json_dumps(obj: Any, /) -> bytes:
    """Serialize object to JSON formatted bytes."""
    return json.dumps(
        obj,
        indent=1,
        sort_keys=True,
        ensure_ascii=True,
        separators=(',', ': '),
    ).encode('ascii')


def _json_value(value: Any, dtype: numpy.dtype[Any], /) -> Any:
    """Return value which is serializable to JSON."""
    if value is None:
        return value
    if dtype.kind == 'b':
        return bool(value)
    if dtype.kind in 'ui':
        return int(value)
    if dtype.kind == 'f':
        if numpy.isnan(value):
            return 'NaN'
        if numpy.isposinf(value):
            return 'Infinity'
        if numpy.isneginf(value):
            return '-Infinity'
        return float(value)
    if dtype.kind == 'c':
        value = numpy.array(value, dtype)
        return (
            _json_value(value.real, value.real.dtype),
            _json_value(value.imag, value.imag.dtype),
        )
    return value


def _dtype_str(dtype: numpy.dtype[Any], /) -> str:
    """Return dtype as string with native byte order."""
    if dtype.itemsize == 1:
        byteorder = '|'
    else:
        byteorder = {'big': '>', 'little': '<'}[sys.byteorder]
    return byteorder + dtype.str[1:]


def _ndindex(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    /,
    separator: str = '.',
) -> Iterator[str]:
    """Return iterator over all chunk index strings."""
    chunked = tuple(
        (i + j - 1) // j for i, j in zip(shape, chunks, strict=True)
    )
    for indices in numpy.ndindex(chunked):
        yield separator.join(str(index) for index in indices)


def _chunks(
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    shaped: tuple[int, int, int, int, int],
    /,
) -> tuple[int, ...]:
    """Return chunks with same length as shape."""
    ndim = len(shape)
    if ndim == 0:
        return ()  # empty array
    if 0 in shape:
        return (1,) * ndim
    d = 0 if shaped[1] == 1 else 1
    i = min(ndim, 3 + d)
    if (
        len(chunks) == 2 + d
        and i != 2 + d
        and shape[-1] == 1
        and shape[-i:] == shaped[-i:]
    ):
        # planarconfig=contig with one sample
        chunks = (*chunks, 1)
    if ndim < len(chunks):
        # remove leading dimensions of size 1 from chunks
        i = 0
        for size in chunks:
            if size > 1:
                break
            i += 1
        chunks = chunks[i:]
        if ndim < len(chunks):
            msg = f'{shape=!r} is shorter than {chunks=!r}'
            raise ValueError(msg)
    # prepend size 1 dimensions to chunks to match length of shape
    return (1,) * (ndim - len(chunks)) + chunks


def _enum_name(
    value: int | str | enum.Enum | None, enum_cls: type[enum.Enum], /
) -> str | None:
    """Normalize int, str, or enum member to canonical enum name string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, enum_cls):
        return value.name
    if isinstance(value, enum.Enum):
        return enum_cls(value.value).name
    return enum_cls(value).name


def _setattrs(obj: object, /, **kwargs: Any) -> None:
    """Set attributes on a frozen dataclass instance."""
    for k, v in kwargs.items():
        object.__setattr__(obj, k, v)


def register_codec() -> None:
    """Register zarr 3 tifffile codec."""
    from zarr.registry import register_codec

    register_codec('tifffile', Tiff)
