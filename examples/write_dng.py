# tifffile/examples/write_dng.py

"""Write Digital Negative (DNG) compatible file using tifffile.

Writes a synthetic, 12-bit CFA (Bayer mosaic) array as a DNG 1.7 file.
DNG-specific TIFF/EP tags are passed via TiffWriter's extratags parameter.
Full-resolution and thumbnail previews are written as JPEG-compressed SubIFDs.
Validated with dng_validate (Adobe DNG SDK 1.7.1).

"""

from __future__ import annotations

__all__ = ['write_dng']

from typing import TYPE_CHECKING, Any

import numpy as np

from tifffile import TiffWriter, rational

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from numpy.typing import NDArray


def write_dng(
    filename: str | os.PathLike[str],
    cfa: NDArray[np.uint16],
    *,
    preview: NDArray[np.uint8] | None = None,
    thumbnail: NDArray[np.uint8] | None = None,
    cfa_pattern: str | Sequence[int] = 'RGGB',
    color_matrix: Sequence[float] | None = None,
    as_shot_neutral: Sequence[float] = (1.0, 1.0, 1.0),
    black_level: int = 0,
    white_level: int = 65535,
    unique_camera_model: str = 'Generic Camera',
    cfa_compression: str | None = None,
    cfa_bitspersample: int = 16,
) -> None:
    """Write CFA image to DNG 1.7 file.

    Parameters:

        filename:
            Name of file to write.
        cfa:
            Raw Bayer mosaic array of shape ``(H, W)`` and dtype ``uint16``.
        preview:
            Full-resolution RGB24 image.
        thumbnail:
            Thumbnail RGB24 image.
        cfa_pattern:
            CFA color pattern for 2x2 repeat tile.
            Either a 4-character string such as ``'RGGB'`` (default),
            ``'BGGR'``, ``'GRBG'``, or ``'GBRG'``;
            or a sequence of integer color indices (R=0, G=1, B=2).
        color_matrix:
            Nine floats for row-major 3x3 ColorMatrix1 (XYZ D50 -> camera RGB).
            Defaults to the standard sRGB matrix calibrated under D65.
        as_shot_neutral:
            White-balance coefficients.
            Three non-negative floats <= 1.0 giving neutral color in
            camera space.
        black_level:
            Black-point value (per sample, applied to all planes).
        white_level:
            Maximum (saturated) pixel value.
        unique_camera_model:
            ASCII string stored in ``UniqueCameraModel`` DNG tag.
        cfa_compression:
            Compression for raw CFA image.
            ``None`` (uncompressed; default),
            ``'jpegxl_dng'`` (lossless JPEG XL), or ``'jpeg'`` (lossless JPEG).
        cfa_bitspersample:
            Number of bits per sample for raw CFA image.

    """
    if cfa.ndim != 2 or cfa.dtype != np.uint16:
        msg = 'cfa must be a 2-D uint16 array'
        raise ValueError(msg)
    if black_level >= white_level:
        msg = 'black_level must be less than white_level'
        raise ValueError(msg)

    if isinstance(cfa_pattern, str):
        try:
            cfa_pattern = tuple(
                {'R': 0, 'G': 1, 'B': 2}[c] for c in cfa_pattern.upper()
            )
        except KeyError as exc:
            msg = f'unknown color letter in {cfa_pattern=!r}'
            raise ValueError(msg) from exc
    else:
        cfa_pattern = tuple(cfa_pattern)
    if len(cfa_pattern) != 4:
        msg = f'cfa_pattern must have 4 elements, got {len(cfa_pattern)}'
        raise ValueError(msg)
    # plane colors are unique indices in sorted order, e.g. (0,1,2) for RGB
    cfa_plane_colors = tuple(sorted(set(cfa_pattern)))

    if color_matrix is None:
        # sRGB ColorMatrix1 for D65 illuminant (D50-adapted XYZ, per DNG spec)
        color_matrix = (
            0.4361,
            0.3851,
            0.1431,
            0.2225,
            0.7169,
            0.0606,
            0.0139,
            0.0971,
            0.7141,
        )

    # validate optional preview images
    if preview is not None and (
        preview.ndim != 3 or preview.shape[2] != 3 or preview.dtype != np.uint8
    ):
        msg = 'preview must be a uint8 (H, W, 3) array'
        raise ValueError(msg)
    if thumbnail is not None and (
        thumbnail.ndim != 3
        or thumbnail.shape[2] != 3
        or thumbnail.dtype != np.uint8
    ):
        msg = 'thumbnail must be a uint8 (H, W, 3) array'
        raise ValueError(msg)

    # DNG global tags written to IFD0 (identification and camera color profile)
    # Tuple layout: (code, dtype, count, value, writeonce)
    ifd0_tags: list[tuple[Any, ...]] = [
        # DNG identification
        ('DNGVersion', 1, 4, (1, 7, 1, 0), True),
        ('DNGBackwardVersion', 1, 4, (1, 4, 0, 0), True),
        ('UniqueCameraModel', 2, 0, unique_camera_model, True),
        ('Orientation', 3, 1, 1, True),
        # Camera color profile
        ('CalibrationIlluminant1', 3, 1, 21, True),  # D65
        ('ColorMatrix1', 10, 9, [rational(v) for v in color_matrix], True),
        ('AsShotNeutral', 5, 3, [rational(v) for v in as_shot_neutral], True),
    ]

    # CFA-specific tags written to raw image IFD
    cfa_tags: list[tuple[Any, ...]] = [
        # CFA geometry (TIFF/EP tags required for piCFA)
        ('CFARepeatPatternDim', 3, 2, (2, 2), True),
        ('CFAPattern', 1, 4, cfa_pattern, True),
        ('CFAPlaneColor', 1, len(cfa_plane_colors), cfa_plane_colors, True),
        ('CFALayout', 3, 1, 1, True),
        # Sensor levels
        ('BlackLevel', 5, 1, rational(black_level), True),
        ('WhiteLevel', 4, 1, white_level, True),
        # Crop / active area
        # ActiveArea = full sensor; DefaultCrop is inset by 4 pixels on
        # each side to provide padding for the demosaic interpolation kernel
        ('ActiveArea', 4, 4, (0, 0, *cfa.shape), True),  #  [T,L,B,R]
        ('DefaultCropOrigin', 4, 2, (4, 4), True),
        ('DefaultCropSize', 4, 2, (cfa.shape[1] - 8, cfa.shape[0] - 8), True),
    ]

    # validate compression
    cfa_kwargs: dict[str, Any] = {}
    if cfa_compression == 'jpegxl_dng':
        cfa_kwargs['compression'] = 'jpegxl_dng'
        cfa_kwargs['compressionargs'] = {
            'lossless': True,
            'bitspersample': cfa_bitspersample,
        }
    elif cfa_compression == 'jpeg':
        cfa_kwargs['compression'] = 'jpeg'
        cfa_kwargs['compressionargs'] = {
            'lossless': True,
            'bitspersample': cfa_bitspersample,
        }
    elif cfa_compression is not None:
        msg = f'unsupported {cfa_compression=!r}'
        raise ValueError(msg)

    # write DNG file
    with TiffWriter(
        filename, bigtiff=False, byteorder='<', kind='generic'
    ) as tif:
        if thumbnail is not None:
            # DNG recommends thumbnail in IFD0 with raw and preview in SubIFDs
            tif.write(
                thumbnail,
                photometric='rgb',
                subfiletype=1,
                subifds=1 if preview is None else 2,
                extratags=ifd0_tags,
                compression='jpeg',
                compressionargs={'level': 90},
            )
            cfa_subifds = None
            cfa_software = False  # don't write Software tag in SubIFDs
        else:
            # no thumbnail: raw goes in IFD0 with all tags
            cfa_tags += ifd0_tags
            cfa_subifds = None if preview is None else 1
            cfa_software = None

        tif.write(
            cfa,
            photometric='cfa',
            subfiletype=0,
            subifds=cfa_subifds,
            extratags=cfa_tags,
            software=cfa_software,
            **cfa_kwargs,
        )

        if preview is not None:
            tif.write(
                preview,
                photometric='rgb',
                subfiletype=1,
                compression='jpeg',
                compressionargs={'level': 90},
                software=False,
            )


def synthetic_cfa(height: int, width: int) -> np.ndarray:
    """Return realistic-looking 12-bit RGGB Bayer mosaic."""
    rng = np.random.default_rng(42)
    # Smooth base luminance
    yy = np.linspace(0.2, 0.8, height)[:, None]
    xx = np.linspace(0.2, 0.8, width)[None, :]
    base = (yy * xx * 3750).astype(np.uint16)
    noise = rng.integers(0, 16, (height, width), dtype=np.uint16)
    mosaic = np.clip(base + noise, 0, 4095).astype(np.uint16)
    # simulate channel offsets for RGGB:
    # R=top-left, G=top-right&bot-left, B=bot-right
    mosaic[0::2, 0::2] = np.clip(
        mosaic[0::2, 0::2].astype(np.int32) + 125, 0, 4095
    )  # R
    mosaic[1::2, 1::2] = np.clip(
        mosaic[1::2, 1::2].astype(np.int32) - 94, 0, 4095
    )  # B
    return mosaic  # type: ignore[no-any-return]


def rggb2rgb(
    cfa: np.ndarray,
    *,
    wb_gains: tuple[float, float, float] = (2.0, 1.0, 1.6),
    white_level: int = 4095,
) -> np.ndarray:
    """Return simple demosaic of RGGB Bayer mosaic to uint8 RGB.

    Parameters:
        cfa:
            RGGB Bayer mosaic array.
        wb_gains:
            Per-channel (R, G, B) multiplicative gains applied before clipping.
            The default gains are typical daylight values for a sensor with
            equal-illuminant green reference.
        white_level:
            Sensor white level.
            Pixels are normalised to this value before scaling to 8-bit so
            that the preview approximates what a raw converter would produce.

    """
    h, w = cfa.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    # R (top-left of each 2x2 block)
    rgb[0::2, 0::2, 0] = cfa[0::2, 0::2]
    rgb[0::2, 1::2, 0] = cfa[0::2, 0::2]
    rgb[1::2, 0::2, 0] = cfa[0::2, 0::2]
    rgb[1::2, 1::2, 0] = cfa[0::2, 0::2]
    # G (average of the two green pixels)
    g = (
        cfa[0::2, 1::2].astype(np.float32) + cfa[1::2, 0::2].astype(np.float32)
    ) / 2.0
    rgb[0::2, 0::2, 1] = g
    rgb[0::2, 1::2, 1] = g
    rgb[1::2, 0::2, 1] = g
    rgb[1::2, 1::2, 1] = g
    # B (bottom-right of each 2x2 block)
    rgb[0::2, 0::2, 2] = cfa[1::2, 1::2]
    rgb[0::2, 1::2, 2] = cfa[1::2, 1::2]
    rgb[1::2, 0::2, 2] = cfa[1::2, 1::2]
    rgb[1::2, 1::2, 2] = cfa[1::2, 1::2]
    # apply white-balance gains and normalise to 8-bit
    rgb *= np.array(wb_gains, dtype=np.float32) * (255.0 / white_level)
    return rgb.clip(0, 255).astype(np.uint8)


def main() -> int:
    """Write and validate DNG file with synthetic CFA data and previews."""
    import shutil
    import subprocess

    out = 'test.dng'
    cfa = synthetic_cfa(512, 512)
    preview = rggb2rgb(cfa, white_level=4095)
    thumb = preview[::8, ::8, :].copy()

    write_dng(
        out,
        cfa,
        preview=preview,
        thumbnail=thumb,
        cfa_pattern='RGGB',
        cfa_compression='jpegxl_dng',
        cfa_bitspersample=12,
        white_level=4095,
        unique_camera_model='Synthetic Camera',
    )

    # run dng_validate if available
    path = shutil.which('dng_validate')
    if path:
        result = subprocess.run(  # noqa: S603
            [path, '-v', out],
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        print(output)
        if result.returncode == 0:
            if 'error' in output.lower() or 'warning' in output.lower():
                # scan for actual error/warning lines
                issues = [
                    ln
                    for ln in output.splitlines()
                    if any(w in ln.lower() for w in ('error', 'warning'))
                ]
                if issues:
                    print('Issues found:')
                    for ln in issues:
                        print(' ', ln)
                    return 1
            print('dng_validate: OK')
        else:
            print(f'dng_validate exited with code {result.returncode}')
            return result.returncode
    else:
        print('\ndng_validate not found - skipping.')
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main())
