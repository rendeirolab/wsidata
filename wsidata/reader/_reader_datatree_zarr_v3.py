__all__ = ["to_datatree"]

import asyncio
import json
import math
from dataclasses import asdict
from typing import AsyncIterator, Iterable, Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer import Buffer, BufferPrototype, default_buffer_prototype

from .base import ReaderBase


class SlideZarrStore(Store):
    """
    Read-only Zarr-v3-compatible store backed by a Slider Reader.

    Keys produced:
      - ".zgroup" -> JSON identifying zarr format (v3)
      - "{level}/.zarray" -> JSON metadata for array at `level`
      - "{level}/{r}.{c}" -> bytes for chunk at chunk-row=r, chunk-col=c

    The store only implements reading: set/delete raise.
    """

    def __init__(
        self,
        reader: ReaderBase,
        *,
        chunks: Tuple[int, int] = (1024, 1024),
    ) -> None:
        super().__init__(read_only=True)
        self._reader = reader
        self.chunks = tuple(chunks)
        self.channels = 3

        self._level_shape = list(self._reader.properties.level_shape)
        self._level_downsample = list(self._reader.properties.level_downsample)
        self.n_level = self._reader.properties.n_level

    @classmethod
    def open(
        cls,
        reader: ReaderBase,
        *,
        chunks: Tuple[int, int] = (1024, 1024),
    ) -> "SlideZarrStore":
        """
        Synchronous convenience constructor that creates the store and opens the underlying slide.
        """
        return cls(
            reader,
            chunks=chunks,
        )

    async def _open(self) -> None:
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SlideZarrStore) and other._reader == self._reader

    # --- properties describing capabilities ---
    @property
    def supports_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return False

    @property
    def supports_listing(self) -> bool:
        return True

    # --- Helper utilities ---
    def _parse_chunk_key(self, key: str) -> Optional[Tuple[int, int, int]]:
        """
        Expect chunk keys like 'LEVEL/ROW.COL' => returns (level, row, col)
        """
        try:
            level_part, rc = key.split("/", 1)
            if rc.startswith("."):
                return None
            row_s, col_s = rc.split(".", 1)
            return int(level_part), int(row_s), int(col_s)
        except Exception:
            return None

    def _array_meta(self, level: int) -> dict:
        """
        Build zarr v3 .zarray metadata for the given level.
        We expose shape as (height, width, channels) and chunk shape (ch_h, ch_w, channels).
        dtype fixed as uint8 for typical H&E/brightfield slides.
        """
        w, h = self._level_shape[level]
        ch_h, ch_w = self.chunks
        channels = self.channels
        shape = [h, w, channels]
        chunks = [ch_h, ch_w, channels]
        meta = {
            "zarr_format": 3,
            "shape": shape,
            "chunks": chunks,
            "dtype": "<u1",  # uint8 little-endian
            "compressor": None,
            "filters": None,
            "order": "C",
            "fill_value": None,
            # "dimension_names": ["x", "y", "c"],
            "dimension_separator": "/",
        }
        return meta

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: "ByteRequest | None" = None,
    ) -> Buffer | None:
        """
        Return bytes for the key. We ignore byte_range for simplicity (could be supported).
        Keys supported:
          - ".zgroup"
          - "{level}/.zarray"
          - "{level}/{row}.{col}" (chunk)
        """
        await self._ensure_open()
        assert self._slide is not None

        # group metadata
        if key == ".zgroup":
            return bytes(json.dumps({"zarr_format": 3}).encode("utf-8"))

        # array metadata
        if key.endswith("/.zarray"):
            level_s = key.split("/", 1)[0]
            try:
                level = int(level_s)
            except ValueError:
                return None
            meta = self._array_meta(level)
            b = json.dumps(meta).encode("utf-8")
            return b

        # chunk key
        parsed = self._parse_chunk_key(key)
        if parsed is None:
            return None
        level, row, col = parsed
        # validate level
        if level < 0 or level >= len(self._level_shape):
            return None
        w, h = self._level_shape[level]
        ch_h, ch_w = self.chunks

        # chunk pixel dimensions (may be smaller at edges)
        chunk_w = min(ch_w, w - col * ch_w)
        chunk_h = min(ch_h, h - row * ch_h)
        if chunk_w <= 0 or chunk_h <= 0:
            return None

        # compute location in level-0 coordinates:
        downsample = int(round(self._level_downsample[level]))
        x0 = col * ch_w * downsample
        y0 = row * ch_h * downsample

        arr = await asyncio.to_thread(
            self.reader.read_region,
            self._slide,
            x0,
            y0,
            chunk_w,
            chunk_h,
            level,
        )

        # Ensure shape (h, w, channels) and C-order
        if arr.ndim == 2:
            # grayscale -> add channel axis
            arr = arr[:, :, None]
        arr = np.ascontiguousarray(arr)
        # If user wants a channels-last zarr array of shape (H, W, C) that's what we return
        b = arr.tobytes()
        return b

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, "ByteRequest | None"]],
    ) -> list[Buffer | None]:
        # Simple implementation: sequentially call get. Could be optimized by batching.
        results = []
        for k, br in key_ranges:
            val = await self.get(k, prototype, br)
            results.append(val)
        return results

    async def exists(self, key: str) -> bool:
        await self._ensure_open()
        if key == ".zgroup":
            return True
        if key.endswith("/.zarray"):
            level_s = key.split("/", 1)[0]
            try:
                level = int(level_s)
            except ValueError:
                return False
            return 0 <= level < len(self._level_shape)
        parsed = self._parse_chunk_key(key)
        if parsed is None:
            return False
        level, row, col = parsed
        if not (0 <= level < len(self._level_shape)):
            return False
        w, h = self._level_shape[level]
        ch_h, ch_w = self.chunks
        max_row = math.ceil(h / ch_h) - 1
        max_col = math.ceil(w / ch_w) - 1
        return 0 <= row <= max_row and 0 <= col <= max_col

    # writes are not supported:
    async def set(self, key: str, value: Buffer) -> None:
        raise ValueError("store is read-only")

    async def delete(self, key: str) -> None:
        raise ValueError("store is read-only")

    # --- listing helpers ---
    async def list(self) -> AsyncIterator[str]:
        # yield all keys: .zgroup, then for each level .zarray and all chunk keys
        await self._ensure_open()
        yield ".zgroup"
        for lvl in range(len(self._level_shape)):
            yield f"{lvl}/.zarray"
            w, h = self._level_shape[lvl]
            ch_h, ch_w = self.chunks
            rows = math.ceil(h / ch_h)
            cols = math.ceil(w / ch_w)
            for r in range(rows):
                for c in range(cols):
                    yield f"{lvl}/{r}.{c}"

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # Normalize prefix
        if prefix != "" and not prefix.endswith("/"):
            prefix = prefix + "/"
        # We will filter the full list; for huge slides you might want to compute directly
        async for k in self.list():
            if k.startswith(prefix) or (prefix == "" and k):
                yield k

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # Return immediate children under prefix (no nested '/')
        if prefix != "" and not prefix.endswith("/"):
            prefix = prefix + "/"
        seen = set()
        plen = len(prefix)
        async for k in self.list_prefix(prefix):
            rest = k[plen:]
            if rest == "":
                continue
            next_token = rest.split("/", 1)[0]
            if next_token not in seen:
                seen.add(next_token)
                yield next_token

    def close(self) -> None:
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception:
                pass
        super().close()


async def _fetch_chunk_as_array(
    store: SlideZarrStore, level: int, row: int, col: int
) -> np.ndarray:
    proto = default_buffer_prototype()
    key = f"{level}/{row}.{col}"
    buf = await store.get(key, proto)
    if buf is None:
        # produce an array of zeros of the expected shape
        ch_h, ch_w = store.chunks
        channels = store.channels
        w, h = store._level_dims[level]
        chunk_w = min(ch_w, w - col * ch_w)
        chunk_h = min(ch_h, h - row * ch_h)
        return np.zeros((chunk_h, chunk_w, channels), dtype=np.uint8)
    # Buffer 'buf' supports the buffer protocol; convert to numpy
    b = bytes(buf)
    # compute the expected shape
    w, h = store._level_dims[level]
    ch_h, ch_w = store.chunks
    chunk_w = min(ch_w, w - col * ch_w)
    chunk_h = min(ch_h, h - row * ch_h)
    channels = store.channels
    arr = np.frombuffer(b, dtype=np.uint8)
    arr = arr.reshape((chunk_h, chunk_w, channels))
    return arr


def level_to_xarray(store: SlideZarrStore, level: int):
    """
    Return an xarray.DataArray for the given level.
    This function is synchronous: it uses asyncio.run for simplicity; adjust if you already have an event loop.
    """
    w, h = store._level_shape[level]
    ch_h, ch_w = store.chunks
    channels = store.channels

    rows = math.ceil(h / ch_h)
    cols = math.ceil(w / ch_w)

    # Build a list of delayed blocks
    blocks = []
    for r in range(rows):
        row_blocks = []
        for c in range(cols):
            # delayed construction of numpy array block via dask.delayed
            delayed_block = da.from_delayed(
                delayed(
                    lambda s, L, R, C: asyncio.run(_fetch_chunk_as_array(s, L, R, C))
                )(store, level, r, c),
                shape=(min(ch_h, h - r * ch_h), min(ch_w, w - c * ch_w), channels),
                dtype=np.uint8,
            )
            row_blocks.append(delayed_block)
        # horizontally concatenate row blocks
        row_concat = da.concatenate(row_blocks, axis=1)
        blocks.append(row_concat)
    # vertically concatenate rows
    arr = da.concatenate(blocks, axis=0)
    # arr is (H, W, C)
    coords = {
        "y": np.arange(h),
        "x": np.arange(w),
        "c": np.arange(channels),
    }
    da_xr = xr.DataArray(arr, dims=("y", "x", "c"), coords=coords)
    return da_xr


def to_datatree(
    reader,
    chunks=(1024, 1024),
):
    store = SlideZarrStore.open(reader, chunks=chunks)

    images = {}
    for level in range(reader.properties.n_level):
        img = level_to_xarray(store, level)
        scale_factor = reader.properties.level_downsample[level]
        if scale_factor == 1:
            transform = Identity()
        else:
            transform = Scale([scale_factor, scale_factor], axes=("y", "x"))
        scale_image = Image2DModel.parse(
            img,
            transformations={"global": transform},
            c_coords=["r", "g", "b"],
        )
        images[f"scale{level}"] = xr.Dataset({"image": scale_image})

    slide_image = xr.DataTree.from_dict(images)
    slide_image.attrs = asdict(reader.properties)
    return slide_image
