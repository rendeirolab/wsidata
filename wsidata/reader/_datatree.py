__all__ = ["to_datatree"]

from dataclasses import asdict
from math import ceil

import dask.array as da
import numpy as np
import xarray as xr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale


def _read_block(reader, level, block_info=None):
    _, (y0, y1), (x0, x1) = block_info[None]["array-location"]
    ds = reader.properties.level_downsample[level]
    # Readers map level-0 offsets to the level with int(x / ds), so ceil
    # lands on the first level-0 pixel of the block's origin. OpenSlide
    # paints at fractional offsets instead, staying within 1 / ds px.
    region = reader.get_region(
        ceil(x0 * ds), ceil(y0 * ds), x1 - x0, y1 - y0, level=level
    )
    return region.transpose(2, 0, 1)


def to_datatree(reader, chunks=(1024, 1024)) -> xr.DataTree:
    """Lazy multiscale SpatialData image of the slide, read by blocks."""
    levels = {}
    for level, (height, width) in enumerate(reader.properties.level_shape):
        # map_blocks keeps the reader in the task arguments: the graph
        # pickles and dask never pushes user slicing into the read
        data = da.map_blocks(
            _read_block,
            reader,
            level,
            chunks=da.core.normalize_chunks((3, *chunks), (3, height, width)),
            dtype=np.uint8,
            meta=np.empty((0, 0, 0), dtype=np.uint8),
        )
        ds = reader.properties.level_downsample[level]
        transform = Identity() if ds == 1 else Scale([ds, ds], axes=("y", "x"))
        image = Image2DModel.parse(
            xr.DataArray(
                data,
                dims=("c", "y", "x"),
                coords={"y": np.arange(height), "x": np.arange(width)},
            ),
            c_coords=["r", "g", "b"],
            transformations={"global": transform},
        )
        levels[f"scale{level}"] = xr.Dataset({"image": image})

    tree = xr.DataTree.from_dict(levels)
    tree.attrs = asdict(reader.properties)
    return tree
