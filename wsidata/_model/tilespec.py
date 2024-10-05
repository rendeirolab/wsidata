import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TileSpec:
    """Data class for storing tile specifications.

    Parameters
    ----------
    height : int
        The height of the tile.
    width : int
        The width of the tile.
    stride_height : int
        The height of the stride.
    stride_width : int
        The width of the stride.
    raw_{height, width} : int
        The height/width of the tile when retrieving images.
    raw_stride_{height, width}: int
        The height/width of the stride when retrieving images.
    tissue_name : str
        The name of the tissue.
    level : int, default: 0
        The level of the tile.
    downsample : float, default: 1
        The downsample factor.
    mpp : float, default: None
        The microns per pixel.

    """

    height: int
    width: int
    stride_height: int
    stride_width: int
    raw_height: int
    raw_width: int
    raw_stride_height: int
    raw_stride_width: int
    tissue_name: str
    level: int = 0
    downsample: float = 1
    mpp: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(asdict(self))
