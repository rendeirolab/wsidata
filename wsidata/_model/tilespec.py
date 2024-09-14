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
    raw_height : int
        The height of the raw image.
    raw_width : int
        The width of the raw image.
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
    raw_height: int
    raw_width: int
    tissue_name: str
    level: int = 0
    downsample: float = 1
    mpp: Optional[float] = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(asdict(self))
