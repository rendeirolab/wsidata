from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from functools import cached_property
from numbers import Integral, Number
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .._utils import find_stack_level

if TYPE_CHECKING:
    from .core import WSIData


@dataclass
class TileSpec:
    """Data class for storing tile specifications.

    .. note::
        Tile coordinates (x, y) from shape bounds are always in level 0 coordinates.

    To enable efficient tils generation, there are 3 levels of tile size:

    1. The destined tile size and level requested by the user
    2. The tile size and level used by the image reader to optimize the performance
    3. The actual tile size at the level 0

    This enables user to request tile size and level that are not exist in the image pyramids.
    For example, if our slide is mpp=0.5 (20X) with pyramids of mpp=[0.5, 2.0, 4.0],
    and user request mpp=1.0 (10X) with tile size 512x512. There is no direct level to read from,
    we need to read from mpp=0.5 with tile size of 1024x1024 and downsample to 512x512.


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
    mpp : float, default: None
        The requested microns per pixel of tiles.
    ops_level : int, default: 0
        The level of the tiling operation.
    ops_downsample : float, default: 1
        The downsample factor to transform the operation level
        tiles to the requested level.
    tissue_name : str, optional
        The name of the tissue.

    Attributes
    ----------
    ops_{height, width} : int
        The height/width of the tile when retrieving images.
    ops_stride_{height, width}: int
        The height/width of the stride when retrieving images.
    base_{height, width} : int
        The height/width of the tile at the level 0.
    base_stride_{height, width} : int
        The height/width of the stride at the level 0.

    """

    height: int
    width: int
    stride_height: int
    stride_width: int

    mpp: float | None = None

    ops_level: int = 0  # level of the tiling operation
    ops_downsample: float = 1  # downsample to requested level

    base_level: int = 0  # level of the base tile, always 0
    base_downsample: float = 1.0  # downsample to requested level

    tissue_name: str | None = None

    def __post_init__(self):
        # coerce attributes to correct type
        self.height = int(self.height)
        self.width = int(self.width)
        self.stride_height = int(self.stride_height)
        self.stride_width = int(self.stride_width)
        self.ops_level = int(self.ops_level)
        self.ops_downsample = float(self.ops_downsample)
        self.base_level = int(self.base_level)
        self.base_downsample = float(self.base_downsample)

    def __repr__(self):
        return (
            f"Tile at {self.mpp} mpp, {self.height}×{self.width} (h×w)\n"
            f"Stride: {self.stride_height}×{self.stride_width} ({self.overlap_y}×{self.overlap_x} overlap)\n"
            f"Operation size: {self.ops_height}×{self.ops_width}, level={self.ops_level}\n"
            f"Base size: {self.base_height}×{self.base_width}, level={self.base_level}\n"
            f"Target tissue: '{self.tissue_name}'"
        )

    def _repr_html_(self):
        BORDER_WIDTH = 2
        scale = 100 / max(self.width, self.height)
        scaled_w = np.floor(self.width * scale) - BORDER_WIDTH
        scaled_h = np.floor(self.height * scale) - BORDER_WIDTH
        scaled_stride_w = np.floor(
            self.stride_width * scale
        )  # add 2 to account for the border
        scaled_stride_h = np.floor(
            self.stride_height * scale
        )  # add 2 to account for the border

        tile_style = (
            f"width: {int(scaled_w)}px; "
            f"height: {int(scaled_h)}px; "
            f"position: absolute; "
            f"box-sizing: content-box;"  # make sure the border is included in the size
            f"border: {BORDER_WIDTH}px solid #000; "
        )
        container_style = (
            f"width: {int(scaled_w + scaled_stride_w) + 2 * BORDER_WIDTH}px; "
            f"height: {int(scaled_h + scaled_stride_h) + 2 * BORDER_WIDTH}px; "
            f"position: relative;"
            f"border: 0;"
        )
        html = f"""
        <div style="display: flex; align-item: center; gap: 10px;">
            <div style='{container_style}'>
                <div style='{tile_style}; top: 0px; left: 0px; border-color: #C68FE6'>
                    <p style='padding-left: 3pt'>Tile 1</p>
                </div>
                <div style='{tile_style}; top: 0px; left: {scaled_stride_w}px; border-color: #F6DC43'>
                    <p style='padding-left: 3pt'>Tile 2</p>
                </div>
                <div style='{tile_style}; top: {scaled_stride_h}px; left: 0px; border-color: #6A9C89'>
                    <p style='padding-left: 3pt'>Tile 3</p>
                </div>
            </div>
            <div>
                <b>Tile at</b>: {self.mpp} mpp<br>
                <b>Tile size</b>: {self.height}×{self.width} (h×w)<br>
                <b>Stride</b>: {self.stride_height}×{self.stride_width} ({self.overlap_y}×{self.overlap_x} overlap)<br>
                <b>Operation size</b>: {self.ops_height}×{self.ops_width}, level={self.ops_level}<br>
                <b>Base size</b>: {self.base_height}×{self.base_width}, level={self.base_level}<br>
                <b>Target tissue</b>: '{self.tissue_name}'
            </div>
        </div>
        """
        return html

    @classmethod
    def from_wsidata(
        cls,
        wsi: WSIData,
        tile_px: int | Tuple[int, int],
        stride_px: Optional[int | Tuple[int, int]] = None,
        overlap: Optional[float | Tuple[float, float]] = None,
        mpp: Optional[float] = None,
        ops_level: Optional[int] = None,
        slide_mpp: Optional[float] = None,
        tissue_name: Optional[str] = None,
    ) -> TileSpec:
        """Create a TileSpec from a WSIData object.

        To tile from the whole slide image, the user needs to specify the tile size and stride size.
        mpp only need to be specified if the user wants to make sure
        the tile size is harmonized across different slides.

        If ops_level is not specified, the optimal level will be calculated based on the requested mpp
        to maximize the performance.

        Parameters
        ----------
        wsi : WSIData
            The whole-slide image data object.
        tile_px : int or (int, int)
            Tile size in pixels.  If a tuple, interpreted as ``(width, height)``.
        stride_px : int or (int, int), optional
            Stride size in pixels.  If a tuple, interpreted as ``(width, height)``.
        overlap : float or (float, float), optional
            Overlap fraction or pixel count.  Cannot be specified together with *stride_px*.
        mpp : float, optional
            Requested microns per pixel.
        ops_level : int, optional
            Override the pyramid level used for reading.
        slide_mpp : float, optional
            Override the slide's recorded mpp.
        tissue_name : str, optional
            Name of the tissue to tile.

        """
        # Check if the input params are valid
        tile_w, tile_h, stride_w, stride_h = _preprocess_tile_params(
            tile_px, stride_px, overlap
        )

        # If user does not override slide mpp, use the recorded slide mpp
        if slide_mpp is None:
            slide_mpp = wsi.properties.mpp

        # If user does not specify mpp, default to ops level is 0
        if mpp is None:
            mpp = wsi.properties.mpp
            if ops_level is not None:
                raise ValueError("Please specify mpp if ops_level is specified.")
            ops_level = 0
            ops_downsample = 1
        # Or if user specify mpp, but the slide mpp is not available, use the default level 0
        elif slide_mpp is None:
            if ops_level is None:
                ops_level = 0
                ops_downsample = 1
            else:
                ops_downsample = wsi.properties.level_downsample[ops_level]

            warning_text = f"Slide mpp is not available, using level {ops_level}."
            # If user specify the mpp at the same time but slide mpp is not available
            # we will inform the user that the mpp will be ignored
            if mpp is not None:
                warning_text += f" Requested mpp={mpp} will be ignored."
            warnings.warn(warning_text, stacklevel=find_stack_level())
        else:
            # If user didn't specify ops level but mpp, an optimized ops level will be calculated
            wsi_downsample = np.asarray(wsi.properties.level_downsample)
            req_downsample = mpp / slide_mpp
            if req_downsample < 1:
                raise ValueError(
                    f"Requested mpp={mpp} is smaller than the slide mpp={slide_mpp}. "
                    f"Up-sampling is not supported."
                )

            gap = wsi_downsample - req_downsample
            gap[gap > 0] = np.inf

            suggest_level = np.argmin(np.abs(gap))
            suggest_downsample = mpp / (slide_mpp * wsi_downsample[suggest_level])
            if ops_level is None:
                # Apply the optimized level
                ops_level = suggest_level
                ops_downsample = suggest_downsample
            else:
                # check if user specified ops level is valid
                ops_level = wsi.reader.translate_level(ops_level)
                ops_downsample = mpp / (
                    slide_mpp * wsi.properties.level_downsample[ops_level]
                )
                if ops_downsample < 1:
                    if suggest_level == 0:
                        consider_text = "Consider using level 0."
                    else:
                        consider_text = f"Consider using ops_level<={suggest_level}."
                    raise ValueError(
                        f"Requested tile size at mpp={mpp} "
                        f"on ops_level={ops_level} will require up-sampling. "
                        + consider_text
                    )

        base_downsample = ops_downsample * wsi.properties.level_downsample[ops_level]

        return cls(
            height=tile_h,
            width=tile_w,
            stride_height=stride_h,
            stride_width=stride_w,
            mpp=mpp,
            ops_level=ops_level,
            ops_downsample=ops_downsample,
            base_downsample=base_downsample,
            tissue_name=tissue_name,
        )

    def to_dict(self):
        """Convert the TileSpec to a dictionary."""
        return asdict(self)

    def to_json(self):
        """Convert the TileSpec to a JSON string."""
        return json.dumps(asdict(self))

    @cached_property
    def is_overlap_x(self) -> bool:
        """Check if the tiles are overlapped along the x-axis."""
        return self.stride_width < self.width

    @cached_property
    def is_overlap_y(self) -> bool:
        """Check if the tiles are overlapped along the y-axis."""
        return self.stride_height < self.height

    @cached_property
    def is_overlap(self) -> bool:
        """Check if the tiles are overlapped."""
        return self.is_overlap_x or self.is_overlap_y

    @cached_property
    def overlap_x(self) -> int:
        """The overlap pixel size along the x-axis."""
        return self.width - self.stride_width

    @cached_property
    def overlap_y(self) -> int:
        """The overlap pixel size along the y-axis."""
        return self.height - self.stride_height

    @cached_property
    def ops_height(self) -> int:
        """The height of the tile when retrieving images."""
        return int(self.height * self.ops_downsample)

    @cached_property
    def ops_width(self) -> int:
        """The width of the tile when retrieving images."""
        return int(self.width * self.ops_downsample)

    @cached_property
    def ops_stride_height(self) -> int:
        """The height of the stride when retrieving images."""
        return int(self.stride_height * self.ops_downsample)

    @cached_property
    def ops_stride_width(self) -> int:
        """The width of the stride when retrieving images."""
        return int(self.stride_width * self.ops_downsample)

    @cached_property
    def base_height(self) -> int:
        """The height of the tile at the level 0."""
        return int(self.height * self.base_downsample)

    @cached_property
    def base_width(self) -> int:
        """The width of the tile at the level 0."""
        return int(self.width * self.base_downsample)

    @cached_property
    def base_stride_height(self) -> int:
        """The height of the stride at the level 0."""
        return int(self.stride_height * self.base_downsample)

    @cached_property
    def base_stride_width(self) -> int:
        """The width of the stride at the level 0."""
        return int(self.stride_width * self.base_downsample)


def _preprocess_tile_params(
    tile_px: int | Tuple[int, int],
    stride_px: Optional[int | Tuple[int, int]] = None,
    overlap: Optional[float | Tuple[float, float]] = None,
) -> Tuple[int, int, int, int]:
    """Parse tile and stride parameters.

    Parameters
    ----------
    tile_px : int or (int, int)
        Tile size.  If a tuple, interpreted as ``(width, height)``.
    stride_px : int or (int, int), optional
        Stride size.  If a tuple, interpreted as ``(width, height)``.
    overlap : float or (float, float), optional
        Overlap fraction (< 1) or pixel count (>= 1).

    Returns
    -------
    (tile_w, tile_h, stride_w, stride_h) : tuple of int
    """
    if isinstance(tile_px, Integral):
        tile_w, tile_h = (tile_px, tile_px)
    elif isinstance(tile_px, Sequence):
        tile_w, tile_h = (tile_px[0], tile_px[1])
    else:
        raise TypeError("tile_px must be an integer or a tuple of two integers.")

    if tile_w <= 0 or tile_h <= 0:
        raise ValueError("tile_px must be positive.")

    if stride_px is not None and overlap is not None:
        raise ValueError("Cannot specify both stride_px and overlap.")

    stride_w, stride_h = tile_w, tile_h
    if stride_px is not None:
        if isinstance(stride_px, Integral):
            stride_w, stride_h = (stride_px, stride_px)
        elif isinstance(stride_px, Sequence):
            stride_w, stride_h = (stride_px[0], stride_px[1])
        else:
            raise TypeError("stride_px must be an integer or a tuple of two integers.")
        if stride_w <= 0 or stride_h <= 0:
            raise ValueError("stride_px must be positive.")

    if overlap is not None:
        if isinstance(overlap, Number):
            overlap_w, overlap_h = (overlap, overlap)
        elif isinstance(overlap, Sequence):
            overlap_w, overlap_h = (overlap[0], overlap[1])
        else:
            raise TypeError("overlap must be a number or a tuple of two numbers.")

        if overlap_w < 0 or overlap_h < 0:
            raise ValueError("overlap must be non-negative.")
        # calculate stride px from overlap

        if overlap_w < 1:
            stride_w = int(tile_w * (1 - overlap_w))
        else:
            stride_w = int(tile_w - overlap_w)

        if overlap_h < 1:
            stride_h = int(tile_h * (1 - overlap_h))
        else:
            stride_h = int(tile_h - overlap_h)

    return tile_w, tile_h, stride_w, stride_h


class TileRequest(NamedTuple):
    x: int
    y: int
    level: int
    width: int
    height: int
    dsize: Tuple[int, int] | None


def shapes2tiles(
    wsi: WSIData,
    shape_key: str,
    image_size: int | Tuple[int, int] | None = None,  # Width, Height
) -> List[TileRequest]:
    """Convert arbitrary shapes into tile read specifications.

    When a :class:`TileSpec` is stored for *shape_key*, tiles are derived from
    it directly (uniform grid).  Otherwise, each shape's bounding box is used
    and the optimal pyramid level is selected per-shape.

    Parameters
    ----------
    wsi : WSIData
        The whole-slide image data object.
    shape_key : str
        Key into ``wsi.shapes`` for the shape collection.
    image_size : tuple of (int, int), optional
        Desired output ``(width, height)``.  Only used when no TileSpec is
        available.  When given, the best pyramid level is chosen so the read
        region is closest to *image_size* and then resized.

    Returns
    -------
    list[TileRequest]
        One :class:`TileOp` per shape, with coordinates at level 0.
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    shapes = wsi.shapes[shape_key]
    tile_spec = wsi.tile_spec(shape_key)

    # -- TileSpec path: uniform grid --
    if tile_spec is not None:
        xy = shapes.bounds[["minx", "miny"]].to_numpy()
        dsize: Tuple[int, int] | None = None
        ops_w = tile_spec.ops_width
        ops_h = tile_spec.ops_height
        # Only need resize when ops size differs from requested size
        if ops_w != tile_spec.width or ops_h != tile_spec.height:
            dsize = (tile_spec.width, tile_spec.height)
        return [
            TileRequest(
                x=int(x),
                y=int(y),
                level=tile_spec.ops_level,
                width=ops_w,
                height=ops_h,
                dsize=dsize,
            )
            for x, y in xy
        ]

    # -- No TileSpec: derive from shape bounds --
    warnings.warn(
        f"TileSpec not found for {shape_key}, "
        f"will proceed with assumption that "
        f"the coordinates system for {shape_key} is the same as WSI.",
        stacklevel=find_stack_level(),
    )

    bounds_arr = shapes.bounds[["minx", "miny", "maxx", "maxy"]].to_numpy()
    widths = bounds_arr[:, 2] - bounds_arr[:, 0]
    heights = bounds_arr[:, 3] - bounds_arr[:, 1]

    # Guard against degenerate shapes (zero or negative extent)
    if np.any(widths <= 0) or np.any(heights <= 0):
        bad = int(np.sum((widths <= 0) | (heights <= 0)))
        raise ValueError(
            f"Tiles of '{shape_key}' has {bad} shape(s) with zero or negative extent. "
            "Filter degenerate geometries beforehand."
        )

    if image_size is not None:
        level_downsample = np.asarray(wsi.properties.level_downsample)
        # if not np.all(np.diff(level_downsample) >= 0):
        #     raise ValueError(
        #         "level_downsample is not sorted in ascending order; "
        #         "cannot determine optimal read level."
        #     )

        breakpoints_width = level_downsample * image_size[0]
        breakpoints_height = level_downsample * image_size[1]

        # Vectorized level selection
        level_w = np.searchsorted(breakpoints_width, widths, side="right") - 1
        level_h = np.searchsorted(breakpoints_height, heights, side="right") - 1
        levels = np.clip(np.minimum(level_w, level_h), 0, len(level_downsample) - 1)
        downsamples = level_downsample[levels]
        ops_widths = np.round(widths / downsamples).astype(int)
        ops_heights = np.round(heights / downsamples).astype(int)

        return [
            TileRequest(
                x=int(bounds_arr[i, 0]),
                y=int(bounds_arr[i, 1]),
                level=int(levels[i]),
                width=int(ops_widths[i]),
                height=int(ops_heights[i]),
                dsize=image_size,
            )
            for i in range(len(bounds_arr))
        ]
    else:
        # No target image_size: read at level 0, full shape extent, no resize
        int_widths = np.round(widths).astype(int)
        int_heights = np.round(heights).astype(int)
        return [
            TileRequest(
                x=int(bounds_arr[i, 0]),
                y=int(bounds_arr[i, 1]),
                level=0,
                width=int(int_widths[i]),
                height=int(int_heights[i]),
                dsize=None,
            )
            for i in range(len(bounds_arr))
        ]
