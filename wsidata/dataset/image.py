from functools import cached_property

import cv2
from torch.utils.data import Dataset

from .._model import WSIData, shapes2tiles


class TileImagesDataset(Dataset):
    """
    Dataset for tiles from the whole slide image.

    Uses :func:`shapes2tiles` to resolve tile read specifications,
    supporting both uniform grids (with :class:`TileSpec`) and
    arbitrary shape collections.

    .. note::
        ``image_size`` is used **only** for pyramid level selection — the
        reader picks the closest level whose native resolution is ≥ the
        requested size.  No resize is performed after reading; add a
        ``Resize`` step in *transform* if you need exact output dimensions.

    Parameters
    ----------
    wsi : WSIData
    key : str
        The key of the tile table.
    target_key : str
        The key of the target table.
    transform: callable
        The transformation for the input tiles.  Use this to resize,
        augment, or convert tiles (e.g. ``torchvision.transforms.v2``).
    target_transform: callable
        The transformation for the target.
    color_norm: str
        The color normalization method.
    image_size : int or tuple of (int, int), optional
        Hint for optimal pyramid level selection via :func:`shapes2tiles`.

    Returns
    -------
    TileImagesDataset

    """

    def __init__(
        self,
        wsi: WSIData,
        key: str = "tiles",
        target_key: str = None,
        transform=None,
        color_norm=None,
        target_transform=None,
        image_size: int | tuple[int, int] = None,
    ):
        # Do not assign wsi to self to avoid pickling
        tiles_gdf = wsi[key]
        self.color_norm = color_norm

        # Resolve tile read specs via shapes2tiles (same as iter.tile_images)
        # image_size only affects pyramid level selection, no resize applied
        self._tile_requests = shapes2tiles(wsi, key, image_size=image_size)

        # Tile coordinates at level 0 (from shape bounds)
        bounds = tiles_gdf.bounds
        self._minx = bounds["minx"].to_numpy()
        self._miny = bounds["miny"].to_numpy()
        self._maxx = bounds["maxx"].to_numpy()
        self._maxy = bounds["maxy"].to_numpy()

        # tissue_id per tile (if available)
        if "tissue_id" in tiles_gdf.columns:
            self.tissue_ids = tiles_gdf["tissue_id"].to_numpy()
        else:
            self.tissue_ids = None

        self.targets = None
        if target_key is not None:
            self.targets = tiles_gdf[target_key].to_numpy()
        self.transform = transform
        self.target_transform = target_transform

        # Send reader to the worker instead of wsi
        self.reader = wsi.reader
        self.reader.detach_reader()

    @cached_property
    def _cn_func(self):
        return self._get_cn_func()

    def _get_cn_func(self):
        if self.color_norm is not None:
            from .._normalizer import ColorNormalizer

            cn = ColorNormalizer(method=self.color_norm)
            return lambda x: cn(x)
        else:
            return lambda x: x

    def __len__(self):
        return len(self._tile_requests)

    def __getitem__(self, idx):
        tile_req = self._tile_requests[idx]

        # Read region at optimal level determined by shapes2tiles
        tile = self.reader.get_region(
            tile_req.x,
            tile_req.y,
            tile_req.width,
            tile_req.height,
            level=tile_req.level,
        )
        # Resize to target size if needed
        if tile_req.dsize is not None:
            tile = cv2.resize(tile, tile_req.dsize)

        tile = self._cn_func(tile)

        # Downsample: ratio of level-0 extent to output pixel size
        # (computed post-resize, before transform — matches IterAccessor.tile_images)
        out_w = tile.shape[1]
        tile_w_base = self._maxx[idx] - self._minx[idx]
        downsample = tile_w_base / out_w if out_w > 0 else 1.0

        if self.transform:
            tile = self.transform(tile)

        x = int(self._minx[idx])
        y = int(self._miny[idx])
        tissue_id = int(self.tissue_ids[idx]) if self.tissue_ids is not None else -1

        result = {
            "image": tile,
            "x": x,
            "y": y,
            "tissue_id": tissue_id,
            "downsample": downsample,
        }

        if self.targets is not None:
            tile_target = self.targets[idx]
            if self.target_transform:
                tile_target = self.target_transform(tile_target)
            result["target"] = tile_target

        return result


class TileImageDiskDataset(Dataset):
    def __init__(
        self,
        wsi: WSIData,
        output_dir: str = None,  # Default to a temporary directory
    ):
        pass
