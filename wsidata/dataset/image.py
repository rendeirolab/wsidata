from functools import cached_property

from torch.utils.data import Dataset
from torchvision.transforms import Resize

from .._model import WSIData


class TileImagesDataset(Dataset):
    """
    Dataset for tiles from the whole slide image.


    Parameters
    ----------
    wsi : WSIData
    key : str
        The key of the tile table.
    target_key : str
        The key of the target table.
    transform: callable
        The transformation for the input tiles.
    target_transform: callable
        The transformation for the target.
    color_norm: str
        The color normalization method.

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
    ):
        # Do not assign wsi to self to avoid pickling
        tiles = wsi[key]
        self.tiles = tiles[["x", "y", "tissue_id"]].to_numpy()
        self.spec = wsi.tile_spec(key)
        self.color_norm = color_norm

        self.targets = None
        if target_key is not None:
            self.targets = tiles[target_key].to_numpy()
        self.transform = transform
        self.target_transform = target_transform
        self._resize = Resize((self.spec.height, self.spec.width), antialias=True)

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
        return len(self.tiles)

    def __getitem__(self, idx):
        x, y, tid = self.tiles[idx]
        tile = self.reader.get_region(
            x, y, self.spec.raw_width, self.spec.raw_height, level=self.spec.level
        )
        tile = self.reader.resize_img(tile, dsize=(self.spec.height, self.spec.width))
        tile = self._cn_func(tile)
        if self.transform:
            tile = self.transform(tile)
        if self.targets is not None:
            tile_target = self.targets[idx]
            if self.target_transform:
                tile_target = self.target_transform(tile_target)
            return {
                "image": tile,
                "target": tile_target,
                "x": int(x),
                "y": int(y),
                "tissue_id": int(tid),
            }
        return {"image": tile, "x": int(x), "y": int(y), "tissue_id": int(tid)}
