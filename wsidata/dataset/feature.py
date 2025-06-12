from torch.utils.data import Dataset

from .._model import WSIData


class TileFeatureDataset(Dataset):
    """
    Dataset for features extracted from tiles.

    Parameters
    ----------
    wsi : WSIData
    feature_key : str
    target_key : str
    target_transform: callable

    Returns
    -------
    TileFeatureDataset

    """

    def __init__(
        self,
        wsi: WSIData,
        feature_key: str,
        tile_key: str = "tiles",
        target_key: str = None,
        target_transform=None,
    ):
        feature_key = wsi._check_feature_key(feature_key, tile_key)
        tables = wsi.tables[feature_key]
        self.X = tables.X
        self.tables = tables.obs
        self.targets = None
        if target_key in self.tables:
            self.targets = self.tables[target_key].to_numpy()
        self.target_transform = target_transform

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.targets is not None:
            y = self.targets[idx]
            if self.target_transform is not None:
                y = self.target_transform(y)
            return x, y
        return x
