class DatasetAccessor(object):
    """
    Accessor for dataset objects.
    """

    def __init__(self, obj):
        self._obj = obj

    def tile_images(
        self,
        tile_key: str = "tiles",
        target_key: str = None,
        transform=None,
        target_transform=None,
        color_norm=None,
    ):
        """
        Create a TileImagesDataset from the current object.

        Parameters
        ----------
        tile_key : str, default: "tiles"
            The key of the tile table.
        target_key : str
            The key of the target table.
        transform: callable
            The transformation for the input tiles.
        target_transform: callable
            The transformation for the target.
        color_norm: str

        Returns
        -------
        TileImagesDataset

        """
        from ..dataset.image import TileImagesDataset

        return TileImagesDataset(
            self._obj,
            key=tile_key,
            target_key=target_key,
            transform=transform,
            target_transform=target_transform,
            color_norm=color_norm,
        )

    def tile_feature(
        self,
        feature_key: str,
        target_key: str = None,
        target_transform=None,
    ):
        """
        Create a TileFeatureDataset from the current object.

        Parameters
        ----------
        feature_key : str
            The key of the feature table.
        target_key : str
            The key of the target table.
        target_transform: callable
            The transformation for the target.

        Returns
        -------
        TileFeatureDataset

        """

        from ..dataset.feature import TileFeatureDataset

        return TileFeatureDataset(
            self._obj,
            feature_key=feature_key,
            target_key=target_key,
            target_transform=target_transform,
        )
