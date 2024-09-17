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
        tile_key
        target_key
        transform
        target_transform
        color_norm

        Returns
        -------

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
        feature_key: str
        target_key: str
        target_transform: callable

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
