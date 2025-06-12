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
        tile_key: str = "tiles",
        target_key: str = None,
        target_transform=None,
    ):
        """
        Create a TileFeatureDataset from the current object.

        Parameters
        ----------
        feature_key : str
            The key of the feature table.
        tile_key : str, default: "tiles"
            The key of the tile table.
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
            tile_key=tile_key,
            target_key=target_key,
            target_transform=target_transform,
        )

    def tile_feature_graph(
        self,
        feature_key: str,
        tile_key: str = "tiles",
        graph_key: str = None,
        target_key: str = None,
    ):
        """
        Create a PyTorch Geometric Data object from the graph data in WSIData.

        Parameters
        ----------
        feature_key : str
            The key for the tile features.
        tile_key : str, default: "tiles"
            The key for the tiles.
        graph_key : str, optional
            The key for tile graph, defaults to "{tile_key}_graph".
        target_key : str, optional
            The key for the target data in the observation table.

        Returns
        -------
        :class:`torch_geometric.data.Data`
            A PyTorch Geometric Data object containing:
            - x: Node features (image features)
            - edge_index: Graph connectivity
            - edge_attr: Edge attributes (distances)
            - y: Target values (if target_key is provided)

        """

        from ..dataset.graph import graph_data

        return graph_data(
            self._obj,
            feature_key=feature_key,
            tile_key=tile_key,
            graph_key=graph_key,
            target_key=target_key,
        )
