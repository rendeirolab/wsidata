import pandas as pd
from anndata import AnnData


class FetchAccessor(object):
    """Accessor for getting information from WSIData object.

    Usage: `wsidata.fetch`

    """

    def __init__(self, obj):
        self._obj = obj

    def n_tissue(self, key: str) -> int:
        """
        Return the number of tissue regions in the tissue table.

        Parameters
        ----------
        key: str
            The tile key.

        Returns
        -------
        int
            The number of tissue regions.

        """
        return len(self._obj.shapes[key])

    def n_tiles(self, key: str) -> int:
        """
        Return the number of tiles in the tile table.

        Parameters
        ----------
        key: str
            The tile key.

        Returns
        -------
        int
            The number of tiles.
        """
        return self.n_tissue(key)

    def pyramids(self) -> pd.DataFrame:
        """
        Return the pyramid levels of the whole slide image.

        Returns
        -------
        pd.DataFrame
            A table of pyramid levels (index) with columns:

            - height : The height of the level (px).
            - width : The width of the level (px).
            - downsample : The downsample factor of the level.

        """
        heights, widths = zip(*self._obj.properties.level_shape)
        return pd.DataFrame(
            {
                "height": pd.Series(heights, dtype=int),
                "width": pd.Series(widths, dtype=int),
                "downsample": self._obj.properties.level_downsample,
            },
            index=pd.RangeIndex(self._obj.properties.n_level, name="level"),
        )

    def features_anndata(
        self, feature_key, tile_key="tiles", tile_graph=True
    ) -> AnnData:
        """Return the feature table as an AnnData object.

        Parameters
        ----------
        feature_key : str
            The feature key.
        tile_key : str, default: "tiles"
            The tile key.
        tile_graph : bool, default: True
            If True, include spatial graph information.

        Returns
        -------
        AnnData
            An AnnData object with the following components (if present):

            - X : The feature table.
            - obs : The data stored in the tile table.
            - obsm : The x,y coordinates for each tile.
            - obsp : The spatial graph information.
            - uns : Metadata including tile specifications and slide properties.

        """

        sdata = self._obj
        feature_key = self._obj._check_feature_key(feature_key, tile_key)
        feature_adata = sdata.tables[feature_key]
        X = feature_adata.X  # Must be a numpy array
        var = feature_adata.var

        # layers slot
        layers = feature_adata.layers

        # obs slot
        tile_table = sdata.shapes[tile_key]
        tile_xy = tile_table.bounds[["minx", "miny"]].to_numpy()
        obs = tile_table.drop(columns=["geometry"])
        # To suppress anndata warning
        obs.index = obs.index.astype(str)

        # obsm slot
        obsm = {"spatial": tile_xy}

        # obsp slot
        obsp = {}

        # varm slot
        varm = feature_adata.varm

        # uns slot
        uns = {
            "tile_spec": self._obj.tile_spec(tile_key).to_dict(),
            "slide_properties": self._obj.properties.to_dict(),
        }

        if tile_graph:
            conns_key = "spatial_connectivities"
            dists_key = "spatial_distances"
            graph_key = f"{tile_key}_graph"
            if graph_key in sdata:
                graph_table = sdata.tables[graph_key]
                obsp[conns_key] = graph_table.obsp[conns_key]
                obsp[dists_key] = graph_table.obsp[dists_key]
                uns["spatial"] = graph_table.uns["spatial"]

        return AnnData(
            X=X,
            layers=layers,
            var=var,
            varm=varm,
            obs=obs,
            obsm=obsm,
            obsp=obsp,
            uns=uns,
        )
