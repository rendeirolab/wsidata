import pytest
import torch
from anndata import AnnData


class TestFetchAccessor:
    def test_pyramids(self, wsidata):
        tables = wsidata.fetch.pyramids()

        assert len(tables) >= 1

        assert tables.index.name == "level"
        assert "width" in tables.columns
        assert "height" in tables.columns

    def test_get_features_anndata(self, wsidata):
        tables = wsidata.fetch.features_anndata("resnet50")

        assert isinstance(tables, AnnData)
        assert tables.X is not None
        assert tables.obs is not None
        assert tables.obsm is not None
        assert tables.obsp is not None
        assert tables.uns is not None
        assert "tile_spec" in wsidata.attrs
        assert "slide_properties" in wsidata.attrs

    def test_get_n_tissue(self, wsidata):
        wsidata.fetch.n_tissue("tissues")

    def test_get_n_tiles(self, wsidata):
        wsidata.fetch.n_tiles("tiles")


class TestIterAccessor:
    @pytest.mark.parametrize("mask_bg", [True, False])
    @pytest.mark.parametrize("format", ["yxc", "cyx"])
    def test_iter_tissues(self, wsidata, mask_bg, format):
        for it in wsidata.iter.tissue_images("tissues", mask_bg=mask_bg, format=format):
            pass
        if format == "yxc":
            assert it.image.shape == (2902, 1946, 3)
        else:
            assert it.image.shape == (3, 2902, 1946)

    def test_iter_tissues_plot(self, wsidata):
        it = next(wsidata.iter.tissue_images("tissues", mask_bg=True))
        it.plot()

    @pytest.mark.parametrize("color_norm", ["macenko", "reinhard"])
    def test_iter_tiles(self, wsidata, color_norm):
        for it in wsidata.iter.tile_images("tiles", color_norm=color_norm):
            pass

    def test_iter_tiles_plot(self, wsidata):
        it = next(wsidata.iter.tile_images("tiles"))
        it.plot()

    def test_iter_contours(self, wsidata):
        for _ in wsidata.iter.tissue_contours("tissues"):
            pass

    def test_iter_contours_plot(self, wsidata):
        it = next(wsidata.iter.tissue_contours("tissues"))
        it.plot()


class TestDatasetAccessor:
    def test_ds_tile_images(self, wsidata):
        dataset = wsidata.ds.tile_images("tiles")
        assert dataset.spec is not None

    def test_ds_tile_feature(self, wsidata):
        dataset = wsidata.ds.tile_feature("resnet50")

        # Check that the dataset has the expected attributes
        assert hasattr(dataset, "X")
        assert hasattr(dataset, "tables")

        # Check that the dataset has the expected length
        assert len(dataset) > 0

        # Check that __getitem__ returns the expected data type
        item = dataset[0]
        assert isinstance(item, (list, tuple)) or item.ndim >= 1

    def test_ds_tile_feature_graph(self, wsidata):
        try:
            import numpy as np
            import scipy.sparse as sp
            import torch_geometric
        except ImportError:
            pytest.skip("torch_geometric or scipy not installed")

        # Get the feature data
        feature_key = "resnet50"
        tile_key = "tiles"
        graph_key = f"{tile_key}_graph"

        # Create an AnnData object for the graph
        feature_key = wsidata._check_feature_key(feature_key, tile_key)
        features = wsidata.tables[feature_key]

        # Get the number of tiles
        n_tiles = features.X.shape[0]

        # Create a simple connectivity matrix (each tile connects to the next one)
        # This is just a simple example - in a real scenario, you'd compute actual connections
        row = np.arange(n_tiles - 1)
        col = np.arange(1, n_tiles)
        data = np.ones(n_tiles - 1)

        # Create sparse matrices for connectivity and distances
        connectivities = sp.csr_matrix((data, (row, col)), shape=(n_tiles, n_tiles))

        # Create distances (using simple Euclidean distance for this example)
        distances = sp.csr_matrix(
            (np.arange(1, n_tiles, dtype=float), (row, col)), shape=(n_tiles, n_tiles)
        )

        # Create an AnnData object with the graph data
        graph_adata = AnnData(X=np.zeros((n_tiles, 1)))  # Placeholder X matrix
        graph_adata.obsp["spatial_connectivities"] = connectivities
        graph_adata.obsp["spatial_distances"] = distances
        graph_adata.uns["spatial"] = {"method": "test"}

        # Add the graph data to the WSIData object
        wsidata.tables[graph_key] = graph_adata

        # Test with default parameters
        data = wsidata.ds.tile_feature_graph("resnet50")

        # Check that the returned object has the expected attributes
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")

        # Check that the attributes have the expected types
        assert isinstance(data.edge_index, torch.Tensor)
        assert isinstance(data.x, torch.Tensor)
        assert isinstance(data.edge_attr, torch.Tensor)

        # Check that x has the expected shape (n_nodes, n_features)
        assert data.x.dim() == 2

        # Check that edge_index has the expected shape (2, n_edges)
        assert data.edge_index.dim() == 2
        assert data.edge_index.size(0) == 2

        # Check that we have the expected number of edges
        assert data.edge_index.size(1) == n_tiles - 1
