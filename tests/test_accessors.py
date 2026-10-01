import pickle
import sys
import warnings
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData
from scipy import sparse

from wsidata import open_wsi
from wsidata.io import add_features, add_shapes, subset_tiles, update_shapes_data


@pytest.fixture
def own_wsidata(test_slide, test_store):
    """A WSIData of its own, whose elements a test may change in memory"""
    return open_wsi(test_slide, store=test_store)


@pytest.fixture
def stub_torch_geometric(monkeypatch):
    """Run graph_data without torch_geometric: Data returns its arguments"""
    stub = SimpleNamespace(Data=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "torch_geometric", stub)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", stub)


def chain_graph(n, sparse_format):
    """A tile graph linking tile i to tile i + 1, at a distance of i + 1"""
    i = np.arange(n - 1)
    return AnnData(
        obs=pd.DataFrame(index=np.arange(n).astype(str)),
        obsp={
            "spatial_connectivities": sparse_format(
                (np.ones(n - 1), (i, i + 1)), shape=(n, n)
            ),
            "spatial_distances": sparse_format((i + 1.0, (i, i + 1)), shape=(n, n)),
        },
    )


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

    @pytest.mark.parametrize("n_kept", [None, 20], ids=["reordered", "subset"])
    def test_features_anndata_follows_tiles(self, own_wsidata, n_kept):
        """Regression: features_anndata put the rows of the feature table next
        to the tiles by position, not by tile_id, so once the tiles were
        reordered each tile got the features of another, and once they were
        subset AnnData raised on the different lengths.
        """
        features = own_wsidata.tables["resnet50_tiles"]
        features.layers["doubled"] = features.X * 2
        order = np.random.default_rng(0).permutation(features.n_obs)[:n_kept]
        subset_tiles(own_wsidata, "tiles", order)
        tile_ids = own_wsidata.shapes["tiles"]["tile_id"].to_numpy()

        adata = own_wsidata.fetch.features_anndata("resnet50")

        row_of = dict(zip(features.obs["tile_id"], range(features.n_obs)))
        rows = [row_of[t] for t in tile_ids]
        np.testing.assert_array_equal(adata.obs["tile_id"], tile_ids)
        np.testing.assert_array_equal(adata.X, features.X[rows])
        np.testing.assert_array_equal(adata.layers["doubled"], features.X[rows] * 2)

    def test_features_anndata_raises_for_tiles_without_features(self, own_wsidata):
        """Regression: a tile with no row in the feature table got the features
        in the row at its position, those of another tile.
        """
        tile_ids = own_wsidata.shapes["tiles"]["tile_id"].to_numpy().copy()
        tile_ids[0] = tile_ids.max() + 1
        update_shapes_data(own_wsidata, "tiles", {"tile_id": tile_ids})

        with pytest.raises(ValueError, match="No features"):
            own_wsidata.fetch.features_anndata("resnet50")

    def test_features_anndata_of_shapes_without_tile_spec(self, own_wsidata):
        """Regression: features_anndata read the TileSpec of the shapes with no
        check, so shapes that have none, such as cells, raised AttributeError.
        """
        cells = gpd.GeoDataFrame(geometry=own_wsidata.shapes["tiles"].geometry.values)
        add_shapes(own_wsidata, "cells", cells)
        features = np.random.default_rng(0).random((len(cells), 8), dtype=np.float32)
        add_features(own_wsidata, "resnet50_cells", "cells", features)

        adata = own_wsidata.fetch.features_anndata("resnet50", tile_key="cells")

        assert "tile_spec" not in adata.uns
        # The cells have no tile_id, so the features follow them by position
        np.testing.assert_array_equal(adata.X, features)

    def test_features_anndata_by_position_needs_one_row_per_tile(self, own_wsidata):
        """Regression: tiles without tile_id and a feature table of another
        length raised AnnData's error on the length of obs, which does not say
        that the rows are matched by position.
        """
        tiles = own_wsidata.shapes["tiles"]
        own_wsidata.shapes["tiles"] = tiles.drop(columns="tile_id").iloc[:20]

        with pytest.raises(ValueError, match="by position"):
            own_wsidata.fetch.features_anndata("resnet50")

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
        with pytest.warns(FutureWarning, match="color_norm"):
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
        assert len(dataset) > 0
        item = dataset[0]
        assert "image" in item
        assert "x" in item
        assert "y" in item
        assert "tissue_id" in item
        assert "downsample" in item
        assert item["downsample"] > 0

    def test_ds_tile_images_pickles_after_read(self, wsidata):
        """Regression: reading a tile cached a lambda as the color normalizer,
        so a dataset that had read a tile could not go to spawned DataLoader
        workers."""
        dataset = wsidata.ds.tile_images("tiles")
        tile = dataset[0]["image"]

        in_worker = pickle.loads(pickle.dumps(dataset))
        np.testing.assert_array_equal(in_worker[0]["image"], tile)

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
        sp = pytest.importorskip(
            "scipy.sparse", reason="requires scipy and torch_geometric to be installed"
        )
        pytest.importorskip(
            "torch_geometric",
            reason="requires scipy and torch_geometric to be installed",
        )

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

    @pytest.mark.usefixtures("stub_torch_geometric")
    @pytest.mark.parametrize(
        "sparse_format", [sparse.csr_matrix, sparse.csr_array], ids=lambda f: f.__name__
    )
    def test_ds_tile_feature_graph_of_sparse_arrays(self, own_wsidata, sparse_format):
        """Regression: graph_data read the distances of the edges with .A1,
        which np.matrix has but ndarray does not, so a tile graph stored as
        scipy sparse arrays, such as csr_array, raised AttributeError.
        """
        n = len(own_wsidata.shapes["tiles"])
        own_wsidata.tables["tiles_graph"] = chain_graph(n, sparse_format)

        data = own_wsidata.ds.tile_feature_graph("resnet50")

        i = np.arange(n - 1)
        np.testing.assert_array_equal(data["edge_index"], [i, i + 1])
        np.testing.assert_array_equal(data["edge_attr"], (i + 1.0)[:, None])

    @pytest.mark.usefixtures("stub_torch_geometric")
    @pytest.mark.parametrize("n_kept", [None, 20], ids=["reordered", "subset"])
    def test_ds_tile_feature_graph_follows_tiles(self, own_wsidata, n_kept):
        """Regression: the tile graph is in the order of the tiles, but
        graph_data took the node features in the order of the feature table,
        so once the tiles were reordered or subset each node got the features
        of another tile. Subset tiles also raised KeyError on the targets:
        torch reads a Series at the index labels 0 to n - 1, and a subset of
        the tiles lacks some of them.
        """
        features = own_wsidata.tables["resnet50_tiles"]
        order = np.random.default_rng(0).permutation(features.n_obs)[:n_kept]
        subset_tiles(own_wsidata, "tiles", order)
        tile_ids = own_wsidata.shapes["tiles"]["tile_id"].to_numpy()
        own_wsidata.tables["tiles_graph"] = chain_graph(
            len(tile_ids), sparse.csr_matrix
        )

        # The tile_id of each node, as its target
        data = own_wsidata.ds.tile_feature_graph("resnet50", target_key="tile_id")

        row_of = dict(zip(features.obs["tile_id"], range(features.n_obs)))
        rows = [row_of[t] for t in tile_ids]
        np.testing.assert_array_equal(data["y"], tile_ids)
        np.testing.assert_array_equal(data["x"], features.X[rows])


@pytest.mark.parametrize(
    "read",
    [
        lambda wsi, **kw: wsi.ds.tile_images("tiles", **kw),
        lambda wsi, **kw: next(wsi.iter.tile_images("tiles", **kw)),
        lambda wsi, **kw: next(wsi.iter.tissue_images("tissues", **kw)),
    ],
    ids=["ds.tile_images", "iter.tile_images", "iter.tissue_images"],
)
def test_color_norm_is_deprecated(wsidata, read):
    """color_norm warns, at the line of the caller; without it, nothing warns"""
    with pytest.warns(FutureWarning, match="color_norm") as record:
        read(wsidata, color_norm="macenko")
    assert record.pop(FutureWarning).filename == __file__

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read(wsidata)
    assert not [w for w in caught if "color_norm" in str(w.message)]


def test_color_normalizer_is_deprecated():
    """wsidata.ColorNormalizer warns, at the line of the caller"""
    with pytest.warns(FutureWarning, match="ColorNormalizer") as record:
        from wsidata import ColorNormalizer  # noqa: F401
    assert record.pop(FutureWarning).filename == __file__
