from anndata import AnnData


class TestGetAccessor:
    def test_pyramids(self, wsidata):
        tables = wsidata.get.pyramids()

        assert len(tables) >= 1

        assert tables.index.name == "level"
        assert "width" in tables.columns
        assert "height" in tables.columns

    def test_feature_anndata(self, wsidata):
        tables = wsidata.get.feature_anndata("tiles")

        assert isinstance(tables, AnnData)
        assert "X" in tables
        assert "obs" in tables
        assert "obsm" in tables
        assert "obsp" in tables
        assert "uns" in tables
        assert "tile_spec" in tables.uns
        assert "slide_properties" in tables.uns

    def test_get_features_anndata(self, wsidata):
        features = wsidata.get.features_anndata("resnet50")
        assert features is not None

    def test_get_n_tissues(self, wsidata):
        n_tissues = wsidata.get.n_tissues("tissues")
        assert n_tissues == 1

    def test_get_n_tiles(self, wsidata):
        n_tiles = wsidata.get.n_tiles("tiles")
        assert n_tiles == 20


class TestIterAccessor:
    def test_iter_tissues(self, wsidata):
        for _ in wsidata.iter.tissue_images("tissues"):
            pass

    def test_iter_tiles(self, wsidata):
        for _ in wsidata.iter.tile_images("tiles"):
            pass

    def test_iter_contours(self, wsidata):
        for _ in wsidata.iter.tissue_contours("tissues"):
            pass


class TestDatasetAccessor:
    def test_ds_tile_images(self, wsidata):
        dataset = wsidata.ds.tile_images("tiles")
        assert dataset.tiles.shape == (20, 2)
        assert dataset.spec is not None
