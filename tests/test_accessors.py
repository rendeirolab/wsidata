import pytest
from anndata import AnnData


class TestGetAccessor:
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
        assert "tile_spec" in tables.uns
        assert "slide_properties" in tables.uns

    def test_get_n_tissue(self, wsidata):
        wsidata.fetch.n_tissue("tissues")

    def test_get_n_tiles(self, wsidata):
        wsidata.fetch.n_tiles("tiles")


class TestIterAccessor:
    @pytest.mark.parametrize("color_norm", ["macenko", "reinhard"])
    def test_iter_tissues(self, wsidata, color_norm):
        for _ in wsidata.iter.tissue_images("tissues", color_norm=color_norm):
            pass

        next(wsidata.iter.tissue_images("tissues", mask_bg=True))

    def test_iter_tiles(self, wsidata):
        for _ in wsidata.iter.tile_images("tiles"):
            pass

    def test_iter_contours(self, wsidata):
        for _ in wsidata.iter.tissue_contours("tissues"):
            pass


class TestDatasetAccessor:
    def test_ds_tile_images(self, wsidata):
        dataset = wsidata.ds.tile_images("tiles")
        assert dataset.spec is not None
