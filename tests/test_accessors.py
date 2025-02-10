import pytest
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

    def test_iter_tiles(self, wsidata):
        for it in wsidata.iter.tile_images("tiles"):
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
