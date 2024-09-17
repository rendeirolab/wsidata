class TestGetAccessor:
    def test_get_pyramids(self, wsidata):
        pyramids = wsidata.get.pyramids()
        assert pyramids is not None

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
