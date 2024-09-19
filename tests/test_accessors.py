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
