import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

from wsidata import TileSpec
from wsidata import io


class TestWSIData:
    n_tiles = 100

    def test_repr(self, wsidata):
        repr(wsidata)

    def test_add_tissues(self, wsidata):
        tissue = np.array(
            [
                [0, 0],
                [0, 10],
                [10, 10],
                [10, 0],
            ]
        )

        tissue_holes = [
            np.array(
                [
                    [2, 2],
                    [2, 8],
                    [8, 8],
                    [8, 2],
                ]
            )
        ]

        tissues = [Polygon(tissue, tissue_holes)]

        io.add_tissues(wsidata, "test_tissue", tissues)

        tissue_table = wsidata["test_tissue"]

        assert "tissue_id" in tissue_table.columns

    def test_add_tiles(self, wsidata):
        tiles = np.random.randint(0, 255, (self.n_tiles, 2), dtype=np.int32)

        io.add_tiles(
            wsidata,
            "test_tile",
            tiles,
            tile_spec=TileSpec.from_wsidata(wsidata, 25, tissue_name="test_tissue"),
            tissue_ids=np.random.randint(0, 2, self.n_tiles),
        )

        tile_table = wsidata["test_tile"]
        assert "id" in tile_table.columns
        assert "tissue_id" in tile_table.columns
        assert "x" in tile_table.columns
        assert "y" in tile_table.columns

    @pytest.mark.parametrize("format", ["dict", "dataframe"])
    def test_update_shape_data(self, format, wsidata):
        data = {"key1": np.random.rand(self.n_tiles)}

        if format == "dict":
            io.update_shapes_data(wsidata, "test_tile", data=data)
        elif format == "dataframe":
            io.update_shapes_data(wsidata, "test_tile", data=pd.DataFrame(data))

    def test_add_features(self, wsidata):
        features = np.random.rand(self.n_tiles, 1024)

        io.add_features(wsidata, "test_feature", "test_tile", features)

    def test_save(self, wsidata, tmpdir):
        wsidata.write(tmpdir / "test.wsi")
