import numpy as np
import pandas as pd
import pytest
from shapely import Polygon

from wsidata import TileSpec


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

        wsidata.add_tissues("test_tissue", tissues)

        tissue_table = wsidata.sdata["test_tissue"]

        assert "tissue_id" in tissue_table.columns

    def test_add_tiles(self, wsidata):
        tiles = np.random.randint(0, 255, (self.n_tiles, 2), dtype=np.int32)

        wsidata.add_tiles(
            "test_tile",
            tiles,
            tile_spec=TileSpec(
                height=256,
                width=256,
                raw_height=256,
                raw_width=256,
                tissue_name="test_tissue",
                level=0,
                downsample=1,
                mpp=0.5,
            ),
            tissue_ids=np.random.randint(0, 2, self.n_tiles),
        )

        tile_table = wsidata.sdata["test_tile"]
        assert "id" in tile_table.columns
        assert "tissue_id" in tile_table.columns
        assert "x" in tile_table.columns
        assert "y" in tile_table.columns

    @pytest.mark.parametrize("format", ["dict", "dataframe"])
    def test_update_shape_data(self, format, wsidata):
        data = {"key1": np.random.rand(self.n_tiles)}

        if format == "dict":
            wsidata.update_shapes_data("test_tile", data=data)
        elif format == "dataframe":
            wsidata.update_shapes_data("test_tile", data=pd.DataFrame(data))

    def test_add_features(self, wsidata):
        features = np.random.rand(self.n_tiles, 1024)

        wsidata.add_features("test_feature", "test_tile", features)

    def test_save(self, wsidata, tmpdir):
        wsidata.save(tmpdir / "test.wsi")
