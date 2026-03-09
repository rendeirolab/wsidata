import numpy as np
import pandas as pd
import pytest
import tifffile
from shapely import Polygon
from spatialdata import read_zarr
from spatialdata.models import Image2DModel

from wsidata import TileSpec, io, open_wsi


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
        assert "tile_id" in tile_table.columns
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


def _make_test_wsi(path, size=(256, 256)):
    """Create a minimal RGB TIFF file for testing."""
    img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    res = 1e7 / 200
    tifffile.imwrite(
        path,
        img,
        tile=(64, 64),
        photometric="rgb",
        compression="deflate",
        bigtiff=True,
        resolution=(res, res),
        resolutionunit="CENTIMETER",
    )


def test_write_preserves_existing_images(tmp_path):
    """write() should not zero out images stored in a previous session.

    Regression test: calling write() on an existing zarr store used to clear
    the store before re-writing, which zeroed out dask-backed elements that
    were loaded from that same store but not modified in the current session.
    """
    wsi_path = tmp_path / "tissue.tiff"
    store_dir = tmp_path / "store.zarr"

    _make_test_wsi(wsi_path)

    # First session: write a non-zero ROI image.
    wsi = open_wsi(wsi=wsi_path, store=str(store_dir), reader="openslide")
    roi = np.random.rand(3, 256, 256).astype(np.float32) * 100.0
    da = Image2DModel.parse(
        roi, dims=("c", "y", "x"), c_coords=[f"c{i}" for i in range(3)]
    )
    wsi.images["roi"] = da
    wsi.write(store_dir)

    first_max = float(read_zarr(store_dir)["roi"].max().compute())
    assert first_max > 0, "First write produced a zero ROI unexpectedly"

    # Second session: re-open and call write() without touching the ROI.
    wsi2 = open_wsi(wsi=wsi_path, store=str(store_dir), reader="openslide")
    wsi2.write(store_dir)

    second_max = float(read_zarr(store_dir)["roi"].max().compute())
    assert second_max > 0, "Second write dropped / zeroed the existing ROI"
    assert abs(first_max - second_max) < 1e-6, "ROI value changed after second write"
