import numpy as np
import pandas as pd
import pytest
from shapely import Polygon
from spatialdata import read_zarr

from wsidata import TileSpec, WSIData, io, open_wsi
from wsidata.reader import OpenSlideReader


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

    # def test_save(self, wsidata, tmpdir):
    #     wsidata.write(tmpdir / "test.zarr")


def _write_store(slide, store):
    """Write a store of the slide whose slide properties are not the slide's."""
    wsi = open_wsi(slide, store=store)
    wsi.set_mpp(0.123)
    # set_bounds raises KeyError: it writes to a table that does not exist
    wsi.attrs["slide_properties"]["bounds"] = [1, 2, 30, 40]
    # As written by a reader that saw another pyramid
    wsi.attrs["slide_properties"].update(
        n_level=2, level_shape=[[2967, 2220], [741, 555]], level_downsample=[1.0, 4.0]
    )
    wsi.write()
    wsi.close()


def test_reopen_keeps_set_mpp_and_bounds(test_slide, tmp_path):
    """Regression: WSIData looked for the stored slide properties among the
    elements, not the attrs, so it never found them and replaced them with the
    reader's on every open: the mpp from set_mpp and the bounds were lost once
    the store was opened again. The other properties are still the reader's,
    so a store written by another reader does not bring its pyramid along.
    """
    store = tmp_path / "s.zarr"
    _write_store(test_slide, store)

    wsi = open_wsi(test_slide, store=store)
    assert wsi.properties.mpp == 0.123
    assert wsi.properties.bounds == [1, 2, 30, 40]
    assert wsi.properties.n_level == 1
    assert wsi.properties.level_shape == [[2967, 2220]]
    assert wsi.properties.level_downsample == [1.0]
    # The next write keeps them
    assert wsi.attrs["slide_properties"]["mpp"] == 0.123
    assert wsi.attrs["slide_properties"]["bounds"] == [1, 2, 30, 40]
    wsi.close()


def test_reopen_keeps_the_reader_mpp_and_bounds_a_store_lacks(test_slide, tmp_path):
    """A store may hold no mpp, from a reader that found none, or no bounds:
    the reader's are kept.
    """
    store = tmp_path / "s.zarr"
    wsi = open_wsi(test_slide, store=store)
    wsi.attrs["slide_properties"]["mpp"] = None
    del wsi.attrs["slide_properties"]["bounds"]
    wsi.write()
    wsi.close()

    wsi = open_wsi(test_slide, store=store)
    assert wsi.properties.mpp == 0.499
    assert wsi.properties.bounds == [0, 0, 2220, 2967]
    wsi.close()


def test_slide_properties_source_slide_ignores_the_store(test_slide, tmp_path):
    """With slide_properties_source="slide", the mpp and bounds in the store
    are not used, and the next write replaces them with the slide's.
    """
    store = tmp_path / "s.zarr"
    _write_store(test_slide, store)

    wsi = WSIData.from_spatialdata(
        read_zarr(store), OpenSlideReader(test_slide), slide_properties_source="slide"
    )
    assert wsi.properties.mpp == 0.499
    assert wsi.properties.bounds == [0, 0, 2220, 2967]
    assert wsi.attrs["slide_properties"]["mpp"] == 0.499
    wsi.close()


def test_store_folder_keeps_the_stores_of_slides_apart(
    test_slide, test_pyramid_slide, tmp_path
):
    """Regression: a store path that did not exist yet became the store of the
    first slide, so every later slide opened with it loaded that store and
    wrote into it. A new path without a .zarr suffix is a folder of stores.
    """
    folder = tmp_path / "data"
    for slide in (test_slide, test_pyramid_slide):
        wsi = open_wsi(slide, store=str(folder))
        wsi.write()
        wsi.close()
    assert sorted(p.name for p in folder.iterdir()) == [
        "GTEX-1117F-0526.zarr",
        "sample.zarr",
    ]


def test_store_of_another_slide_raises(test_slide, test_pyramid_slide, tmp_path):
    """A store keeps the shape of its slide, so the store of another slide
    raises instead of being loaded and overwritten. Here it is data, the one
    store that store="data" used to give every slide.
    """
    store = tmp_path / "data"
    wsi = open_wsi(test_slide, store=None)
    wsi.write(store)
    wsi.close()
    with pytest.raises(ValueError, match="belongs to a slide of shape"):
        open_wsi(test_pyramid_slide, store=str(store))
