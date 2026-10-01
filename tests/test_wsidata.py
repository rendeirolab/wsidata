import warnings

import numpy as np
import pandas as pd
import pytest
import zarr
from shapely import Polygon, box
from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01
from spatialdata.models import Image2DModel, PointsModel
from xarray import DataArray

from wsidata import TileSpec, WSIData, io, open_wsi
from wsidata.reader import OpenSlideReader


@pytest.mark.parametrize(
    "bounds",
    [(10, 20, 300, 400), np.array([10, 20, 300, 400])],
    ids=["tuple", "numpy"],
)
def test_set_bounds(bounds, test_slide, tmp_path):
    store = tmp_path / "sample.zarr"
    wsi = open_wsi(test_slide, store=store)
    wsi.set_bounds(bounds)

    # Lists of plain ints: attrs are written to the store as JSON
    assert wsi.properties.bounds == [10, 20, 300, 400]
    assert wsi.attrs["slide_properties"]["bounds"] == [10, 20, 300, 400]

    wsi.write()
    assert read_zarr(store).attrs["slide_properties"]["bounds"] == [10, 20, 300, 400]


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


def _write_store(slide, store):
    """Write a store of the slide whose slide properties are not the slide's."""
    wsi = open_wsi(slide, store=store)
    wsi.set_mpp(0.123)
    wsi.set_bounds([1, 2, 30, 40])
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


def test_write_does_not_warn_about_format(test_slide, tmp_path):
    """spatialdata 0.7.0 renamed format to sdata_formats, write uses the new name"""
    wsi = open_wsi(test_slide, store=tmp_path / "s.zarr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wsi.write()
    assert not [
        w
        for w in caught
        if issubclass(w.category, (DeprecationWarning, FutureWarning))
        and "format" in str(w.message)
    ]


@pytest.mark.parametrize("name", ["sdata_formats", "format"])
def test_write_passes_sdata_formats(test_slide, tmp_path, name):
    """The formats reach spatialdata, also by the deprecated name format"""
    wsi = open_wsi(test_slide, store=tmp_path / "s.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        wsi.write(**{name: SpatialDataContainerFormatV01()})
    attrs = zarr.open_group(tmp_path / "s.zarr").attrs["spatialdata_attrs"]
    assert attrs["version"] == "0.1"


@pytest.mark.parametrize(
    "fmt", [None, SpatialDataContainerFormatV01()], ids=["None", "V01"]
)
def test_write_format_is_deprecated(test_slide, tmp_path, fmt):
    """format warns, even as None, at the line of the caller"""
    wsi = open_wsi(test_slide, store=tmp_path / "s.zarr")
    with pytest.warns(FutureWarning, match="sdata_formats") as record:
        wsi.write(format=fmt)
    assert record.pop(FutureWarning).filename == __file__


@pytest.mark.parametrize(
    "fmt", [None, SpatialDataContainerFormatV01()], ids=["None", "V01"]
)
def test_write_rejects_format_with_sdata_formats(test_slide, tmp_path, fmt):
    """format and its new name sdata_formats cannot both be passed"""
    wsi = open_wsi(test_slide, store=tmp_path / "s.zarr")
    with pytest.raises(TypeError, match="sdata_formats"):
        wsi.write(sdata_formats=SpatialDataContainerFormatV01(), format=fmt)


def test_write_without_store_goes_to_path(tmp_path):
    """Opened from a SpatialData, a WSIData has its path but no store"""
    store = tmp_path / "s.zarr"
    image = Image2DModel.parse(np.zeros((3, 64, 64), np.uint8), dims=("c", "y", "x"))
    sdata = SpatialData(images={"image": image})
    sdata.write(store)
    wsi = open_wsi(sdata, image_key="image")
    io.add_tissues(wsi, "tissues", [box(0, 0, 10, 10)])
    wsi.write()

    assert read_zarr(store).shapes["tissues"].area.tolist() == [100.0]


def test_write_saved_thumbnail(test_slide, tmp_path):
    """Regression: with spatialdata <0.7.3 and ome-zarr >=0.14, writing any
    image failed with "TypeError: Expected an iterable of integers", and a
    single-scale image was written as a pyramid.
    """
    store = tmp_path / "s.zarr"
    wsi = open_wsi(
        test_slide,
        store=store,
        attach_thumbnail=True,
        save_thumbnail=True,
        thumbnail_size=200,
    )
    thumbnail = wsi.images["wsi_thumbnail"]
    wsi.write()
    wsi.close()

    written = read_zarr(store).images["wsi_thumbnail"]
    # Single scale, as attached
    assert isinstance(written, DataArray)
    np.testing.assert_array_equal(written.values, thumbnail.values)


def test_write_back_into_own_store(test_slide, tmp_path):
    store = tmp_path / "slide.zarr"
    wsi = open_wsi(test_slide, store=store)
    io.add_tissues(wsi, "tissues", [box(0, 0, 10, 10)])
    wsi.write()

    # The usual workflow: reopen the slide's store and save new results into it
    wsi = open_wsi(test_slide, store=store)
    io.add_tissues(wsi, "more_tissues", [box(0, 0, 20, 20)])
    wsi.write()
    io.add_tissues(wsi, "tissues", [box(0, 0, 30, 30)])
    wsi.write_element("tissues", overwrite=True)

    shapes = read_zarr(store).shapes
    assert shapes["tissues"].area.tolist() == [900.0]
    assert shapes["more_tissues"].area.tolist() == [400.0]


def test_write_keeps_store_that_lazy_elements_read_from(
    test_slide, tmp_path, monkeypatch
):
    # A relative store, as open_wsi("slide.svs") gives, while backing files are
    # absolute
    monkeypatch.chdir(tmp_path)
    store = "slide.zarr"
    wsi = open_wsi(test_slide, store=store)
    cells = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    wsi.points["cells"] = PointsModel.parse(cells)
    wsi.write()

    # Reopened, the points are read lazily from parquet files inside the store,
    # which overwriting the store would delete before reading them
    wsi = open_wsi(test_slide, store=store)
    with pytest.raises(ValueError):
        wsi.write(overwrite=True)

    assert read_zarr(store).points["cells"].compute()["x"].tolist() == [1.0, 2.0]
