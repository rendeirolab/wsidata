import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box
from spatialdata.models import ShapesModel

from wsidata import io
from wsidata._model.tile import TileRequest, TileSpec, shapes2tiles


class TestShapes2Tiles:
    """Tests for the shapes2tiles utility function."""

    def test_with_tilespec(self, wsidata):
        """TileSpec path: returns TileRequests matching spec parameters."""
        tiles = wsidata["tiles"]
        spec = wsidata.tile_spec("tiles")
        assert spec is not None, "Fixture tiles should have a TileSpec"

        result = shapes2tiles(wsidata, "tiles")

        assert isinstance(result, list)
        assert len(result) == len(tiles)
        for tr in result:
            assert isinstance(tr, TileRequest)
            assert tr.level == spec.ops_level
            assert tr.width == spec.ops_width
            assert tr.height == spec.ops_height

    def test_no_tilespec(self, wsidata):
        """No TileSpec, no image_size: read at level 0, full extent, no resize."""
        shapes_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 500, 300), box(100, 100, 800, 600)],
        )
        shapes_gdf = ShapesModel.parse(shapes_gdf)
        key = "_test_no_spec"
        wsidata.shapes[key] = shapes_gdf

        with pytest.warns(UserWarning, match="TileSpec not found"):
            result = shapes2tiles(wsidata, key)

        assert len(result) == 2
        for tr in result:
            assert tr.level == 0
            assert tr.dsize is None

        # First shape: 500x300
        assert result[0].width == 500
        assert result[0].height == 300
        # Second shape: 700x500
        assert result[1].width == 700
        assert result[1].height == 500

        # Cleanup
        del wsidata.shapes[key]

    def test_no_tilespec_with_image_size(self, wsidata):
        """No TileSpec + image_size: optimal level selected, dsize set."""
        shapes_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 2000, 2000)],
        )
        shapes_gdf = ShapesModel.parse(shapes_gdf)
        key = "_test_no_spec_imgsize"
        wsidata.shapes[key] = shapes_gdf

        with pytest.warns(UserWarning, match="TileSpec not found"):
            result = shapes2tiles(wsidata, key, image_size=(256, 256))

        assert len(result) == 1
        tr = result[0]
        assert tr.dsize == (256, 256)
        # Level should be >= 0 (optimizer picks best)
        assert tr.level >= 0

        # Cleanup
        del wsidata.shapes[key]

    def test_degenerate_shape_raises(self, wsidata):
        """Zero-extent shape should raise ValueError."""
        # box with zero width
        shapes_gdf = gpd.GeoDataFrame(
            geometry=[box(100, 100, 100, 200)],
        )
        shapes_gdf = ShapesModel.parse(shapes_gdf)
        key = "_test_degenerate"
        wsidata.shapes[key] = shapes_gdf

        with pytest.warns(UserWarning, match="TileSpec not found"):
            with pytest.raises(ValueError, match="zero or negative extent"):
                shapes2tiles(wsidata, key)

        # Cleanup
        del wsidata.shapes[key]


class TestDimensionConventions:
    """Tests enforcing width/height ordering conventions across the codebase.

    Conventions:
    - SlideProperties.shape = (height, width)
    - get_region(x, y, width, height) → numpy (H, W, C)
    - TileSpec: tile_px tuple = (width, height)
    - shapes2tiles image_size = (width, height) — matches cv2.resize dsize
    - shapely box(minx, miny, maxx, maxy): width along x, height along y
    """

    def test_slide_properties_shape_convention(self, wsidata):
        """properties.shape is (height, width) and matches level_shape[0]."""
        props = wsidata.reader.properties
        assert props.shape == list(props.level_shape[0])
        # shape[0] = height, shape[1] = width
        # For a real slide, both should be positive
        assert props.shape[0] > 0
        assert props.shape[1] > 0

    def test_get_region_returns_correct_shape(self, wsidata):
        """get_region(x, y, width=W, height=H) returns array with shape (H, W, 3)."""
        W, H = 64, 32
        img = wsidata.reader.get_region(0, 0, W, H, level=0)
        assert img.shape == (H, W, 3), (
            f"get_region(width={W}, height={H}) should return (H={H}, W={W}, 3), "
            f"got {img.shape}"
        )

    def test_tile_spec_tuple_convention(self, wsidata):
        """TileSpec.from_wsidata(tile_px=(W, H)) → spec.width=W, spec.height=H."""
        spec = TileSpec.from_wsidata(wsidata, tile_px=(512, 256))
        assert spec.width == 512, f"Expected width=512, got {spec.width}"
        assert spec.height == 256, f"Expected height=256, got {spec.height}"

    def test_tile_spec_scalar_convention(self, wsidata):
        """TileSpec.from_wsidata(tile_px=N) → spec.width=N, spec.height=N."""
        spec = TileSpec.from_wsidata(wsidata, tile_px=224)
        assert spec.width == 224
        assert spec.height == 224

    def test_shapes2tiles_image_size_convention(self, wsidata):
        """shapes2tiles image_size=(W, H) → TileRequest.dsize=(W, H) for cv2."""
        shapes_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1000, 800)])
        shapes_gdf = ShapesModel.parse(shapes_gdf)
        key = "_test_imgsize_conv"
        wsidata.shapes[key] = shapes_gdf

        with pytest.warns(UserWarning, match="TileSpec not found"):
            result = shapes2tiles(wsidata, key, image_size=(512, 256))

        assert result[0].dsize == (512, 256), (
            f"image_size=(512, 256) should produce dsize=(512, 256), "
            f"got {result[0].dsize}"
        )

        del wsidata.shapes[key]

    def test_add_tiles_box_geometry(self, wsidata):
        """add_tiles creates boxes where maxx-minx=base_width, maxy-miny=base_height."""
        spec = TileSpec.from_wsidata(wsidata, tile_px=(100, 50))
        xys = np.array([[0, 0], [200, 300]], dtype=np.int32)
        tissue_ids = np.array([0, 0])

        io.add_tiles(
            wsidata, "_test_box_geom", xys, tile_spec=spec, tissue_ids=tissue_ids
        )

        tiles = wsidata["_test_box_geom"]
        for _, row in tiles.iterrows():
            geom = row.geometry
            box_width = geom.bounds[2] - geom.bounds[0]  # maxx - minx
            box_height = geom.bounds[3] - geom.bounds[1]  # maxy - miny
            assert box_width == spec.base_width, (
                f"Box width {box_width} != base_width {spec.base_width}"
            )
            assert box_height == spec.base_height, (
                f"Box height {box_height} != base_height {spec.base_height}"
            )

        del wsidata.shapes["_test_box_geom"]
