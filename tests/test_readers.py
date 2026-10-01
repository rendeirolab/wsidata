import pickle
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wsidata import open_wsi
from wsidata.reader import (
    FastSlideReader,
    OpenSlideReader,
    ReaderBase,
    SlideProperties,
    TiffSlideReader,
    to_datatree,
)
from wsidata.reader._reader_registry import READERS, ReaderRegistry
from wsidata.reader.base import convert_image


def try_import(mod):
    try:
        import_module(mod)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def skip_reader(reader):
    if reader == "bioformats":
        return not try_import("scyjava")
    elif reader == "pylibczi":
        return not try_import("pylibCZIrw")
    else:
        return not try_import(reader)


def run_reader_test(reader, test_slide):
    wsi = open_wsi(test_slide, reader=reader)
    wsi.read_region(0, 0, 10, 10, level=0)
    wsi.associated_images
    wsi.thumbnail
    wsi.get_thumbnail(as_array=True)
    assert wsi.reader.translate_level(-1) == wsi.properties.n_level - 1


@pytest.mark.skipif(skip_reader("openslide"), reason="openslide not installed")
def test_openslide(test_slide):
    run_reader_test("openslide", test_slide)


@pytest.mark.skipif(skip_reader("openslide"), reason="openslide not installed")
def test_single_scene_reader_rejects_nonzero_scene(test_slide):
    with pytest.raises(ValueError, match="does not support scene selection"):
        open_wsi(test_slide, reader="openslide", scene=1, store=None)


@pytest.mark.skipif(skip_reader("openslide"), reason="openslide not installed")
@pytest.mark.parametrize(
    "x, y, width, height",
    [(0, 0, 4979, 4989), (1001, 2003, 4500, 4300)],  # all of level 1; an offset
)
def test_openslide_reads_big_regions_in_chunks(
    test_pyramid_slide, monkeypatch, x, y, width, height
):
    """Memory regression: a big region went to OpenSlide in one read, and
    converting it to RGB held several copies of it, ~16 bytes per px.

    OpenSlide splits a read into 4096 px chunks, each painted at its own
    fractional offset in the level, so reading those chunks one at a time must
    give the pixels of one read.
    """
    reader = OpenSlideReader(test_pyramid_slide)
    slide = reader.reader
    one_read = convert_image(slide.read_region((x, y), 1, (width, height)))
    sizes = []
    read_region = slide.read_region

    def record_size(location, level, size):
        sizes.append(size)
        return read_region(location, level, size)

    monkeypatch.setattr(slide, "read_region", record_size)
    region = reader.get_region(x, y, width, height, level=1)

    assert len(sizes) == 4
    assert all(w <= 4096 and h <= 4096 for w, h in sizes)
    np.testing.assert_array_equal(region, one_read)


@pytest.mark.skipif(skip_reader("tiffslide"), reason="tiffslide not installed")
def test_tiffslide_big_region_matches_one_read(test_pyramid_slide):
    """TiffSlide maps offsets to the level with int(x / ds), so OpenSlide's
    chunk offsets would shift its rows by one: its big regions must match one
    read."""
    reader = TiffSlideReader(test_pyramid_slide)
    one_read = convert_image(reader.reader.read_region((0, 0), 1, (4979, 4989)))
    region = reader.get_region(0, 0, 4979, 4989, level=1)
    np.testing.assert_array_equal(region, one_read)


@pytest.mark.skipif(skip_reader("tiffslide"), reason="tiffslide not installed")
def test_tiffslide(test_slide):
    run_reader_test("tiffslide", test_slide)


@pytest.mark.skipif(skip_reader("fastslide"), reason="fastslide not installed")
def test_fastslide(test_slide):
    run_reader_test("fastslide", test_slide)


@pytest.mark.skipif(skip_reader("fastslide"), reason="fastslide not installed")
def test_fastslide_reopens_after_detach(test_slide):
    """Regression: detach_reader() closed the slide, but get_region and
    get_thumbnail kept using the scene view of the closed slide, so any read
    after it raised "slide reader is closed". TileImagesDataset detaches the
    reader before it reads tiles, so every tile dataset failed. The view also
    cannot be pickled, so the reader could not go to spawned workers.
    """
    reader = FastSlideReader(test_slide)
    region = reader.get_region(100, 200, 64, 64, level=0)
    thumbnail = reader.get_thumbnail(256)
    reader.detach_reader()

    np.testing.assert_array_equal(reader.get_region(100, 200, 64, 64, level=0), region)
    np.testing.assert_array_equal(reader.get_thumbnail(256), thumbnail)
    reader.detach_reader()
    in_worker = pickle.loads(pickle.dumps(reader))
    assert "macro" in in_worker.associated_images
    np.testing.assert_array_equal(in_worker.get_region(100, 200, 64, 64), region)


@pytest.mark.skipif(skip_reader("fastslide"), reason="fastslide not installed")
@pytest.mark.parametrize(
    "slide, level, x, y, inside, read",
    [  # 128 x 128 px regions; sample.svs is 2220 x 2967 px
        ("test_slide", 0, 2156, 100, np.s_[:, :64], (2156, 100, 64, 128)),
        ("test_slide", 0, 100, 2903, np.s_[:64, :], (100, 2903, 128, 64)),
        ("test_slide", 0, -50, -30, np.s_[30:, 50:], (0, 0, 78, 98)),
        ("test_slide", 0, 3000, 0, None, None),
        # level 1 is 4979 px wide at downsample 4.0005: x = 19800 is in px 4949,
        # x = -10 in px -3
        ("test_pyramid_slide", 1, 19800, 400, np.s_[:, :30], (19800, 400, 30, 128)),
        ("test_pyramid_slide", 1, -10, 400, np.s_[:, 3:], (0, 400, 125, 128)),
    ],
)
def test_fastslide_pads_regions_outside_the_slide(
    request, slide, level, x, y, inside, read
):
    """Regression: fastslide cropped a region reaching past the slide edge,
    and raised for one entirely outside it or at negative coordinates. Like
    the other readers, get_region returns the requested size, black outside.
    """
    reader = FastSlideReader(request.getfixturevalue(slide))
    region = reader.get_region(x, y, 128, 128, level=level)

    expected = np.zeros((128, 128, 3), dtype=np.uint8)
    if inside is not None:
        expected[inside] = reader.get_region(*read, level=level)
    np.testing.assert_array_equal(region, expected)


@pytest.mark.skipif(skip_reader("bioformats"), reason="scyjava not installed")
@pytest.mark.skipif(sys.version_info >= (3, 13), reason="Not supported on Python 3.13+")
def test_bioformats(test_slide):
    run_reader_test("bioformats", test_slide)
    # TODO: Add test for bioformats on vsi format
    #       Add test for bioformats against openslide reader


@pytest.mark.skipif(skip_reader("cucim"), reason="cucim not installed")
def test_cucim(test_slide):
    run_reader_test("cucim", test_slide)


@pytest.mark.skipif(skip_reader("isyntax"), reason="pyisyntax not installed")
def test_isyntax(test_isyntax):
    run_reader_test("isyntax", test_isyntax)


@pytest.mark.skipif(skip_reader("pylibczi"), reason="pylibCZIrw not installed")
def test_pylibczi(test_czi):
    run_reader_test("pylibczi", test_czi)


@pytest.mark.skipif(skip_reader("pylibczi"), reason="pylibCZIrw not installed")
def test_pylibczi_scenes(test_multiscene_czi):
    scene_0 = open_wsi(test_multiscene_czi, reader="pylibczi", scene=0, store=None)
    scene_1 = open_wsi(test_multiscene_czi, reader="pylibczi", scene=1, store=None)

    assert scene_0.scene == 0
    assert scene_1.scene == 1
    assert scene_0.n_scenes == scene_1.n_scenes == 2
    assert scene_0.scene_names == ("Scene 0", "Scene 1")
    assert scene_0.properties.shape == [48, 64]
    assert scene_1.properties.shape == [56, 80]
    np.testing.assert_array_equal(scene_0.read_region(0, 0, 1, 1)[0, 0], [30, 20, 10])
    np.testing.assert_array_equal(
        scene_1.read_region(0, 0, 1, 1)[0, 0], [120, 110, 100]
    )

    scene_0.close()
    scene_1.close()


@pytest.mark.skipif(skip_reader("pylibczi"), reason="pylibCZIrw not installed")
def test_invalid_scene(test_multiscene_czi):
    with pytest.raises(ValueError, match="available scenes are 0 through 1"):
        open_wsi(test_multiscene_czi, reader="pylibczi", scene=2, store=None)


@pytest.mark.skipif(skip_reader("fastslide"), reason="fastslide not installed")
def test_fastslide_scenes(test_multiscene_czi):
    wsi = open_wsi(test_multiscene_czi, reader="fastslide", scene=1, store=None)

    assert wsi.scene == 1
    assert wsi.n_scenes == 2
    assert wsi.scene_names == ("Scene 0", "Scene 1")
    assert wsi.properties.shape == [56, 80]
    assert wsi.read_region(0, 0, 8, 8).shape == (8, 8, 3)
    wsi.close()


@pytest.mark.skipif(skip_reader("fastslide"), reason="fastslide not installed")
def test_auto_reader_with_scene(test_multiscene_czi):
    wsi = open_wsi(test_multiscene_czi, scene=1, store=None)
    assert wsi.scene == 1
    assert wsi.n_scenes == 2
    assert wsi.reader.supports_scenes
    wsi.close()


@pytest.mark.skipif(skip_reader("bioformats"), reason="scyjava not installed")
@pytest.mark.skipif(sys.version_info >= (3, 13), reason="Not supported on Python 3.13+")
def test_bioformats_scenes(test_multiscene_czi):
    wsi = open_wsi(test_multiscene_czi, reader="bioformats", scene=1, store=None)

    assert wsi.scene == 1
    assert wsi.n_scenes == 2
    assert len(wsi.scene_names) == 2
    assert wsi.properties.shape == [56, 80]
    assert wsi.read_region(0, 0, 8, 8).shape == (8, 8, 3)
    wsi.close()


@pytest.mark.skipif(skip_reader("pylibczi"), reason="pylibCZIrw not installed")
def test_scene_store_name(test_multiscene_czi):
    wsi = open_wsi(test_multiscene_czi, reader="pylibczi", scene=1)
    assert Path(wsi.wsi_store).name == "multi_scene.scene-1.zarr"
    wsi.close()


def reader_case(reader, *args):
    """pytest.param(reader, *args), skipped where the reader cannot run"""
    skip = skip_reader(reader) or (
        reader == "bioformats" and sys.version_info >= (3, 13)
    )
    return pytest.param(
        reader, *args, marks=pytest.mark.skipif(skip, reason=f"{reader} not available")
    )


@pytest.mark.parametrize(
    "reader, slide, scene",
    [
        reader_case("openslide", "test_slide", None),
        reader_case("tiffslide", "test_slide", None),
        reader_case("fastslide", "test_slide", None),
        reader_case("isyntax", "test_isyntax", None),
        # The scenes of this CZI have different pixels: a copy that lost its
        # scene reads other pixels
        reader_case("fastslide", "test_multiscene_czi", 1),
        reader_case("pylibczi", "test_multiscene_czi", 1),
        reader_case("bioformats", "test_multiscene_czi", 1),
    ],
)
def test_reader_pickles_after_read(request, reader, slide, scene):
    """Regression: a read opens the slide, and pickling the reader then failed
    on the open slide handle ("ctypes objects containing pointers cannot be
    pickled" for openslide), so a TileImagesDataset that had read a tile could
    not go to spawned DataLoader workers, the default on macOS and Windows.
    A copy must open the slide again on its first read, whichever it is:
    openslide read associated images from the slide it had not opened.
    """
    original = READERS[reader](request.getfixturevalue(slide), scene=scene)
    region = original.get_region(0, 0, 32, 32)
    to_worker = pickle.dumps(original)

    np.testing.assert_array_equal(
        pickle.loads(to_worker).get_region(0, 0, 32, 32), region
    )
    assert list(pickle.loads(to_worker).associated_images) == list(
        original.associated_images
    )


@pytest.mark.parametrize(
    "reader, slide",
    [
        reader_case("openslide", "test_slide"),
        reader_case("tiffslide", "test_slide"),
        reader_case("fastslide", "test_slide"),
        reader_case("isyntax", "test_isyntax"),
    ],
)
def test_reader_pickles_after_associated_images(request, reader, slide):
    """Regression: AssociatedImages.__getattr__ read self._images, which does
    not exist yet while unpickling, so pickle's lookup of __setstate__
    recursed until RecursionError.
    """
    original = READERS[reader](request.getfixturevalue(slide))
    images = original.associated_images

    in_worker = pickle.loads(pickle.dumps(original))
    assert list(in_worker.associated_images) == list(images)


def test_store_scene_validation():
    from wsidata.io._wsi import _validate_store_scene

    reader = SimpleNamespace(n_scenes=2, scene=1)
    with pytest.raises(ValueError, match="has no scene metadata"):
        _validate_store_scene(SimpleNamespace(attrs={}), reader, "legacy.zarr")
    with pytest.raises(ValueError, match="belongs to scene 0"):
        _validate_store_scene(
            SimpleNamespace(attrs={"slide_properties": {"scene": 0}}),
            reader,
            "scene.zarr",
        )


class _EchoRegionReader(ReaderBase):
    """get_region returns its arguments instead of pixels."""

    name = "echo"
    pkg_namespaces = "os"

    def __init__(self, properties):
        self.file = "echo"
        self.properties = properties

    def get_region(self, x, y, width, height, level=0, **kwargs):
        return x, y, width, height, level

    def get_thumbnail(self, size, **kwargs):
        pass

    def create_reader(self):
        pass

    def detach_reader(self):
        pass


@pytest.mark.parametrize(
    "level, in_bounds, expected",
    [
        (0, False, (0, 0, 1503, 1000, 0)),
        (1, False, (0, 0, 375, 250, 1)),
        (0, True, (9, 200, 1494, 601, 0)),
        # ceil(601 / 4) = 151; ceil(1494 / 4) = 374, clipped to 375 - int(9 / 4)
        (1, True, (9, 200, 373, 151, 1)),
    ],
)
def test_get_level_region(level, in_bounds, expected):
    # get_region takes x, y at level 0 but width, height at the requested level
    reader = _EchoRegionReader(
        SlideProperties(
            shape=[1000, 1503],
            n_level=2,
            level_shape=[[1000, 1503], [250, 375]],
            level_downsample=[1.0, 4.0],
            bounds=[9, 200, 1494, 601],  # level-0 x, y, width, height
        )
    )
    assert reader.get_level(level, in_bounds=in_bounds) == expected


def test_spatialdata(test_slide):
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel

    img = np.random.randint(0, 256, (3, 512, 512), dtype=np.uint8)
    images = Image2DModel.parse(img, dims=("c", "y", "x"))

    big_img = np.random.randint(0, 256, (3, 5120, 5120), dtype=np.uint8)
    ms_images = Image2DModel.parse(big_img, dims=("c", "y", "x"), scale_factors=[2, 2])

    sdata = SpatialData(images={"img": images, "ms_img": ms_images})

    wsi = open_wsi(sdata, image_key="img")
    wsi.read_region(0, 0, 10, 10, level=0)
    wsi.get_thumbnail(as_array=True)

    wsi = open_wsi(sdata, image_key="ms_img")
    wsi.read_region(0, 0, 10, 10, level=0)
    wsi.get_thumbnail(as_array=True)

    assert wsi.reader.translate_level(-1) == wsi.properties.n_level - 1
    assert wsi.scene == 0
    assert wsi.n_scenes == 1
    with pytest.raises(ValueError, match="does not exist"):
        open_wsi(sdata, image_key="img", scene=1)


@pytest.mark.skipif(skip_reader("tiffslide"), reason="tiffslide not installed")
def test_to_datatree_levels_match_one_read(test_pyramid_slide):
    """TiffSlide maps offsets to the level with int(x / ds): a chunk origin
    rounded at a non-integer downsample (4.0005) reads a row or column early."""
    reader = TiffSlideReader(test_pyramid_slide)
    tree = to_datatree(reader)
    for level in range(1, reader.properties.n_level):
        height, width = reader.properties.level_shape[level]
        one_read = convert_image(
            reader.reader.read_region((0, 0), level, (width, height))
        )
        np.testing.assert_array_equal(
            tree[f"scale{level}"]["image"].values, one_read.transpose(2, 0, 1)
        )


def test_to_datatree_pickles(test_pyramid_slide):
    """Spawned workers receive the lazy image through the standard pickle."""
    image = to_datatree(OpenSlideReader(test_pyramid_slide))["scale2"]["image"]
    restored = pickle.loads(pickle.dumps(image))
    np.testing.assert_array_equal(
        restored.data.blocks[0, 0, 0].compute(), image.data.blocks[0, 0, 0].compute()
    )


def test_attached_image_layout(test_pyramid_slide):
    """sopa.io.wsi reads slides through open_wsi(attach_images=True)."""
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import Identity, Scale, get_transformation

    wsi = open_wsi(test_pyramid_slide, store=None, attach_images=True)
    image = wsi.to_spatialdata()["wsi"]
    Image2DModel().validate(image)
    assert "raw" in image.attrs
    assert list(image.children) == ["scale0", "scale1", "scale2"]
    for level, (height, width) in enumerate(wsi.properties.level_shape):
        level_image = image[f"scale{level}"]["image"]
        assert level_image.dims == ("c", "y", "x")
        assert level_image.shape == (3, height, width)
        assert list(level_image.c.values) == ["r", "g", "b"]
    assert get_transformation(image["scale0"]["image"]) == Identity()
    ds = wsi.properties.level_downsample[1]
    assert get_transformation(image["scale1"]["image"]) == Scale(
        [ds, ds], axes=("y", "x")
    )


def test_attached_image_is_not_written(test_pyramid_slide, tmp_path):
    store = tmp_path / "slide.zarr"
    wsi = open_wsi(test_pyramid_slide, store=str(store), attach_images=True)
    wsi.write()
    assert store.exists()
    assert not (store / "images" / "wsi").exists()


# ---- Extension-based reader detection tests ----


class TestExtensionIndex:
    def test_ext_index_built(self):
        """Extension index maps known extensions to correct readers."""
        READERS._ext_index = None  # force rebuild
        READERS._build_ext_index()
        idx = READERS._ext_index

        # .czi should map to pylibczi (if registered)
        if "pylibczi" in READERS:
            assert "pylibczi" in idx.get(".czi", [])

        # .isyntax should map to isyntax
        if "isyntax" in READERS:
            assert "isyntax" in idx.get(".isyntax", [])

        # .svs should include openslide first (highest priority)
        if "openslide" in READERS:
            svs_readers = idx.get(".svs", [])
            assert "openslide" in svs_readers
            assert svs_readers[0] == "openslide"

    def test_ext_index_excludes_none_and_empty(self):
        """Readers with extensions=None or () are not in the index."""
        READERS._ext_index = None
        READERS._build_ext_index()
        idx = READERS._ext_index

        all_indexed_readers = set()
        for names in idx.values():
            all_indexed_readers.update(names)

        # bioformats has extensions=None → not indexed
        assert "bioformats" not in all_indexed_readers

    def test_ext_index_invalidation(self):
        """Registering a new reader invalidates the index."""
        from wsidata.reader.base import ReaderBase

        READERS._build_ext_index()
        assert READERS._ext_index is not None

        # Create a dummy reader
        class DummyReader(ReaderBase):
            name = "dummy"
            pkg_namespaces = "os"  # always available
            extensions = (".dummy",)

            def get_region(self, *a, **kw):
                pass

            def get_thumbnail(self, *a, **kw):
                pass

            def create_reader(self):
                pass

            def detach_reader(self):
                pass

        READERS["dummy"] = DummyReader
        assert READERS._ext_index is None  # invalidated

        READERS._build_ext_index()
        assert "dummy" in READERS._ext_index.get(".dummy", [])

        # Cleanup
        del READERS["dummy"]

    def test_get_extension(self):
        """Extension extraction works for various paths."""
        get_ext = ReaderRegistry._get_extension
        assert get_ext("slide.svs") == ".svs"
        assert get_ext("/path/to/slide.czi") == ".czi"
        assert get_ext("slide.ome.tiff") == ".ome.tiff"
        assert get_ext("slide.ome.zarr") == ".ome.zarr"
        assert get_ext("no_extension") == ""
        assert get_ext("/path/to/slide.NDPI") == ".ndpi"

    def test_priority_order_in_ext_index(self):
        """Readers sharing an extension are sorted by priority."""
        READERS._ext_index = None
        READERS._build_ext_index()
        idx = READERS._ext_index

        svs_readers = idx.get(".svs", [])
        if len(svs_readers) >= 2:
            # Verify order matches priority
            priority_rank = {name: i for i, name in enumerate(ReaderRegistry.priority)}
            ranks = [
                priority_rank.get(n, len(ReaderRegistry.priority)) for n in svs_readers
            ]
            assert ranks == sorted(ranks)
