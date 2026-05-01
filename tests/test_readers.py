import sys
from importlib import import_module

import pytest

from wsidata import open_wsi
from wsidata.reader._reader_registry import READERS, ReaderRegistry


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


@pytest.mark.skipif(skip_reader("tiffslide"), reason="tiffslide not installed")
def test_tiffslide(test_slide):
    run_reader_test("tiffslide", test_slide)


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


def test_spatialdata(test_slide):
    import numpy as np
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
