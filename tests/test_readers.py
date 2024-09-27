import pytest
from importlib import import_module

from wsidata import open_wsi


def try_import(mod):
    try:
        import_module(mod)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def skip_reader(reader):
    if reader == "bioformats":
        return not try_import("scyjava")
    else:
        return not try_import(reader)


def run_reader_test(reader, test_slide):
    wsi = open_wsi(test_slide, reader=reader)
    wsi.read_region(0, 0, 10, 10, level=0)
    wsi.thumbnail
    wsi.get_thumbnail(as_array=True)
    assert wsi.reader.translate_level(-1) == 0


@pytest.mark.skipif(skip_reader("openslide"), reason="openslide not installed")
def test_openslide(test_slide):
    run_reader_test("openslide", test_slide)


@pytest.mark.skipif(skip_reader("tiffslide"), reason="tiffslide not installed")
def test_tiffslide(test_slide):
    run_reader_test("tiffslide", test_slide)


@pytest.mark.skipif(skip_reader("bioformats"), reason="scyjava not installed")
def test_bioformats(test_slide):
    run_reader_test("bioformats", test_slide)


@pytest.mark.skipif(skip_reader("cucim"), reason="cucim not installed")
def test_cucim(test_slide):
    run_reader_test("cucim", test_slide)
