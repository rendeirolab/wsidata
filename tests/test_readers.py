import sys
from importlib import import_module

import pytest

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
