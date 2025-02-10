from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture(scope="session")
def test_slide():
    return Path(__file__).parent / "data" / "sample.svs"


@pytest.fixture(scope="session")
def test_store(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("data")

    with ZipFile(Path(__file__).parent / "data" / "sample.zarr.zip", "r") as zip_ref:
        zip_ref.extractall(tmpdir / "sample.zarr")

    return str(tmpdir / "sample.zarr")


@pytest.fixture(scope="class")
def wsidata(test_slide, test_store):
    from wsidata import open_wsi

    wsi = open_wsi(test_slide, store=test_store)

    return wsi
