from pathlib import Path

import pooch
import pytest

ROOT_URL = "https://lazyslide.blob.core.windows.net/lazyslide-data"


@pytest.fixture(scope="session")
def test_slide():
    path = Path(__file__).parent / "data"
    s = pooch.retrieve(
        f"{ROOT_URL}/sample.svs", fname="sample.svs", known_hash=None, path=path
    )

    return s


@pytest.fixture(scope="session")
def test_store():
    path = Path(__file__).parent / "data"
    _ = pooch.retrieve(
        f"{ROOT_URL}/sample.zarr.zip",
        fname="sample.zarr.zip",
        known_hash=None,
        path=path,
        processor=pooch.Unzip(extract_dir="sample.zarr"),
    )

    return str(path / "sample.zarr")


@pytest.fixture(scope="class")
def wsidata(test_slide, test_store):
    from wsidata import open_wsi

    wsi = open_wsi(test_slide, store=test_store)

    return wsi
