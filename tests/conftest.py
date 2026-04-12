from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

REPO_ID = "RendeiroLab/LazySlide-data"


@pytest.fixture(scope="session")
def test_slide():
    return hf_hub_download(REPO_ID, "sample.svs", repo_type="dataset")


@pytest.fixture(scope="session")
def test_isyntax():
    return hf_hub_download(REPO_ID, "testslide.isyntax", repo_type="dataset")


@pytest.fixture(scope="session")
def test_czi():
    # The sample.czi fixture is expected to be a small Bgr24 CZI file
    # hosted alongside the other test slides in RendeiroLab/LazySlide-data.
    # Until that file is uploaded, skip the pylibczi test cleanly so CI
    # stays green.
    try:
        return hf_hub_download(REPO_ID, "sample.czi", repo_type="dataset")
    except EntryNotFoundError:
        pytest.skip("sample.czi fixture not yet uploaded to LazySlide-data")


@pytest.fixture(scope="session")
def test_store():
    slide_zarr_zip = hf_hub_download(REPO_ID, "sample.zarr.zip", repo_type="dataset")
    slide_zarr = Path(slide_zarr_zip.replace(".zip", ""))
    # Unzip the zarr file if it is a zip file
    # But only if it is not already unzipped
    if not slide_zarr.exists():
        from zipfile import ZipFile

        with ZipFile(slide_zarr_zip, "r") as zip_ref:
            zip_ref.extractall(slide_zarr.parent)
    return slide_zarr


@pytest.fixture(scope="class")
def wsidata(test_slide, test_store):
    from wsidata import open_wsi

    wsi = open_wsi(test_slide, store=test_store)

    return wsi
