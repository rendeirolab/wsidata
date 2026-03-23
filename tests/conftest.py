from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download

REPO_ID = "RendeiroLab/LazySlide-data"


@pytest.fixture(scope="session")
def test_slide():
    return hf_hub_download(REPO_ID, "sample.svs", repo_type="dataset")


@pytest.fixture(scope="session")
def test_isyntax():
    return hf_hub_download(REPO_ID, "testslide.isyntax", repo_type="dataset")


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
def wsidata(test_slide):
    from wsidata import open_wsi

    wsi = open_wsi(test_slide)

    return wsi
