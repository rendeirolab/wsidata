from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

REPO_ID = "RendeiroLab/LazySlide-data"

# Zeiss CZI test file (LGPL-licensed, not redistributable via our HF repo)
CZI_URL = "https://github.com/ZEISS/pylibczirw/raw/main/test_data/c1_bgr24.czi"


@pytest.fixture(scope="session")
def test_slide():
    return hf_hub_download(REPO_ID, "sample.svs", repo_type="dataset")


@pytest.fixture(scope="session")
def test_isyntax():
    return hf_hub_download(REPO_ID, "testslide.isyntax", repo_type="dataset")


@pytest.fixture(scope="session")
def test_czi():
    # Zeiss c1_bgr24.czi from pylibczirw repo (LGPL license, fetched directly)
    from urllib.request import urlretrieve

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "c1_bgr24.czi"
    if not dest.exists():
        urlretrieve(CZI_URL, dest)
    return str(dest)


@pytest.fixture(scope="session")
def test_multiscene_czi(tmp_path_factory):
    """Create a tiny, deterministic two-scene BGR CZI."""
    czi = pytest.importorskip("pylibCZIrw.czi")

    path = tmp_path_factory.mktemp("czi") / "multi_scene.czi"
    scene_0 = np.zeros((48, 64, 3), dtype=np.uint8)
    scene_0[...] = (10, 20, 30)
    scene_1 = np.zeros((56, 80, 3), dtype=np.uint8)
    scene_1[...] = (100, 110, 120)

    with czi.create_czi(str(path)) as doc:
        doc.write(scene_0, location=(-100, 20), scene=0)
        doc.write(scene_1, location=(50, -10), scene=1)
        doc.write_metadata(scale_x=0.5e-6, scale_y=0.5e-6)
    return str(path)


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
