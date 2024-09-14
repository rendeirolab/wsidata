from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_slide():
    return Path(__file__).parent / "data" / "CMU-1-Small-Region.svs"


@pytest.fixture(scope="class")
def wsidata(test_slide):
    from wsidata import open_wsi

    return open_wsi(test_slide)
