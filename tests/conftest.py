from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def test_slide():
    return Path(__file__).parent / "data" / "CMU-1-Small-Region.svs"


@pytest.fixture(scope="class")
def wsidata(test_slide):
    from wsidata import open_wsi, TileSpec
    from wsidata.io import add_tissues, add_tiles, add_features
    from shapely import box

    wsi = open_wsi(test_slide)

    # Mock tissue
    add_tissues(
        wsi,
        "tissues",
        tissues=[
            box(0, 0, 100, 125),
            box(100, 0, 200, 225),
            box(200, 200, 300, 325),
        ],
    )
    # Mock tiles
    spec = TileSpec.from_wsidata(wsi, 25)
    add_tiles(
        wsi,
        "tiles",
        xys=np.asarray(
            [
                [0, 0],
                [0, 25],
                [0, 50],
                [100, 0],
                [100, 25],
                [100, 50],
                [200, 200],
                [200, 225],
                [200, 250],
            ]
        ),
        tile_spec=spec,
        tissue_ids=[0, 0, 0, 1, 1, 1, 2, 2, 2],
    )
    # Mock features
    add_features(wsi, "features", "tiles", features=np.random.rand(9, 2048))

    return wsi
