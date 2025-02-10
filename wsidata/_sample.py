from zipfile import ZipFile

from platformdirs import user_cache_path

from wsidata import open_wsi

CACHE_PATH = user_cache_path("wsidata", ensure_exists=True)


def sample():
    # Download the sample data
    from urllib.request import urlretrieve

    root = "https://github.com/rendeirolab/wsidata/blob/main/tests/data"

    svs_url = f"{root}/sample.svs?raw=true"
    zarr_url = f"{root}/sample.zarr.zip?raw=true"

    urlretrieve(svs_url, CACHE_PATH / "sample.svs")
    urlretrieve(zarr_url, CACHE_PATH / "sample.zarr.zip")

    with ZipFile(CACHE_PATH / "sample.zarr.zip", "r") as zip_ref:
        zip_ref.extractall(CACHE_PATH / "sample.zarr")

    return open_wsi(CACHE_PATH / "sample.svs", store=str(CACHE_PATH / "sample.zarr"))
