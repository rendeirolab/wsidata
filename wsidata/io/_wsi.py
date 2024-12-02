from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from anndata import AnnData
from fsspec.core import url_to_fs
from rich.progress import track
from spatialdata import read_zarr, SpatialData
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale

from ._download import CacheDownloader
from .._model import WSIData
from ..reader import get_reader, to_datatree


def open_wsi(
    wsi,
    store=None,
    reader=None,
    download=True,
    name=None,
    cache_dir=None,
    pbar=True,
    attach_images=False,
    image_key="wsi",
    save_images=False,
    attach_thumbnail=True,
    thumbnail_key="wsi_thumbnail",
    thumbnail_size=2000,
    save_thumbnail=True,
    **kwargs,
):
    """Open a whole slide image.

    You can open a whole slide image from a URL or a local file.
    If load from remote URL, the image will be downloaded and cached (default to current the working directory).
    You can also attach images and thumbnail to the SpatialData object. By default, only the thumbnail is attached,
    the thumbnail is a downsampled version of the whole slide image, the original image is not attached to save disk space
    when you save the WSIData object on disk.

    Parameters
    ----------
    wsi : str or Path
        The URL to whole slide image.
    store : str, optional
        The backed file path, by default will create
        a zarr file with the same name as the slide file.
        You can either supply a file path or a directory.
        If a directory is supplied, the zarr file will be created in that directory.
        This is useful when you want to store all zarr files in a specific location.
    reader : str, optional
        Reader to use, by default "auto", choosing available reader, first openslide, then tifffile.
    download : bool, optional
        Whether to download the slide.
    name : str, optional
        The name of the slide.
    cache_dir : str, optional
        The cache directory, by default file will be stored in working direction.
    pbar : bool, optional
        Whether to show progress bar, by default True.
    attach_images : bool, optional, default: False
        Whether to attach whole slide image to image slot in the spatial data object.
    image_key : str, optional
        The key to store the whole slide image, by default "wsi_thumbnail".
    save_images : bool, optional, default: False
        Whether to save the whole slide image to on the disk.
        Only works for wsi.save() method.
    attach_thumbnail : bool, optional, default: True
        Whether to attach thumbnail to image slot in the spatial data object.
    thumbnail_key : str, optional
        The key to store the thumbnail, by default "wsi_thumbnail".
    thumbnail_size : int, optional, default: 2000
        The size of the thumbnail.
    save_thumbnail : bool, optional, default: True
        Whether to save the thumbnail to on the disk.

        Only works for wsi.save() method.

    Returns
    -------
    :class:`WSIData`
        Whole slide image data.

    Examples
    --------

    .. code-block:: python

        >>> from wsidata import open_wsi
        >>> wsi = open_wsi("https://bit.ly/3ZvbzVc")

    """

    # Check if the slide is a file or URL
    wsi = str(wsi)
    fs, wsi_path = url_to_fs(wsi)
    if not fs.exists(wsi_path):
        raise ValueError(f"Slide {wsi} not existed or not accessible.")

    # Early attempt with reader
    format = Path(wsi).suffix
    ReaderCls = get_reader(reader, format=format)

    if download and fs.protocol != "file":
        downloader = CacheDownloader(wsi_path, name=name, cache_dir=cache_dir)
        wsi = downloader.download(pbar)

    reader_obj = ReaderCls(wsi)
    wsi = Path(wsi)
    if store is None:
        store = wsi.with_suffix(".zarr")
    else:
        # We also support write all backed file to a directory
        backed_file_p = Path(store)
        if backed_file_p.is_dir():
            if is_zarr_dir(backed_file_p):
                store = backed_file_p
            else:
                zarr_name = Path(wsi).with_suffix(".zarr").name
                store = backed_file_p / zarr_name
        else:
            store = backed_file_p

    if store.exists():
        sdata = read_zarr(store)
    else:
        sdata = SpatialData()

    exclude_elements = []

    if attach_images and image_key not in sdata:
        images_datatree = to_datatree(reader_obj)
        sdata.images[image_key] = images_datatree
        if not save_images:
            exclude_elements.append(image_key)

    if attach_thumbnail and thumbnail_key not in sdata:
        thumbnail = reader_obj.get_thumbnail(thumbnail_size)
        thumbnail_shape = thumbnail.shape
        origin_shape = reader_obj.properties.shape
        scale_x, scale_y = (
            origin_shape[0] / thumbnail_shape[0],
            origin_shape[1] / thumbnail_shape[1],
        )

        if thumbnail is not None:
            sdata.images[thumbnail_key] = Image2DModel.parse(
                np.asarray(thumbnail).transpose(2, 0, 1),
                dims=("c", "y", "x"),
                transformations={"global": Scale([scale_x, scale_y], axes=("x", "y"))},
            )
            if not save_thumbnail:
                exclude_elements.append(thumbnail_key)

    slide_data = WSIData.from_spatialdata(sdata, reader_obj)
    slide_data.set_exclude_elements(exclude_elements)
    slide_data.set_wsi_store(store)
    return slide_data


def agg_wsi(
    slides_table,
    feature_key,
    tile_key="tiles",
    agg_key="agg",
    wsi_col=None,
    store_col=None,
    error="raise",
):
    """
    Aggregate feature from a whole slide image.

    Parameters
    ----------
    slides_table: pd.DataFrame
        The table of slides, including information of whole slide image and .zarr paths and metadata.

        Backed file path to the anndata file is optional, by default the same directory as the
        whole slide image.
    feature_key: str
        The feature key on which aggregation should be run on.
    tile_key: str
        The tile key.
    agg_key: str
        The output aggregation key in the varm slot.
    wsi_col: str
        The column name of the whole slide image paths.
    store_col: str
        The column name of the backed file.
    error: str
        Whether to raise error when file not existed.

    Returns
    -------
    AnnData
        The aggregated feature space.
    """
    if wsi_col is None and store_col is None:
        raise ValueError("Either wsi_col or backed_file_col must be provided.")

    if store_col is not None:
        backed_files = slides_table[store_col]
    elif wsi_col is not None:
        backed_files = slides_table[wsi_col].apply(
            lambda x: Path(x).with_suffix(".zarr")
        )

    jobs = []
    with ThreadPoolExecutor() as executor:
        for backed_f in backed_files:
            job = executor.submit(
                _agg_wsi, backed_f, feature_key, tile_key, agg_key, error
            )
            jobs.append(job)

    results = []
    for job in track(
        jobs,
        description=f"Aggregation of {len(jobs)} slides",
    ):
        results.append(job.result())

    if error != "raise":
        mask = np.asarray([r is not None for r in results])
        X = np.vstack(np.asarray(results, dtype=object)[mask])
        slides_table = slides_table[mask]

    else:
        X = np.vstack(results)

    # Convert index to string
    slides_table.index = slides_table.index.astype(str)
    return AnnData(X, obs=slides_table)


def _agg_wsi(f, feature_key, tile_key, agg_key, error="raise"):
    """
    Aggregate feature from a whole slide image.

    Parameters
    ----------
    f: str
        The backed file path to the anndata file stored as .zarr.
    feature_key: str
        The feature key on which aggregation should be run on.
    tile_key: str
        The tile key.
    agg_key:
        The output aggregation key in the varm slot.
    error: str
        Whether to raise error when file not existed.

    Returns
    -------
    np.ndarray
        The aggregated feature space.

    """
    if not Path(f).exists():
        if error == "raise":
            raise ValueError(f"File {f} not existed.")
        else:
            return None
    try:
        import zarr
        from anndata import read_zarr

        tables = zarr.open(f"{f}/tables")
        available_keys = list(tables.keys())
        if feature_key not in available_keys:
            feature_key = f"{feature_key}_{tile_key}"
        s = read_zarr(f"{f}/tables/{feature_key}")
        return s.varm[agg_key]
    except Exception as e:
        if error == "raise":
            raise e
        else:
            return None


def is_zarr_dir(path):
    """
    Detect if the given directory is a Zarr storage using the Zarr library.

    Parameters:
        path (str): The path to the directory.

    Returns:
        bool: True if the directory is a Zarr storage, False otherwise.
    """
    import zarr

    try:
        zarr.open_group(path, mode="r")
        return True
    except Exception:
        return False
